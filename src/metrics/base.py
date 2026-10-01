"""Shared machinery for the time-binned metrics: bin geometry, decoding, edit counts."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from src.constants import PREFIX_LEN

import numpy as np
import torch
from kaldialign import batch_error_rate
from normalizer import data_utils
from torchmetrics import Metric

if TYPE_CHECKING:  # avoids a runtime import of the dataset module from a metric
    from transformers import WhisperTokenizerFast

    from src.dataset import VocabMap

# torch tensor, numpy array or plain sequence — anything torch.as_tensor accepts.
Edges = torch.Tensor | np.ndarray | list[float] | None


def schedule_bin_edges(
    sample_t_fn: Callable | None, num_bins: int = 10
) -> torch.Tensor:
    """Bin edges following the training schedule, so no bin sits empty.

    flow_step draws u ~ U[0, 1] and takes t = sample_t_fn(u), which makes sample_t_fn the
    quantile function of t: the edges tau^-1(k/n) ARE the k/n quantiles, by construction.
    Equal-width bins are the wrong grid under that schedule — the reparameterization puts
    nothing below the dead zone, so ~65% of the bins stay empty and every sample crowds
    into the top two or three.

    Pass sample_t_fn=None (or leave it off) for plain equal-width bins.
    """
    if sample_t_fn is None:
        return torch.linspace(0, 1, num_bins + 1)
    edges = np.asarray(sample_t_fn(np.linspace(0.0, 1.0, num_bins + 1)), dtype=np.float64)
    if not np.all(np.diff(edges) > 0):
        raise ValueError(
            f"sample_t_fn is flat somewhere on [0, 1], so {num_bins} schedule bins collapse "
            f"onto duplicate edges:\n  {np.round(edges, 4)}\nAsk for fewer bins."
        )
    return torch.as_tensor(edges, dtype=torch.float32)


class TimeBinnedMetric(Metric):
    """Base for metrics. All they share is the t -> bin mapping.

    Equal-width bins by default. Pass `edges` to bin some other way; in particular
    schedule_bin_edges(sample_t_fn, n), which is what keeps every bin populated once t is
    drawn through a reparameterization rather than uniformly.

    `num_bins=1` is the degenerate case, and is how a quantity that cannot vary with t is
    measured — mu_psi's forward takes audio and nothing else, so its prediction is the
    same whichever bin its batch drew. One bin makes `overall()` the whole quantity, and
    `metrics.utils.is_binned` reads False, which suppresses the histogram and the
    low_t/high_t pair downstream.
    """

    def __init__(self, num_bins: int = 10, edges: Edges = None, **kwargs: object) -> None:
        """Build the bin geometry; `edges` overrides `num_bins` when given."""
        super().__init__(**kwargs)
        if edges is None:
            edges = torch.linspace(0, 1, num_bins + 1)
            self.schedule_bins = False
        else:
            edges = torch.as_tensor(edges, dtype=torch.float32)
            if not bool((edges[1:] > edges[:-1]).all()):
                raise ValueError(f"bin edges must be strictly increasing, got {edges}")
            num_bins = edges.numel() - 1
            self.schedule_bins = True
        self.num_bins = num_bins
        self._edges = edges
        # Running sum and count of the t values seen per bin, so bin_centers can report
        # where the samples actually were rather than assuming they filled the bin evenly.
        self.add_state("t_sum", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("t_seen", default=torch.zeros(num_bins), dist_reduce_fx="sum")

    def _bin_indices(self, t: torch.Tensor) -> torch.Tensor:
        """Map t to the index of the bin [edges[k], edges[k+1]) it falls in.

        right=True so a t landing exactly on an edge goes to the bin above it, matching
        the half-open convention (and the floor that equal-width binning used to do).

        Records each t as it goes. Every subclass calls this exactly once per update, so
        accumulating here is what keeps `bin_centers` honest without each of them having
        to remember to do it.
        """
        t = t.float()
        idx = torch.bucketize(t, self._edges[1:-1].to(t.device), right=True)
        self.t_sum.scatter_add_(0, idx, t)
        self.t_seen.scatter_add_(0, idx, torch.ones_like(t))
        return idx

    def set_edges(self, edges: Edges) -> None:
        """Re-bin in place, keeping the same number of bins.

        Used when tau is refreshed mid-run: the schedule's quantiles move, so the bins
        have to move with them or they no longer hold ~1/n of the samples each. The bin
        count is fixed because the accumulator states are sized by it — call reset()
        first, since counts collected under the old edges do not describe the new ones.
        """
        edges = torch.as_tensor(edges, dtype=torch.float32)
        if edges.numel() - 1 != self.num_bins:
            raise ValueError(
                f"re-binning must keep {self.num_bins} bins, got {edges.numel() - 1}"
            )
        if not bool((edges[1:] > edges[:-1]).all()):
            raise ValueError(f"bin edges must be strictly increasing, got {edges}")
        self._edges = edges
        self.schedule_bins = True

    def bin_edges(self) -> torch.Tensor:
        """(num_bins + 1,) the bin boundaries in t."""
        return self._edges.detach().cpu()

    def bin_centers(self) -> torch.Tensor:
        """(num_bins,) the mean t seen in each bin, falling back to its midpoint.

        The midpoint is only the representative t if the samples fill the bin evenly, and
        under a schedule they do not: bins are quantiles of u, so t inside one is spread
        as tau^-1 across a uniform slice of u, and the widths differ by an order of
        magnitude — the lowest bin can span [0.19, 0.66] where the top ones span ~0.01.
        Attributing a bin's measured recovery to its midpoint then places it at a t the
        samples were never near, which matters because that pairing is what the measured
        tau is rebuilt from.

        Empty bins have no samples to average, so they keep the midpoint; before the first
        update that is every bin.
        """
        edges = self.bin_edges()
        midpoints = (edges[:-1] + edges[1:]) / 2
        seen = self.t_seen.detach().cpu()
        mean_t = self.t_sum.detach().cpu() / seen.clamp(min=1)
        return torch.where(seen > 0, mean_t, midpoints)


def ids_to_text(
    ids: torch.Tensor,
    tokenizer: "WhisperTokenizerFast",
    pad_id: int,
    vocab_map: "VocabMap | None" = None,
) -> list[str]:
    """Turn canvas token ids into transcripts, cut at the first <|endoftext|>.

    Cutting rather than relying on skip_special_tokens alone: an untrained model emits
    tokens all over the canvas, and everything past the first end-of-text is not part of
    the transcript a sampler would have produced.

    vocab_map is explicit rather than a module global — compact ids have to become
    Whisper ids before decoding, and getting it wrong decodes silently wrong text.
    """
    if vocab_map is not None:
        ids = vocab_map.decode(ids)
        pad_id = vocab_map.to_full[pad_id].item()

    texts = []
    for row in ids.tolist():
        tail = row[PREFIX_LEN:]
        if pad_id in tail:
            row = row[: PREFIX_LEN + tail.index(pad_id)]
        texts.append(
            tokenizer.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
    return texts


def edit_counts(ref: str, pred: str) -> tuple[int, int]:
    """Return (errors, reference words) for one normalized pair, as score_results counts.

    kaldialign with merge_compounds=True, not jiwer. This is what the leaderboard scores
    with, and it forgives split compounds in either direction — "white paper" vs
    "whitepaper" is 0 errors here and 33% under jiwer. score_eval.py prints a jiwer
    column too, but only as a cross-check against the pre-kaldialign CSV; the kaldialign
    column is the one --expect asserts on and the one the published figures came from.
    """
    r = batch_error_rate([tuple(ref.split())], [tuple(pred.split())], merge_compounds=True)
    return r["total"], r["ref_len"]


def corpus_wer(refs: list[str], preds: list[str]) -> float:
    """Percent corpus WER, the way scripts/score_eval.py's kaldialign column computes it.

    Named `corpus_wer`, not `wer`: `src.metrics.wer` is the submodule holding BinnedWER,
    and a function of that name would be unreachable through the package.

    Whisper's EnglishTextNormalizer folds case, strips punctuation, expands contractions
    and rewrites number words as digits, so it must be applied to both sides — and never
    to the training targets, which stay raw.
    """
    pairs = [
        (data_utils.normalizer(r), data_utils.normalizer(p))
        for r, p in zip(refs, preds, strict=True)
    ]
    # An empty reference has no words to divide by; the leaderboard drops them at write
    # time, and kaldialign reports err_rate=inf for one.
    pairs = [(r, p) for r, p in pairs if r.strip()]
    r = batch_error_rate(
        [tuple(r.split()) for r, _ in pairs],
        [tuple(p.split()) for _, p in pairs],
        merge_compounds=True,
    )
    return 100 * r["err_rate"]


@torch.no_grad()
def score_samples(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    tokenizer: "WhisperTokenizerFast",
    pad_id: int,
    vocab_map: "VocabMap | None" = None,
) -> tuple[float, float]:
    """(WER %, token error rate) for sampled ids against the references.

    Both sides go through the decode and normalisation the binned WER uses, so the number
    is comparable to what `scripts/score_eval.py` reports.

    The token error counts only positions whose target is a real token. Averaging over the
    whole canvas instead would fold in the padding tail, which the model learns first and
    which dominates the count — on a 224-wide canvas over ~43-token utterances it is 81% of
    positions, so getting every real token wrong would still report 0.19 rather than 1.0.
    """
    preds = ids_to_text(pred_ids, tokenizer, pad_id, vocab_map)
    refs = ids_to_text(target_ids, tokenizer, pad_id, vocab_map)
    pairs = [
        (r, p)
        for r, p in zip(refs, preds, strict=True)
        if data_utils.normalizer(r).strip()
    ]
    wer = corpus_wer([r for r, _ in pairs], [p for _, p in pairs]) if pairs else float("nan")
    scored = target_ids != pad_id
    n_scored = int(scored.sum())
    token_error = (
        float("nan")
        if n_scored == 0
        else (((pred_ids != target_ids) & scored).sum() / n_scored).item()
    )
    return wer, token_error
