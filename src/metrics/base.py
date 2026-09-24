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
    """Base for the three metrics below — all they share is the t -> bin mapping.

    Equal-width bins by default. Pass `edges` to bin some other way; in particular
    schedule_bin_edges(sample_t_fn, n), which is what keeps every bin populated once t is
    drawn through a reparameterization rather than uniformly.
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
        # a plain attribute, not a Metric state: it is fixed geometry, nothing to reset
        self._edges = edges

    def _bin_indices(self, t: torch.Tensor) -> torch.Tensor:
        """Map t to the index of the bin [edges[k], edges[k+1]) it falls in.

        right=True so a t landing exactly on an edge goes to the bin above it, matching
        the half-open convention (and the floor that equal-width binning used to do).
        """
        return torch.bucketize(t.float(), self._edges[1:-1].to(t.device), right=True)

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
        """(num_bins,) the midpoint of each bin, for plotting against t."""
        edges = self.bin_edges()
        return (edges[:-1] + edges[1:]) / 2


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
