"""Turning canvas token ids into transcripts, and scoring them the way the leaderboard does."""

from typing import TYPE_CHECKING

from src.constants import PREFIX_LEN

import torch
from kaldialign import batch_error_rate
from normalizer import data_utils

if TYPE_CHECKING:
    from transformers import WhisperTokenizerFast

    from src.data import VocabMap


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
    """Percent corpus WER, the way scripts/evaluation/score_eval.py's kaldialign column computes it.

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
    is comparable to what `scripts/evaluation/score_eval.py` reports.

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
