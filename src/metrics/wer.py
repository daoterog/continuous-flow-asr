"""Corpus WER binned by flow time."""

from typing import TYPE_CHECKING

import torch
from normalizer import data_utils

from .base import Edges, TimeBinnedMetric, edit_counts, ids_to_text

if TYPE_CHECKING:
    from transformers import WhisperTokenizerFast

    from src.dataset import VocabMap


class BinnedWER(TimeBinnedMetric):
    """Corpus WER as a function of the flow time t, per bin.

    Accumulates edit counts rather than per-utterance rates. Corpus WER is
    sum(errors)/sum(ref_words) — averaging per-utterance WERs is a different number, and
    would let a three-word utterance weigh as much as a thirty-word one. Summing the
    counts and dividing at the end is exactly equal to scoring a whole bin in one call,
    but costs two floats per bin instead of every string seen in the epoch.
    """

    # The contract the training loop drives every binned metric through: `inputs` names
    # the keys of the per-batch record it wants fed to update(), and label/ylim let the
    # plotting helpers annotate the axis without knowing what the metric is.
    inputs = ("pred_ids", "target_ids", "t")
    label = "WER %"
    ylim = (0, 105)

    def __init__(
        self,
        tokenizer: "WhisperTokenizerFast",
        pad_id: int,
        vocab_map: "VocabMap | None" = None,
        num_bins: int = 10,
        edges: Edges = None,
        **kwargs: object,
    ) -> None:
        """Hold what decoding needs, plus a running error/word count per t-bin.

        vocab_map is needed because update() has to turn compact ids back into Whisper
        ids before the tokenizer sees them.
        """
        super().__init__(num_bins=num_bins, edges=edges, **kwargs)
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.vocab_map = vocab_map
        self.add_state("errors", default=torch.zeros(self.num_bins), dist_reduce_fx="sum")
        self.add_state("ref_words", default=torch.zeros(self.num_bins), dist_reduce_fx="sum")

    def update(
        self, pred_ids: torch.Tensor, target_ids: torch.Tensor, t: torch.Tensor
    ) -> None:
        """Accumulate one batch. pred_ids/target_ids: (B, L) ids; t: (B,) flow times."""
        preds = ids_to_text(pred_ids, self.tokenizer, self.pad_id, self.vocab_map)
        refs = ids_to_text(target_ids, self.tokenizer, self.pad_id, self.vocab_map)

        for b, ref, pred in zip(self._bin_indices(t).tolist(), refs, preds, strict=True):
            ref, pred = data_utils.normalizer(ref), data_utils.normalizer(pred)
            if not ref.strip():
                continue
            errors, ref_words = edit_counts(ref, pred)
            self.errors[b] += errors
            self.ref_words[b] += ref_words

    def compute(self) -> torch.Tensor:
        """(num_bins,) percent WER; empty bins report 0."""
        return 100 * self.errors / self.ref_words.clamp(min=1)

    def seen(self) -> torch.Tensor:
        """(num_bins,) bool — which bins actually received a sample."""
        return self.ref_words > 0

    def overall(self) -> float:
        """Corpus WER across every bin — the number to compare against the harness."""
        total = self.ref_words.sum()
        if total == 0:
            return float("nan")
        return (100 * self.errors.sum() / total).item()
