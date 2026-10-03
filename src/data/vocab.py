"""The compact vocabulary: the ids a corpus actually uses, and the map back."""

from collections.abc import Sequence

import torch
from transformers import WhisperTokenizerFast


class VocabMap:
    """Lossless remap of the Whisper ids that occur in the corpus onto [0, size).

    LibriSpeech is uppercase and unpunctuated, so it touches a small fraction of
    Whisper's 51,866 tokens. The rest are slots the simplex has to carry and the model can
    never use. Dropping them is exact — every id round-trips — and it is what makes a
    (B, L, V) field affordable on a laptop.

    Build it from the union of every split that will be encoded, or `encode` will trip on
    a validation token that never appeared in training.
    """

    def __init__(self, texts: Sequence[str], tokenizer: WhisperTokenizerFast) -> None:
        """Measure which Whisper ids `texts` uses and build both directions of the map."""
        used = sorted({i for ids in tokenizer(list(texts)).input_ids for i in ids})
        self.to_full = torch.tensor(used, dtype=torch.long)
        rev = torch.full((len(tokenizer),), -1, dtype=torch.long)
        rev[self.to_full] = torch.arange(len(used))
        self.to_compact = rev
        self.size = len(used)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Whisper ids -> compact ids."""
        out = self.to_compact.to(ids.device)[ids]
        if not bool((out >= 0).all()):
            raise ValueError(
                "token outside the compact vocabulary — build the VocabMap from the "
                "union of every split you encode, not from train alone"
            )
        return out

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        """Compact ids -> Whisper ids."""
        return self.to_full.to(ids.device)[ids]
