"""Time-binned metrics: loss, token error and WER as functions of the flow time t.

`base` is the bin geometry they share, `scoring` turns ids into scored transcripts,
`loss`, `token_error` and `wer` are the metrics themselves, and `utils` builds and
validates a bank of them. Measuring only — what reaches a logger is `src.reporting`.
"""

from .base import TimeBinnedMetric, schedule_bin_edges
from .loss import BinnedLoss
from .scoring import corpus_wer, edit_counts, ids_to_text, score_samples
from .token_error import BinnedTokenError
from .utils import (
    SPLITS,
    check_metrics,
    edge_values,
    make_metrics,
    section_for,
    split_key,
)
from .wer import BinnedWER

__all__ = [
    "SPLITS",
    "BinnedLoss",
    "BinnedTokenError",
    "BinnedWER",
    "TimeBinnedMetric",
    "check_metrics",
    "corpus_wer",
    "edge_values",
    "edit_counts",
    "ids_to_text",
    "make_metrics",
    "schedule_bin_edges",
    "score_samples",
    "section_for",
    "split_key",
]
