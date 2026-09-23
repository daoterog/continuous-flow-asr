"""Continuous-flow ASR.

Importing anything under ``src`` runs this first, which is what makes the two
import-time side effects in ``src.constants`` reliable: ``open_asr_leaderboard`` lands on
``sys.path`` before any module body says ``from normalizer import ...``, and ``HF_HOME``
is set before ``transformers`` resolves its cache. Doing it here rather than at the top
of each module also keeps the import blocks in plain isort order.
"""

from . import constants  # noqa: F401
