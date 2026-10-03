"""Report LibriSpeech clip durations, and how many exceed Whisper's 30 s window.

Whisper's feature extractor pads or truncates every clip to exactly 30 s (480,000 samples
-> 3000 mel frames -> 1500 encoder frames). LibriSpeech's training sets were segmented at
silences longer than 0.3 s with a **35 s** cap (Panayotov et al., 2015), so clips can be
longer than the window. Those keep their full transcript while losing the audio past 30 s,
which is an example no model can fit: the text describes speech that is not in the input.

The precompute records `valid` frames only when trimming, so an untrimmed cache (NO_TRIM=1,
which is what the flow needs) cannot answer this after the fact. This reads the source
audio instead, and only the FLAC headers, so it never decodes a sample.

    python -m scripts.data.audio_durations --configs clean --splits train.100
    python -m scripts.data.audio_durations --configs clean other \
        --splits train.100 train.360 train.500 validation test
"""

import argparse
import io
import sys
from collections.abc import Sequence
from pathlib import Path


# Importing from src runs src/__init__.py, which sets HF_HOME; `datasets` is imported
# inside durations() so it cannot beat this to the environment variable.
from src.constants import CONFIGS, SPLITS, data_root_for

import numpy as np
import soundfile as sf

WHISPER_WINDOW_S = 30.0
LIBRISPEECH_CAP_S = 35.0  # the corpus paper's segmentation limit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Which (config, split) pairs to measure, and where the corpus lives."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=["clean"])
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=["train.100"])
    p.add_argument("--env", choices=["local", "snellius"], default="local")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument(
        "--limit", type=int, default=None,
        help="measure only this many clips per split, for a quick look",
    )
    return p.parse_args(argv)


def durations(config: str, split: str, limit: int | None) -> np.ndarray:
    """Seconds per clip, read from FLAC headers without decoding audio."""
    from datasets import Audio, load_dataset

    ds = load_dataset("openslr/librispeech_asr", config, split=split)
    # decode=False hands back the raw bytes, so soundfile reads the header alone.
    ds = ds.cast_column("audio", Audio(decode=False))
    if limit is not None:
        ds = ds.select(range(min(limit, ds.num_rows)))

    out = np.empty(ds.num_rows, dtype=np.float64)
    for i, row in enumerate(ds):
        info = sf.info(io.BytesIO(row["audio"]["bytes"]))
        out[i] = info.frames / info.samplerate
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Print per-split duration percentiles and the over-window counts."""
    args = parse_args(argv)
    root = args.data_root or data_root_for(args.env)
    print(f"corpus root {root}   Whisper window {WHISPER_WINDOW_S:.0f}s\n")
    print(f"{'split':<22} {'clips':>7} {'mean':>6} {'p99':>6} {'max':>6} "
          f"{'>30s':>7} {'>30s %':>7} {'audio lost':>11}")

    everything = []
    for config in args.configs:
        for split in args.splits:
            try:
                d = durations(config, split, args.limit)
            except Exception as exc:  # a split that was never downloaded
                first = " ".join(str(exc).split("\n")[0].split())
                print(f"{config + '/' + split:<22} SKIPPED {first[:60]}")
                continue
            everything.append(d)
            over = d > WHISPER_WINDOW_S
            lost = (d[over] - WHISPER_WINDOW_S).sum()
            print(f"{config + '/' + split:<22} {len(d):>7} {d.mean():>6.2f} "
                  f"{np.percentile(d, 99):>6.2f} {d.max():>6.2f} {over.sum():>7} "
                  f"{100 * over.mean():>6.2f}% {lost / 60:>9.1f}m")

    if not everything:
        return 1
    d = np.concatenate(everything)
    over = d > WHISPER_WINDOW_S
    print(f"\n{'ALL':<22} {len(d):>7} {d.mean():>6.2f} {np.percentile(d, 99):>6.2f} "
          f"{d.max():>6.2f} {over.sum():>7} {100 * over.mean():>6.2f}%")
    if d.max() > LIBRISPEECH_CAP_S + 0.5:
        print(f"\nNOTE: max {d.max():.2f}s exceeds the corpus paper's {LIBRISPEECH_CAP_S}s "
              f"segmentation cap — worth checking before trusting it.")
    if over.any():
        print(f"\n{over.sum()} clips are truncated by the extractor and keep their full "
              f"transcript.\nThey ask the model to produce words whose audio is absent; "
              f"drop them or split them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
