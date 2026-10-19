"""Does a second pass over the same utterances come back from memory, per read method?

Training reads random utterances; whether the job's RAM holds them after the first epoch
decides whether train.960 is limited by GPFS or by the GPU. Two ways to read a slice:
`mmap`, faulting it in through a memory map (`np.load(mmap_mode="r")`), and `pread`, one
positioned read at its byte offset (`np.fromfile(offset=...)`). Each method gets its own
random set of utterances, read `--passes` times by a pool of worker processes the way the
DataLoader's workers do; pass 1 is cold, and later passes are as fast as the cache makes
them.

    python -m scripts.diagnostics.bench_reads --n 12000 --workers 16 --passes 3
"""

import argparse
import glob
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = "/scratch-shared/cont-flow-asr/data/features/whisper-large-v3"
_maps: dict[str, np.ndarray] = {}  # per worker process, as in AudioFeatureDataset


def _shard(path: str) -> np.ndarray:
    if path not in _maps:
        _maps[path] = np.load(path, mmap_mode="r")
    return _maps[path]


def via_mmap(u: tuple[str, int, int]) -> int:
    """Copy the slice off a memory map that stays open in this worker."""
    path, offset, length = u
    return np.array(_shard(path)[offset : offset + length]).nbytes


def via_pread(u: tuple[str, int, int]) -> int:
    """Read the slice with one positioned read, the header taken from the open map."""
    path, offset, length = u
    a = _shard(path)
    d = a.shape[1]
    start = a.offset + offset * d * a.itemsize
    return np.fromfile(path, a.dtype, count=length * d, offset=start).nbytes


def main() -> int:
    """Time each method's passes over its own random utterances."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--n", type=int, default=12000, help="utterances per method")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--passes", type=int, default=3)
    args = p.parse_args()

    utts = []
    for idx in sorted(glob.glob(f"{args.root}/*/train.*/[0-9]*.json")):
        if not Path(idx).with_suffix(".npy").exists():  # an index-only directory
            continue
        with open(idx) as f:
            index = json.load(f)
        utts += [(idx[:-5] + ".npy", u["offset"], u["length"]) for u in index["utts"]]
    random.seed(0)
    random.shuffle(utts)
    print(f"{len(utts)} utterances; {args.n} per method, {args.workers} workers", flush=True)

    with Pool(args.workers) as pool:
        for i, (name, fn) in enumerate((("mmap", via_mmap), ("pread", via_pread))):
            chunk = utts[i * args.n : (i + 1) * args.n]
            for k in range(args.passes):
                random.shuffle(chunk)
                t = time.time()
                b = sum(pool.imap_unordered(fn, chunk, chunksize=4))
                dt = time.time() - t
                print(f"{name:6s} pass {k + 1}  {len(chunk) / dt:6.0f} utt/s  "
                      f"{b / dt / 1e9:5.2f} GB/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
