"""How fast do random utterances come off GPFS, by how many reads are in flight?

Training reads random utterances through memory maps, one at a time per DataLoader worker,
so a job keeps as many reads in flight as it has workers. A node reading at 0.43 GB/s with
16 of them is either out of bandwidth or waiting on latency; only the second gets faster
with more reads in flight. Each configuration `PxT` reads with P worker processes of T
threads each (P*T reads in flight), on its own disjoint set of random utterances, so its
first pass is cold; the second pass says whether the reads stayed in the job's RAM.

    python -m scripts.diagnostics.bench_reads --n 3000 --configs 16x1 32x1 64x1 16x4 16x8
"""

import argparse
import glob
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = "/scratch-shared/cont-flow-asr/data/features/whisper-large-v3"
_maps: dict[str, np.ndarray] = {}  # per worker process, as in AudioFeatureDataset
_threads: ThreadPoolExecutor | None = None


def _shard(path: str) -> np.ndarray:
    if path not in _maps:
        _maps[path] = np.load(path, mmap_mode="r")
    return _maps[path]


def read_one(u: tuple[str, int, int]) -> int:
    """Copy one utterance off its memory map, the way AudioFeatureDataset does."""
    path, offset, length = u
    return np.array(_shard(path)[offset : offset + length]).nbytes


def _init(threads: int) -> None:
    global _threads
    _threads = ThreadPoolExecutor(threads) if threads > 1 else None


def read_chunk(chunk: list[tuple[str, int, int]]) -> int:
    """One worker's share: through its threads when it has them, else one at a time."""
    if _threads is None:
        return sum(read_one(u) for u in chunk)
    return sum(_threads.map(read_one, chunk))


def main() -> int:
    """Time a cold and a warm pass for each configuration."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default=ROOT)
    p.add_argument("--n", type=int, default=3000, help="utterances per configuration")
    p.add_argument("--configs", nargs="+", default=["16x1", "32x1", "64x1", "16x4", "16x8"],
                   help="PxT: P worker processes of T threads each")
    p.add_argument("--chunk", type=int, default=32, help="utterances per task")
    p.add_argument("--seed", type=int, default=0,
                   help="picks the utterances; change it between jobs, since a node can keep "
                        "a previous job's reads in RAM and make a cold pass look warm")
    args = p.parse_args()

    utts = []
    for idx in sorted(glob.glob(f"{args.root}/*/train.*/[0-9]*.json")):
        if not Path(idx).with_suffix(".npy").exists():  # an index-only directory
            continue
        with open(idx) as f:
            index = json.load(f)
        utts += [(idx[:-5] + ".npy", u["offset"], u["length"]) for u in index["utts"]]
    random.seed(args.seed)
    random.shuffle(utts)
    print(f"{len(utts)} utterances; {args.n} fresh ones per configuration", flush=True)
    if len(args.configs) * args.n > len(utts):
        raise SystemExit("not enough utterances for disjoint sets; lower --n")

    for k, config in enumerate(args.configs):
        procs, threads = (int(x) for x in config.split("x"))
        mine = utts[k * args.n : (k + 1) * args.n]
        with Pool(procs, initializer=_init, initargs=(threads,)) as pool:
            for name in ("cold", "warm"):
                random.shuffle(mine)
                chunks = [mine[i : i + args.chunk] for i in range(0, len(mine), args.chunk)]
                t = time.time()
                b = sum(pool.imap_unordered(read_chunk, chunks))
                dt = time.time() - t
                print(f"{config:>6}  {procs * threads:>4} in flight  {name}  "
                      f"{len(mine) / dt:6.0f} utt/s  {b / dt / 1e9:5.2f} GB/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
