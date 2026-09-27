"""Print the parameter count and training footprint of every size in configs/sizes/.

Needs no GPU and no data: the models are built on the meta device, so nothing is
allocated and `large` costs the same to inspect as `tiny`. Use it to decide which sizes
are worth a real timing run before spending a job on scripts/bench_train.py.

    python -m scripts.model_sizes
    python -m scripts.model_sizes --vocab 51866   # what DRAX report, for comparison

The optimizer column is the fixed cost: under AMP, Lightning keeps fp32 parameters and
gradients, and AdamW adds two fp32 moments, so 16 bytes per parameter before a single
activation. Activations depend on batch and canvas and are what bench_train measures.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import src  # noqa: F401, E402  # sets HF_HOME before transformers is imported
from src.models import build_audio_decoder, build_denoiser  # noqa: E402

SIZES_DIR = REPO_ROOT / "configs" / "sizes"
MU_DIR = REPO_ROOT / "configs" / "mu"
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
BYTES_PER_PARAM = 16  # fp32 weight + fp32 grad + two fp32 AdamW moments


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Which vocabulary to size the embedding and output tables against."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--vocab", type=int, default=1192,
        help="vocabulary width. The default is the compact map's size; pass 51866 to see "
             "the same geometries over Whisper's full vocabulary",
    )
    p.add_argument(
        "--tokens", type=float, default=1.35e6,
        help="training tokens to report params-per-token against; the default is "
             "train.100 (28539 utts at ~47 tokens each)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build each size on meta and print its parameter table."""
    args = parse_args(argv)
    paths = sorted(SIZES_DIR.glob("*.yaml"))
    if not paths:
        print(f"no size configs under {SIZES_DIR}")
        return 1

    # A size file carries geometry only; dropout and whisper_dim live in base.yaml, and
    # --size layers over it the same way at train time.
    base = OmegaConf.load(BASE_CONFIG).model.denoiser
    rows = []
    for path in paths:
        denoiser = OmegaConf.merge(base, OmegaConf.load(path).model.denoiser)
        with torch.device("meta"):
            model = build_denoiser(denoiser, args.vocab)
        total = sum(p.numel() for p in model.parameters())
        blocks = sum(p.numel() for p in model.blocks.parameters())
        rows.append((path.stem, denoiser, total, blocks))

    rows.sort(key=lambda r: r[2])
    print(f"vocab {args.vocab}, {args.tokens / 1e6:.2f}M training tokens\n")
    print(f"{'size':<8} {'hidden':>6} {'blk':>4} {'heads':>6} {'params':>9} "
          f"{'in blocks':>10} {'optim GiB':>10} {'params/token':>13}")
    for name, cfg, total, blocks in rows:
        print(f"{name:<8} {cfg.hidden_size:>6} {cfg.n_blocks:>4} {cfg.n_heads:>6} "
              f"{total / 1e6:>8.1f}M {100 * blocks / total:>9.0f}% "
              f"{total * BYTES_PER_PARAM / 2**30:>10.2f} {total / args.tokens:>13.0f}")

    mu_paths = sorted(MU_DIR.glob("*.yaml"))
    if not mu_paths:
        return 0
    # mu's canvas width decides its learned query table, so it is a real parameter here
    # rather than a detail — unlike the denoiser, which is RoPE and carries no positions.
    canvas = int(OmegaConf.load(BASE_CONFIG).data.canvas_len)
    mu_rows = []
    for path in mu_paths:
        node = OmegaConf.load(path).model.audio_decoder
        with torch.device("meta"):
            model = build_audio_decoder(node, args.vocab, canvas)
        total = sum(p.numel() for p in model.parameters())
        head = sum(p.numel() for p in model.head.parameters())
        mu_rows.append((path.stem, node, total, head))

    mu_rows.sort(key=lambda r: r[2])
    print(f"\nmu_psi (--mu), canvas {canvas}\n")
    print(f"{'mu':<8} {'hidden':>6} {'blk':>4} {'heads':>6} {'params':>9} "
          f"{'in head':>9} {'optim GiB':>10}")
    for name, cfg, total, head in mu_rows:
        print(f"{name:<8} {cfg.hidden:>6} {cfg.n_blocks:>4} {cfg.n_heads:>6} "
              f"{total / 1e6:>8.1f}M {100 * head / total:>8.0f}% "
              f"{total * BYTES_PER_PARAM / 2**30:>10.2f}")
    if args.vocab < 10_000:
        print("\nThe 'in head' column is why DRAX's '28M for p_mid' does not transfer: at")
        print("their 51,866-token vocabulary the projection is ~85% of that number, so 28M")
        print("buys one transformer block. Run with --vocab 51866 to see it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
