"""Pull a run's components down from wandb, the other half of `upload_components`.

Training writes `runs/<run_id>/components/` on whichever machine it ran on, and
`scripts/upload_components.py` pushes that to the wandb run. This brings it back, so a
checkpoint trained on the cluster can be loaded on a laptop — which is the whole point of
the components format, and was the missing step.

Files land under the same `runs/<run_id>/components/` the training would have written, so
`load_components(run_dir(run_id))` works afterwards with no path juggling.

    python -m scripts.checkpointing.fetch_components q5bsvywz
    python -m scripts.checkpointing.fetch_components q5bsvywz --variant best --dry-run
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from src.checkpointing import COMPONENTS, MANIFEST, TRAINING_STATE, VOCAB_MAP

# What `upload_components` writes them under: it passes base_path=<run_dir>, so the run's
# own `components/` prefix survives. (Its docstring claims `individual_components/`; the
# constant of that name is declared there and never used.)
REMOTE_DIR = "components"


def wanted(variant: str, training_state: bool) -> list[str]:
    """Remote names to fetch: always the manifest and vocab map, plus the chosen weights."""
    names = [MANIFEST, VOCAB_MAP]
    variants = ("best", "final") if variant == "all" else (variant,)
    names += [f"{name}_{v}.pth" for name in COMPONENTS for v in variants]
    if training_state:
        names.append(TRAINING_STATE)
    return names


def backfill_run_config(out_dir: Path, run: "wandb.apis.public.Run") -> bool:
    """Add `run_config` to a manifest written before that key existed.

    `manifest_for` records the whole resolved config so a components directory is enough to
    rebuild the data pipeline, not just the models. Manifests from before that change carry
    only the eleven contract keys, and `sample_eval._config_from_components` and
    `measure_tau_bias` both read `manifest.run_config` — so an older teacher is unusable
    with them despite the weights being fine.

    The config is recoverable: `build_logger` passes the fully resolved cfg to wandb, so
    `run.config` is exactly what the manifest would have stored. Checked against the
    manifest's own contract keys before writing, since a mismatch would mean the run and the
    files have been paired up wrongly.
    """
    path = out_dir / MANIFEST
    manifest = OmegaConf.load(path)
    if "run_config" in manifest:
        return False
    cfg = OmegaConf.create(run.config)
    for key, got in (
        ("canvas_len", cfg.data.canvas_len),
        ("compact_vocab", cfg.data.compact_vocab),
        ("tokenizer", str(cfg.data.model)),
    ):
        if manifest.get(key) != got:
            raise SystemExit(
                f"wandb config disagrees with the manifest on {key}: "
                f"{got!r} vs {manifest.get(key)!r}. Refusing to backfill — the run and "
                f"these files may not belong together."
            )
    manifest.run_config = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(manifest, path)
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Which run to fetch, how much of it, and where to put it."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_id", help="wandb run id, which is also the local directory name")
    p.add_argument("--save-dir", default="runs", help="root to write the run under")
    p.add_argument(
        "--variant", default="all", choices=["all", "best", "final"],
        help="which weights to pull; the manifest and vocab map always come",
    )
    p.add_argument(
        "--include-training-state", action="store_true",
        help=f"also pull {TRAINING_STATE}, which is large and only useful for resuming",
    )
    p.add_argument("--entity", default="cont-flow-asr-")
    p.add_argument("--project", default="continuous-flow-asr")
    p.add_argument(
        "--force", action="store_true", help="re-download files that are already present"
    )
    p.add_argument(
        "--no-backfill", action="store_true",
        help="leave an older manifest without the run_config key it predates",
    )
    p.add_argument("--dry-run", action="store_true", help="list what would be pulled")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the run, match its files against what was asked for, and download them."""
    args = parse_args(argv)
    api = wandb.Api(timeout=60)
    run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    out_dir = Path(args.save_dir) / args.run_id / REMOTE_DIR

    remote = {Path(f.name).name: f for f in run.files() if f.name.startswith(f"{REMOTE_DIR}/")}
    if not remote:
        raise SystemExit(
            f"run {run.name} ({args.run_id}) has no {REMOTE_DIR}/ files. Upload them from "
            f"the machine that trained it:\n"
            f"  python -m scripts.upload_components {args.run_id} --variant {args.variant}"
        )

    names = wanted(args.variant, args.include_training_state)
    chosen = [remote[n] for n in names if n in remote]
    missing = [n for n in names if n not in remote and not n.endswith(".pth")]

    print(f"run        {run.name} ({args.run_id}), {run.state}")
    print(f"remote     {len(remote)} file(s) under {REMOTE_DIR}/")
    print(f"target     {out_dir}")
    if missing:
        # A compact-vocab checkpoint without its map is unusable, so say so rather than
        # leaving it to fail later at load time.
        print(f"  WARNING: not on the run: {', '.join(missing)}")
    print()
    total = 0
    for f in chosen:
        local = out_dir / Path(f.name).name
        have = local.exists() and local.stat().st_size == f.size
        total += 0 if have else f.size
        print(f"  {Path(f.name).name:<24} {f.size:>14,} bytes  "
              f"{'present' if have and not args.force else 'fetch'}")
    print(f"\n  {total:,} bytes to download")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in chosen:
        local = out_dir / Path(f.name).name
        if local.exists() and local.stat().st_size == f.size and not args.force:
            continue
        # root is the run directory: wandb recreates the `components/` prefix under it,
        # which lands the file exactly where training would have put it.
        f.download(root=str(out_dir.parent), replace=True)

    if not args.no_backfill and backfill_run_config(out_dir, run):
        print(f"\n  backfilled run_config into {MANIFEST} from the wandb run "
              f"(it predates that key)")

    print(f"\nwrote {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name:<24} {path.stat().st_size:>14,} bytes")
    print(
        f"\nload with:\n"
        f"  from src.checkpointing import load_components, run_dir\n"
        f"  manifest, built, tau = load_components(run_dir({args.run_id!r}, {args.save_dir!r}))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
