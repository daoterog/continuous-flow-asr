"""Upload a run's saved components to its wandb run, after the fact.

Training writes `runs/<run_id>/components/` locally and does not upload it: a cluster job
should not depend on the network to keep its weights, and most runs are never looked at
again. This pushes one run's files up when you do want them there — to hand a checkpoint
to someone, or to pull it from a different machine.

    python -m scripts.upload_components <run_id>
    python -m scripts.upload_components <run_id> --variant best --include-training-state

The run id is the wandb one, which is also the directory name. Files land under
`individual_components/` on the run so they are easy to spot among wandb's own.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wandb  # noqa: E402

from src.checkpoint import (  # noqa: E402
    COMPONENTS,
    MANIFEST,
    TRAINING_STATE,
    VOCAB_MAP,
    load_manifest,
    run_dir,
)

REMOTE_DIR = "individual_components"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Which run to upload, and how much of it."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_id", help="wandb run id, which is also the local directory name")
    p.add_argument("--save-dir", default="runs", help="where training wrote the run")
    p.add_argument(
        "--variant", default="all", choices=["all", "best", "final"],
        help="which weights to send; the manifest and vocab map always go",
    )
    p.add_argument(
        "--include-training-state", action="store_true",
        help=f"also send {TRAINING_STATE}, which is large and only useful for resuming",
    )
    p.add_argument("--entity", default="cont-flow-asr-")
    p.add_argument("--project", default="continuous-flow-asr")
    p.add_argument("--dry-run", action="store_true", help="list what would be sent")
    return p.parse_args(argv)


def files_to_send(directory: Path, variant: str, training_state: bool) -> list[Path]:
    """The manifest and vocab map, plus the requested weights."""
    chosen = [directory / MANIFEST]
    vocab = directory / VOCAB_MAP
    if vocab.exists():
        chosen.append(vocab)
    variants = ("best", "final") if variant == "all" else (variant,)
    for name in COMPONENTS:
        for v in variants:
            path = directory / f"{name}_{v}.pth"
            if path.exists():
                chosen.append(path)
    state = directory / TRAINING_STATE
    if training_state and state.exists():
        chosen.append(state)
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the run, list the files, and upload unless this is a dry run."""
    args = parse_args(argv)
    directory = run_dir(args.run_id, args.save_dir)
    manifest = load_manifest(directory)
    chosen = files_to_send(directory, args.variant, args.include_training_state)

    total = sum(p.stat().st_size for p in chosen)
    print(f"run {args.run_id}  (variant={manifest.get('variant')}, "
          f"size={manifest.get('size')}, mu={manifest.get('mu_size')}, "
          f"vocab={manifest.vocab_size})")
    print(f"from {directory}\n")
    for path in chosen:
        print(f"  {path.name:<28} {path.stat().st_size / 2**20:>8.1f} MiB")
    print(f"  {'TOTAL':<28} {total / 2**20:>8.1f} MiB")
    if args.dry_run:
        print("\ndry run; nothing sent")
        return 0

    # resume="allow" attaches to the existing run rather than starting a new one, so the
    # files land on the page that produced them instead of an orphan.
    run = wandb.init(
        id=args.run_id, entity=args.entity, project=args.project,
        resume="allow", job_type="upload-components",
    )
    for path in chosen:
        run.save(str(path), base_path=str(directory.parent), policy="now")
    run.finish()
    print(f"\nsent {len(chosen)} files to {args.project}/{args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
