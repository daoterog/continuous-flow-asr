"""Back up the components of every run the paper uses, as wandb artifacts.

`runs/<id>/components/` lives only on the machine that trained it, and home is a 200 GiB
quota that has to be cleaned. This uploads each paper run's whole components directory —
weights, manifest, vocab map, tau history — as a `model` artifact `components-<id>` in the
run's own project, where `wandb artifact get` restores it. The training state is left out:
it is large and only resuming needs it.

The runs are every `run:` in paper/results.yaml plus the runs whose weights the paper's
figures load (WEIGHTS_USED); `--run project/id` adds others. A run whose artifact already
exists is skipped, so the command can be rerun after a failure.

    python -m scripts.checkpointing.backup_components --dry-run
    python -m scripts.checkpointing.backup_components
"""

import argparse
import re
import sys
from pathlib import Path

import wandb

from src.checkpointing import TRAINING_STATE

ENTITY = "cont-flow-asr-"
RESULTS = Path("paper/results.yaml")
# Loaded by jobs/batch/sample_trajectory.sh, probe_audio_info.sh and project_tau.
WEIGHTS_USED = [
    "flow-asr-tau-schedule/y9hyu9p0",  # ours
    "flow-asr-tau-schedule/16keaex7",  # linear
    "flow-asr-tau-schedule/1nz9v8pz",  # linear + audio dropout 0.1, CFG
    "flow-asr-tau-schedule/6n9h2pko",  # ours + audio dropout 0.05, CFG
    "flow-asr-mu-only/txuowwbe",  # mu-only
    "flow-asr-flm-parity/5v9lar7i",  # the audio-information probe
    "flow-asr-flm-parity/olc06gip",  # project_tau
]


def paper_runs(extra: list[str]) -> list[str]:
    """`project/id` for every run results.yaml names, the weights in use, and `extra`."""
    named = re.findall(r"run: ([\w-]+/[a-z0-9]{8})", RESULTS.read_text())
    return sorted(set(named) | set(WEIGHTS_USED) | set(extra))


def main() -> int:
    """List each run's components, then upload the ones not backed up yet."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--save-dir", type=Path, default=Path("runs"))
    p.add_argument("--run", action="append", default=[], help="another project/id")
    p.add_argument("--dry-run", action="store_true", help="list what would be sent")
    args = p.parse_args()

    api = wandb.Api()
    missing, done, total = [], [], 0
    for ref in paper_runs(args.run):
        project, run_id = ref.split("/")
        directory = args.save_dir / run_id / "components"
        if not directory.is_dir():
            missing.append(ref)
            continue
        name = f"{ENTITY}/{project}/components-{run_id}"
        if api.artifact_exists(f"{name}:latest"):
            done.append(ref)
            continue
        files = [f for f in directory.rglob("*") if f.is_file() and f.name != TRAINING_STATE]
        size = sum(f.stat().st_size for f in files)
        total += size
        print(f"{ref:<36} {len(files):>3} files  {size / 2**30:6.2f} GiB")
        if args.dry_run:
            continue
        run = wandb.init(
            entity=ENTITY, project=project, job_type="backup-components",
            name=f"backup-{run_id}",
        )
        artifact = wandb.Artifact(f"components-{run_id}", type="model",
                                  metadata={"run_id": run_id})
        # immutable: upload from the file where it is, instead of first copying it into
        # wandb's staging directory — a second copy of every weight, in a full home.
        for f in files:
            artifact.add_file(str(f), name=str(f.relative_to(directory)), policy="immutable")
        run.log_artifact(artifact)
        run.finish()

    print(f"\n{'would send' if args.dry_run else 'sent'} {total / 2**30:.2f} GiB")
    if done:
        print(f"already backed up: {len(done)}")
    if missing:
        print(f"no components directory here ({len(missing)}): {' '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
