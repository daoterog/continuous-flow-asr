"""The compute a finished run spent, from its components directory: FLOPs per step and in total.

Runs from before `src.flops.FlopCount` existed logged no compute; this counts it for them
the same way the callback does — one real training step and one sampling call, through
`FlopCounterMode` — on a model rebuilt from the run's own config, at its own batch size and
latents. FLOPs depend on the shapes, not the weights, so nothing is loaded.

The total is per-step FLOPs times the steps the run took, read from its wandb run when
`--run` is given (`trainer/global_step` at the end), else `--steps`.

    python -m scripts.diagnostics.count_flops --components runs/<id>/components data.env=snellius
    python -m scripts.diagnostics.count_flops --components ... --run flow-asr-tau-schedule/<id>
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

import train as T
from scripts.evaluation.sample_eval import _config_from_components
from src.flops import count_flops

ENTITY = "cont-flow-asr-"


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild the run's model, count one step and one sampling call, print the totals."""
    flags, overrides = T.split_overrides(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--run", default=None, help="project/run_id, for the steps the run took")
    p.add_argument("--steps", type=int, default=None, help="steps taken, without --run")
    p.add_argument("--sample-steps", type=int, default=16)
    args = p.parse_args(flags)
    args.config, args.overrides = None, overrides

    cfg = _config_from_components(args)
    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    model = T.build_model(dm, cfg).to("cuda" if torch.cuda.is_available() else "cpu")
    batch = {
        k: (v.to(model.device) if torch.is_tensor(v) else v)
        for k, v in next(iter(dm.train_dataloader())).items()
    }
    flops = count_flops(model, batch, args.sample_steps)

    steps = args.steps
    if args.run:
        import wandb

        steps = int(wandb.Api().run(f"{ENTITY}/{args.run}").summary["trainer/global_step"])
    print(f"batch {batch['tokens'].size(0)}  frames {batch['audio'].size(1)}  V {dm.vocab_size}")
    print(f"train FLOPs per step         {flops['train_per_step']:.3e}")
    sample = flops["sample_per_utt"]
    print(f"sample FLOPs per utterance   {sample:.3e}  ({args.sample_steps} steps)")
    if steps:
        total = flops["train_per_step"] * steps
        print(f"train FLOPs total            {total:.3e}  ({steps:,} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
