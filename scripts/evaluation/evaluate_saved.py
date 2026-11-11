"""Run train.py's end-of-training evaluation on a run that has already finished.

`train.evaluate_saved` is what a run calls once `fit` returns: it reloads `_best` and
`_final` from the components directory and samples each on val and test. This calls the
same function on a directory written earlier, so the path can be checked against a run
whose numbers are already known — the reloaded `_final` sampled on val must match the
`sampled_*/wer` the run logged at its last sampling epoch, since the weights, the curve and
the probe's noise seed are all the same.

    python -m scripts.evaluation.evaluate_saved --components runs/<id>/components
    python -m scripts.evaluation.evaluate_saved --components ... data.env=snellius
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import train as T
from scripts.evaluation.sample_eval import _config_from_components


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild the run from its components and evaluate it as train.py would."""
    flags, overrides = T.split_overrides(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--roles", nargs="+", default=["val", "test"])
    args = p.parse_args(flags)
    args.config, args.overrides = None, overrides

    cfg = _config_from_components(args)
    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    model = T.build_model(dm, cfg)
    T.evaluate_saved(model, dm, args.components, logger=None, roles=args.roles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
