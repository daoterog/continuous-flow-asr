"""Turn a Lightning .ckpt into the components/ directory the current code reads.

Runs up to `skilled-lake-22` wrote a `.ckpt` and no components; later ones the reverse
(jobs/batch/sample_eval.sh:21). So the best checkpoint we have is in the one format
`load_components` cannot read, and its config predates keys the current code expects. This
converts it, so one loader serves every run and the result uploads with the existing
`scripts/upload_components.py`.

Three things the conversion has to get right, none of them automatic:

  * **legacy key names.** `load_components` is `strict=True` and applies no remapping, so an
    older checkpoint fails on it. `load_state_into` does apply `strip_compile_prefix` and
    `remap_legacy_keys`, which is why the weights go through that path here.
  * **config drift.** A manifest records the whole resolved config, and `manifest_for` reads
    keys an old config may simply not have. Missing keys are filled from the current
    `configs/base.yaml` and every one of them is printed. Silent defaulting is how a batch
    of cluster runs was lost; this is loud on purpose.
  * **the vocabulary map.** A compact-vocab checkpoint is meaningless without it, and the
    `.ckpt` does not contain one — `save_hyperparameters` ignores it. It is rebuilt from the
    checkpoint's own data config, which also gives `audio_dim` and an independent check on
    `vocab_size` and `pad_id`. `--vocab-map` supplies one instead when the features are not
    to hand.

    python -m scripts.checkpointing.ckpt_to_components --ckpt runs/.../last.ckpt --dry-run
    python -m scripts.checkpointing.ckpt_to_components --ckpt runs/.../last.ckpt --run-id q5bsvywz
"""

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import DictConfig, OmegaConf

import train as T
from src.checkpointing import (
    COMPONENTS,
    VOCAB_MAP,
    load_state_into,
    manifest_for,
    save_components,
    save_manifest,
)
from src.models import build_audio_decoder, build_denoiser

# The config every old one is completed against. Not a variant file: only keys that are
# missing get filled, so the variant's own choices survive whatever it already set.
BASE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"


def leaf_keys(cfg: DictConfig | dict, prefix: str = "") -> set[str]:
    """Dotted paths of every leaf in a config, for diffing one against another."""
    out: set[str] = set()
    items = cfg.items() if isinstance(cfg, dict) else OmegaConf.to_container(cfg).items()
    for key, value in items:
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out |= leaf_keys(value, f"{path}.")
        else:
            out.add(path)
    return out


def reconcile(old: DictConfig, defaults: DictConfig) -> tuple[DictConfig, list[str]]:
    """Fill keys `old` lacks from `defaults`, and name every one that had to be invented.

    `old` wins wherever it has an opinion — this completes a config, it does not update one.
    A run's own `data.subset_n` or `schedule` settings are what it actually trained under and
    must not be overwritten by whatever the defaults happen to say today.
    """
    merged = OmegaConf.merge(defaults, old)
    invented = sorted(leaf_keys(merged) - leaf_keys(old))
    return merged, invented


def component_models(
    cfg: DictConfig, state: dict, vocab_size: int, canvas_len: int
) -> dict[str, torch.nn.Module]:
    """Rebuild each component from the config and load its slice of the checkpoint.

    Goes through `load_state_into` rather than `load_state_dict` so the compile prefix and
    the pre-`trim-sound-9` `ff.2` -> `ff.3` rename are undone — the two fixups that make an
    old checkpoint loadable at all.
    """
    built: dict[str, torch.nn.Module] = {}
    for name in COMPONENTS:
        if name == "denoiser":
            model = build_denoiser(cfg.model.denoiser, vocab_size)
        elif cfg.model.get("audio_decoder") is not None:
            model = build_audio_decoder(cfg.model.audio_decoder, vocab_size, canvas_len)
        else:
            continue
        try:
            load_state_into(model, state, name)
        except KeyError:
            # The linear variant has no mu_nn; a denoiser that is missing is a real error.
            if name == "denoiser":
                raise
            continue
        model.eval()
        built[name] = model
    return built


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The checkpoint to read, where to write it, and how to recover the vocabulary."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", type=Path, required=True, help="the Lightning .ckpt to convert")
    p.add_argument(
        "--run-id",
        help="write to runs/<run-id>/components; defaults to the .ckpt's own run directory",
    )
    p.add_argument("--out", type=Path, help="explicit output directory, overriding --run-id")
    p.add_argument("--save-dir", default="runs", help="root --run-id is resolved under")
    p.add_argument(
        "--variant",
        default="best",
        choices=("best", "final"),
        help="the suffix to write the weights under",
    )
    p.add_argument(
        "--vocab-map",
        type=Path,
        help="an existing vocab_map.pt to copy, instead of rebuilding it from the data",
    )
    p.add_argument(
        "--no-data",
        action="store_true",
        help="skip building the datamodule; needs --vocab-map and infers audio_dim",
    )
    p.add_argument("--dry-run", action="store_true", help="report and write nothing")
    return p.parse_args(argv)


def resolve_out(args: argparse.Namespace) -> Path:
    """`--out`, else `<save-dir>/<run-id>/components`, else beside the checkpoint's run."""
    if args.out is not None:
        return args.out
    run_id = args.run_id
    if run_id is None:
        # runs/<project>/<run_id>/checkpoints/x.ckpt -> <run_id>
        parents = args.ckpt.resolve().parents
        run_id = parents[1].name if len(parents) > 1 else "converted"
    return Path(args.save_dir) / run_id / "components"


def main(argv: Sequence[str] | None = None) -> int:
    """Read the checkpoint, complete its config, rebuild the models, write the directory."""
    args = parse_args(argv)
    if not args.ckpt.exists():
        raise SystemExit(f"no checkpoint at {args.ckpt}")

    # weights_only=False: a .ckpt pickles the hyperparameters, which is the whole reason
    # this format is being migrated away from. Only run it on your own checkpoints.
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters") or {}
    if "cfg" not in hparams:
        raise SystemExit(
            f"{args.ckpt} has no hyper_parameters.cfg; found {sorted(hparams)}. "
            f"It was not written by FlowMatching."
        )

    old_cfg = OmegaConf.create(hparams["cfg"])
    cfg, invented = reconcile(old_cfg, T.load_config(BASE_CONFIG))

    print(f"checkpoint  {args.ckpt}")
    print(f"  epoch {checkpoint.get('epoch')}  global_step {checkpoint.get('global_step')}")
    print(f"  variant {cfg.get('variant')}  size {cfg.get('size')}  mu {cfg.get('mu_size')}")
    if invented:
        print(f"\n  {len(invented)} key(s) absent from the checkpoint, filled from base.yaml:")
        for key in invented:
            print(f"    {key} = {OmegaConf.select(cfg, key)!r}")
    else:
        print("\n  config is complete against the current base.yaml")

    vocab_size = int(hparams.get("vocab_size", 0)) or None
    pad_id = hparams.get("pad_id")
    canvas_len = int(hparams.get("canvas_len") or cfg.data.canvas_len)
    vocab_map = None
    audio_dim = None

    if not args.no_data:
        # Rebuilding the data is what recovers the vocabulary map, and it independently
        # confirms vocab_size and pad_id — a mismatch means the checkpoint and the features
        # disagree, and every token id in it would be read against the wrong vocabulary.
        print("\n  rebuilding the datamodule to recover the vocabulary map")
        dm = T.build_datamodule(cfg)
        vocab_map, audio_dim = dm.vocab_map, dm.d_model
        for label, want, got in (
            ("vocab_size", vocab_size, dm.vocab_size),
            ("pad_id", pad_id, dm.pad_id),
        ):
            if want is not None and int(want) != int(got):
                raise SystemExit(
                    f"{label} mismatch: the checkpoint says {want}, the data gives {got}. "
                    f"Converting would pair the weights with the wrong vocabulary."
                )
            print(f"    {label} {got}  (checkpoint: {want})")
        vocab_size, pad_id = dm.vocab_size, dm.pad_id

    if vocab_size is None or pad_id is None:
        raise SystemExit(
            "the checkpoint carries no vocab_size/pad_id; rerun without --no-data so they "
            "can be read off the dataset"
        )
    if audio_dim is None:
        audio_dim = int(cfg.model.denoiser.get("whisper_dim") or 1280)
        print(f"    audio_dim {audio_dim}  (from the config; no data was built)")

    built = component_models(cfg, checkpoint["state_dict"], vocab_size, canvas_len)
    print(f"\n  rebuilt {', '.join(sorted(built))}")

    out_dir = resolve_out(args)
    manifest = manifest_for(cfg, vocab_size, pad_id, audio_dim, canvas_len)

    if args.dry_run:
        print(f"\n  would write to {out_dir}:")
        for name in sorted(built):
            print(f"    {name}_{args.variant}.pth")
        print("    manifest.yaml")
        if vocab_map is not None or args.vocab_map is not None:
            print(f"    {VOCAB_MAP}")
        print("\ndry run, nothing written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(out_dir, manifest, vocab_map)
    if vocab_map is None and args.vocab_map is not None:
        shutil.copy2(args.vocab_map, out_dir / VOCAB_MAP)
    written = save_components(SimpleNamespace(**built), out_dir, variant=args.variant)

    print(f"\nwrote {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name:<24} {path.stat().st_size:>12,} bytes")
    if not written:
        raise SystemExit("no component weights were written")
    print(
        f"\nupload with:\n"
        f"  python -m scripts.upload_components {out_dir.parent.name} "
        f"--variant {args.variant}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
