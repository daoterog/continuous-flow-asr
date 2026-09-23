"""Train the continuous flow ASR model.

    python train.py --config configs/linear.yaml
    python train.py --config configs/audio_cond.yaml train.num_epochs=10
    python train.py --config configs/three_term.yaml \
        --train-configs clean other --train-splits train.100 train.360

The YAML is the source of truth. `--*-configs` / `--*-splits` mirror the flags on
scripts/precompute_whisper.py and override the `data.splits` node; anything else is a
plain OmegaConf dotlist override.
"""

import argparse
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import src.constants  # noqa: F401  # sets HF_HOME; must precede transformers
from src.constants import CONFIGS, SPLITS, data_root_for

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from transformers import WhisperTokenizerFast

from src.dataset import ROLES, AudioFeatureDataModule
from src.pl_module import FlowMatching
from src.schedule import make_tau_inverse, sweep_corruption

# `train.num_epochs=10`, `data.subset_n.train=64` — an OmegaConf dotlist assignment.
DOTLIST = re.compile(r"^[A-Za-z_][\w.]*=")


def load_config(path: Path) -> DictConfig:
    """Load a YAML, resolving a `_base_:` chain relative to each file's own directory."""
    path = Path(path)
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    if base is None:
        return cfg
    return OmegaConf.merge(load_config(path.parent / base), cfg)


def split_overrides(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Separate `a.b=c` dotlist overrides from the argparse flags.

    Pulled out before argparse sees them because `--train-splits` takes `nargs="+"` and
    would otherwise swallow any override that followed it. Splitting first means the two
    kinds of argument can appear in any order.
    """
    flags, overrides = [], []
    for token in argv:
        if not token.startswith("-") and DOTLIST.match(token):
            overrides.append(token)
        else:
            flags.append(token)
    return flags, overrides


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse the flags; `a.b=c` tokens are returned as OmegaConf dotlist overrides."""
    flags, overrides = split_overrides(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True, help="YAML config to run")
    for role in ROLES:
        p.add_argument(
            f"--{role}-configs", nargs="+", choices=CONFIGS, default=None,
            help=f"override data.splits.{role}.configs",
        )
        p.add_argument(
            f"--{role}-splits", nargs="+", choices=SPLITS, default=None,
            help=f"override data.splits.{role}.splits",
        )
    p.add_argument("--offline", action="store_true", help="log without a network")
    p.add_argument("--print-config", action="store_true", help="resolve, print and exit")
    return p.parse_args(flags), overrides


def build_config(argv: Sequence[str] | None = None) -> tuple[DictConfig, argparse.Namespace]:
    """YAML, then the split flags, then the dotlist overrides."""
    args, overrides = parse_args(argv)
    cfg = load_config(args.config)

    for role in ROLES:
        for field in ("configs", "splits"):
            value = getattr(args, f"{role}_{field}")
            if value is not None:
                cfg.data.splits[role][field] = list(value)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if args.offline:
        cfg.logging.offline = True
    # The config's stem rides along as a hyperparameter so runs can be grouped and
    # filtered by variant; the run's *name* is left for wandb to generate.
    if cfg.get("variant") is None:
        cfg.variant = args.config.stem
    return cfg, args


def build_sample_t_fn(dm: AudioFeatureDataModule, cfg: DictConfig) -> object | None:
    """The prior tau^-1, measured by corrupting real targets with no model in the loop."""
    if not cfg.schedule.enabled:
        return None
    torch.manual_seed(cfg.seed)
    tokens = dm.sample_tokens("train", cfg.schedule.sweep_n, seed=cfg.seed)
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    ts = np.linspace(0.0, 1.0, cfg.schedule.num_bins + 1).tolist()
    recovery = sweep_corruption(tokens, dm.vocab_size, ts)
    tau_inverse = make_tau_inverse(ts, recovery, dm.vocab_size)
    lo, hi = float(tau_inverse(0.0)), float(tau_inverse(1.0))
    print(f"tau: sweep over {tokens.size(0)} utts, t in [{lo:.3f}, {hi:.3f}] carries the draw")
    return tau_inverse


def setup_reproducibility(cfg: DictConfig) -> None:
    """Seed every RNG and fix the matmul precision, before anything touches CUDA.

    `workers=True` gives each DataLoader worker its own derived seed, which is what makes
    the data order — and so the whole run — repeat. Under `train.deterministic` the
    reductions are pinned too; cuBLAS needs a fixed workspace for that and reads the
    environment variable once, at CUDA init.
    """
    torch.set_float32_matmul_precision(cfg.train.get("matmul_precision", "highest"))
    if cfg.train.get("deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pl.seed_everything(cfg.seed, workers=True)


def build_datamodule(cfg: DictConfig) -> AudioFeatureDataModule:
    """Tokenizer plus the three roles, set up so the measured sizes are available."""
    tokenizer = WhisperTokenizerFast.from_pretrained(
        cfg.data.model, language="english", task="transcribe"
    )
    subset_n = {k: v for k, v in (cfg.data.subset_n or {}).items() if v is not None}
    return AudioFeatureDataModule(
        data_root=data_root_for(cfg.data.env, cfg.data.data_root),
        model=cfg.data.model,
        splits=OmegaConf.to_container(cfg.data.splits),
        tokenizer=tokenizer,
        canvas_len=cfg.data.canvas_len,
        batch_size=cfg.data.batch_size,
        audio_stride=cfg.data.audio_stride,
        compact_vocab=cfg.data.compact_vocab,
        subset_n=subset_n,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
        pin_memory=torch.cuda.is_available(),
    ).prepare()


def build_trainer(cfg: DictConfig, logger: object) -> pl.Trainer:
    """Trainer with gradient clipping, LR logging and a val-loss checkpoint."""
    callbacks = [ModelCheckpoint(monitor="total_loss/val", mode="min", save_last=True)]
    if logger:  # LearningRateMonitor raises if there is nowhere to write
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision or "32-true",
        gradient_clip_val=cfg.train.grad_clip,
        fast_dev_run=cfg.train.fast_dev_run,
        log_every_n_steps=cfg.train.log_every_n_steps,
        deterministic=cfg.train.get("deterministic", False),
        benchmark=cfg.train.get("benchmark", False),
        logger=logger,
        callbacks=callbacks,
    )


def build_logger(cfg: DictConfig) -> Logger:
    """WandbLogger, online unless `logging.offline` (or --offline) is set.

    WANDB_MODE has to be in the environment before the run is created, so it is set here
    rather than passed down.
    """
    if cfg.logging.offline:
        os.environ["WANDB_MODE"] = "offline"
    return WandbLogger(
        entity=cfg.logging.get("entity"),
        project=cfg.logging.project,
        # name=None lets wandb generate one; `variant` in the config identifies the run.
        name=cfg.logging.get("name"),
        group=cfg.logging.get("group") or cfg.get("variant"),
        tags=[cfg.variant] if cfg.get("variant") else None,
        save_dir=cfg.logging.save_dir,
        offline=bool(cfg.logging.offline),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the config, build everything from it, and fit."""
    cfg, args = build_config(argv)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return 0

    setup_reproducibility(cfg)

    dm = build_datamodule(cfg)
    print(dm.summary())

    sample_t_fn = build_sample_t_fn(dm, cfg)
    model = FlowMatching(
        cfg,
        vocab_size=dm.vocab_size,
        pad_id=dm.pad_id,
        audio_dim=dm.d_model,
        canvas_len=cfg.data.canvas_len,
        tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map,
        sample_t_fn=sample_t_fn,
    )

    trainer = build_trainer(cfg, build_logger(cfg))
    trainer.fit(model, datamodule=dm)

    # Test is deliberately not run here — call it explicitly once the model is final:
    #   trainer.test(model, datamodule=dm, ckpt_path="best")
    return 0


if __name__ == "__main__":
    sys.exit(main())
