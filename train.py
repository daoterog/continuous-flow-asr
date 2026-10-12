"""Train the continuous flow ASR model.

    python train.py --config configs/linear.yaml
    python train.py --config configs/audio_cond.yaml train.num_epochs=10
    python train.py --config configs/linear.yaml --size small
    python train.py --config configs/three_term.yaml \
        --train-configs clean other --train-splits train.100 train.360

The YAML is the source of truth. `--*-configs` / `--*-splits` mirror the flags on
scripts/data/precompute_whisper.py and override the `data.splits` node; `--size` layers a
denoiser geometry from configs/sizes/ over the variant; anything else is a plain
OmegaConf dotlist override.
"""

import argparse
import os
import re
import socket
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import src.constants  # noqa: F401  # sets HF_HOME; must precede transformers
from src.constants import CONFIGS, SPLITS, data_root_for

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from transformers import WhisperTokenizerFast

from src.checkpointing import ComponentCheckpoint, load_manifest, manifest_for
from src.data import ROLES, AudioFeatureDataModule
from src.flow.coefficients import make_schedule
from src.flow.schedule import TimeReparametrization
from src.flow_mapping import DISTILL_GROUPS, FlowMapping
from src.flow_matching import FlowMatching

# Used to parse OmegaConf dotlist assignments like
# `train.num_epochs=10`, `data.subset_n.train=64`
DOTLIST = re.compile(r"^[A-Za-z_][\w.]*=")

# Read NN sizes
SIZES_DIR = Path(__file__).resolve().parent / "configs" / "sizes"
SIZES = sorted(p.stem for p in SIZES_DIR.glob("*.yaml"))
MU_DIR = Path(__file__).resolve().parent / "configs" / "mu"
MU_SIZES = sorted(p.stem for p in MU_DIR.glob("*.yaml"))
# Where a run executes and what kind of run it is: configs/profiles/{where,run}/. They hold
# the settings that depend on those two facts and nothing else; see configs/profiles/README.md.
PROFILES_DIR = Path(__file__).resolve().parent / "configs" / "profiles"
WHERE = sorted(p.stem for p in (PROFILES_DIR / "where").glob("*.yaml"))
RUN_KINDS = sorted(p.stem for p in (PROFILES_DIR / "run").glob("*.yaml"))


def detect_where() -> str:
    """`snellius` inside a SLURM job or on a Snellius node, else `local`."""
    if os.environ.get("SLURM_JOB_ID") or socket.getfqdn().endswith(".snellius.surf.nl"):
        return "snellius"
    return "local"


def load_config(path: Path) -> DictConfig:
    """Load a YAML, resolving a `_base_:` chain relative to each file's own directory."""
    return OmegaConf.merge(*config_layers(path))


def config_layers(path: Path) -> list[DictConfig]:
    """A YAML and its `_base_:` chain as separate layers, the root base first."""
    path = Path(path)
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    if base is None:
        return [cfg]
    return [*config_layers(path.parent / base), cfg]


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
    p.add_argument(
        "--size", choices=SIZES, default=None,
        help="denoiser geometry from configs/sizes/. Layers over the variant config, "
            "which owns the flow and mu; a dotlist override still wins over it",
    )
    p.add_argument(
        "--mu", choices=MU_SIZES, default=None,
        help="mu_psi geometry from configs/mu/. Only meaningful on a variant whose "
            "flow.gamma is set, since that is what puts mu in the interpolant",
    )
    p.add_argument(
        "--where", choices=WHERE, default=None,
        help="the machine profile from configs/profiles/where/; detected when omitted "
            "(snellius inside a SLURM job, else local)",
    )
    p.add_argument(
        "--run", choices=RUN_KINDS, default=None,
        help="the run-kind profile from configs/profiles/run/ (sweep = a 100-epoch "
            "screen, full = a 300-epoch run); omitted, base.yaml and the variant decide",
    )
    p.add_argument("--offline", action="store_true", help="log without a network")
    p.add_argument("--print-config", action="store_true", help="resolve, print and exit")
    p.add_argument(
        "--print-steps", action="store_true",
        help="print the run's total optimizer steps and exit; jobs/submit/train.sh sizes "
            "its walltime from this, so it must not need a GPU",
    )
    return p.parse_args(flags), overrides


def build_config(argv: Sequence[str] | None = None) -> tuple[DictConfig, argparse.Namespace]:
    """base.yaml, the --where/--run profiles, the variant YAML, --size, --mu, the split
    flags, then the dotlist overrides — each layer winning over the ones before it.

    The profiles sit under the variant rather than over it: they replace defaults that
    depend on the machine or the kind of run, and a variant that pins one of those keys on
    purpose (flow_mapping.yaml leaves compile off) keeps it.
    """
    args, overrides = parse_args(argv)
    layers = config_layers(args.config)
    where = args.where or detect_where()
    profiles = [load_config(PROFILES_DIR / "where" / f"{where}.yaml")]
    if args.run is not None:
        profiles.append(load_config(PROFILES_DIR / "run" / f"{args.run}.yaml"))
    cfg = OmegaConf.merge(layers[0], *profiles, *layers[1:])
    cfg.profile = {"where": where, "run": args.run}

    # Parse cli args, load denoiser config, then mu config.
    # Order matters because denoiser args must match some of mu args.
    if args.size is not None:
        cfg = OmegaConf.merge(cfg, load_config(SIZES_DIR / f"{args.size}.yaml"))
        cfg.size = args.size
    if args.mu is not None:
        if cfg.flow.get("gamma") is None:
            raise ValueError(
                f"--mu {args.mu} on {args.config.name}, whose flow.gamma is null: mu would "
                f"be built and trained but never enter the interpolant. Use a variant that "
                f"sets gamma (audio_cond, three_term), or drop --mu."
            )
        cfg = OmegaConf.merge(cfg, load_config(MU_DIR / f"{args.mu}.yaml"))
        cfg.mu_size = args.mu

    for role in ROLES:
        for field in ("configs", "splits"):
            value = getattr(args, f"{role}_{field}")
            if value is not None:
                cfg.data.splits[role][field] = list(value)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if args.offline:
        cfg.logging.offline = True
    # Variant is referenced as stem in the config, here we just override it
    if cfg.get("variant") is None:
        cfg.variant = args.config.stem

    validate_distill_split(cfg)
    require_recipe_on_cluster(cfg, args.config)
    return cfg, args


def require_recipe_on_cluster(cfg: DictConfig, config_path: Path) -> None:
    """Refuse a Snellius config that was not built from a configs/recipes/ file.

    Every cluster run up to 2026-10-10 trained base.yaml's laptop model (a tiny denoiser,
    stride-4 pooled latents) because nothing said otherwise. A recipe states what is
    trained; this makes the cluster insist on one. Keyed on the resolved config, not the
    --where flag, so `data.env=snellius` as a plain override is caught too. A deliberate
    one-off passes `recipe=<name>` to say so.
    """
    on_cluster = cfg.profile.where == "snellius" or cfg.data.get("env") == "snellius"
    if on_cluster and not cfg.get("recipe"):
        raise SystemExit(
            f"{config_path} is not a recipe, and cluster runs must come from one: pass "
            f"--config configs/recipes/<name>.yaml (fulldata, distill), or add "
            f"recipe=<name> to state that this one-off is deliberate. Without a recipe the "
            f"run inherits base.yaml's laptop defaults (tiny denoiser, pooled latents)."
        )


def validate_distill_split(cfg: DictConfig) -> None:
    """`distill.split` must be a partition: every row trains exactly one of the three terms.

    Checked here rather than in the module because it is a statement about the config, and
    because a bad split should fail before anything loads a checkpoint or touches a GPU. The
    fractions are also the relative weight of the three objectives, so one that does not sum
    to 1 is silently reweighting them rather than erroring.
    """
    split = cfg.get("distill", {}).get("split")
    if split is None:
        return
    missing = [k for k in DISTILL_GROUPS if k not in split]
    if missing:
        raise ValueError(f"distill.split is missing {missing}; it needs {list(DISTILL_GROUPS)}")
    parts = {k: float(split[k]) for k in DISTILL_GROUPS}
    total = sum(parts.values())
    if any(v < 0.0 for v in parts.values()) or abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"distill.split must be non-negative and sum to 1, got {parts} (sum {total:.6f})"
        )


def build_model(
    dm: AudioFeatureDataModule, cfg: DictConfig
) -> "FlowMatching | FlowMapping":
    """The module this config describes: a distillation run when it configures a teacher.

    Keyed on `distill` rather than on a name of its own, because a config that names a
    teacher *is* a distillation config and a second key could contradict the first.

    The two do not take the same arguments, which is why this is a function and not a
    name->class map: `FlowMatching` is handed the reparametrization built from its own
    config, while `FlowMapping` refuses one and loads the teacher's, since sampling a
    student on any other grid makes its WER incomparable to the model it came from.
    """
    common = dict(
        vocab_size=dm.vocab_size,
        pad_id=dm.pad_id,
        canvas_len=cfg.data.canvas_len,
        tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map,
    )
    if cfg.get("distill") is not None:
        validate_distill_teacher(dm, cfg)
        print(f"distilling from {cfg.distill.teacher}")
        return FlowMapping(cfg, **common)
    return FlowMatching(cfg, time_reparam=build_time_reparam(dm, cfg), **common)


def validate_distill_teacher(dm: AudioFeatureDataModule, cfg: DictConfig) -> None:
    """The student has to be fed what the teacher was fed.

    It is initialised from the teacher's weights and trained against the teacher's own
    outputs, so a different input distribution makes both meaningless — and this is the one
    mismatch that raises nothing on its own. `data.audio_stride` pools latents over *frames*
    (`src/data/features.py:80`), so stride 1 and stride 4 differ by 4x in sequence length at
    the same width: 1,280 either way, which is why checking `audio_dim` would pass and prove
    nothing. Attention takes any length, so a stride-4 teacher fed stride-1 features runs
    happily on 1,500 frames where it learned 375.

    Live risk, not hypothetical: every teacher so far was trained at stride 4, and
    `configs/profiles/where/snellius.yaml` now sets stride 1.

    `audio_dim` is checked too. Stride does not move it, but a different encoder would, and
    that would otherwise surface as a shape error inside `load_state_dict`.
    """
    manifest = load_manifest(cfg.distill.teacher)
    checks = (
        ("data.audio_stride", manifest.run_config.data.get("audio_stride"),
         cfg.data.get("audio_stride")),
        ("audio_dim", manifest.audio_dim, dm.d_model),
    )
    for key, want, got in checks:
        if want is None or int(want) == int(got):
            continue
        raise ValueError(
            f"the teacher at {cfg.distill.teacher} was trained with {key}={want} but this "
            f"run uses {got}. The student is distilled against that teacher's outputs, so "
            f"it has to see the same features — pass {key}={want}, or distil from a teacher "
            f"trained at {key}={got}."
        )


def build_time_reparam(dm: AudioFeatureDataModule, cfg: DictConfig) -> TimeReparametrization:
    """The prior tau: what the interpolant alone leaves recoverable, with no model.

    The identity when `schedule.enabled` is false, so t is drawn uniformly. Under
    `schedule.uniform_floor` the curve carries the floor from the first step, as every
    refreshed curve does after it.
    """
    if not cfg.schedule.enabled:
        return TimeReparametrization.identity()
    gamma_fn = make_schedule(cfg.flow.get("gamma"))
    reparam = TimeReparametrization.exact(
        dm.vocab_size,
        alpha_fn=make_schedule(cfg.flow.get("alpha", "linear_down")),
        beta_fn=make_schedule(cfg.flow.get("beta", "linear_up")),
        gamma_fn=gamma_fn,
        mu_accuracy=float(cfg.schedule.get("mu_accuracy", 0.0)),
    ).with_uniform_floor(float(cfg.schedule.get("uniform_floor", 0.0)))
    path = "with mu" if gamma_fn is not None else "linear"
    print(f"tau: exact curve on the {path} path over {dm.vocab_size:,} tokens: {reparam!r}")
    return reparam


def setup_reproducibility(cfg: DictConfig) -> None:
    """Seed every RNG and fix the matmul precision, before anything touches CUDA."""
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
        eval_batch_size=cfg.data.get("eval_batch_size"),
        audio_stride=cfg.data.audio_stride,
        compact_vocab=cfg.data.compact_vocab,
        subset_n=subset_n,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
        pin_memory=torch.cuda.is_available(),
        steps_per_epoch=cfg.data.get("steps_per_epoch"),
        gpu_prefetch=bool(cfg.data.get("gpu_prefetch", False)),
        prefetch_factor=cfg.data.get("prefetch_factor"),
        eval_num_workers=cfg.data.get("eval_num_workers"),
    ).prepare()


class Heartbeat(pl.Callback):
    """Used to identify possibly failed training runs.

    When a cluster run goes quiet, the first question is whether *training* stopped or
    only *logging* did — and the wandb history cannot answer it, because if the logger is
    the thing that died the history ends either way.

    This writes straight to the job's stdout, so the SLURM .out file settles it:
    if these lines keep coming after the metrics stop, the training loop is alive and
    the logger is at fault; if they stop too, the loop itself is stuck, and the last one
    names the epoch it stuck on.
    """

    def __init__(self) -> None:
        """Record the start so each line carries elapsed time as well as wall clock."""
        super().__init__()
        self.started = time.monotonic()

    def on_train_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        """Epoch, global step, elapsed minutes, and the training loss if it is in hand."""
        loss = trainer.callback_metrics.get("total_loss/train")
        loss_s = f"  loss {float(loss):.4f}" if loss is not None else ""
        print(
            f"[heartbeat] {datetime.now():%H:%M:%S}  "
            f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs}  "
            f"step {trainer.global_step}  "
            f"{(time.monotonic() - self.started) / 60:.1f} min{loss_s}",
            flush=True,
        )


def components_dir(cfg: DictConfig, logger: object) -> Path:
    """Where this run writes its loadable parts: `<save_dir>/<run_id>/components`."""
    run_id = getattr(getattr(logger, "experiment", None), "id", None) or "local"
    return Path(cfg.logging.save_dir) / str(run_id) / "components"


def build_trainer(
    cfg: DictConfig, logger: object, extra_callbacks: Sequence[pl.Callback] = ()
) -> pl.Trainer:
    """Trainer with gradient clipping, LR logging and a val-loss checkpoint."""
    checkpoint = cfg.train.get("checkpoint", True)
    callbacks: list[pl.Callback] = [Heartbeat(), *extra_callbacks]
    if checkpoint:
        callbacks.append(ModelCheckpoint(monitor="total_loss/val", mode="min", save_last=True))
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    stop_on = cfg.train.get("early_stop_on")
    if stop_on:
        callbacks.append(
            EarlyStopping(
                monitor=stop_on,
                mode=cfg.train.get("early_stop_mode", "min"),
                stopping_threshold=float(cfg.train.get("early_stop_threshold", 0.0)),
                patience=10**6,
                check_on_train_epoch_end=False,
                verbose=True,
            )
        )
    return pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision or "32-true",
        # Lightning wants a number, not None: 0 is how it spells "no clipping".
        gradient_clip_val=cfg.train.grad_clip or 0,
        fast_dev_run=cfg.train.fast_dev_run,
        log_every_n_steps=cfg.train.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.get("check_val_every_n_epoch", 1),
        enable_checkpointing=checkpoint,
        deterministic=cfg.train.get("deterministic", False),
        benchmark=cfg.train.get("benchmark", False),
        logger=logger,
        callbacks=callbacks,
    )


def build_logger(cfg: DictConfig) -> Logger:
    """WandbLogger, online unless `logging.offline` (or --offline) is set."""
    if cfg.logging.offline:
        os.environ["WANDB_MODE"] = "offline"
    return WandbLogger(
        entity=cfg.logging.get("entity"),
        project=cfg.logging.project,
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

    if args.print_steps:
        # Sizing only: build the data, count the batches, print, exit.
        # No model, no CUDA, so this runs on a login node.
        dm = build_datamodule(cfg)
        per_epoch = len(dm.train_dataloader())
        print(f"steps_per_epoch {per_epoch}")
        print(f"epochs {cfg.train.num_epochs}")
        print(f"total_steps {per_epoch * cfg.train.num_epochs}")
        return 0

    print(f"profile: where={cfg.profile.where} run={cfg.profile.run or '(none)'}  "
          f"epochs {cfg.train.num_epochs}  compile {cfg.train.compile}  "
          f"train WER bins {cfg.metrics.wer_bins.get('train', 0)}")
    setup_reproducibility(cfg)

    dm = build_datamodule(cfg)
    print(dm.summary())

    model = build_model(dm, cfg)

    logger = build_logger(cfg)
    extra: list[pl.Callback] = []
    if cfg.train.get("save_components", True):
        out_dir = components_dir(cfg, logger)
        extra.append(
            ComponentCheckpoint(
                out_dir=out_dir,
                manifest=manifest_for(
                    cfg, dm.vocab_size, dm.pad_id, dm.d_model, cfg.data.canvas_len
                ),
                vocab_map=dm.vocab_map,
                monitor="total_loss/val",
            )
        )
        print(f"components  {out_dir}")

    trainer = build_trainer(cfg, logger, extra)
    trainer.fit(model, datamodule=dm)

    # Test is deliberately not run here — call it explicitly once the model is final:
    #   trainer.test(model, datamodule=dm, ckpt_path="best")
    return 0


if __name__ == "__main__":
    sys.exit(main())
