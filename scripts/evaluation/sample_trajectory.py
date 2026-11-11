r"""WER across sampling step counts, WER along one trajectory, and the transcripts it passes.

Three readings of the same saved models, for the paper's sampling figure:

  sampling_steps.csv       final WER at each step count N
  sampling_trajectory.csv  at each N in --trace-steps, the WER of the denoiser's running
                           prediction and of the state's argmax at every step k
  sampling_examples.tsv    for --n-show utterances, the decoded prediction at every step

Each model is named `label:components_dir`. How it is sampled is read from the run's own
config, never typed here: its `sampling.method`, its saved curve, and guidance only where
it was trained for it. A run trained with `train.audio_dropout` is sampled unguided (w = 1)
and at every --cfg-strengths weight, so what guidance buys is measured on the same model
from the same noise; a run trained without it has no unconditional branch to guide
against and is sampled at w = 1 alone. `label:dir@1.5,2` sets that model's weights, and
naming a weight for a model trained without audio dropout is an error. A mu-only run is one
argmax pass, so it gives one row at N = 1 and no trajectory.

Every model, step count and weight starts from the same x0: the noise is seeded per batch,
and the batch is fixed by --batch-size rather than by each run's own.

    python -m scripts.evaluation.sample_trajectory \
        --model ours:runs/<id>/components linear:runs/<id>/components \
        --role val test data.env=snellius data.subset_n.val=null
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from omegaconf import DictConfig

import train as T
from scripts.evaluation.sample_eval import _config_from_components
from src.checkpointing import load_components
from src.metrics import corpus_wer, ids_to_text, score_samples
from src.mu_only import MuOnly

STEP_FIELDS = ("label", "role", "method", "cfg", "trained_dropout", "num_steps", "wer",
               "token_err")
TRACE_FIELDS = ("label", "role", "method", "cfg", "trained_dropout", "num_steps", "step",
                "t", "wer_pred", "wer_state")
EXAMPLE_FIELDS = ("label", "role", "cfg", "utt", "step", "t", "wer", "text")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Models, splits, step counts; `a.b=c` tokens are dotlist overrides for every model."""
    flags, overrides = T.split_overrides(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", nargs="+", required=True, help="label:components_dir[@w,...]")
    p.add_argument("--variant", default="best", choices=["best", "final"])
    p.add_argument("--role", nargs="+", default=["val"], choices=["train", "val", "test"])
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--trace-steps", type=int, nargs="+", default=[16],
                   help="the step counts whose every step is recorded")
    p.add_argument(
        "--cfg-strengths", type=float, nargs="+", default=[1.5, 2.0],
        help="guided weights for models trained with audio dropout, on top of w = 1",
    )
    p.add_argument("--n-utts", type=int, default=None, help="cap each split (default: all)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-show", type=int, default=6, help="utterances whose steps are written")
    p.add_argument("--seed", type=int, default=0, help="seeds x0, per batch")
    p.add_argument("--out", type=Path, default=Path("paper/data"))
    args = p.parse_args(flags)
    args.overrides = overrides
    return args


def parse_model(spec: str) -> tuple[str, Path, list[float] | None]:
    """`label:dir[@w,...]` -> (label, dir, the weights asked for, or None for the default).

    A colon, not `=`: train.split_overrides takes any `a=b` token for an OmegaConf override.
    """
    label, sep, rest = spec.partition(":")
    if not sep or not label:
        raise SystemExit(f"--model takes label:components_dir[@w,...], got {spec!r}")
    path, _, weights = rest.partition("@")
    return label, Path(path), [float(w) for w in weights.split(",")] if weights else None


def trained_dropout(cfg: DictConfig) -> float:
    """The audio-dropout rate the run trained with: 0 means it has no unconditional branch."""
    return float((cfg.train.get("audio_dropout") or {}).get("p", 0.0))


def guidance(label: str, cfg: DictConfig, asked: list[float] | None,
             default: list[float]) -> list[float]:
    """The weights to sample this model at: w = 1, plus guided ones if it trained for them."""
    guided = [w for w in (asked if asked is not None else default) if w != 1.0]
    if trained_dropout(cfg) > 0.0:
        return [1.0, *guided]
    if asked is not None and guided:
        raise SystemExit(
            f"{label}: guidance {guided} asked for, but the run trained without "
            f"train.audio_dropout, so it has no unconditional branch to guide against"
        )
    return [1.0]


def load_model(components: Path, variant: str, cfg: DictConfig, dm) -> torch.nn.Module:  # noqa: ANN001
    """The run from its own config, saved weights and curve, as train.evaluate_saved loads it."""
    model = T.build_model(dm, cfg)
    manifest, built, saved_tau = load_components(components, variant=variant)
    if manifest.vocab_size != dm.vocab_size:
        raise SystemExit(
            f"{components}: weights are for vocab {manifest.vocab_size}, data gives {dm.vocab_size}"
        )
    for name, sub in built.items():
        getattr(model, name).load_state_dict(sub.state_dict(), strict=True)
    if saved_tau is not None:
        model.time_reparam = saved_tau
    return model


def spread(n_rows: int, n: int) -> list[int]:
    """`n` indices spread evenly over `n_rows`, as sample_eval.show_examples picks them."""
    return sorted({round(k * (n_rows - 1) / max(n - 1, 1)) for k in range(n)}) if n else []


def score_role(
    label: str, model: torch.nn.Module, cfg: DictConfig, dm, role: str,  # noqa: ANN001
    weights: list[float], args: argparse.Namespace, out: dict[str, list],
) -> None:
    """Sample one split at every N and weight, appending rows to `out`."""
    device = next(model.parameters()).device
    loader = {"train": dm.train_dataloader, "val": dm.val_dataloader,
              "test": dm.test_dataloader}[role]()
    batches = [{"audio": b["audio"], "tokens": b["tokens"]} for b in loader]
    # Scored where the model is: the vocabulary map's tables move with it.
    ref = torch.cat([b["tokens"] for b in batches]).to(device)
    refs = ids_to_text(ref, dm.tokenizer, dm.pad_id, dm.vocab_map)
    picks = spread(len(refs), args.n_show)
    is_mu = isinstance(model, MuOnly)
    method = "argmax" if is_mu else str(cfg.get("sampling", {}).get("method", "interpolant"))
    dropout = trained_dropout(cfg)
    # Sample under the precision validation ran at, so N = 16 reproduces the logged WER.
    bf16 = "bf16" in str(cfg.train.get("precision") or "") and device.type == "cuda"
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)
    print(f"\n{label}  role {role}  {len(refs):,} utts  method {method}  "
          f"trained audio dropout {dropout:g}  cfg {weights}  bf16 {bf16}")

    def text_rows(ids: torch.Tensor, w: float, step: int | str, t: float | str) -> None:
        hyps = ids_to_text(ids[picks], dm.tokenizer, dm.pad_id, dm.vocab_map)
        for i, h in zip(picks, hyps, strict=True):
            out["examples"].append(
                (label, role, w, i, step, t, round(corpus_wer([refs[i]], [h]), 1), h)
            )

    for w in weights:
        for n in [1] if is_mu else args.steps:
            traced = n in args.trace_steps and not is_mu
            preds, steps_seen = [], []
            for b_idx, batch in enumerate(batches):
                gen = torch.Generator(device=device).manual_seed(args.seed + b_idx)
                audio = batch["audio"].to(device).float()
                trace = [] if traced else None
                with torch.no_grad(), autocast:
                    if is_mu:
                        pred = model.sample(audio)
                    else:
                        pred = model.sample(audio, n, generator=gen, cfg_strength=w,
                                            trace=trace)
                preds.append(pred.cpu())
                if traced:
                    steps_seen.append(trace)
            pred = torch.cat(preds).to(device)
            wer, token_err = score_samples(pred, ref, dm.tokenizer, dm.pad_id, dm.vocab_map)
            print(f"  N {n:>3}  w {w:g}  WER {wer:6.2f}  token err {token_err:.4f}")
            out["steps"].append((label, role, method, w, dropout, n, round(wer, 3),
                                 round(token_err, 5)))
            if is_mu:
                text_rows(pred, w, "final", "")
                continue
            if not traced:
                continue
            for k in range(len(steps_seen[0])):
                t = steps_seen[0][k]["t"]
                step_pred = torch.cat([tr[k]["pred"] for tr in steps_seen]).to(device)
                step_state = torch.cat([tr[k]["state"] for tr in steps_seen]).to(device)
                wer_pred, _ = score_samples(step_pred, ref, dm.tokenizer, dm.pad_id, dm.vocab_map)
                wer_state, _ = score_samples(step_state, ref, dm.tokenizer, dm.pad_id,
                                             dm.vocab_map)
                out["trace"].append((label, role, method, w, dropout, n, k + 1, round(t, 5),
                                     round(wer_pred, 3), round(wer_state, 3)))
                text_rows(step_pred, w, k + 1, round(t, 4))
            text_rows(pred, w, "final", 1.0)
            print(f"  traced N {n}: running prediction WER "
                  + " ".join(f"{r[8]:.1f}" for r in out["trace"][-len(steps_seen[0]):]))

    if not any(r[0] == "reference" and r[1] == role for r in out["examples"]):
        for i in picks:
            out["examples"].append(("reference", role, "", i, "", "", 0.0, refs[i]))


def write(path: Path, fields: Sequence[str], rows: list, delimiter: str = ",") -> None:
    """One table to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(fields)
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Score every model on every split, then write the three tables."""
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict[str, list] = {"steps": [], "trace": [], "examples": []}
    for spec in args.model:
        label, components, asked = parse_model(spec)
        ns = argparse.Namespace(components=components, config=None, overrides=args.overrides)
        cfg = _config_from_components(ns)
        for role in args.role:
            if args.n_utts is not None:
                cfg.data.subset_n[role] = args.n_utts
        cfg.data.batch_size = cfg.data.eval_batch_size = args.batch_size
        weights = guidance(label, cfg, asked, args.cfg_strengths)

        T.setup_reproducibility(cfg)
        dm = T.build_datamodule(cfg)
        model = load_model(components, args.variant, cfg, dm).to(device).eval()
        for role in args.role:
            score_role(label, model, cfg, dm, role, weights, args, out)
        del model
        torch.cuda.empty_cache()

    write(args.out / "sampling_steps.csv", STEP_FIELDS, out["steps"])
    write(args.out / "sampling_trajectory.csv", TRACE_FIELDS, out["trace"])
    write(args.out / "sampling_examples.tsv", EXAMPLE_FIELDS, out["examples"], "\t")
    return 0


if __name__ == "__main__":
    sys.exit(main())
