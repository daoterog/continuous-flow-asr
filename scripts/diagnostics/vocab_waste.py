"""How much of a full-vocabulary model's output goes to tokens the corpus never uses.

LibriSpeech touches 1,193 of Whisper's 51,866 tokens, so 97.7% of the output projection's
rows are never a target and receive only negative gradient. That is the structural reason
a full-vocabulary run might fail where the compact one succeeds — as opposed to simply
having had less training — and the two look identical from the loss curve.

This measures it directly, at several flow times:

  * how much softmax mass lands outside the corpus vocabulary,
  * how often the argmax itself lands there,
  * what the token error would be if those logits were masked out.

A model that has learned to suppress the unreachable tokens shows near-zero mass and no
gain from masking: then the problem is budget, not structure. A model that leaks mass
there is spending its capacity on tokens that cannot occur, and masking recovers it.

    python -m scripts.diagnostics.vocab_waste --ckpt runs/.../last.ckpt
    python -m scripts.diagnostics.vocab_waste --components runs/<id>/components \
        --config configs/audio_cond.yaml
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train as T  # noqa: E402
from scripts.evaluation.sample_eval import (  # noqa: E402
    _build,
    _config_from_components,
    _model_from_components,
)
from src.checkpointing import load_state_into  # noqa: E402
from src.data import VocabMap  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Which model to inspect and at which flow times."""
    flags, overrides = T.split_overrides(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path)
    src.add_argument("--components", type=Path)
    p.add_argument("--variant", default="best", choices=["best", "final"])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--role", choices=["train", "val", "test"], default="val")
    p.add_argument("--times", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--batches", type=int, default=8, help="batches to average over")
    args = p.parse_args(flags)
    args.overrides = overrides
    return args


def corpus_ids(dm: object, vocab_size: int) -> torch.Tensor:
    """Bool mask over the model's vocabulary, True where the corpus can produce a token.

    With a compact vocabulary every id is reachable by construction, so the mask is all
    True and the numbers below are trivially zero — which is itself the point of the
    comparison.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    vmap: VocabMap | None = dm.vocab_map
    if vmap is None:
        # Full vocabulary: only the ids the VocabMap *would* have kept are reachable.
        seen = set()
        for role in ("train", "val", "test"):
            for text in dm.texts(role):
                seen.update(dm.tokenizer(text).input_ids)
        mask[sorted(seen)] = True
    else:
        mask[:] = True
    return mask


def main(argv: Sequence[str] | None = None) -> int:
    """Report out-of-corpus mass and the token error masking would recover."""
    args = parse_args(argv)
    if args.components is not None:
        cfg = _config_from_components(args)
    else:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = checkpoint["hyper_parameters"]["cfg"]
        if args.overrides:
            from omegaconf import OmegaConf

            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    if args.components is not None:
        model = _model_from_components(args, cfg, dm)
    else:
        model = _build(cfg, dm)
        for name in ("denoiser", "mu_nn"):
            if getattr(model, name, None) is not None:
                load_state_into(getattr(model, name), checkpoint["state_dict"], name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    reachable = corpus_ids(dm, dm.vocab_size).to(device)
    n_reach = int(reachable.sum())
    print(f"vocabulary {dm.vocab_size:,}   reachable from the corpus {n_reach:,} "
          f"({100 * n_reach / dm.vocab_size:.1f}%)")
    if n_reach == dm.vocab_size:
        print("compact vocabulary: nothing is unreachable, so there is nothing to waste.")
    print(f"\n{'t':>6} {'mass outside':>13} {'argmax outside':>15} {'P_e':>8} "
          f"{'P_e masked':>11} {'recovered':>10}")

    loader = {"train": dm.train_dataloader, "val": dm.val_dataloader,
              "test": dm.test_dataloader}[args.role]()
    batches = []
    for i, b in enumerate(loader):
        if i >= args.batches:
            break
        batches.append({k: v.to(device) for k, v in b.items()})

    for t_val in args.times:
        mass, arg_out, pe, pe_masked, n = 0.0, 0.0, 0.0, 0.0, 0
        for b in batches:
            tokens, audio = b["tokens"], b["audio"].float()
            t = torch.full((tokens.size(0),), t_val, device=device)
            with torch.no_grad():
                mu = None
                if model.mu_nn is not None:
                    mu = F.softmax(model.mu_nn(audio), dim=-1)
                x1 = F.one_hot(tokens, dm.vocab_size).float()
                i_t = model.make_interpolant(x1=x1, t=t, mu=mu)
                logits = model.denoiser(x_t=i_t, time=t, audio_embeddings=audio)
                probs = logits.softmax(-1)
                mass += float(probs[..., ~reachable].sum(-1).mean())
                pred = logits.argmax(-1)
                # Real positions only, matching BinnedTokenError: padding is ~80% of the
                # canvas and is learned first, so counting it puts a floor under P_e and
                # makes the numbers here incomparable with everything else logged.
                real = tokens != dm.pad_id
                n_real = real.sum().clamp(min=1)
                arg_out += float((~reachable[pred] & real).sum() / n_real)
                pe += float(((pred != tokens) & real).sum() / n_real)
                masked = logits.masked_fill(~reachable, float("-inf")).argmax(-1)
                pe_masked += float(((masked != tokens) & real).sum() / n_real)
            n += 1
        k = max(n, 1)
        print(f"{t_val:>6.2f} {mass / k:>12.4f} {100 * arg_out / k:>14.2f}% "
              f"{pe / k:>8.4f} {pe_masked / k:>11.4f} {(pe - pe_masked) / k:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
