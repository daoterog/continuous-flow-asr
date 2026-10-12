"""How much of the transcript each source carries, at fixed flow times: audio, mu, the field.

The binned metrics cannot answer this. Their lowest bin under simulated tau spans
t in [0, ~0.5], where the field already carries some data, so its recovery mixes what the
denoiser read off the audio with what it read off x_t. Here t is fixed, and the inputs are
taken apart one at a time:

  denoiser          the model as it trains: x_t built with mu, audio cross-attended
  denoiser, no mu   mu swapped for the uniform 1/V vector; at t=0 this is audio alone
  field             argmax of x_t itself, with the real mu: what a closed-form tau assumes
  mu                mu's own argmax, and how peaked its softmax is

The last block answers a question the closed form cannot: `TimeReparametrization.exact`
treats mu as one-hot, a full +1 on its predicted token, while the real mu is a softmax.
`field @ sigma` reads x_0 = mu + sigma * eps with the real mu for a few sigma, which is
how readable mu actually is when the noise is rescaled (flow.alpha.scale).

Run on train and val both: a gap at low t is the audio-only mapping being memorised.

    python -m scripts.diagnostics.probe_audio_info --components runs/<id>/components
    python -m scripts.diagnostics.probe_audio_info --components ... --n-utts 4096 data.env=snellius
"""

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import train as T
from scripts.evaluation.sample_eval import _config_from_components, _model_from_components
from src.flow_matching import FlowMatching

TS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SIGMAS = (1.0, 0.7, 0.5, 0.4, 0.3)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """A components directory, how much of each split, and OmegaConf overrides."""
    flags, overrides = T.split_overrides(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--variant", default="best", choices=["best", "final"])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--roles", nargs="+", default=["train", "val"])
    p.add_argument("--n-utts", type=int, default=2048, help="per role")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("figures/audio_info"))
    args = p.parse_args(flags)
    args.overrides = overrides
    return args


class Tally:
    """Hits over real tokens, summed across batches, per named reading."""

    def __init__(self) -> None:
        """Start empty."""
        self.hits: dict[str, float] = {}
        self.reals = 0.0

    def add(self, name: str, pred: torch.Tensor, tokens: torch.Tensor, real: torch.Tensor) -> None:
        """Count the real positions where `pred` is the target."""
        self.hits[name] = self.hits.get(name, 0.0) + float(((pred == tokens) & real).sum())

    def recovery(self) -> dict[str, float]:
        """Hits over real tokens, per reading."""
        return {k: v / self.reals for k, v in self.hits.items()}


@torch.no_grad()
def probe(model: FlowMatching, loader: Iterable[dict], n_utts: int, seed: int) -> dict[str, float]:
    """Every reading at every t in `TS`, over the first `n_utts` utterances of `loader`."""
    tally = Tally()
    peak, on_target = 0.0, 0.0
    gen = torch.Generator(device=model.device).manual_seed(seed)
    seen = 0
    for batch in loader:
        if seen >= n_utts:
            break
        audio = batch["audio"][: n_utts - seen].to(model.device).float()
        tokens = batch["tokens"][: n_utts - seen].to(model.device)
        seen += tokens.size(0)
        real = tokens != model.pad_id
        tally.reals += float(real.sum())
        x1 = F.one_hot(tokens, model.vocab_size).float()

        mu = F.softmax(model.mu_nn(audio).float(), dim=-1)
        tally.add("mu", mu.argmax(-1), tokens, real)
        peak += float(mu.max(-1).values[real].sum())
        on_target += float(mu.gather(-1, tokens[..., None]).squeeze(-1)[real].sum())
        flat = torch.full_like(mu, 1.0 / model.vocab_size)

        eps = torch.randn(x1.shape, device=x1.device, generator=gen)
        for sigma in SIGMAS:
            tally.add(f"field@sigma={sigma} t=0", (mu + sigma * eps).argmax(-1), tokens, real)

        for t in TS:
            tt = torch.full((tokens.size(0),), t, device=model.device)
            time = model.time_reparam.u_of(tt) if model.time_input == "uniform" else tt
            a, b = model.alpha_fn(t), model.beta_fn(t)
            g = model.gamma_fn(t) if model.gamma_fn is not None else 0.0
            for name, prior in (("denoiser", mu), ("denoiser no mu", flat)):
                x_t = a * eps + b * x1 + g * prior
                pred = model.denoiser(x_t=x_t, time=time, audio_embeddings=audio).argmax(-1)
                tally.add(f"{name} t={t}", pred, tokens, real)
            tally.add(f"field t={t}", (a * eps + b * x1 + g * mu).argmax(-1), tokens, real)
    out = tally.recovery()
    out["mu peak prob"] = peak / tally.reals
    out["mu prob on target"] = on_target / tally.reals
    return out


def report(results: dict[str, dict[str, float]], out: Path) -> None:
    """Print the tables and draw recovery against t, one line per reading and role."""
    roles = list(results)
    print(f"\n{'recovery':24s}" + "".join(f"{r:>10s}" for r in roles))
    for key in ("mu", "mu peak prob", "mu prob on target"):
        print(f"{key:24s}" + "".join(f"{results[r][key]:10.3f}" for r in roles))
    print(
        f"\n{'t':>5s}"
        + "".join(
            f"{f'{name} ({r})':>24s}"
            for name in ("denoiser", "denoiser no mu", "field")
            for r in roles
        )
    )
    for t in TS:
        print(
            f"{t:5.2f}"
            + "".join(
                f"{results[r][f'{name} t={t}']:24.3f}"
                for name in ("denoiser", "denoiser no mu", "field")
                for r in roles
            )
        )
    print("\nfield argmax at t=0 with the real mu, x_0 = mu + sigma * eps")
    for sigma in SIGMAS:
        print(
            f"  sigma {sigma:.1f}"
            + "".join(f"{results[r][f'field@sigma={sigma} t=0']:10.3f}" for r in roles)
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, ls in zip(roles, ("-", "--", ":"), strict=False):
        for name, color in (("denoiser", "C0"), ("denoiser no mu", "C1"), ("field", "C2")):
            ax.plot(
                TS,
                [results[r][f"{name} t={t}"] for t in TS],
                ls,
                color=color,
                label=f"{name} ({r})",
            )
        ax.axhline(results[r]["mu"], color="C3", ls=ls, lw=1, label=f"mu argmax ({r})")
    ax.set_xlabel("t")
    ax.set_ylabel("token recovery")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "audio_info.png", dpi=120)


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild the run, probe each role, write the tables and the figure."""
    args = parse_args(argv)
    cfg = _config_from_components(args)
    cfg.data.batch_size = args.batch_size
    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    model = _model_from_components(args, cfg, dm)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader, "test": dm.test_dataloader}
    results = {r: probe(model, loaders[r](), args.n_utts, args.seed) for r in args.roles}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audio_info.json").write_text(json.dumps(results, indent=1))
    report(results, args.out)
    print(f"\nwrote {args.out}/audio_info.json, audio_info.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
