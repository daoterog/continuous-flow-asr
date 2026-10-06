"""Three ways to estimate the denoiser's P_e curve, and what tau each one implies.

A measured refresh rebuilds tau from the denoiser's own error. How that error is estimated
turns out to matter more than anything else about the refresh, so this measures all three
candidates on the same checkpoint:

  schedule   t drawn through tau^-1 (as training does), binned on tau's own quantiles.
             This is `denoiser_perr`. It was the original source and it ratcheted: the
             bins only cover where tau already draws, so the refresh never sees below t0.
  training   the same draws, binned uniformly. This is `denoiser_uperr`, what the refresh
             reads today. Uniform bins cannot invent samples, so wherever tau puts little
             mass the bin is noisy, and where it puts none the bin is empty.
  sweep      t on a fixed uniform grid, independent of tau entirely. Equal precision
             everywhere by construction, at the cost of its own forward passes.

It also checks the two endpoints against the sampler, which integrates on
`TimeReparametrization.sampling_grid(n)`: tau^-1(0) is where sampling starts and tau^-1(1) is where it
ends. If the schedule does not reach t=1 the sampler stops short of the clean endpoint.

    python -m scripts.schedule.measure_tau_bias --components runs/<id>/components
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import train as T
from src.checkpointing import load_components, load_manifest
from src.flow.schedule import TimeReparametrization
from src.flow.tau_measured import MeasuredTau
from src.pl_module import FlowMatching

ESTIMATORS = ("schedule", "training", "sweep")


def draw_records(
    model: FlowMatching, batch: dict, reparam: TimeReparametrization, repeats: int,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, correct, real) per utterance, with t drawn exactly as training draws it."""
    audio_all = batch["audio"].to(model.device).float()
    tokens_all = batch["tokens"].to(model.device)
    ts, hits, reals = [], [], []
    for r in range(repeats):
        g = torch.Generator(device="cpu").manual_seed(r)
        u = torch.rand(tokens_all.size(0), generator=g).to(model.device)
        t_all = reparam.t_of(u)
        t_all = t_all if torch.is_tensor(t_all) else torch.as_tensor(t_all)
        t_all = t_all.to(model.device).float()
        # What the denoiser's time input reads, as training feeds it.
        time_all = u if model.time_input == "uniform" else t_all
        for i in range(0, tokens_all.size(0), chunk):
            audio, tokens = audio_all[i : i + chunk], tokens_all[i : i + chunk]
            t, time = t_all[i : i + chunk], time_all[i : i + chunk]
            x1 = F.one_hot(tokens, model.vocab_size).float()
            mu = F.softmax(model.mu_nn(audio), dim=-1) if model.mu_nn is not None else None
            i_t = model.make_interpolant(x1=x1, t=t, mu=mu)
            pred = model.denoiser(x_t=i_t, time=time, audio_embeddings=audio).argmax(dim=-1)
            real = tokens != model.pad_id
            ts.append(t.cpu().numpy())
            hits.append(((pred == tokens) & real).sum(dim=1).cpu().numpy())
            reals.append(real.sum(dim=1).cpu().numpy())
    return np.concatenate(ts), np.concatenate(hits), np.concatenate(reals)


def bin_records(
    t: np.ndarray, hits: np.ndarray, reals: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, recovery, samples) per bin — empty bins dropped, as the metric does."""
    idx = np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, edges.size - 2)
    centres, recovery, counts = [], [], []
    for b in range(edges.size - 1):
        m = idx == b
        counts.append(int(m.sum()))
        if m.any() and reals[m].sum() > 0:
            centres.append(float(t[m].mean()))
            recovery.append(float(hits[m].sum() / reals[m].sum()))
    return np.asarray(centres), np.asarray(recovery), np.asarray(counts)


def tau_of(ts: np.ndarray, rec: np.ndarray, vocab: int) -> TimeReparametrization:
    """The reparametrization a curve implies, under the measured strategy's own rules."""
    return TimeReparametrization.from_recovery(ts, rec, vocab, p_e0=float(1.0 - rec[0]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Which run to probe, how finely, and how many draws to bin."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--variant", default="final", choices=["best", "final"])
    p.add_argument("--n-utts", type=int, default=256)
    p.add_argument("--num-bins", type=int, default=32)
    p.add_argument("--repeats", type=int, default=20, help="draw passes for the binned ones")
    p.add_argument("--steps", type=int, default=16, help="sampler steps for the grid check")
    p.add_argument("--out", type=Path, default=Path("figures/tau_bias.png"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Estimate the curve three ways and report the tau and sampler grid each gives."""
    args = parse_args(argv)
    manifest = load_manifest(args.components)
    cfg = OmegaConf.create(manifest.run_config)
    cfg.schedule.num_bins = args.num_bins

    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    reparam = T.build_time_reparam(dm, cfg)
    model = FlowMatching(
        cfg, vocab_size=dm.vocab_size, pad_id=dm.pad_id, canvas_len=cfg.data.canvas_len,
        tokenizer=dm.tokenizer, vocab_map=dm.vocab_map, time_reparam=reparam,
    )
    _, built = load_components(args.components, variant=args.variant)
    for name, sub in built.items():
        getattr(model, name).load_state_dict(sub.state_dict(), strict=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    cfg.schedule.probe_n_utts = args.n_utts
    probe = MeasuredTau(cfg.schedule)
    batch = probe._draw_batch(dm)
    t0_now = reparam.t_start
    print(f"run trained under {cfg.schedule.refresh_curve_from}, "
          f"its schedule draws t in [{t0_now:.3f}, {reparam.t_end:.3f}]\n")

    curves = {}
    with torch.no_grad():
        t, hits, reals = draw_records(model, batch, reparam, args.repeats, 32)
        sched_edges = np.asarray(
            reparam.t_of(np.linspace(0, 1, args.num_bins + 1)), dtype=np.float64
        )
        sched_edges = np.maximum.accumulate(sched_edges)
        curves["schedule"] = bin_records(t, hits, reals, sched_edges)
        curves["training"] = bin_records(
            t, hits, reals, np.linspace(0, 1, args.num_bins + 1)
        )
        grid_ts, grid_rec = probe.curve(model, dm)
        curves["sweep"] = (grid_ts, grid_rec, np.full(grid_ts.size, args.n_utts))

    print(f"{'estimator':<10} {'bins':>5} {'lowest t':>9} {'empty':>6} "
          f"{'tau^-1(0)':>10} {'tau^-1(1)':>10}")
    taus = {}
    for name in ESTIMATORS:
        c, r, counts = curves[name]
        taus[name] = tau_of(c, r, dm.vocab_size)
        empty = int((counts == 0).sum()) if name != "sweep" else 0
        print(f"{name:<10} {c.size:>5} {c[0]:9.3f} {empty:>6} "
              f"{taus[name].t_start:10.3f} {taus[name].t_end:10.3f}")

    print(f"\nsampler grid ({args.steps} steps), from each tau:")
    for name in ESTIMATORS:
        g = taus[name].sampling_grid(args.steps)[0]
        ends_at_one = "yes" if abs(g[-1] - 1.0) < 1e-6 else f"NO (stops at {g[-1]:.4f})"
        print(f"  {name:<10} starts {g[0]:.4f}  ends {g[-1]:.4f}  reaches t=1: {ends_at_one}")
        print(f"             {np.round(g, 3)}")

    u = np.linspace(0, 1, 2000)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    colours = {"schedule": "crimson", "training": "darkorange", "sweep": "seagreen"}

    ax = axes[0]
    for name in ESTIMATORS:
        c, r, _ = curves[name]
        ax.plot(c, 1 - r, "o-", ms=3, color=colours[name], label=name)
    ax.axvspan(0, t0_now, color="grey", alpha=0.2, label=f"below t0={t0_now:.2f}")
    ax.set(xlabel="t", ylabel="$P_e(t)$", title="the error curve, three ways")
    ax.legend(fontsize=8)

    ax = axes[1]
    for name in ESTIMATORS:
        ax.plot(u, taus[name].t_of(u), color=colours[name], label=name)
    ax.plot(u, u, ls=":", color="grey", lw=1)
    ax.set(xlabel="u", ylabel="t", title=r"the $\tau^{-1}$ each implies")
    ax.legend(fontsize=8)

    ax = axes[2]
    for i, name in enumerate(ESTIMATORS):
        g = taus[name].sampling_grid(args.steps)[0]
        ax.plot(g, np.full_like(g, i), "o", ms=5, color=colours[name])
        ax.text(0.02, i + 0.15, name, fontsize=8, color=colours[name])
    ax.axvline(1.0, ls="--", color="grey", lw=1)
    ax.set(xlabel="t", yticks=[], ylim=(-0.5, len(ESTIMATORS)),
           title=f"where the sampler steps ({args.steps} steps)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
