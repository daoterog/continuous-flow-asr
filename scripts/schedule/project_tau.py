"""Replay a finished run's logs through every tau strategy, to see curves before paying.

A run under `simulated` tau logs, every validation epoch, the denoiser's token error in ten
bins cut at the deciles of the curve it was drawing from (`denoiser_perr/val_bins`), and
the mu accuracy that curve was built at (`tau/mu_accuracy_ema`). The curve is rebuildable
exactly from the latter, so each bin's flow time is known, and the bins together are the
denoiser's own P_e(t) at that epoch. That is all `measured`, `warmup` and `blend` consume,
so each strategy's curve can be rebuilt at every refresh it would have made, with the real
constructors (`TimeReparametrization.exact` / `from_recovery` / `mixed_with`, and the
measured path's `centered_isotonic`).

It is a projection, not a counterfactual run: the error curve came from a model trained
under simulated tau, and a model trained under another curve would drift from it.

Two things the replay cannot see, stated where they bite:

  * Below the first bin's centre P_e is held flat. Under simulated tau that bin spans
    t in [0, ~0.6] and the field carries nothing over most of it, so the error there is
    what the audio alone supports; but its slope inside the bin is not observed.
  * The measured probe reads `schedule.refresh_split`, train in the recipe, while these
    bins are val. A model that fits train better at low t would put more mass there than
    this projects.

`screen.png` draws the arms in `SCREEN` alone; `overlay.png` sets every strategy against
the control. Everything lands in figures/tau_projection/.

    python -m scripts.schedule.project_tau --run flow-asr-flm-parity/olc06gip
    python -m scripts.schedule.project_tau --run flow-asr-flm-parity/olc06gip --floors 0.3 0.4
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.flow.coefficients import make_schedule
from src.flow.monotone import centered_isotonic
from src.flow.schedule import TimeReparametrization

ENTITY = "cont-flow-asr-"
NUM_BINS = 10
# The audio-only region: below it the field's own recovery is at chance (input_perr ~0.95
# in the lowest bins), so whatever the denoiser recovers there it read off the audio.
AUDIO_T_MAX = 0.15
T_GRID = np.linspace(0.0, 1.0, 401)
LOGGED = ("denoiser_perr/val_bins", "input_perr/val_bins", "mu_perr/train_overall")


def fetch(run: str, cache: Path) -> tuple[dict[int, dict], dict]:
    """{epoch: {key: value}} for the keys the replay reads, plus the run's config."""
    path = cache / f"{run.replace('/', '_')}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return {int(k): v for k, v in data["history"].items()}, data["config"]
    import wandb

    r = wandb.Api().run(f"{ENTITY}/{run}")
    history: dict[int, dict] = defaultdict(dict)
    for row in r.scan_history():
        epoch = row.get("epoch")
        if epoch is None:
            continue
        for key, value in row.items():
            if value is None or not (key.startswith("tau/") or key in LOGGED):
                continue
            history[int(epoch)][key] = value["values"] if isinstance(value, dict) else value
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"history": history, "config": dict(r.config)}))
    return dict(history), dict(r.config)


class Replay:
    """One run's logs, and the curves each strategy would have built from them."""

    def __init__(self, history: dict[int, dict], config: dict, vocab_size: int) -> None:
        """Hold the logs and the run's interpolant, which every simulated curve needs."""
        self.history = history
        self.vocab_size = vocab_size
        flow, schedule = config["flow"], config["schedule"]
        self.coeffs = tuple(
            make_schedule(flow.get(k, d))
            for k, d in (("alpha", "linear_down"), ("beta", "linear_up"), ("gamma", None))
        )
        self.run_momentum = float(schedule.get("mu_accuracy_momentum", 0.0))
        self.run_every = int(schedule.get("refresh_every_n_epochs", 5))
        self.epochs = sorted(history)
        self._exact: dict[tuple[float, float], TimeReparametrization] = {}

    def exact(self, accuracy: float, sigma: float = 1.0) -> TimeReparametrization:
        """The simulated curve at mu accuracy `accuracy`, cached to 4 decimals.

        `sigma` rescales alpha, the noise's coefficient, to sigma * (1 - t): x0 is then
        drawn from N(mu, sigma^2 I) rather than N(mu, I). mu trains under a stop-gradient,
        so its accuracy trajectory is the logged one whatever sigma the denoiser sees.
        """
        key = (round(accuracy, 4), sigma)
        if key not in self._exact:
            alpha, beta, gamma = self.coeffs
            self._exact[key] = TimeReparametrization.exact(
                self.vocab_size,
                (lambda t: sigma * alpha(t)) if sigma != 1.0 else alpha,
                beta,
                gamma,
                mu_accuracy=key[0],
            )
        return self._exact[key]

    def mu_accuracy(self, momentum: float) -> dict[int, float]:
        """The accuracy `SimulatedTau` holds after each epoch, under `momentum`.

        The same bias-corrected EMA as `SimulatedTau.smooth_mu_accuracy`, fed the logged
        train mu error epoch by epoch.
        """
        out, ema, steps = {}, 0.0, 0
        for epoch in self.epochs:
            perr = self.history[epoch].get("mu_perr/train_overall")
            if perr is None:
                continue
            steps += 1
            ema = momentum * ema + (1.0 - momentum) * (1.0 - perr)
            out[epoch] = ema / (1.0 - momentum**steps) if momentum else 1.0 - perr
        return out

    def simulated(
        self, momentum: float, every: int, sigma: float = 1.0
    ) -> dict[int, TimeReparametrization]:
        """{refresh epoch: curve} for `simulated` at `momentum`, refreshing every `every`.

        Epoch 0 is the initial curve, at `schedule.mu_accuracy` 0. A refresh at the start of
        epoch r reads the accuracy as it stood after epoch r-1.
        """
        acc = self.mu_accuracy(momentum)
        curves = {0: self.exact(0.0, sigma)}
        for r in range(every, self.epochs[-1] + 1, every):
            if r - 1 in acc:
                curves[r] = self.exact(acc[r - 1], sigma)
        return curves

    def live_curve(self, epoch: int) -> TimeReparametrization:
        """The curve the run itself was drawing from during `epoch`."""
        curves = self.simulated(self.run_momentum, self.run_every)
        return curves[max(r for r in curves if r <= epoch)]

    def denoiser_pe(
        self, epoch: int, key: str = "denoiser_perr/val_bins"
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """(t, P_e) at each bin's mean flow time, for a validation epoch that logged bins."""
        values = self.history[epoch].get(key)
        if values is None:
            return None
        curve = self.live_curve(epoch)
        # Each bin is a tenth of the u draw; its mean t is the mean of tau^-1 over that tenth.
        u = np.linspace(0.0, 1.0, NUM_BINS * 50 + 1)
        t = np.asarray(curve.t_of(u))
        centres = np.array([t[i * 50 : (i + 1) * 50 + 1].mean() for i in range(NUM_BINS)])
        return centres, np.asarray(values, dtype=np.float64)

    def measured(self, epoch: int) -> tuple[TimeReparametrization | None, dict]:
        """What `MeasuredTau.refresh` would build from the bins logged at `epoch`.

        The same steps in the same order: a centered isotonic fit of recovery against t,
        the same two gates, then `from_recovery` with `p_e0` read off the fit's first point.
        The fit is extended flat to t=0 below the first bin and to recovery 1 at t=1.
        """
        reading = self.denoiser_pe(epoch)
        if reading is None:
            return None, {}
        centres, pe = reading
        ts, rec = centered_isotonic(centres, 1.0 - pe, np.ones_like(pe))
        dense = np.interp(T_GRID, np.r_[0.0, ts, 1.0], np.r_[rec[0], rec, 1.0])
        info = {"audio_recovery": float(1.0 - np.interp(AUDIO_T_MAX / 2, centres, pe))}
        if dense.max() < 0.05:
            info["declined"] = "max recovery under 0.05"
            return None, info
        curve = TimeReparametrization.from_recovery(
            T_GRID, dense, self.vocab_size, flat_eps=0.0, p_e0=float(1.0 - dense[0])
        )
        if curve.t_end <= curve.t_start or curve.median_t > 0.9:
            info["declined"] = f"median t {curve.median_t:.3f} over 0.9"
            return None, info
        return curve, info


def strategies(
    replay: Replay,
    every: int,
    floors: list[float],
    targets: list[float],
    measured_floors: list[float],
) -> dict[str, dict[int, tuple[TimeReparametrization, dict]]]:
    """{strategy: {refresh epoch: (curve, scalars)}} for every arm the screen would run."""
    identity = TimeReparametrization.identity()
    val_epochs = [e for e in replay.epochs if "denoiser_perr/val_bins" in replay.history[e]]
    out: dict[str, dict[int, tuple]] = {}
    for m in (0.0, 0.5, 0.8):
        acc = replay.mu_accuracy(m)
        out[f"simulated m={m}"] = {
            r: (c, {"mu_accuracy": acc.get(r - 1, 0.0)}) for r, c in replay.simulated(m, 5).items()
        }
    sim0 = replay.simulated(0.0, 5)
    for floor in floors:
        out[f"floor {floor}"] = {r: (c.mixed_with(identity, floor), {}) for r, c in sim0.items()}

    # The measured-based arms refresh every `every` epochs off the latest validation read.
    sim_every = replay.simulated(0.0, every)
    measured, warmup = {0: (sim0[0], {})}, {0: (sim0[0], {"source": 0.0})}
    blends = {t: {0: (sim0[0], {"alpha": 0.0})} for t in targets}
    switched = False
    for r in range(every, replay.epochs[-1] + 1, every):
        reads = [e for e in val_epochs if e < r]
        if not reads or r not in sim_every:
            continue
        curve, info = replay.measured(reads[-1])
        sim = sim_every[r]
        if curve is not None:
            measured[r] = (curve, info)
        switched = switched or curve is not None
        warmup[r] = (curve if switched and curve is not None else sim, {"source": float(switched)})
        for target in targets:
            alpha = 0.0 if curve is None else min(1.0, info["audio_recovery"] / target)
            mixed = sim if curve is None else sim.mixed_with(curve, alpha)
            blends[target][r] = (mixed, {**info, "alpha": alpha})
    out["measured"] = measured
    out["warmup"] = warmup
    for target, curves in blends.items():
        out[f"blend target={target}"] = curves
    for floor in measured_floors:
        out[f"measured + floor {floor}"] = {
            r: (c.mixed_with(identity, floor), info) for r, (c, info) in measured.items()
        }
    return out


# Where the draw lands, in the regions the curves differ over: below the field's dead-zone
# edge (audio only), the field's transition, and above where the field saturates.
REGIONS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0))


def region_mass(curve: TimeReparametrization) -> list[float]:
    """The fraction of a batch drawn in each of `REGIONS`: u(hi) - u(lo)."""
    return [float(curve.u_of(hi) - curve.u_of(lo)) for lo, hi in REGIONS]


def plot_overlay(arms: dict, control: str, epochs: list[int], out: Path) -> None:
    """Every strategy against the control, one row each: density, control dashed."""
    # warmup is left out: on these logs it switches at its first refresh and is `measured`.
    names = [n for n in arms if n != control and n != "warmup" and not n.startswith("sim")]
    fig, axes = plt.subplots(
        len(names),
        len(epochs),
        figsize=(3.6 * len(epochs), 2.2 * len(names)),
        sharex=True,
        sharey=True,
    )
    for i, name in enumerate(names):
        for j, epoch in enumerate(epochs):
            ax = axes[i, j]
            for arm, style in ((control, dict(color="k", ls="--", lw=1)), (name, {})):
                curves = arms[arm]
                c = curves[max(k for k in curves if k <= epoch)][0]
                ax.plot(T_GRID, density(c), **style)
            c = arms[name][max(k for k in arms[name] if k <= epoch)][0]
            low = c.u_of(0.5)
            ax.text(0.02, 0.85, f"t<0.5: {low:.0%}", transform=ax.transAxes, fontsize=8)
            if i == 0:
                ax.set_title(f"epoch {epoch}")
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    axes[0, 0].set_ylim(0, 8)
    fig.suptitle(f"p(t) per strategy (solid) against {control} (dashed)")
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=110)


# The screen's arms: (name, sigma, uniform floor). All simulated tau at momentum 0,
# refreshed every 5 epochs.
SCREEN = (
    ("control", 1.0, 0.0),
    ("floor 0.3", 1.0, 0.3),
    ("floor 0.4", 1.0, 0.4),
    ("sigma 0.7", 0.7, 0.0),
    ("sigma 0.5", 0.5, 0.0),
    ("sigma 0.4", 0.4, 0.0),
)


def screen_arms(replay: Replay) -> dict[str, dict[int, tuple]]:
    """{arm: {refresh epoch: (curve, scalars)}} for the arms in `SCREEN`."""
    acc = replay.mu_accuracy(0.0)
    identity = TimeReparametrization.identity()
    return {
        name: {
            r: (c.mixed_with(identity, floor), {"mu_accuracy": acc.get(r - 1, 0.0)})
            for r, c in replay.simulated(0.0, 5, sigma).items()
        }
        for name, sigma, floor in SCREEN
    }


def plot_screen(arms: dict, steps: int, out: Path) -> None:
    """One row per arm: u(t) and p(t) through training, and the sampling grid it walks.

    Curves are coloured by the refresh epoch, so the left-to-right drift is how the arm
    moves its mass as mu learns. The control at the last epoch is dashed on every row.
    """
    cmap = plt.get_cmap("viridis")
    control = arms["control"][max(arms["control"])][0]
    fig, axes = plt.subplots(len(arms), 3, figsize=(15, 2.5 * len(arms)), sharex=True)
    for row, (name, curves) in zip(axes, arms.items(), strict=True):
        last = max(curves)
        for r, (c, _) in sorted(curves.items()):
            color = cmap(r / max(last, 1))
            row[0].plot(T_GRID, c.u_of(T_GRID), color=color, lw=1)
            row[1].plot(T_GRID, density(c), color=color, lw=1)
        row[0].plot(T_GRID, control.u_of(T_GRID), "k--", lw=1)
        row[1].plot(T_GRID, density(control), "k--", lw=1)
        for r, y in ((0, 1.0), (last, 0.0)):
            grid = curves[r][0].sampling_grid(steps)[0]
            row[2].vlines(grid, y - 0.4, y + 0.4, color=cmap(r / max(last, 1)))
        grid = control.sampling_grid(steps)[0]
        row[2].vlines(grid, -1.0, -0.6, color="k", lw=1)
        row[2].set_yticks([1.0, 0.0, -0.8], ["epoch 0", f"epoch {last}", "control"])
        row[2].set_ylim(-1.1, 1.5)
        final = curves[last][0]
        shares = ", ".join(
            f"[{lo},{hi}) {m:.0%}" for (lo, hi), m in zip(REGIONS, region_mass(final), strict=True)
        )
        row[0].set_ylabel(name, fontsize=9)
        row[1].set_ylim(0, 8)
        row[1].set_title(f"epoch {last}: {shares}", fontsize=8)
    axes[0, 0].set_title("u = tau(t), coloured by refresh epoch")
    axes[0, 2].set_title(f"{steps}-step sampling grid")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out / "screen.png", dpi=110)


def density(curve: TimeReparametrization) -> np.ndarray:
    """p(t) on T_GRID: the slope of u(t), which is the draw's density over flow time."""
    u = np.asarray(curve.u_of(T_GRID))
    return np.gradient(u, T_GRID)


def export(arms: dict, replay: Replay, steps: int) -> dict:
    """Everything the page draws, as plain lists."""
    series = {}
    for name, curves in arms.items():
        series[name] = {
            str(r): {
                "u": np.round(np.asarray(c.u_of(T_GRID)), 5).tolist(),
                "grid": np.round(c.sampling_grid(steps)[0], 4).tolist(),
                "q": np.round(c.quantiles([0.1, 0.25, 0.5, 0.75, 0.9]), 4).tolist(),
                **{k: v for k, v in info.items() if isinstance(v, float | int | str)},
            }
            for r, (c, info) in sorted(curves.items())
        }
    pe = {}
    for e in replay.epochs:
        reading = replay.denoiser_pe(e)
        field = replay.denoiser_pe(e, "input_perr/val_bins")
        if reading is not None:
            pe[str(e)] = {
                "t": np.round(reading[0], 4).tolist(),
                "denoiser": np.round(reading[1], 4).tolist(),
                "field": None if field is None else np.round(field[1], 4).tolist(),
            }
    return {"t": T_GRID.round(4).tolist(), "arms": series, "pe": pe}


def plot(arms: dict, epochs: list[int], out: Path) -> None:
    """Figure 1: every strategy's draw density at a few epochs. Figure 2: evolution."""
    names = list(arms)
    fig, axes = plt.subplots(2, len(epochs), figsize=(4.2 * len(epochs), 7), sharex=True)
    for j, epoch in enumerate(epochs):
        for name in names:
            curves = arms[name]
            r = max(k for k in curves if k <= epoch)
            c = curves[r][0]
            axes[0, j].plot(T_GRID, c.u_of(T_GRID), label=name)
            axes[1, j].plot(T_GRID, density(c), label=name)
        axes[0, j].set_title(f"epoch {epoch}")
        axes[1, j].set_xlabel("t")
        axes[1, j].set_yscale("log")
        axes[1, j].set_ylim(1e-2, 50)
    axes[0, 0].set_ylabel("u = tau(t)")
    axes[1, 0].set_ylabel("p(t)")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "strategies.png", dpi=120)

    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.4), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, name in zip(axes, names, strict=True):
        last = max(arms[name])
        for r, (c, _) in sorted(arms[name].items()):
            ax.plot(T_GRID, c.u_of(T_GRID), color=cmap(r / max(last, 1)), lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("t")
    axes[0].set_ylabel("u = tau(t)")
    fig.tight_layout()
    fig.savefig(out / "evolution.png", dpi=120)


def main() -> None:
    """Fetch, replay, write the figures and the page's data."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="project/run_id of a simulated-tau run")
    p.add_argument("--vocab-size", type=int, default=1193)
    p.add_argument("--every", type=int, default=10, help="refresh stride of measured arms")
    p.add_argument("--floors", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4])
    p.add_argument("--measured-floors", type=float, nargs="+", default=[0.2])
    p.add_argument("--targets", type=float, nargs="+", default=[0.8])
    p.add_argument("--control", default="simulated m=0.0")
    p.add_argument("--steps", type=int, default=16, help="sampling grid shown per curve")
    p.add_argument("--epochs", type=int, nargs="+", default=[10, 30, 60, 99])
    p.add_argument("--out", type=Path, default=Path("figures/tau_projection"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    history, config = fetch(args.run, args.out / "cache")
    replay = Replay(history, config, args.vocab_size)

    # The rebuild has to reproduce what the run logged, or every bin's t is wrong.
    for r, curve in replay.simulated(replay.run_momentum, replay.run_every).items():
        logged = [history.get(r, {}).get(f"tau/t_q{q}") for q in (10, 50, 90)]
        if None not in logged:
            err = np.abs(curve.quantiles([0.1, 0.5, 0.9]) - np.asarray(logged)).max()
            if err > 5e-3:
                raise SystemExit(f"epoch {r}: rebuilt quantiles off the logged ones by {err}")

    arms = strategies(replay, args.every, args.floors, args.targets, args.measured_floors)
    plot(arms, args.epochs, args.out)
    plot_overlay(arms, args.control, args.epochs, args.out)
    plot_screen(screen_arms(replay), args.steps, args.out)
    print(f"\nshare of the draw per t region, epoch {args.epochs[-1]}")
    print(f"{'':26s}" + "".join(f"{f'[{lo},{hi})':>12s}" for lo, hi in REGIONS))
    for name, curves in arms.items():
        c = curves[max(k for k in curves if k <= args.epochs[-1])][0]
        print(f"{name:26s}" + "".join(f"{m:12.1%}" for m in region_mass(c)))
    data = export(arms, replay, args.steps)
    data["run"] = args.run
    (args.out / "projection.json").write_text(json.dumps(data))
    print(f"wrote {args.out}/strategies.png, evolution.png, projection.json")


if __name__ == "__main__":
    main()
