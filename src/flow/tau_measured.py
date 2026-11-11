"""Tau from the denoiser's own error, swept at t the current schedule did not choose."""

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.flow.monotone import centered_isotonic
from src.flow.schedule import TimeReparametrization
from src.flow.tau_base import TauScheduler

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.flow_matching import FlowMatching


class MeasuredTau(TauScheduler):
    """Sweeps the denoiser at t ~ U[0, 1] on a fresh subset, and inverts the curve.

    The corruption sweep only knows what the interpolant destroys. This reads the model, so
    it accounts for the audio the denoiser is about to see: at low t the field carries
    nothing and recovery is whatever the audio alone supports, which is well above chance
    and nothing a model-free sweep can predict.

    It draws its own t rather than reading a metric filled by training. A training-fed
    curve inherits wherever the schedule happens to draw, and the schedule draws nothing
    below its own t0 — which closed a loop: the region below went unmeasured, was
    extrapolated flat, and normalising by that curve's worst error pushed t0 further right
    every refresh (0.250 -> 0.500 -> 0.625 -> 0.750, then stuck). Binning that curve
    uniformly does not help, because uniform bins cannot invent samples that were never
    drawn: measured on a trained checkpoint it still left 11 of 32 bins empty, with the
    lowest populated bin at t=0.214, against t=0.015 for this sweep.

    Choosing t here instead gives uniform coverage by construction — no empty bins, no
    extrapolation at either end, equal precision across the interval — and both anchors are
    real readings, so the sampler's grid (which is this same tau^-1) spans [0, 1]. The
    subset is redrawn every refresh, so successive curves are independent samples of the
    data rather than repeated readings of one fixed batch.

    It needs no mu, so it refreshes on a linear run too. The cost is
    `ceil(n_utts / chunk)` forward passes per refresh, which against a refresh every N
    epochs is noise.
    """

    # Always the denoiser's own curve; `WarmupTau` is what moves between sources.
    source = "measured"

    def __init__(self, cfg: DictConfig, seed: int = 0) -> None:
        """Count refreshes, so each one draws a different subset and different t."""
        super().__init__(cfg, seed=seed)
        self._refreshes = 0

    @property
    def n_utts(self) -> int:
        """Utterances per sweep. Well above `num_bins`, so no bin comes up empty."""
        return int(self.cfg.get("probe_n_utts", 512))

    @property
    def min_recovery(self) -> float:
        """Recovery the best bin must reach before its curve is worth inverting.

        A weak gate, and deliberately kept as only the first one. It reads `rec.max()`,
        which sits at t ~ 1 where the field is already almost clean, so it is high for any
        model from the first epoch and says more about the top of the interval than about
        the denoiser. `max_median_t` is what actually decides whether a curve is usable.
        """
        return float(self.cfg.get("measured_min_recovery", 0.05))

    @property
    def max_median_t(self) -> float:
        """How concentrated the schedule a curve produces may be before it is refused.

        An early measured curve recovers nothing until t is almost 1 and everything after,
        so tau stays at 0 across the interval and jumps at the end — which inverts to a
        schedule that draws its whole batch from that last sliver. Measured on full data at
        refresh-every-epoch, the median drawn t was 0.993 at epoch 1 against 0.805 by epoch
        5, so the first curve would have trained almost entirely where the model already
        succeeded. Neither existing guard sees it: `min_recovery` reads the peak, which is
        high precisely because t ~ 1 is easy, and the span check only catches a curve that
        is flat everywhere.

        The bound is on the median rather than on t0, because t0 is near 0 throughout — it
        is the spread that collapses, not the start. 0.9 is loose against the simulated
        curve's own 0.767, so it refuses only the degenerate shape and not a schedule that
        is merely top-heavy, which a real one legitimately is.
        """
        return float(self.cfg.get("measured_max_median_t", 0.9))

    def refresh(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule | None"
    ) -> tuple[TimeReparametrization | None, str | None]:
        """Sweep, bin, invert."""
        if datamodule is None:
            return None, "tau refresh skipped: no datamodule to sweep"
        # `_sweep` also scores argmax(i_t) — the interpolant's own recovery on the same
        # tokens and t — which nothing reads since the self-check was dropped. Kept because
        # it is the one quantity the closed form predicts exactly, so it is what a probe
        # self-check would hold itself against; the cost is one argmax per chunk.
        t, hits, _input_hits, reals = self._sweep_split(
            model, datamodule, chunk=getattr(datamodule, "eval_batch_size", 32)
        )
        self._refreshes += 1

        ts, rec = self.monotone_curve(t, hits, reals)
        best = float(rec.max())
        if best < self.min_recovery:
            return None, (
                f"tau refresh skipped: the model recovers at most {best:.3f} of tokens "
                f"across a uniform sweep, under the {self.min_recovery} needed to anchor tau"
            )
        reparam = TimeReparametrization.from_recovery(
            ts, rec, model.vocab_size, flat_eps=0.0, p_e0=float(1.0 - rec[0])
        )
        if reparam.t_end <= reparam.t_start:
            return None, (
                f"tau refresh skipped: the model recovers {rec.min():.3f}-{rec.max():.3f} "
                f"of tokens with no trend across t, so tau^-1 is constant at "
                f"{reparam.t_start:.4f} and carries no schedule"
            )
        median_t = reparam.median_t
        if median_t > self.max_median_t:
            return None, (
                f"tau refresh skipped: the curve puts half the draw above t={median_t:.4f}, "
                f"over the {self.max_median_t} allowed — it recovers nothing until t is "
                f"nearly 1, so training it would mean training where the model already wins"
            )
        return reparam, None

    def _draw_indices(self, datamodule: "AudioFeatureDataModule") -> np.ndarray:
        """A fresh random subset of the refresh split, different on every refresh."""
        n_split = len(datamodule.datasets[self.refresh_split])
        rng = np.random.default_rng([self.seed, self._refreshes])
        return rng.choice(n_split, size=min(self.n_utts, n_split), replace=False)

    def _draw_batch(self, datamodule: "AudioFeatureDataModule") -> dict:
        """The same subset collated whole, for scripts that want it in memory at once."""
        subset = datamodule.datasets[self.refresh_split]
        return datamodule.collate_fn([subset[int(i)] for i in self._draw_indices(datamodule)])

    def _sweep_split(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule", chunk: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`_sweep` over a fresh subset, collated one chunk at a time.

        Collating the whole subset first is what `_draw_batch` does, and at full latents
        8,192 utterances of audio is ~31 GB of host memory per run. The t for every
        utterance is drawn up front, so the readings match a whole-subset sweep exactly.
        """
        subset = datamodule.datasets[self.refresh_split]
        idx = self._draw_indices(datamodule)
        gen = torch.Generator(device="cpu").manual_seed(self.seed + self._refreshes)
        t_all = torch.rand(len(idx), generator=gen)
        parts = []
        for i in range(0, len(idx), chunk):
            batch = datamodule.collate_fn([subset[int(j)] for j in idx[i : i + chunk]])
            parts.append(self._sweep(model, batch, chunk=chunk, t=t_all[i : i + chunk]))
        return tuple(np.concatenate(p) for p in zip(*parts, strict=True))

    @torch.no_grad()
    def _sweep(
        self,
        model: "FlowMatching",
        batch: dict,
        chunk: int = 32,
        t: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(t, denoiser hits, interpolant hits, real tokens) per utterance, at t ~ U[0, 1].

        `chunk` is how many utterances cross to the device and through the model at once.
        Nothing here holds a backward graph, so it is not bounded by what training fits —
        `refresh` passes the datamodule's `eval_batch_size` and the default is only the
        floor for a caller that has no datamodule to ask.

        It is a transfer size as much as a batch size: moving the whole probe to the device
        at once cost 3.66 GiB against 0.32 GiB streamed, which is what the chunking is for.

        `t` gives each utterance its flow time; drawn here, from this refresh's seed, when
        it is not.
        """
        audio_all = batch["audio"]
        tokens_all = batch["tokens"]
        if t is None:
            gen = torch.Generator(device="cpu").manual_seed(self.seed + self._refreshes)
            t = torch.rand(tokens_all.size(0), generator=gen)
        t_all = t

        was_training = model.training
        model.eval()
        ts, hits, input_hits, reals = [], [], [], []
        try:
            for i in range(0, tokens_all.size(0), chunk):
                audio = audio_all[i : i + chunk].to(model.device).float()
                tokens = tokens_all[i : i + chunk].to(model.device)
                t = t_all[i : i + chunk].to(model.device)
                x1 = F.one_hot(tokens, model.vocab_size).float()
                mu_logits, sigma = model.mu_and_sigma(audio)
                mu = F.softmax(mu_logits, dim=-1) if mu_logits is not None else None
                i_t = model.make_interpolant(x1=x1, t=t, mu=mu, sigma=sigma)
                # The sweep picks t, not u, so a `uniform` denoiser is fed u back off the
                # curve it is currently trained against.
                time = (
                    model.time_reparam.u_of(t) if model.time_input == "uniform" else t
                )
                pred = model.denoiser(
                    x_t=i_t, time=time, audio_embeddings=audio
                ).argmax(dim=-1)
                real = tokens != model.pad_id
                ts.append(t.cpu().numpy())
                hits.append(((pred == tokens) & real).sum(dim=1).cpu().numpy())
                input_hits.append(
                    ((i_t.argmax(dim=-1) == tokens) & real).sum(dim=1).cpu().numpy()
                )
                reals.append(real.sum(dim=1).cpu().numpy())
        finally:
            model.train(was_training)
        return (
            np.concatenate(ts),
            np.concatenate(hits),
            np.concatenate(input_hits),
            np.concatenate(reals),
        )

    def monotone_curve(
        self, t: np.ndarray, hits: np.ndarray, reals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(t, recovery) knots for tau, fitted to the raw sweep with no binning at all.

        One observation per utterance — its own recovery, weighted by how many real tokens
        it carries, which makes the centered isotonic fit the binomial MLE. Binning first
        would pool neighbours into a fixed width chosen in advance; this lets the fit pool
        exactly as far as the data supports and no further, so `num_bins` stops being a
        precision knob for the schedule and stays only for the metrics' histograms.
        """
        order = np.argsort(t)
        t_sorted = t[order].astype(np.float64)
        weights = reals[order].astype(np.float64)
        # Utterances that are all padding carry no reading; give them zero weight rather
        # than dividing by zero to get one.
        recovery = np.divide(
            hits[order], weights, out=np.zeros_like(weights), where=weights > 0
        )
        return centered_isotonic(t_sorted, recovery, weights)
