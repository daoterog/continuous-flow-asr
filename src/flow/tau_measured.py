"""Tau from the denoiser's own error, swept at t the current schedule did not choose."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.flow.monotone import centered_isotonic
from src.flow.recovery import recovery_curve
from src.flow.schedule import make_tau_inverse
from src.flow.tau_base import TauScheduler

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.pl_module import FlowMatching


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
        """Recovery the best bin must reach before its curve is worth inverting."""
        return float(self.cfg.get("measured_min_recovery", 0.05))

    def refresh(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule | None"
    ) -> tuple[Callable | None, str | None]:
        """Sweep, bin, invert."""
        if datamodule is None:
            return None, "tau refresh skipped: no datamodule to sweep"
        t, hits, input_hits, reals = self._sweep(model, self._draw_batch(datamodule))
        self._refreshes += 1

        ts, rec = self.monotone_curve(t, hits, reals)
        best = float(rec.max())
        if best < self.min_recovery:
            return None, (
                f"tau refresh skipped: the model recovers at most {best:.3f} of tokens "
                f"across a uniform sweep, under the {self.min_recovery} needed to anchor tau"
            )
        tau_inverse = make_tau_inverse(
            ts, rec, model.vocab_size, flat_eps=0.0, p_e0=float(1.0 - rec[0])
        )
        span = float(tau_inverse(1.0)) - float(tau_inverse(0.0))
        if span <= 0.0:
            return None, (
                f"tau refresh skipped: the model recovers {rec.min():.3f}-{rec.max():.3f} "
                f"of tokens with no trend across t, so tau^-1 is constant at "
                f"{float(tau_inverse(0.0)):.4f} and carries no schedule"
            )
        return tau_inverse, None

    def _draw_batch(self, datamodule: "AudioFeatureDataModule") -> dict:
        """A fresh random subset, different on every refresh."""
        subset = datamodule.datasets[self.refresh_split]
        rng = np.random.default_rng([self.seed, self._refreshes])
        n = min(self.n_utts, len(subset))
        idx = rng.choice(len(subset), size=n, replace=False)
        return datamodule.collate_fn([subset[int(i)] for i in idx])

    @torch.no_grad()
    def _sweep(
        self, model: "FlowMatching", batch: dict, chunk: int = 32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(t, denoiser hits, interpolant hits, real tokens) per utterance, at t ~ U[0, 1]."""
        audio_all = batch["audio"]
        tokens_all = batch["tokens"]
        gen = torch.Generator(device="cpu").manual_seed(self.seed + self._refreshes)
        t_all = torch.rand(tokens_all.size(0), generator=gen)

        was_training = model.training
        model.eval()
        ts, hits, input_hits, reals = [], [], [], []
        try:
            for i in range(0, tokens_all.size(0), chunk):
                audio = audio_all[i : i + chunk].to(model.device).float()
                tokens = tokens_all[i : i + chunk].to(model.device)
                t = t_all[i : i + chunk].to(model.device)
                x1 = F.one_hot(tokens, model.vocab_size).float()
                mu = (
                    F.softmax(model.mu_nn(audio), dim=-1)
                    if model.mu_nn is not None
                    else None
                )
                i_t = model.make_interpolant(x1=x1, t=t, mu=mu)
                pred = model.denoiser(
                    x_t=i_t, time=t, audio_embeddings=audio
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
