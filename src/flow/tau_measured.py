"""Tau from the denoiser's own error, swept at t the current schedule did not choose."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

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
        ts, rec = self.curve(model, datamodule)
        self._refreshes += 1

        if ts.size == 0:
            return None, "tau refresh skipped: the sweep filled no bins"
        best = float(rec.max())
        if best < self.min_recovery:
            return None, (
                f"tau refresh skipped: the model recovers at most {best:.3f} of tokens "
                f"across a uniform sweep, under the {self.min_recovery} needed to anchor tau"
            )
        # Both ends are measured rather than assumed, so anchor_zero anchors to a real
        # reading and the top needs no carrying up to t=1.
        return make_tau_inverse(
            ts, rec, model.vocab_size, p_e0=float(1.0 - rec[0]), anchor_zero=True
        ), None

    def curve(
        self, model: "FlowMatching", datamodule: "AudioFeatureDataModule"
    ) -> tuple[np.ndarray, np.ndarray]:
        """(mean t, recovery) per populated bin, from one fresh uniform sweep.

        Public because it is the honest P_e curve, useful on its own for inspecting a
        checkpoint — see scripts/schedule/measure_tau_bias.py.
        """
        t, hits, reals = self._sweep(model, self._draw_batch(datamodule))
        return self._bin(t, hits, reals)

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, tokens recovered, real tokens) per utterance, at t drawn uniformly."""
        audio_all = batch["audio"].to(model.device).float()
        tokens_all = batch["tokens"].to(model.device)
        gen = torch.Generator(device="cpu").manual_seed(self.seed + self._refreshes)
        t_all = torch.rand(tokens_all.size(0), generator=gen).to(model.device)

        was_training = model.training
        model.eval()
        ts, hits, reals = [], [], []
        try:
            for i in range(0, tokens_all.size(0), chunk):
                audio, tokens = audio_all[i : i + chunk], tokens_all[i : i + chunk]
                t = t_all[i : i + chunk]
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
                reals.append(real.sum(dim=1).cpu().numpy())
        finally:
            model.train(was_training)
        return np.concatenate(ts), np.concatenate(hits), np.concatenate(reals)

    def _bin(
        self, t: np.ndarray, hits: np.ndarray, reals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform bins over [0, 1] -> (mean t, recovery) for the populated ones.

        Recovery is pooled over tokens, not averaged over utterances, so a long utterance
        counts for more than a short one — the same weighting `BinnedTokenError` uses.
        """
        n = int(self.cfg.num_bins)
        edges = np.linspace(0.0, 1.0, n + 1)
        idx = np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, n - 1)
        centres, recovery = [], []
        for b in range(n):
            m = idx == b
            if m.any() and reals[m].sum() > 0:
                centres.append(float(t[m].mean()))
                recovery.append(float(hits[m].sum() / reals[m].sum()))
        return np.asarray(centres), np.asarray(recovery)
