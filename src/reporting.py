"""Everything that writes numbers to the logger, kept out of the model.

`MetricReporting` is a mixin for a LightningModule: it owns the metric bank, updates it
each batch, and at each epoch end emits the scalars and the per-bin histograms. It leans
on `self.log`, `self.logger`, `self.trainer` and `self.current_epoch`, so it only works
mixed into a LightningModule and must come before it in the bases.

It sits above `src.metrics` rather than inside it — the metrics are measurements, this is
the layer that decides what reaches wandb and in what shape, which is also why it may
reach for a `SamplingProbe` or the `TauScheduler` while the metrics package may not.
"""

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from transformers import WhisperTokenizerFast

from src.data import VocabMap
from src.flow.sampling_probe import SamplingProbe
from src.flow.schedule import TimeReparametrization
from src.flow.tau_base import TauScheduler
from src.metrics.base import TimeBinnedMetric
from src.metrics.utils import (
    bins_key,
    check_metrics,
    edge_values,
    is_binned,
    make_metrics,
    section_for,
    split_key,
)


class MetricReporting:
    """Metric bank plus the logging hooks, mixed into a LightningModule.

    Keys come out as `{group}_{metric}/{split}_{stat}` so one wandb section is one metric
    of one group, with train and val next to each other. A group is a model on the
    one-time path (`denoiser`, `mu`) and one of the distillation regimes on the flow-map
    path (`diagonal`, `offdiag`, `boundary`).

    The per-bin stats carry the name of the axis they resolve, so `_low_gap` and
    `_high_gap` cannot be read as flow times.
    """

    metrics: nn.ModuleDict
    names: tuple[str, ...]
    sampling_probe: SamplingProbe
    tau_scheduler: TauScheduler

    def init_metrics(
        self,
        names: tuple[str, ...],
        tokenizer: WhisperTokenizerFast,
        pad_id: int,
        vocab_map: VocabMap | None = None,
        time_reparam: TimeReparametrization | None = None,
        **bins: int | dict | None,
    ) -> None:
        """Build and validate the metric bank. Call once, from `__init__`."""
        self.names = tuple(names)
        self.metrics = make_metrics(
            self.names,
            tokenizer,
            pad_id,
            vocab_map,
            time_reparam=time_reparam,
            **bins,
        )
        check_metrics(self.metrics, time_reparam)

    def update_metrics(
        self, out: dict, target_ids: torch.Tensor, bin_value: torch.Tensor, split: str
    ) -> None:
        """Feed one batch's per-group records to every metric of this split.

        `target_ids` and `bin_value` are the defaults; a record may override either. That
        is what lets one `out` hold groups measured on different axes against different
        slices of the batch — flow-map distillation bins its off-diagonal group on the
        jump size and its diagonal group on the flow time, each over its own rows, while
        the one-time path overrides neither.
        """
        for name, part in out.items():
            record = {
                "pred_ids": part["pred_ids"],
                "target_ids": part.get("target_ids", target_ids),
                "bin_value": part.get("bin_value", bin_value),
            }
            if "loss_per_seq" in part:
                record["loss_per_seq"] = part["loss_per_seq"]
            for key, metric in self.metrics.items():
                if key.startswith(f"{split}_{name}_"):
                    metric.update(*(record[k] for k in metric.inputs))

    def log_losses(self, loss: torch.Tensor, out: dict, split: str, batch_size: int) -> None:
        """The objective and each model's term, on the same `{section}/{split}` shape."""
        self.log(
            f"total_loss/{split}", loss, on_step=False, on_epoch=True, batch_size=batch_size
        )
        for name in self.names:
            if name not in out:
                # A group can be absent from a batch — distillation assigns rows by a
                # permutation, so a 1.6% share can come up empty. Nothing to report.
                continue
            if f"{split}_{name}_loss" in self.metrics:
                # `log_metrics` already emits {name}_loss/{split}_overall from BinnedLoss,
                # so we don't need to log it again.
                continue
            self.log(
                f"{name}_loss/{split}",
                out[name]["loss"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def reset_metrics(self, split: str) -> None:
        """Clear one split's accumulators. Called at the start of its epoch."""
        for key, metric in self.metrics.items():
            if key.startswith(f"{split}_"):
                metric.reset()

    def log_metrics(self, split: str) -> None:
        """Emit one split's scalars and histograms. `reset_metrics` clears them."""
        for key, metric in self.metrics.items():
            if not key.startswith(f"{split}_"):
                continue
            section = section_for(key)
            self.log(f"{section}/{split}_overall", metric.overall())
            if is_binned(metric):
                low, high = edge_values(metric)
                if low is not None:
                    self.log(f"{section}/{split}_low_{metric.axis}", low)
                    self.log(f"{section}/{split}_high_{metric.axis}", high)
                self._log_bins(bins_key(section, split, metric), metric)

    def _log_bins(self, key: str, metric: TimeBinnedMetric) -> None:
        """Send one metric's per-bin values to wandb as a histogram over the bin index.

        `np_histogram=(values, edges)` hands wandb the bars directly rather than letting
        it re-bin. The edges are the bin *index*, 0..num_bins, not t: schedule bins are
        quantiles of t, so their widths differ by an order of magnitude and a to-scale t
        axis crushes the whole populated region into a sliver near t=1. Equal widths also
        mean equal sample counts per bar, which is the honest weighting.

        The t each bar covers is deliberately not recorded alongside it. Every tau refresh
        re-cuts the edges, so there is no one geometry the run's histograms share; what
        the bars span over time is read off the `tau/t_q*` series instead.
        """
        if not isinstance(self.logger, WandbLogger) or self.trainer.sanity_checking:
            return
        num_bins = metric.num_bins
        if num_bins > wandb.Histogram.MAX_LENGTH:
            raise ValueError(
                f"{key} has {num_bins} bins; wandb histograms cap at "
                f"{wandb.Histogram.MAX_LENGTH}. Lower metrics.*_bins."
            )
        edges = np.arange(num_bins + 1, dtype=float)
        values = metric.compute().detach().cpu().numpy()
        values = np.where(metric.seen().detach().cpu().numpy(), values, 0.0)
        self.logger.experiment.log(
            {
                key: wandb.Histogram(np_histogram=(values.tolist(), edges.tolist())),
                "epoch": self.current_epoch,
            },
            commit=False,
        )

    def log_mu_accuracy(self) -> None:
        """Report the schedule's running mu accuracy, when it is tracking one.

        Reads only: the average is the schedule's state, updated by
        `TauScheduler.observe_mu`, which has to run first to include the epoch that just
        finished. Mu's raw error is already a metric, so only the average is reported
        here — `mu_perr/{split}_overall` is the same reading before smoothing.
        """
        if self.tau_scheduler.mu_accuracy is not None:
            self.log("tau/mu_accuracy_ema", self.tau_scheduler.mu_accuracy)

    def log_tau_source(self) -> None:
        """Report which curve the live schedule came from, as a series rather than a log line.

        `self.log` takes scalars only, so the strategy's own `source` string is mapped to
        `tau/source_measured`: 0 while the interpolant's curve is driving, 1 once the
        denoiser's is. Flat in a `simulated` or `measured` run, a single step in a `warmup`
        one — which puts the handover on the same x-axis as `tau/t0` and `sampled/wer`,
        where it can be read against them.
        """
        self.log("tau/source_measured", float(self.tau_scheduler.source == "measured"))

    def log_sampled(self) -> None:
        """Log the WER a full integration from noise reaches on the fixed val slice.

        The binned metrics score one denoising step from an x_t built out of the target,
        so at high t they largely measure their own input; this is the number inference
        would produce. `SamplingProbe` owns the mechanics — what this adds is `self.log`.
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or self.trainer.sanity_checking:
            return
        if not self.sampling_probe.is_due(self.current_epoch):
            return
        probe = self.sampling_probe
        for method in probe.methods:
            for n in probe.step_counts:
                for grid in probe.grids:
                    for w in probe.cfg_strengths:
                        wer, token_error, n_utts = probe.score(
                            self, datamodule, method, num_steps=n, grid=grid, cfg_strength=w
                        )
                        key = probe.key(method, n, grid, w)
                        self.log(f"{key}/wer", wer, batch_size=n_utts)
                        self.log(f"{key}/perr", token_error, batch_size=n_utts)

    def reset_audio_only(self) -> None:
        """Clear the audio-only tally; `update_audio_only` says what each slot counts."""
        self._audio_only = torch.zeros(6, device=self.device)

    @torch.no_grad()
    def update_audio_only(self, batch: dict) -> None:
        """Add one batch to the denoiser's audio-only tally at t = 0.

        At t=0 the field holds no data term at all, so whatever the denoiser recovers it
        read off the audio: through its own cross-attention, plus whatever mu's softmax
        shows through the noise. The binned metrics cannot isolate this — their lowest bin
        spans up to t~0.5, where x_t already carries some of the target. Three priors:

          * mu, as trained.
          * the uniform vector: mu's share taken out. Out of distribution for a model that
            always saw mu clearly (low sigma), so a drop there is partly that shift.
          * another utterance's mu (the batch rolled by one): the same shape and confidence
            as the real one, but wrong. In distribution, so how often the denoiser copies
            its argmax, where that argmax is not the target, is how far it takes mu on
            trust rather than checking it against the audio.

        Slots: hits under each prior, copies of the wrong mu, positions where the wrong
        mu's argmax differs from the target, real tokens.

        The time input is 0 under either `flow.time_input`: every curve a run holds has
        u(0) = 0, since `exact` and `MeasuredTau` normalise tau by the error at t=0.
        """
        audio, tokens = batch["audio"].float(), batch["tokens"]
        real = tokens != self.pad_id

        t0 = torch.zeros(tokens.size(0), device=self.device)
        noise = self.alpha_fn(t0)[:, None, None] * torch.randn(
            *tokens.shape, self.vocab_size, device=self.device
        )
        mu = F.softmax(self.mu_nn(audio), dim=-1)
        # A mu that knows nothing: the field at t=0 then carries no hint at all.
        uniform = torch.full_like(noise, 1.0 / self.vocab_size)
        wrong = mu.roll(1, dims=0)

        preds = []
        for prior in (mu, uniform, wrong):
            x_0 = noise + self.gamma_fn(t0)[:, None, None] * prior
            preds.append(self.denoiser(x_t=x_0, time=t0, audio_embeddings=audio).argmax(-1))
        misleading = real & (wrong.argmax(-1) != tokens)
        self._audio_only += torch.stack(
            [
                *[((pred == tokens) & real).sum() for pred in preds],
                ((preds[2] == wrong.argmax(-1)) & misleading).sum(),
                misleading.sum(),
                real.sum(),
            ]
        ).float()

    def log_audio_only(self) -> None:
        """Log the tally under `audio_only/`: recovery per prior, and the copy rate."""
        with_mu, uniform, wrong, copies, misleading, reals = self._audio_only.tolist()
        if reals > 0:
            self.log("audio_only/val_recovery", with_mu / reals)
            self.log("audio_only/val_recovery_no_mu", uniform / reals)
            self.log("audio_only/val_recovery_wrong_mu", wrong / reals)
        if misleading > 0:
            self.log("audio_only/val_copies_wrong_mu", copies / misleading)

    def log_tau_shape(self, reparam: TimeReparametrization) -> None:
        """Record what a schedule actually draws, not just where it starts.

        t0 alone says where the support begins; these say how the draw is spread inside
        it, which is what moves when tau is refreshed. The quantiles are read off the curve
        exactly; the histogram is of a seeded draw, so successive epochs differ because the
        schedule moved, not because the draw did.
        """
        for q in (10, 25, 50, 75, 90):
            self.log(f"tau/t_q{q}", float(reparam.quantiles(q / 100)))
        if not isinstance(self.logger, WandbLogger) or self.trainer.sanity_checking:
            return
        u = torch.rand(4096, generator=torch.Generator().manual_seed(0))
        drawn = reparam.t_of(u).numpy()
        counts, edges = np.histogram(drawn, bins=40, range=(0.0, 1.0))
        self.logger.experiment.log(
            {
                "tau/t_drawn": wandb.Histogram(
                    np_histogram=(counts.tolist(), edges.tolist())
                ),
                "epoch": self.current_epoch,
            },
            commit=False,
        )

    def _define_bin_metrics(self) -> None:
        """Plot the histogram panels against the epoch, not the global step.

        One histogram per epoch is the natural x-axis for these, and it keeps them off
        the step counter that train and val writes share.
        """
        if not isinstance(self.logger, WandbLogger):
            return
        self.logger.experiment.define_metric("epoch")
        self.logger.experiment.define_metric("tau/t_drawn", step_metric="epoch")
        # define_metric globs are suffix-only ("prefix*"), and the histograms sit at the
        # tail of each section's key — so declare them one by one. All are known here.
        for key, metric in self.metrics.items():
            if not is_binned(metric):
                continue
            split, _, _ = split_key(key)
            self.logger.experiment.define_metric(
                bins_key(section_for(key), split, metric), step_metric="epoch"
            )
