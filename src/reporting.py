"""Everything that writes numbers to the logger, kept out of the model.

`MetricReporting` is a mixin for a LightningModule: it owns the metric bank, updates it
each batch, and at each epoch end emits the scalars and the per-bin histograms. It leans
on `self.log`, `self.logger`, `self.trainer` and `self.current_epoch`, so it only works
mixed into a LightningModule and must come before it in the bases.

It sits above `src.metrics` rather than inside it — the metrics are measurements, this is
the layer that decides what reaches wandb and in what shape, which is also why it may
reach for a `SamplingProbe` while the metrics package may not.
"""

from collections.abc import Callable

import numpy as np
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from transformers import WhisperTokenizerFast

from src.dataset import VocabMap
from src.metrics.base import TimeBinnedMetric
from src.metrics.utils import (
    check_metrics,
    edge_values,
    is_binned,
    make_metrics,
    section_for,
    split_key,
)
from src.sampling_probe import SamplingProbe


class MetricReporting:
    """Metric bank plus the logging hooks, mixed into a LightningModule.

    Keys come out as `{model}_{metric}/{split}_{stat}` so one wandb section is one metric
    of one model, with train and val next to each other.
    """

    metrics: nn.ModuleDict
    names: tuple[str, ...]
    sampling_probe: SamplingProbe

    def init_metrics(
        self,
        names: tuple[str, ...],
        tokenizer: WhisperTokenizerFast,
        pad_id: int,
        vocab_map: VocabMap | None = None,
        sample_t_fn: Callable | None = None,
        **bins: int | dict | None,
    ) -> None:
        """Build and validate the metric bank. Call once, from `__init__`."""
        self.names = tuple(names)
        self.metrics = make_metrics(
            self.names,
            tokenizer,
            pad_id,
            vocab_map,
            sample_t_fn=sample_t_fn,
            **bins,
        )
        check_metrics(self.metrics, sample_t_fn)

    def update_metrics(
        self, out: dict, target_ids: torch.Tensor, t: torch.Tensor, split: str
    ) -> None:
        """Feed one batch's per-model records to every metric of this split."""
        for name, part in out.items():
            record = {"pred_ids": part["pred_ids"], "target_ids": target_ids, "t": t}
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
                    self.log(f"{section}/{split}_low_t", low)
                    self.log(f"{section}/{split}_high_t", high)
                self._log_bins(f"{section}/{split}_bins", metric)

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
        wer, token_error, n_utts = self.sampling_probe.score(self, datamodule)
        self.log("sampled/wer", wer, batch_size=n_utts)
        self.log("sampled/perr", token_error, batch_size=n_utts)

    def log_tau_shape(self, tau_inverse: Callable) -> None:
        """Record what a schedule actually draws, not just where it starts.

        t0 alone says where the support begins; these say how the draw is spread inside
        it, which is what moves when tau is refreshed. The sample is seeded so successive
        epochs differ because the schedule moved, not because the draw did.
        """
        u = torch.rand(4096, generator=torch.Generator().manual_seed(0))
        drawn = tau_inverse(u)
        drawn = drawn.detach().cpu().numpy() if torch.is_tensor(drawn) else np.asarray(drawn)
        for q in (10, 25, 50, 75, 90):
            self.log(f"tau/t_q{q}", float(np.percentile(drawn, q)))
        if not isinstance(self.logger, WandbLogger) or self.trainer.sanity_checking:
            return
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
                f"{section_for(key)}/{split}_bins", step_metric="epoch"
            )

    def on_fit_start(self) -> None:
        """Declare the histogram panels' step metric before anything is logged."""
        super().on_fit_start()
        self._define_bin_metrics()

    def on_test_start(self) -> None:
        """Same, for a `trainer.test()` that did not go through `fit`."""
        super().on_test_start()
        self._define_bin_metrics()

    def on_train_epoch_end(self) -> None:
        """Flush the train metrics."""
        super().on_train_epoch_end()
        self.log_metrics("train")

    def on_validation_epoch_end(self) -> None:
        """Flush the val metrics."""
        super().on_validation_epoch_end()
        self.log_metrics("val")

    def on_test_epoch_end(self) -> None:
        """Flush the test metrics."""
        super().on_test_epoch_end()
        self.log_metrics("test")
