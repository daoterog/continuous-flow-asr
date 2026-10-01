"""The flow-matching LightningModule: interpolant, objective and optimizers.

Everything that reports numbers to a logger lives in `src.metrics.reporting`, and the
metric bank is built by `src.metrics.utils`.
"""

from collections.abc import Callable

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Metric
from transformers import WhisperTokenizerFast

from src.dataset import VocabMap
from src.metrics import schedule_bin_edges
from src.metrics.reporting import MetricReporting
from src.metrics.utils import is_binned
from src.models import build_audio_decoder, build_denoiser
from src.sampler import sample_tokens, score_samples
from src.schedule import (
    generation_grid,
    make_tau_inverse,
    sweep_corruption,
)


def make_schedule(spec: DictConfig | dict | str | None) -> Callable | None:
    """Turn a named coefficient schedule into alpha_fn / beta_fn / gamma_fn.

    The three variants are:
      * `linear_up`   -> t          (beta, the weight on the one-hot target)
      * `linear_down` -> 1 - t      (alpha on the noise; gamma in the audio-conditioned run)
      * `parabola`    -> 4c t (1-t) (gamma in the three-term run, peaking at c)

    None means the term is absent, which is how the linear variant drops mu entirely.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = OmegaConf.to_container(spec) if isinstance(spec, DictConfig) else dict(spec)
    kind = spec.pop("type")

    # `scale` matters most on gamma. At scale 1 the audio_cond schedule has gamma == alpha
    scale = spec.pop("scale", 1.0)

    if kind == "linear_up":
        return lambda t: scale * t
    if kind == "linear_down":
        return lambda t: scale * (1 - t)
    if kind == "parabola":
        c = spec.pop("c", 0.5)
        return lambda t: scale * 4 * c * t * (1 - t)
    raise ValueError(
        f"unknown schedule {kind!r}; expected linear_up, linear_down or parabola"
    )


class FlowMatching(MetricReporting, pl.LightningModule):
    """Continuous flow matching over the token simplex, conditioned on Whisper latents.

    Everything stays continuous: x1 is the one-hot target read as a point on the simplex,
    x0 is Gaussian, and I_t is their interpolation. DRAX's Transformer takes that directly
    — a 3D x_t goes through `x_t @ vocab_embed.weight` instead of an embedding lookup — so
    the continuous field never has to be discretised.

    MetricReporting comes first in the bases so its Lightning hooks win the MRO.
    """

    def __init__(
        self,
        cfg: DictConfig,
        vocab_size: int,
        pad_id: int,
        canvas_len: int,
        tokenizer: WhisperTokenizerFast,
        vocab_map: VocabMap | None = None,
        sample_t_fn: Callable | None = None,
    ) -> None:
        """Build both models, the interpolant coefficients and the metric bank.

        sample_t_fn maps a uniform draw to a flow time (tau^-1 from make_tau_inverse).
        Left as None, t is uniform — which at this vocabulary spends most of every batch
        below the dead zone, where x_t carries nothing and the denoiser can only hear the
        audio.
        """
        super().__init__()
        # tokenizer/vocab_map/sample_t_fn are objects, not hyperparameters: they would
        # not survive a checkpoint round-trip and are rebuilt from the data instead.
        self.save_hyperparameters(
            ignore=["tokenizer", "vocab_map", "sample_t_fn"], logger=False
        )
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.canvas_len = canvas_len
        self.sample_t_fn = sample_t_fn
        self.tokenizer = tokenizer
        self.vocab_map = vocab_map
        self._sample_batch = None
        self._pending_tau = None
        self._mu_accuracy_ema = 0.0
        self._mu_ema_steps = 0
        # The accuracy tau was last rebuilt from. Saved with the run, because rebuilding
        # tau from `schedule.mu_accuracy` on reload gives the *prior*, not the schedule the
        # model actually trained under — they diverge as mu learns.
        self.last_mu_accuracy: float | None = None

        self.denoiser = build_denoiser(cfg.model.denoiser, vocab_size)
        self.mu_nn = (
            build_audio_decoder(cfg.model.audio_decoder, vocab_size, canvas_len)
            if cfg.model.get("audio_decoder") is not None
            else None
        )

        flow = cfg.flow
        self.alpha_fn = make_schedule(flow.get("alpha", "linear_down"))
        self.beta_fn = make_schedule(flow.get("beta", "linear_up"))
        self.gamma_fn = make_schedule(flow.get("gamma"))
        self.stop_grad_mu = bool(flow.get("stop_grad_mu", True))
        self.mu_loss_weight = float(flow.get("mu_loss_weight", 1.0))
        self.pad_loss_weight = float(flow.get("pad_loss_weight", 1.0))
        if not 0.0 <= self.pad_loss_weight <= 1.0:
            raise ValueError(
                f"flow.pad_loss_weight must be in [0, 1], got {self.pad_loss_weight}"
            )

        if self.gamma_fn is not None and self.mu_nn is None:
            raise ValueError(
                "flow.gamma is set but model.audio_decoder is absent — the third "
                "interpolant term needs a mu to weight"
            )

        if cfg.train.get("compile"):
            mode = cfg.train.get("compile_mode", "default")
            self.denoiser = torch.compile(self.denoiser, mode=mode)
            if self.mu_nn is not None:
                self.mu_nn = torch.compile(self.mu_nn, mode=mode)

        self.init_metrics(
            ("denoiser", "mu") if self.mu_nn is not None else ("denoiser",),
            tokenizer,
            pad_id,
            vocab_map,
            sample_t_fn=sample_t_fn,
            loss_bins=cfg.metrics.get("loss_bins", 10),
            perr_bins=cfg.metrics.get("perr_bins", 10),
            wer_bins=OmegaConf.to_container(cfg.metrics.get("wer_bins", {})) or None,
            input_perr_bins=cfg.metrics.get("input_perr_bins", 0),
        )

    def make_interpolant(
        self, x1: torch.Tensor, t: torch.Tensor, mu: torch.Tensor | None = None
    ) -> torch.Tensor:
        """I_t = alpha(t) x0 + beta(t) x1 [+ gamma(t) mu], broadcast over (B, L, V)."""
        x0 = torch.randn_like(x1)
        i_t = self.alpha_fn(t)[:, None, None] * x0 + self.beta_fn(t)[:, None, None] * x1
        if self.gamma_fn is not None and mu is not None:
            if self.stop_grad_mu:
                mu = mu.detach()
            i_t = i_t + self.gamma_fn(t)[:, None, None] * mu
        return i_t

    def token_weights(self, tokens: torch.Tensor) -> torch.Tensor | None:
        """(B, L) loss weights, or None when every position counts the same.

        `flow.pad_loss_weight` scales the padding positions before a weighted mean. At
        weight 1 this is the plain mean over the canvas, where ~70% of the terms are
        "emit end-of-text" on a 224-wide canvas holding ~67-token utterances. Lowering it
        does two things at once, and they are easy to confuse: it removes padding's share
        of the gradient, AND, because the denominator shrinks with it, multiplies what
        each real token contributes by up to canvas_len / n_real ~ 3.3. Some of any gain
        is therefore an effective learning-rate increase on content, which is why the
        sweep that tests this has to vary the learning rate alongside it.

        Padding is still a real target at any weight above 0: emitting end-of-text is how
        a non-autoregressive sample knows where to stop, and a model that never stops
        scores over 100% WER through insertions alone.

        Real tokens weigh 1. A padding position weighs

            floor + (1 - floor) * exp(-(d - 1) / pad_decay)

        where d is its distance into the padding run, so the first pad — the end-of-text
        that tells a non-autoregressive sample where to stop — keeps full weight and the
        far tail decays to `flow.pad_loss_weight`. That is the distinction a flat weight
        cannot make: predicting padding 150 positions past the transcript is free, while
        predicting it one position past is the entire length decision, and a model that
        gets it wrong scores over 100% WER through insertions alone.

        `flow.pad_decay` of 0 or null gives the flat weight at every distance.

        Padding is assumed to be a contiguous suffix, which is what `collate` produces and
        what `ids_to_text` already relies on when it cuts at the first pad.
        """
        floor = float(self.pad_loss_weight)
        decay = float(self.cfg.flow.get("pad_decay", 0.0) or 0.0)
        if floor == 1.0:
            return None
        is_pad = tokens == self.pad_id
        if decay <= 0.0:
            return torch.where(is_pad, tokens.new_tensor(floor, dtype=torch.float),
                            tokens.new_tensor(1.0, dtype=torch.float))
        n_real = (~is_pad).sum(dim=1, keepdim=True)
        pos = torch.arange(tokens.size(1), device=tokens.device).expand_as(tokens)
        distance = (pos - n_real + 1).clamp(min=1).float()
        decayed = floor + (1.0 - floor) * torch.exp(-(distance - 1.0) / decay)
        return torch.where(is_pad, decayed, torch.ones_like(decayed))

    def sequence_ce(self, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """(B,) cross-entropy, one value per row instead of a batch scalar."""
        b, _, vocab_size = logits.shape
        per_token = F.cross_entropy(
            logits.reshape(-1, vocab_size), tokens.reshape(-1), reduction="none"
        ).view(b, -1)
        weights = self.token_weights(tokens)
        if weights is None:
            return per_token.mean(dim=1)
        return (per_token * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)

    def flow_step(self, batch: dict) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """One forward pass: the objective, the per-model records, and the t that drew it."""
        audio = batch["audio"].float()
        tokens = batch["tokens"]
        x1 = F.one_hot(tokens, self.vocab_size).float()

        out = {}
        mu = None
        if self.mu_nn is not None:
            mu_logits = self.mu_nn(audio)
            mu = F.softmax(mu_logits, dim=-1)
            mu_loss_per_seq = self.sequence_ce(mu_logits, tokens)
            out["mu"] = {
                "loss": mu_loss_per_seq.mean(),
                "loss_per_seq": mu_loss_per_seq,
                "pred_ids": mu_logits.argmax(dim=-1),
            }

        u = torch.rand(tokens.size(0), device=self.device)
        t = u if self.sample_t_fn is None else self.sample_t_fn(u)
        i_t = self.make_interpolant(x1=x1, t=t, mu=mu)

        logits = self.denoiser(x_t=i_t, time=t, audio_embeddings=audio)
        d_loss_per_seq = self.sequence_ce(logits, tokens)
        out["denoiser"] = {
            "loss": d_loss_per_seq.mean(),
            "loss_per_seq": d_loss_per_seq,
            "pred_ids": logits.argmax(dim=-1),
        }

        if self.cfg.metrics.get("input_perr_bins"):
            out["input"] = {"pred_ids": i_t.argmax(dim=-1)}

        loss = out["denoiser"]["loss"]
        if self.mu_nn is not None:
            loss = loss + self.mu_loss_weight * out["mu"]["loss"]
        return loss, out, t

    def generation_grid(self, num_steps: int) -> list[float]:
        """The sampler's time grid — uniform in tau when t was drawn through tau^-1."""
        if self.sample_t_fn is None:
            return torch.linspace(0, 1, num_steps + 1).tolist()
        return [float(t) for t in generation_grid(self.sample_t_fn, num_steps)]

    def sample(
        self,
        audio: torch.Tensor,
        num_steps: int,
        hard: bool = False,
        init_from_mu: bool | None = None,
    ) -> torch.Tensor:
        """Integrate from noise to (B, canvas_len) token ids for this batch of audio.

        init_from_mu seeds the first state's endpoint term with mu. The tau grid starts
        at the edge of the dead zone rather than at 0, so training's state there already
        carries beta(t0) * x1 and the sampler otherwise supplies nothing for it.
        """
        mu = None
        if self.mu_nn is not None:
            mu = F.softmax(self.mu_nn(audio), dim=-1)
        if init_from_mu is None:
            init_from_mu = bool(self.cfg.get("sampling", {}).get("init_from_mu", False))
        return sample_tokens(
            self.denoiser,
            audio,
            self.generation_grid(num_steps),
            self.vocab_size,
            self.canvas_len,
            self.alpha_fn,
            self.beta_fn,
            self.gamma_fn,
            mu,
            hard=hard,
            x1_init=mu if init_from_mu else None,
        )

    def _sampling_batch(self) -> dict | None:
        """The val utterances to sample, collated once and reused every time."""
        if self._sample_batch is None:
            datamodule = getattr(self.trainer, "datamodule", None)
            if datamodule is None:
                return None
            subset = datamodule.datasets["val"]
            want = self.cfg.sampling.get("n_utts")
            n = len(subset) if want is None else min(int(want), len(subset))
            self._sample_batch = datamodule.collate_fn([subset[i] for i in range(n)])
        return self._sample_batch

    def _log_sampled(self) -> None:
        """Sample from noise on a fixed val slice and log the WER it actually achieves."""
        cfg = self.cfg.get("sampling")
        if cfg is None or not cfg.get("enabled", False) or self.trainer.sanity_checking:
            return
        every = int(cfg.get("every_n_epochs", 5))
        if every <= 0 or (self.current_epoch + 1) % every:
            return
        batch = self._sampling_batch()
        if batch is None:
            return

        was_training = self.training
        self.eval()
        chunk = int(cfg.get("batch_size") or self.cfg.data.batch_size)
        num_steps = int(cfg.get("num_steps", 16))
        hard = bool(cfg.get("hard", False))
        tokens = batch["tokens"].to(self.device)
        pred = torch.cat(
            [
                self.sample(
                    batch["audio"][i : i + chunk].to(self.device).float(),
                    num_steps,
                    hard=hard,
                )
                for i in range(0, batch["audio"].size(0), chunk)
            ]
        )
        wer, token_error = score_samples(
            pred, tokens, self.tokenizer, self.pad_id, self.vocab_map
        )
        self.log("sampled/wer", wer, batch_size=tokens.size(0))
        self.log("sampled/perr", token_error, batch_size=tokens.size(0))
        self.train(was_training)

    def on_validation_epoch_end(self) -> None:
        """Flush the metrics, sample from noise, then re-derive tau if it is due.

        Order matters. The refresh reads mu's error, so it has to happen after this
        epoch's numbers exist; the re-binning has to happen after `super()` has logged
        and reset, since counts collected under the old edges do not describe the new
        ones. So: read, flush, then swap.
        """
        source = str(self.cfg.schedule.get("refresh_split", "train"))
        raw = {
            split: self.metrics[f"{split}_mu_perr"].overall()
            for split in ("train", "val")
            if f"{split}_mu_perr" in self.metrics
        }
        measured = raw.get(source)
        # Updated on every validation, not only when a refresh is due: the average should
        # see every reading it can, and rebuilding tau is the expensive half. That also
        # makes the momentum's lag count epochs rather than refreshes.
        accuracy = (
            self._smooth_mu_accuracy(1.0 - measured)
            if measured is not None and measured == measured  # not NaN
            else None
        )
        super().on_validation_epoch_end()
        self._log_sampled()
        if accuracy is not None:
            self.log("tau/mu_accuracy_ema", accuracy)
            for split, perr in raw.items():
                self.log(f"tau/mu_accuracy_{split}", 1.0 - perr)
            if len(raw) == 2:
                self.log("tau/mu_generalisation_gap", raw["val"] - raw["train"])
        # Not gated on mu: the `measured` curve is rebuilt from the denoiser's P_e and
        # needs no mu at all, so a linear run refreshes too. Gating here is what would
        # leave `linear` silently on its initial schedule, which is precisely the control
        # the comparison needs.
        if self._refresh_is_due():
            if accuracy is not None:
                self.last_mu_accuracy = accuracy
            self._refresh_schedule(accuracy)

    def _smooth_mu_accuracy(self, measured: float) -> float:
        """Bias-corrected EMA of mu's accuracy, so tau tracks its trend and not SGD wobble."""
        m = float(self.cfg.schedule.get("mu_accuracy_momentum", 0.0))
        if not 0.0 <= m < 1.0:
            raise ValueError(f"schedule.mu_accuracy_momentum must be in [0, 1), got {m}")
        if m == 0.0:
            return measured
        self._mu_accuracy_ema = m * self._mu_accuracy_ema + (1.0 - m) * measured
        self._mu_ema_steps += 1
        return self._mu_accuracy_ema / (1.0 - m**self._mu_ema_steps)

    def on_validation_epoch_start(self) -> None:
        """Clear the val accumulators before the pass fills them."""
        super().on_validation_epoch_start()
        self.reset_metrics("val")

    def on_test_epoch_start(self) -> None:
        """Clear the test accumulators before the pass fills them."""
        super().on_test_epoch_start()
        self.reset_metrics("test")

    def on_train_epoch_start(self) -> None:
        """Clear the train accumulators, then apply a pending schedule refresh.

        Reset here and log at epoch end, so each pass reads as what it is and a metric's
        contents always describe the epoch in progress — which is what `bin_centers` and a
        tau refresh both rely on when they read a live metric.

        The refresh is applied here rather than where it is computed because Lightning runs
        validation before `on_train_epoch_end`: re-binning at that point would change the
        edges under train counts that had not been logged yet. By the start of the next
        epoch both splits have been flushed.
        """
        super().on_train_epoch_start()
        self.reset_metrics("train")
        if self._pending_tau is None:
            return
        new_tau, mu_accuracy = self._pending_tau
        self._pending_tau = None
        self.sample_t_fn = new_tau
        # set_edges leaves the counts alone, and every split has just been or is about to
        # be cleared. An unbinned metric's single bin already spans all of t.
        for metric in self.metrics.values():
            if is_binned(metric):
                metric.set_edges(schedule_bin_edges(new_tau, metric.num_bins))
        lo = float(new_tau(0.0))
        self.log("tau/t0", lo)
        if mu_accuracy is not None:
            self.log("tau/mu_accuracy", mu_accuracy)
        self._log_tau_shape(new_tau)
        self.print(
            f"tau refreshed for epoch {self.current_epoch}: mu accuracy "
            f"{mu_accuracy:.4f}, t in [{lo:.3f}, 1.000] now carries the draw"
        )

    def _smoothed_measured_tau(self, metric: Metric) -> Callable:
        """Tau from the denoiser's recovery, averaged across refreshes on a fixed t grid."""
        seen = metric.seen().detach().cpu().numpy()
        ts = metric.bin_centers().numpy()[seen]
        rec = metric.recovery().detach().cpu().numpy()[seen]
        if rec.size == 0 or rec.max() < 0.05:
            raise ValueError(
                f"the model recovers at most {rec.max() if rec.size else 0:.3f} of tokens "
                f"in any bin; tau would collapse onto t=1. Train further first."
            )
        grid = np.linspace(0.0, 1.0, int(self.cfg.schedule.num_bins) + 1)
        on_grid = np.interp(
            grid,
            np.concatenate([ts, [1.0]]),
            np.concatenate([rec, [1.0]]),
            left=float(rec[0])
        )
        return make_tau_inverse(
            grid.tolist(), on_grid.tolist(), self.vocab_size,
            p_e0=float(1.0 - on_grid[0]),
        )

    def _log_tau_shape(self, tau_inverse: Callable) -> None:
        """Record what the new schedule actually draws, not just where it starts."""
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

    def _refresh_is_due(self) -> bool:
        """Whether tau should be re-derived at the end of this validation."""
        every = int(self.cfg.schedule.get("refresh_every_n_epochs", 0))
        curve = str(self.cfg.schedule.get("refresh_curve_from", "simulated"))
        needs_mu = curve == "simulated"
        return (
            every > 0
            and self.sample_t_fn is not None
            and (self.gamma_fn is not None or not needs_mu)
            and not self.trainer.sanity_checking
            and (self.current_epoch + 1) % every == 0
        )

    def _refresh_schedule(self, mu_accuracy: float | None) -> None:
        """Re-derive tau. `schedule.refresh_curve_from` decides from which P_e curve."""
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        cfg = self.cfg.schedule
        curve = str(cfg.get("refresh_curve_from", "simulated"))

        if curve == "measured":
            key = f"{self.cfg.schedule.get('refresh_split', 'train')}_denoiser_perr"
            if key not in self.metrics:
                self.print(f"tau refresh skipped: no {key} metric to read")
                return
            metric = self.metrics[key]
            new_tau = self._smoothed_measured_tau(metric)
        elif curve == "simulated":
            if self.gamma_fn is not None and mu_accuracy is None:
                self.print("tau refresh skipped: the simulated curve needs mu's accuracy")
                return

            tokens = datamodule.sample_tokens(
                "train", int(cfg.sweep_n), seed=self.cfg.seed
            )
            ts = np.linspace(0.0, 1.0, int(cfg.num_bins) + 1).tolist()
            recovery = sweep_corruption(
                tokens.to(self.device),
                self.vocab_size,
                ts,
                alpha_fn=self.alpha_fn,
                beta_fn=self.beta_fn,
                gamma_fn=self.gamma_fn,
                mu_accuracy=0.0 if mu_accuracy is None else mu_accuracy,
            )
            new_tau = make_tau_inverse(ts, recovery, self.vocab_size)
        else:
            raise ValueError(
                f"schedule.refresh_curve_from must be 'simulated' or 'measured', "
                f"got {curve!r}"
            )

        self._pending_tau = (new_tau, mu_accuracy)

    def _check_finite(self, loss: torch.Tensor, out: dict, split: str) -> None:
        """Stop the run the first time the objective is not finite."""
        if torch.isfinite(loss):
            return
        parts = "  ".join(
            f"{name}_loss={float(rec['loss']):.4g}"
            for name, rec in out.items()
            if "loss" in rec
        )
        raise RuntimeError(
            f"non-finite {split} loss at epoch {self.current_epoch}, "
            f"global step {self.global_step}: total={float(loss):.4g}  {parts}. "
            f"Training stopped rather than carrying NaN into the weights — lower "
            f"train.denoiser_lr, or set train.grad_clip if it is null."
        )

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        loss, out, t = self.flow_step(batch)
        self._check_finite(loss, out, split)
        self.log_losses(loss, out, split, batch_size=batch["tokens"].size(0))
        self.update_metrics(out, batch["tokens"], t, split)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One optimizer step."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One validation batch."""
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One test batch. Only reached through an explicit `trainer.test()`."""
        return self._step(batch, "test")

    def configure_optimizers(self) -> dict:
        """AdamW with a per-model LR group, plus OneCycleLR stepped every batch.

        `estimated_stepping_batches` already accounts for epochs, accumulation and the
        drop_last on the training loader, so the schedule cannot run past the run.
        """
        train = self.cfg.train
        groups = [{"params": self.denoiser.parameters(), "lr": train.denoiser_lr}]
        max_lr = [train.denoiser_lr]
        if self.mu_nn is not None:
            groups.append(
                {"params": self.mu_nn.parameters(), "lr": train.audio_decoder_lr}
            )
            max_lr.append(train.audio_decoder_lr)

        fused = bool(train.get("fused_optimizer", False)) and self.device.type == "cuda"
        if fused and train.get("grad_clip"):
            raise ValueError(
                "train.fused_optimizer and train.grad_clip cannot both be set: a fused "
                "AdamW unscales the gradients inside its own kernel, so there is no "
                "point left at which a clip could be applied. Pick one — fused measured "
                "~28% faster, clipping is the guard on the OneCycle peak LR."
            )
        optimizer = torch.optim.AdamW(
            groups, weight_decay=train.get("weight_decay", 0.0), fused=fused
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup = self.cfg.train.get("warmup_steps")
        pct_start = 0.3 if warmup is None else min(max(int(warmup) / total_steps, 1e-3), 0.5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr if len(max_lr) > 1 else max_lr[0],
            total_steps=total_steps,
            pct_start=pct_start,
        )
        self.print(
            f"OneCycleLR: {total_steps:,} steps, peak {max_lr}, "
            f"warmup {int(pct_start * total_steps):,} ({pct_start:.3f} of the run)"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
