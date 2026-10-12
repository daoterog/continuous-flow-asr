"""The flow-matching LightningModule: interpolant, objective and optimizers.

Everything that reports numbers to a logger lives in `src.reporting`, and the metric bank
it owns is built by `src.metrics.utils`.
"""

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import WhisperTokenizerFast

from src.checkpointing.layout import COMPONENTS
from src.data import VocabMap
from src.data.augment import mask_latents
from src.ema import EMAWeights
from src.flow import build_tau_scheduler
from src.flow.coefficients import make_schedule, waypoint_strength
from src.flow.sampler import sample_tokens, sample_tokens_euler
from src.flow.sampling_probe import SamplingProbe
from src.flow.schedule import TimeReparametrization
from src.metrics.utils import is_binned
from src.models import build_audio_decoder, build_denoiser
from src.reporting import MetricReporting

SAMPLERS = {"interpolant": sample_tokens, "euler": sample_tokens_euler}
TIME_INPUTS = ("reparametrized", "uniform")
LOSS_REDUCTIONS = ("sequence", "token")


def token_weights(
    tokens: torch.Tensor, pad_id: int, floor: float, decay: float
) -> torch.Tensor | None:
    """(B, L) loss weights, or None when every position counts the same.

    Real tokens weigh 1; a pad at distance d into the run weighs

        floor + (1 - floor) * exp(-(d - 1) / decay)

    so the first pad keeps full weight and the tail decays to `floor`
    (`flow.pad_loss_weight`); a `decay` of 0 or null gives that floor flat. The first pad is
    the whole length decision — a model that never stops scores over 100% WER on insertions
    — while a pad 150 positions out is free.

    Lowering the floor also shrinks the weighted mean's denominator, scaling each real token
    by up to canvas_len / n_real ~ 3.3, so part of any gain is an effective learning-rate
    rise and a sweep over this has to vary the learning rate too.

    Padding is assumed to be a contiguous suffix, which is what `collate` produces.

    A free function rather than a method: both `FlowMatching` and `FlowMapping` weigh their
    tokens the same way, and the diagnostics draw the exact weights training uses.
    """
    if floor == 1.0:
        return None
    is_pad = tokens == pad_id
    if decay <= 0.0:
        return torch.where(is_pad, tokens.new_tensor(floor, dtype=torch.float),
                        tokens.new_tensor(1.0, dtype=torch.float))
    n_real = (~is_pad).sum(dim=1, keepdim=True)
    pos = torch.arange(tokens.size(1), device=tokens.device).expand_as(tokens)
    distance = (pos - n_real + 1).clamp(min=1).float()
    decayed = floor + (1.0 - floor) * torch.exp(-(distance - 1.0) / decay)
    return torch.where(is_pad, decayed, torch.ones_like(decayed))


class FlowMatching(MetricReporting, EMAWeights, pl.LightningModule):
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
        time_reparam: TimeReparametrization | None = None,
    ) -> None:
        """Build both models, the interpolant coefficients and the metric bank.

        time_reparam is the map from a uniform draw to a flow time. Left as None it is the
        identity and t is uniform — which at this vocabulary spends most of every batch
        below the dead zone, where x_t carries nothing and the denoiser can only hear the
        audio.
        """
        super().__init__()
        # tokenizer/vocab_map/time_reparam are objects, not hyperparameters: they would
        # not survive a checkpoint round-trip and are rebuilt from the data instead.
        self.save_hyperparameters(
            ignore=["tokenizer", "vocab_map", "time_reparam"], logger=False
        )
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.canvas_len = canvas_len
        self.time_reparam = time_reparam or TimeReparametrization.identity()
        self.tokenizer = tokenizer
        self.vocab_map = vocab_map
        self.sampling_probe = SamplingProbe(cfg.get("sampling"), cfg.data.batch_size)
        self.tau_scheduler = build_tau_scheduler(cfg.schedule, seed=cfg.seed)

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
        # The 4c of the drift's waypoint term, 0 for the variants without one.
        self.waypoint = waypoint_strength(flow.get("gamma"))
        self.stop_grad_mu = bool(flow.get("stop_grad_mu", True))
        self.mu_loss_weight = float(flow.get("mu_loss_weight", 1.0))
        self.pad_loss_weight = float(flow.get("pad_loss_weight", 1.0))
        if not 0.0 <= self.pad_loss_weight <= 1.0:
            raise ValueError(
                f"flow.pad_loss_weight must be in [0, 1], got {self.pad_loss_weight}"
            )

        self.stratified_t = bool(flow.get("stratified_t", False))
        self.loss_reduction = str(flow.get("loss_reduction", "sequence"))
        if self.loss_reduction not in LOSS_REDUCTIONS:
            raise ValueError(
                f"flow.loss_reduction must be one of {LOSS_REDUCTIONS}, "
                f"got {self.loss_reduction!r}"
            )
        self.time_input = str(flow.get("time_input", "reparametrized"))
        if self.time_input not in TIME_INPUTS:
            raise ValueError(
                f"flow.time_input must be one of {TIME_INPUTS}, got {self.time_input!r}"
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

        # Validate, sample and save an average of the weights (src/ema.py); None keeps none.
        self.init_ema(cfg.train.get("ema_decay"))

        self.init_metrics(
            ("denoiser", "mu") if self.mu_nn is not None else ("denoiser",),
            tokenizer,
            pad_id,
            vocab_map,
            time_reparam=self.time_reparam,
            loss_bins=cfg.metrics.get("loss_bins", 10),
            perr_bins=cfg.metrics.get("perr_bins", 10),
            wer_bins=OmegaConf.to_container(cfg.metrics.get("wer_bins", {})) or None,
            input_perr_bins=cfg.metrics.get("input_perr_bins", 10),
        )

    @property
    def components(self) -> tuple[str, ...]:
        """What `save_components` writes: both trained models, `mu_nn` when there is one."""
        return COMPONENTS

    def on_fit_start(self) -> None:
        """Declare the histogram panels' step metric before anything is logged."""
        super().on_fit_start()
        self._define_bin_metrics()

    def on_train_epoch_start(self) -> None:
        """Re-derive the schedule if it is due, then clear the train accumulators.

        In that order: a refresh reads the metrics of the epoch that just finished, which
        are still standing here, and it re-cuts the bins — so the reset has to come after
        it, or the new edges would be applied to counts gathered under the old ones.
        """
        super().on_train_epoch_start()
        if self.tau_scheduler.is_due(self):
            new, skipped = self.tau_scheduler.refresh(
                self, getattr(self.trainer, "datamodule", None)
            )
            if skipped is not None:
                self.print(skipped)
            # Outside the `new is not None` branch: a refresh that was declined still
            # says which curve is driving, and under `warmup` that is the answer.
            self.log_tau_source()
            if new is not None:
                new = new.with_uniform_floor(
                    float(self.cfg.schedule.get("uniform_floor", 0.0))
                )
                self.time_reparam = new
                for metric in self.metrics.values():
                    if is_binned(metric):
                        metric.set_edges(new.bin_edges(metric.num_bins))
                self.log("tau/t0", new.t_start)
                self.log_tau_shape(new)
                self.save_refresh(new)
                self.print(f"tau refreshed for epoch {self.current_epoch}: {new!r}")
                if self.tau_scheduler.mu_accuracy is not None:
                    self.print(f"mu accuracy EMA {self.tau_scheduler.mu_accuracy:.4f}")
        self.reset_metrics("train")

    def save_refresh(self, curve: TimeReparametrization) -> Path:
        """Write a refreshed curve to `<save_dir>/<run_id>/tau_refresh/e<epoch>.npz`.

        The logged quantiles say where a curve sits but not its shape, and only the last
        curve travels with the weights. One file per refresh is what lets
        scripts/schedule/project_tau.py draw how a run's schedule actually moved, against
        what it projected. Each holds the knots, the source and mu's accuracy then.
        """
        experiment = getattr(self.logger, "experiment", None)
        run_id = getattr(experiment, "id", None) or "local"
        out = Path(self.cfg.logging.save_dir) / str(run_id) / "tau_refresh"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"e{self.current_epoch:03d}.npz"
        accuracy = self.tau_scheduler.mu_accuracy
        np.savez(
            path,
            t_knots=curve.t_knots,
            u_knots=curve.u_knots,
            source=curve.source,
            mu_accuracy=np.nan if accuracy is None else accuracy,
        )
        return path

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One optimizer step."""
        return self._step(batch, "train")

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        loss, out, t = self.flow_step(batch)
        self._check_finite(loss, out, split)
        self.log_losses(loss, out, split, batch_size=batch["tokens"].size(0))
        self.update_metrics(out, batch["tokens"], t, split)
        return loss

    def flow_step(self, batch: dict) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """One forward pass: the objective, the per-model records, and the t that drew it."""
        audio = batch["audio"].float()
        if self.training:
            # mu sees the masked audio too: the mask is part of the data, not of the model.
            audio = mask_latents(audio, self.cfg.train.get("audio_mask"))
        tokens = batch["tokens"]
        x1 = F.one_hot(tokens, self.vocab_size).float()

        out = {}
        mu = None
        if self.mu_nn is not None:
            mu_logits = self.mu_nn(audio)
            mu = F.softmax(mu_logits, dim=-1)
            mu_loss, mu_loss_per_seq = self.sequence_ce(mu_logits, tokens)
            out["mu"] = {
                "loss": mu_loss,
                "loss_per_seq": mu_loss_per_seq,
                "pred_ids": mu_logits.argmax(dim=-1),
            }

        u = self.draw_u(tokens.size(0))
        t = self.time_reparam.t_of(u)
        i_t = self.make_interpolant(x1=x1, t=t, mu=mu)

        # `uniform` hands the denoiser the draw itself, as FLM does: at large V the t that
        # matters sits in a narrow band near 1, where a sinusoidal embedding barely moves.
        time = u if self.time_input == "uniform" else t
        logits = self.denoiser(x_t=i_t, time=time, audio_embeddings=audio)
        d_loss, d_loss_per_seq = self.sequence_ce(logits, tokens)
        out["denoiser"] = {
            "loss": d_loss,
            "loss_per_seq": d_loss_per_seq,
            "pred_ids": logits.argmax(dim=-1),
        }

        if self.cfg.metrics.get("input_perr_bins"):
            out["input"] = {"pred_ids": i_t.argmax(dim=-1)}

        loss = out["denoiser"]["loss"]
        if self.mu_nn is not None:
            loss = loss + self.mu_loss_weight * out["mu"]["loss"]
        return loss, out, t

    def sequence_ce(
        self, logits: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The batch objective, and the (B,) per-row cross-entropy the metrics bin.

        `flow.loss_reduction` decides the objective:
          * `sequence` — mean of the per-row weighted means. Every utterance weighs the
            same, so with pads down-weighted a short one's real tokens each count more.
          * `token` — one weighted mean over every token in the batch, as FLM takes it.
        The two are identical while every position weighs 1: every row is canvas_len long.
        """
        b, _, vocab_size = logits.shape
        per_token = F.cross_entropy(
            logits.reshape(-1, vocab_size), tokens.reshape(-1), reduction="none"
        ).view(b, -1)
        weights = token_weights(
            tokens,
            self.pad_id,
            float(self.pad_loss_weight),
            float(self.cfg.flow.get("pad_decay", 0.0) or 0.0),
        )
        if weights is None:
            per_seq = per_token.mean(dim=1)
            return per_seq.mean(), per_seq
        weighted = per_token * weights
        per_seq = weighted.sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)
        if self.loss_reduction == "token":
            return weighted.sum() / weights.sum().clamp(min=1e-8), per_seq
        return per_seq.mean(), per_seq

    def draw_u(self, n: int) -> torch.Tensor:
        """(n,) uniform draws, one per row, before tau^-1 turns them into flow times.

        `flow.stratified_t` puts exactly one draw in each slice [i/n, (i+1)/n), shuffled
        across rows. FLM calls this antithetic sampling. It isn't: antithetic sampling pairs
        u with 1-u. Either way, it stops a batch from piling up on one side of a steep tau.
        """
        u = torch.rand(n, device=self.device)
        if not self.stratified_t:
            return u
        u = (u + torch.arange(n, device=self.device)) / n
        return u[torch.randperm(n, device=self.device)]

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

    def on_train_epoch_end(self) -> None:
        """Flush the train metrics, then take mu's reading for the schedule.

        Observe before logging, so the average reported covers the epoch that just ran;
        the refresh that consumes it is a whole epoch away, at the next epoch's start.
        """
        super().on_train_epoch_end()
        self.log_metrics("train")
        self.tau_scheduler.observe_mu(self.metrics)
        self.log_mu_accuracy()

    def on_validation_epoch_start(self) -> None:
        """Clear the val accumulators before the pass fills them."""
        super().on_validation_epoch_start()
        self.reset_metrics("val")
        if self.mu_nn is not None:
            self.reset_audio_only()

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One validation batch, plus the denoiser's audio-only read of it."""
        loss = self._step(batch, "val")
        if self.mu_nn is not None:
            self.update_audio_only(batch)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Flush the val metrics, then sample from noise."""
        super().on_validation_epoch_end()
        self.log_metrics("val")
        if self.mu_nn is not None:
            self.log_audio_only()
        self.log_sampled()

    def on_test_start(self) -> None:
        """Same, for a `trainer.test()` that did not go through `fit`."""
        super().on_test_start()
        self._define_bin_metrics()

    def on_test_epoch_start(self) -> None:
        """Clear the test accumulators before the pass fills them."""
        super().on_test_epoch_start()
        self.reset_metrics("test")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One test batch. Only reached through an explicit `trainer.test()`."""
        return self._step(batch, "test")

    def on_test_epoch_end(self) -> None:
        """Flush the test metrics."""
        super().on_test_epoch_end()
        self.log_metrics("test")

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

    def sample(
        self,
        audio: torch.Tensor,
        num_steps: int,
        hard: bool = False,
        init_from_mu: bool | None = None,
        method: str | None = None,
        generator: torch.Generator | None = None,
        grid: str = "live",
    ) -> torch.Tensor:
        """Integrate from noise to (B, canvas_len) token ids for this batch of audio.

        init_from_mu seeds the first state's endpoint term with mu. The tau grid starts
        at the edge of the dead zone rather than at 0, so training's state there already
        carries beta(t0) * x1 and the sampler otherwise supplies nothing for it.

        `method` overrides `sampling.method`. `generator` draws x0, so two methods given
        generators seeded alike start from the same noise.

        `grid` picks the flow times walked: `live`, the curve training draws from, or
        `reference`, the control's (`reference_reparam`). Either way a `uniform` denoiser
        is fed u off the live curve at those times, the label it was trained against.
        """
        mu = None
        if self.mu_nn is not None:
            mu = F.softmax(self.mu_nn(audio), dim=-1)
        if init_from_mu is None:
            init_from_mu = bool(self.cfg.get("sampling", {}).get("init_from_mu", False))

        if grid == "live":
            gen_grid, u_grid = self.time_reparam.sampling_grid(num_steps)
        elif grid == "reference":
            gen_grid, _ = self.reference_reparam().sampling_grid(num_steps)
            u_grid = np.asarray(self.time_reparam.u_of(gen_grid))
        else:
            raise ValueError(f"sampling grid must be live or reference, got {grid!r}")

        # Algorithm 6 of the paper, or the equivalent-for-affine-gamma rebuild. They agree
        # exactly on linear and audio_cond and differ at O(h^2) on three_term.
        if method is None:
            method = str(self.cfg.get("sampling", {}).get("method", "interpolant"))
        if method not in SAMPLERS:
            raise ValueError(
                f"sampling.method must be one of {sorted(SAMPLERS)}, got {method!r}"
            )
        return SAMPLERS[method](
            self.denoiser,
            audio,
            gen_grid,
            self.vocab_size,
            self.canvas_len,
            self.alpha_fn,
            self.beta_fn,
            self.gamma_fn,
            mu,
            hard=hard,
            x1_init=mu if init_from_mu else None,
            generator=generator,
            waypoint=self.waypoint,
            time_grid=u_grid if self.time_input == "uniform" else None,
        )

    def reference_reparam(self) -> TimeReparametrization:
        """The control's curve: the exact one at unit noise, no floor, mu as it stands now.

        What this run would be drawing from under the recipe's defaults, so a run that
        changed `flow.alpha.scale` or `schedule.uniform_floor` can also be sampled on the
        grid the control samples on. Rebuilt per call; it takes milliseconds and is only
        asked for on sampling epochs.
        """
        accuracy = self.tau_scheduler.mu_accuracy
        if accuracy is None:
            accuracy = float(self.cfg.schedule.get("mu_accuracy", 0.0))
        return TimeReparametrization.exact(
            self.vocab_size,
            alpha_fn=make_schedule("linear_down"),
            beta_fn=self.beta_fn,
            gamma_fn=self.gamma_fn,
            mu_accuracy=accuracy,
        )

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
