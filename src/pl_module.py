"""The flow-matching LightningModule: interpolant, objective and optimizers.

Everything that reports numbers to a logger lives in `src.reporting`, and the metric bank
it owns is built by `src.metrics.utils`.
"""

from collections.abc import Callable

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import WhisperTokenizerFast

from src.data import VocabMap
from src.flow import build_tau_scheduler
from src.flow.coefficients import make_schedule
from src.flow.sampler import sample_tokens
from src.flow.sampling_probe import SamplingProbe
from src.flow.schedule import generation_grid
from src.metrics import schedule_bin_edges
from src.metrics.utils import is_binned
from src.models import build_audio_decoder, build_denoiser
from src.reporting import MetricReporting


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
        self.sampling_probe = SamplingProbe(cfg.get("sampling"), cfg.data.batch_size)
        self.tau = build_tau_scheduler(cfg.schedule, seed=cfg.seed)

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

        # Make sure to get the correct number of perr bins depending on the startegy
        perr_bins = self.tau.metric_bins or int(cfg.metrics.get("perr_bins", 10))
        self.init_metrics(
            ("denoiser", "mu") if self.mu_nn is not None else ("denoiser",),
            tokenizer,
            pad_id,
            vocab_map,
            sample_t_fn=sample_t_fn,
            loss_bins=cfg.metrics.get("loss_bins", 10),
            perr_bins=perr_bins,
            wer_bins=OmegaConf.to_container(cfg.metrics.get("wer_bins", {})) or None,
            input_perr_bins=cfg.metrics.get("input_perr_bins", 10),
        )

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
        if self.tau.is_due(self):
            new_tau, skipped = self.tau.refresh(
                self, getattr(self.trainer, "datamodule", None)
            )
            if skipped is not None:
                self.print(skipped)
            if new_tau is not None:
                self.sample_t_fn = new_tau
                for metric in self.metrics.values():
                    if is_binned(metric):
                        metric.set_edges(schedule_bin_edges(new_tau, metric.num_bins))
                lo = float(new_tau(0.0))
                self.log("tau/t0", lo)
                self.log_tau_shape(new_tau)
                self.print(
                    f"tau refreshed for epoch {self.current_epoch}"
                    f"t in [{lo:.3f}, 1.000] now carries the draw"
                )
                if self.tau.mu_accuracy is not None:
                    self.print(f"mu accuracy EMA {self.tau.mu_accuracy:.4f}")
        self.reset_metrics("train")

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

    def token_weights(self, tokens: torch.Tensor) -> torch.Tensor | None:
        """(B, L) loss weights, or None when every position counts the same.

        Real tokens weigh 1; a pad at distance d into the run weighs

            floor + (1 - floor) * exp(-(d - 1) / pad_decay)

        so the first pad keeps full weight and the tail decays to `flow.pad_loss_weight`
        (`pad_decay` 0 or null gives that floor flat). The first pad is the whole length
        decision — a model that never stops scores over 100% WER on insertions — while a
        pad 150 positions out is free.

        Lowering the floor also shrinks the weighted mean's denominator, scaling each real
        token by up to canvas_len / n_real ~ 3.3, so part of any gain is an effective
        learning-rate rise and a sweep over this has to vary the learning rate too.

        Padding is assumed to be a contiguous suffix, which is what `collate` produces.
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
        self.tau.observe_mu(self.metrics)
        self.log_mu_accuracy()

    def on_validation_epoch_start(self) -> None:
        """Clear the val accumulators before the pass fills them."""
        super().on_validation_epoch_start()
        self.reset_metrics("val")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One validation batch."""
        return self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Flush the val metrics, then sample from noise."""
        super().on_validation_epoch_end()
        self.log_metrics("val")
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

        if self.sample_t_fn is None:
            gen_grid = torch.linspace(0, 1, num_steps + 1).tolist()
        else:
            gen_grid = [float(t) for t in generation_grid(self.sample_t_fn, num_steps)]

        return sample_tokens(
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
