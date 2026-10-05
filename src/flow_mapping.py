
from collections.abc import Callable

import lightning.pytorch as pl
import torch
from torch.nn import functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import WhisperTokenizerFast

from src.reporting import MetricReporting
from src.data import VocabMap
from src.flow.coefficients import make_schedule, waypoint_strength


class FlowMapping(MetricReporting, pl.LightningModule):

    def __init__(
        self,
        cfg: DictConfig,
        vocab_size: int,
        pad_id: int,
        canvas_len: int,
        tokenizer: WhisperTokenizerFast,
        vocab_map: VocabMap | None = None,
        sample_t_fn: Callable | None = None,
    ):
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

        # Here we need to load a denoiser checkpoint and mu
        self.denoiser = None
        self.mu_nn = None
        self.two_time_denoiser = None

        # Diagonal fraction
        self.diagonal_fraction = None

        flow = cfg.flow
        self.alpha_fn = make_schedule(flow.get("alpha", "linear_down"))
        self.beta_fn = make_schedule(flow.get("beta", "linear_up"))
        self.gamma_fn = make_schedule(flow.get("gamma"))
        # The 4c of the drift's waypoint term, 0 for the variants without one.
        self.waypoint = waypoint_strength(flow.get("gamma"))

        if cfg.train.get("compile"):
            mode = cfg.train.get("compile_mode", "default")
            self.denoiser = torch.compile(self.denoiser, mode=mode)
            self.two_time_denoiser = torch.compile(self.two_time_denoiser, mode=mode)
            if self.mu_nn is not None:
                self.mu_nn = torch.compile(self.mu_nn, mode=mode)

        # TODO: Figure out new metrics
        self.init_metrics(
            ("denoiser", "mu") if self.mu_nn is not None else ("denoiser",),
            tokenizer,
            pad_id,
            vocab_map,
            sample_t_fn=sample_t_fn,
            loss_bins=cfg.metrics.get("loss_bins", 10),
            perr_bins=cfg.metrics.get("perr_bins", 10),
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
        # Maybe refresh schedule but doubt it
        self.reset_metrics("train")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One optimizer step."""
        return self._step(batch, "train")

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        loss, out, startpoint, midpoint, endpoint = self.map_step(batch)
        self._check_finite(loss, out, split)
        self.log_losses(loss, out, split, batch_size=batch["tokens"].size(0))
        # TODO: see what to do abaut update metrics here
        # self.update_metrics(out, batch["tokens"], t, split)
        return loss

    def map_step(self, batch: dict):
        audio = batch["audio"].float()
        tokens = batch["tokens"]
        x1 = F.one_hot(tokens, self.vocab_size).float()

        out = {}
        mu = None
        if self.mu_nn is not None:
            with torch.no_grad():
                mu_logits = self.mu_nn(audio)
                mu = F.softmax(mu_logits, dim=-1)

        startpoint, midpoint, endpoint, weighting = self.sample_interval(batch_size=tokens.size(0))
        i_start_point = self.make_interpolant(x1=x1, t=startpoint, mu=mu)

        diagonal_logits = self.denoiser(
            x_t=i_start_point, time=startpoint, audio_embeddings=audio
        )
        diagonal_probs = F.softmax(diagonal_logits, dim=-1)
        diagonal_loss = - torch.sum(diagonal_probs.detach() * F.log_softmax(diagonal_logits, dim=-1), dim=-1)
        out["diagonal_loss"] = {
            "loss": diagonal_loss.mean(),
            "loss_per_seq": diagonal_loss,
            "pred_ids": diagonal_logits.argmax(dim=-1),
        }

        middle_logits = self.two_time_denoiser(
            x_t=i_start_point,
            startpoint=startpoint,
            endpoint=midpoint,
            audio_embeddings=audio
        )
        midprobs = F.softmax(middle_logits, dim=-1)
        midpreds = midprobs
        if self.hard:
            midpreds = F.one_hot(middle_logits.argmax(dim=-1), self.vocab_size).float()
        mapped_interpolant = (
            (1.0 - midpoint) / (1.0 - startpoint) * i_start_point +
            (midpoint - startpoint) / (1.0 - startpoint) * midpreds
        )
        with torch.no_grad():
            end_logits = self.two_time_denoiser(
                x_t=mapped_interpolant,
                startpoint=midpoint,
                endpoint=endpoint,
                audio_embeddings=audio
            )
            endpreds = F.softmax(end_logits, dim=-1)
            distillation_logits = weighting * midpreds + (1.0 - weighting) * endpreds
            distillation_logits = distillation_logits.detach()
        start_to_end_logits = self.two_time_denoiser(
            x_t=i_start_point,
            startpoint=startpoint,
            endpoint=endpoint,
            audio_embeddings=audio
        )
        off_diagonal_loss = - torch.sum(distillation_logits * F.log_softmax(start_to_end_logits, dim=-1), dim=-1)
        out["off_diagonal_loss"] = {
            "loss": off_diagonal_loss.mean(),
            "loss_per_seq": off_diagonal_loss,
            "pred_ids": start_to_end_logits.argmax(dim=-1),
        }

        loss = out["diagonal_loss"]["loss"] + out["off_diagonal_loss"]["loss"]

        return loss, out, startpoint, midpoint, endpoint, weighting

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

    def sample_interval(self, batch_size: int) -> torch.Tensor:
        """Sample a time in [t_min, t_max] for each batch element."""
        start_point = torch.rand(batch_size, device=self.device)
        step_size = (1.0 - start_point) * torch.rand(batch_size, device=self.device)
        end_point = start_point + step_size
        # Here we might need to first transform start and endpoint to compute
        # accurate estimate of midpoint.
        mid_point = (start_point + end_point) / 2.0
        # Compute the weighting for the interpolant.
        weighting = (
            (1 - end_point) * (mid_point - start_point)) / \
            ((1 - mid_point) * (end_point - start_point)
        )
        return start_point, mid_point, end_point, weighting

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

        # TODO: rethink
        # if self.sample_t_fn is None:
        #     gen_grid = torch.linspace(0, 1, num_steps + 1).tolist()
        # else:
        #     gen_grid = [float(t) for t in generation_grid(self.sample_t_fn, num_steps)]

        # # Algorithm 6 of the paper, or the equivalent-for-affine-gamma rebuild. They agree
        # # exactly on linear and audio_cond and differ at O(h^2) on three_term.
        # method = str(self.cfg.get("sampling", {}).get("method", "interpolant"))
        # if method not in SAMPLERS:
        #     raise ValueError(
        #         f"sampling.method must be one of {sorted(SAMPLERS)}, got {method!r}"
        #     )
        # return SAMPLERS[method](
        #     self.denoiser,
        #     audio,
        #     gen_grid,
        #     self.vocab_size,
        #     self.canvas_len,
        #     self.alpha_fn,
        #     self.beta_fn,
        #     self.gamma_fn,
        #     mu,
        #     hard=hard,
        #     x1_init=mu if init_from_mu else None,
        #     waypoint=self.waypoint,
        # )

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
