"""Flow-map distillation: a two-time denoiser D(x_s, s, t) learned from a one-time teacher.

The one-time model answers "given the state at t, what is the endpoint" and has to be
integrated over many steps. A two-time model answers "given the state at s, what is the
endpoint if I jump straight to t", so in principle one evaluation replaces the whole
integration. On the teacher measured in stage 0 that is a 51.75 -> 18.16 WER gap between one
step and sixteen, and closing it is the entire point of this module.

Follows FMLM's PSD formulation (`src/models/fmlm/algo.py`, the `FMLM` class). The student is
the teacher plus one extra time embedder, zero-initialised, so it *starts* as the teacher and
learns only the dependence on where it is jumping to.
"""

from collections.abc import Callable

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from transformers import WhisperTokenizerFast

from src.checkpointing import load_components
from src.data import VocabMap
from src.flow.coefficients import make_schedule, waypoint_strength
from src.flow.sampling_probe import SamplingProbe
from src.models import build_two_time_denoiser, load_teacher_weights
from src.reporting import MetricReporting

# Floor on the interval denominators. The drawn step size can be exactly 0, which collapses
# the midpoint onto the start and makes both (t - s) and (u - s) vanish
EPS = 1e-8


class FlowMapping(MetricReporting, pl.LightningModule):
    """Distils a frozen one-time denoiser into a two-time one.

    Holds three models: the teacher's denoiser and mu, both frozen, and the student. The
    teacher is not a copy of the student's weights kept in step — it is a fixed checkpoint
    loaded from disk, which is what makes the target stationary.

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
        """Load the teacher, build the student from it, and take the path it was trained on.

        `cfg.flow` and `cfg.model.denoiser` are overwritten from the teacher's manifest: the
        student has to match the teacher's geometry for the weights to transfer, and has to
        denoise the same probability path or it is learning a different map.
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

        distill = cfg.distill
        manifest, teacher = load_components(
            distill.teacher, variant=distill.get("variant", "best")
        )
        for key, want, got in (
            ("vocab_size", int(manifest.vocab_size), int(vocab_size)),
            ("canvas_len", int(manifest.canvas_len), int(canvas_len)),
        ):
            if want != got:
                raise ValueError(
                    f"the teacher at {distill.teacher} was trained with {key}={want} but "
                    f"this run builds {key}={got}. Every token id would be read against a "
                    f"different vocabulary."
                )
        if "denoiser" not in teacher:
            raise ValueError(f"no denoiser weights under {distill.teacher}")

        self.denoiser = teacher["denoiser"]
        self.mu_nn = teacher.get("mu_nn")
        for model in (self.denoiser, self.mu_nn):
            if model is not None:
                model.requires_grad_(False)

        # Built from the teacher's own config node, so `load_teacher_weights` fits: the
        # student is the teacher plus a zero-initialised second time embedder, which makes
        # it *identical* to the teacher at init and lets distillation start from parity
        # rather than from noise.
        self.two_time_denoiser = build_two_time_denoiser(
            manifest.model.denoiser, vocab_size
        )
        load_teacher_weights(self.two_time_denoiser, self.denoiser.state_dict())

        # The path is the teacher's, not this config's.
        flow = OmegaConf.merge(cfg.flow, OmegaConf.create(manifest.flow))
        self.alpha_fn = make_schedule(flow.get("alpha", "linear_down"))
        self.beta_fn = make_schedule(flow.get("beta", "linear_up"))
        self.gamma_fn = make_schedule(flow.get("gamma"))
        # The 4c of the drift's waypoint term, 0 for the variants without one.
        self.waypoint = waypoint_strength(flow.get("gamma"))
        self.pad_loss_weight = float(flow.get("pad_loss_weight", 1.0))
        self.hard = bool(distill.get("hard", False))
        self.diagonal_fraction = float(distill.get("diagonal_fraction", 0.25))

        # Sampling a two-time model needs its own integrator; until stage 3 writes one,
        # `sampling.enabled: false` makes `is_due` False so `sample()` is never reached.
        self.sampling_probe = SamplingProbe(cfg.get("sampling"), cfg.data.batch_size)

        # Only the student trains, so only the student is worth compiling.
        if cfg.train.get("compile"):
            self.two_time_denoiser = torch.compile(
                self.two_time_denoiser, mode=cfg.train.get("compile_mode", "default")
            )

        # TODO(stage 3): how to evaluate a flow map is still open — the binned metrics take
        # one t and this step has three (start, mid, end), and the two loss terms are not
        # comparable to each other. Until that is settled there is no metric bank, so every
        # call that iterates `self.metrics` is commented out rather than fed an empty one.
        # self.init_metrics(...)

    def train(self, mode: bool = True) -> "FlowMapping":
        """Keep the teacher in eval whatever Lightning does to the module.

        `load_components` returns the teacher already in eval, but Lightning calls `.train()`
        at the start of fitting and that recurses into every child — which would switch the
        teacher's dropout back on. This teacher carries dropout 0.1, so its predictions would
        stop being a fixed function of the input, and a distillation target that changes under
        you is the one thing the frozen teacher exists to prevent.
        """
        super().train(mode)
        self.denoiser.eval()
        if self.mu_nn is not None:
            self.mu_nn.eval()
        return self

    def on_fit_start(self) -> None:
        """Nothing to declare yet — the histogram panels arrive with the metric bank."""
        super().on_fit_start()
        # TODO(stage 3): self._define_bin_metrics()

    def on_train_epoch_start(self) -> None:
        """Clear the train accumulators.

        No schedule refresh: tau is the teacher's and does not move, so there is nothing
        here to re-derive.
        """
        super().on_train_epoch_start()
        # TODO(stage 3): self.reset_metrics("train")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One optimizer step."""
        return self._step(batch, "train")

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        """One forward pass, logged as scalars until there is a metric bank to fill."""
        loss, out, _startpoint, _midpoint, _endpoint, _weighting = self.map_step(batch)
        self._check_finite(loss, out, split)
        # TODO: move loss logging into log_losses once it is implemented.
        batch_size = batch["tokens"].size(0)
        self.log(f"total_loss/{split}", loss, on_step=False, on_epoch=True,
                 batch_size=batch_size)
        for name, record in out.items():
            self.log(f"{name}/{split}", record["loss"], on_step=False, on_epoch=True,
                     batch_size=batch_size)
        # TODO(stage 3): self.log_losses(...) and self.update_metrics(...) once the metric
        # bank exists. `update_metrics` takes a single t and this step has three, so the two
        # terms need separate calls against their own times.
        return loss

    def map_step(self, batch: dict) -> tuple:
        """One distillation step: the diagonal term, the off-diagonal term, and the times.

        TODO(stage 2): the objective here is not yet FMLM's. Two things are wrong by design
        until then — every row gets *both* terms rather than the batch being split into
        diagonal / boundary / off-diagonal (`diagonal_fraction` is read and ignored), and the
        diagonal target is the student's own detached softmax from the same forward pass,
        which is entropy minimisation rather than distillation. The target has to come from
        the frozen teacher (`algo.py:1299-1303`).
        """
        audio = batch["audio"].float()
        tokens = batch["tokens"]
        x1 = F.one_hot(tokens, self.vocab_size).float()

        out = {}
        mu = None
        if self.mu_nn is not None:
            with torch.no_grad():
                mu_logits = self.mu_nn(audio)
                mu = F.softmax(mu_logits, dim=-1)

        startpoint, midpoint, endpoint, state, pred, weighting = self.sample_interval(
            batch_size=tokens.size(0)
        )
        i_start_point = self.make_interpolant(x1=x1, t=startpoint, mu=mu)

        diagonal_logits = self.denoiser(
            x_t=i_start_point, time=startpoint, audio_embeddings=audio
        )
        diagonal_probs = F.softmax(diagonal_logits, dim=-1)
        diagonal_loss = -torch.sum(
            diagonal_probs.detach() * F.log_softmax(diagonal_logits, dim=-1), dim=-1
        )
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
        mapped_interpolant = state * i_start_point + pred * midpreds
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
        off_diagonal_loss = -torch.sum(
            distillation_logits * F.log_softmax(start_to_end_logits, dim=-1), dim=-1
        )
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

    def sample_interval(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Draw (s, u, t) with s <= u <= t <= 1, plus the coefficients the algebra needs.

        The three times come back as (B,) because that is what the models take. Everything
        derived from them comes back as (B, 1, 1), because it multiplies (B, L, V) fields —
        a (B,) factor would broadcast against the *vocabulary* axis instead of the batch,
        which is silent whenever B happens not to equal V.

        Three such coefficients, all pure functions of the times, so they are built here
        rather than inline:

          state, pred   the Euler map from s to u, x_u = state * x_s + pred * D(x_s, s, u)
          weighting     lambda_sut, how much of the distillation target comes from the first
                        sub-step rather than the second (algo.py:1330)

        Every denominator is clamped: `step_size` can be 0, which puts u on top of s and
        makes both (t - s) and (u - s) vanish.
        """
        start_point = torch.rand(batch_size, device=self.device)
        step_size = (1.0 - start_point) * torch.rand(batch_size, device=self.device)
        end_point = start_point + step_size
        # TODO(stage 2): FMLM takes the midpoint in tau and only then maps to real time, so
        # the three are not equally spaced in t. Here they are, which is a different split.
        mid_point = (start_point + end_point) / 2.0

        s, u, t = (x[:, None, None] for x in (start_point, mid_point, end_point))
        remaining = (1.0 - s).clamp_min(EPS)
        state = (1.0 - u) / remaining
        pred = (u - s) / remaining
        weighting = ((1.0 - t) * (u - s)) / ((1.0 - u) * (t - s)).clamp_min(EPS)
        return start_point, mid_point, end_point, state, pred, weighting

    def make_interpolant(
        self, x1: torch.Tensor, t: torch.Tensor, mu: torch.Tensor | None = None
    ) -> torch.Tensor:
        """I_t = alpha(t) x0 + beta(t) x1 [+ gamma(t) mu], broadcast over (B, L, V)."""
        x0 = torch.randn_like(x1)
        i_t = self.alpha_fn(t)[:, None, None] * x0 + self.beta_fn(t)[:, None, None] * x1
        if self.gamma_fn is not None and mu is not None:
            i_t = i_t + self.gamma_fn(t)[:, None, None] * mu
        return i_t

    def on_train_epoch_end(self) -> None:
        """Nothing to flush yet; the losses are logged per step."""
        super().on_train_epoch_end()
        # TODO(stage 3): self.log_metrics("train")

    def on_validation_epoch_start(self) -> None:
        """Clear the val accumulators before the pass fills them."""
        super().on_validation_epoch_start()
        # TODO(stage 3): self.reset_metrics("val")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One validation batch."""
        return self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Sample from noise, when there is a sampler to do it with.

        `sampling.enabled: false` makes this return before touching `sample()`, which a
        two-time model does not have yet.
        """
        super().on_validation_epoch_end()
        # TODO(stage 3): self.log_metrics("val")
        self.log_sampled()

    def on_test_start(self) -> None:
        """Same, for a `trainer.test()` that did not go through `fit`."""
        super().on_test_start()
        # TODO(stage 3): self._define_bin_metrics()

    def on_test_epoch_start(self) -> None:
        """Clear the test accumulators before the pass fills them."""
        super().on_test_epoch_start()
        # TODO(stage 3): self.reset_metrics("test")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One test batch. Only reached through an explicit `trainer.test()`."""
        return self._step(batch, "test")

    def on_test_epoch_end(self) -> None:
        """Flush the test metrics."""
        super().on_test_epoch_end()
        # TODO(stage 3): self.log_metrics("test")

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
        """Integrate from noise to (B, canvas_len) token ids. Not written yet.

        TODO(stage 3): a two-time model needs its own integrator and cannot reuse
        `src/flow/sampler.py`. Those samplers call a model with a single time and rebuild the
        state from the interpolant; this one steps on a grid uniform in tau, calls the model
        with *both* ends of each step, and advances by the same Euler map `map_step` already
        uses — `state * x + pred * D`. The last step returns the prediction instead of
        stepping into t=1 (`algo.py:1516`).
        """
        del audio, num_steps, hard, init_from_mu
        raise NotImplementedError(
            "FlowMapping.sample is stage 3. Set sampling.enabled=false, or score the "
            "teacher instead with scripts.evaluation.sample_eval."
        )

    def configure_optimizers(self) -> dict:
        """AdamW with a per-model LR group, plus OneCycleLR stepped every batch.

        `estimated_stepping_batches` already accounts for epochs, accumulation and the
        drop_last on the training loader, so the schedule cannot run past the run.
        """
        train = self.cfg.train
        # The student alone. `denoiser` and `mu_nn` are the frozen teacher: handing their
        # parameters to AdamW would let weight decay move them even with requires_grad off.
        # Its own learning rate, because `denoiser_lr` is the teacher's and distilling from
        # a converged model starts somewhere quite different from training one.
        lr = train.two_time_denoiser_lr
        groups = [{"params": self.two_time_denoiser.parameters(), "lr": lr}]
        max_lr = [lr]

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
