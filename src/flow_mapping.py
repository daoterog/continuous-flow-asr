"""Flow-map distillation: a two-time denoiser D(x_s, s, t) learned from a one-time teacher.

The one-time model answers "given the state at t, what is the endpoint" and has to be
integrated over many steps. A two-time model answers "given the state at s, what is the
endpoint if I jump straight to t", so in principle one evaluation replaces the whole
integration.
"""


from typing import NamedTuple

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from transformers import WhisperTokenizerFast

from src.checkpointing import load_components
from src.data import VocabMap
from src.flow.coefficients import make_schedule, waypoint_strength
from src.flow.sampler import sample_map, sample_map_interpolant
from src.flow.sampling_probe import SamplingProbe
from src.flow.schedule import TimeReparametrization
from src.flow_matching import token_weights
from src.models import build_two_time_denoiser, load_teacher_weights
from src.reporting import MetricReporting

# Floor on the interval denominators. The drawn step size can be exactly 0, which collapses
# the midpoint onto the start and makes both (t - s) and (u - s) vanish
EPS = 1e-8

DISTILL_GROUPS = ("diagonal", "boundary", "offdiag")
METRIC_AXES = {"offdiag": "gap"}
MAP_SAMPLERS = {"interpolant": sample_map_interpolant, "euler": sample_map}


class MapInterval(NamedTuple):
    """One batch's drawn interval: the three times, the coefficients, and the regime."""

    startpoint: torch.Tensor
    midpoint: torch.Tensor
    endpoint: torch.Tensor

    # The Euler map from s to u, x_u = state * x_s + pred * D(x_s, s, u). (B, 1, 1).
    state: torch.Tensor
    pred: torch.Tensor

    weighting: torch.Tensor
    groups: torch.Tensor
    gap: torch.Tensor


class FlowMapping(MetricReporting, pl.LightningModule):
    """Distils a frozen one-time denoiser into a two-time one.

    Holds three models: the teacher's denoiser and mu, both frozen, and the student. The
    teacher is a fixed checkpoint loaded from disk, which is what makes the target stationary.
    """

    def __init__(
        self,
        cfg: DictConfig,
        vocab_size: int,
        pad_id: int,
        canvas_len: int,
        tokenizer: WhisperTokenizerFast,
        vocab_map: VocabMap | None = None,
    ) -> None:
        """Load the teacher, build the student from it, and take the path it was trained on.

        `cfg.flow` and `cfg.model.denoiser` are overwritten from the teacher's manifest: the
        student has to match the teacher's geometry for the weights to transfer, and has to
        denoise the same probability path or it is learning a different map.
        """
        super().__init__()
        # tokenizer/vocab_map are objects, not hyperparameters: they would not survive a
        # checkpoint round-trip and are rebuilt from the data instead.
        self.save_hyperparameters(ignore=["tokenizer", "vocab_map"], logger=False)
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.canvas_len = canvas_len
        self.tokenizer = tokenizer
        self.vocab_map = vocab_map

        self.denoiser: torch.nn.Module | None = None
        self.mu_nn: torch.nn.Module | None = None
        self.two_time_denoiser: torch.nn.Module | None = None
        self.time_reparam: TimeReparametrization | None = None
        self.stratified_t = bool(cfg.flow.get("stratified_t", False))

        manifest, saved_tau = self.load_models()
        distill = cfg.distill

        # The geometry and the path are the teacher's, not this config's
        cfg.model = OmegaConf.create(manifest.model)
        cfg.flow = OmegaConf.merge(cfg.flow, OmegaConf.create(manifest.flow))
        flow = cfg.flow
        self.alpha_fn = make_schedule(flow.get("alpha", "linear_down"))
        self.beta_fn = make_schedule(flow.get("beta", "linear_up"))
        self.gamma_fn = make_schedule(flow.get("gamma"))
        # The 4c of the drift's waypoint term, 0 for the variants without one.
        self.waypoint = waypoint_strength(flow.get("gamma"))
        self.pad_loss_weight = float(flow.get("pad_loss_weight", 1.0))
        self.hard = bool(distill.get("hard", False))

        gamma_spec = flow.get("gamma")
        if gamma_spec is not None and str(gamma_spec) != str(flow.get("alpha", "linear_down")):
            raise NotImplementedError(
                f"flow.gamma is {gamma_spec!r} and flow.alpha is "
                f"{flow.get('alpha', 'linear_down')!r}. The distillation rollout's Euler map "
                f"is only valid when they match (or gamma is absent), because mu has to fold "
                f"into the noise for the interpolant to be invertible."
            )

        # Validated in `train.validate_distill_split`, so it is a partition by the time it
        # gets here; these fractions are therefore also the weight of each objective.
        self.split = tuple(float(distill.split[k]) for k in DISTILL_GROUPS)

        self.sampling_probe = SamplingProbe(cfg.get("sampling"), cfg.data.batch_size)

        # Always the teacher's curve, and we default to an exact curve when the object is not
        # saved
        self.time_reparam = saved_tau or TimeReparametrization.exact(
            vocab_size,
            alpha_fn=self.alpha_fn,
            beta_fn=self.beta_fn,
            gamma_fn=self.gamma_fn,
            mu_accuracy=float(
                manifest.get("mu_accuracy_final")
                or manifest.run_config.schedule.get("mu_accuracy", 0.0)
            ),
        )

        # Only the student trains, so only the student is worth compiling.
        if cfg.train.get("compile"):
            self.two_time_denoiser = torch.compile(
                self.two_time_denoiser, mode=cfg.train.get("compile_mode", "default")
            )

        metrics = cfg.get("metrics", {})
        self.init_metrics(
            DISTILL_GROUPS,
            tokenizer,
            pad_id,
            vocab_map,
            time_reparam=self.time_reparam,
            axes=METRIC_AXES,
            loss_bins=metrics.get("loss_bins", 10),
            perr_bins=metrics.get("perr_bins", 10),
            wer_bins=OmegaConf.to_container(metrics.get("wer_bins", {})) or None,
            input_perr_bins=metrics.get("input_perr_bins", 10),
        )

    def load_models(self) -> tuple[DictConfig, TimeReparametrization | None]:
        """Attach the three models from `distill.teacher`; return the manifest and the curve."""
        distill = self.cfg.distill
        manifest, saved, saved_tau = load_components(
            distill.teacher, variant=distill.get("variant", "best")
        )
        self.check_manifest(manifest)
        if "denoiser" not in saved:
            raise ValueError(
                f"no denoiser weights under {distill.teacher}; found "
                f"{sorted(saved) or 'nothing'} for variant "
                f"{distill.get('variant', 'best')!r}"
            )

        self.denoiser = saved["denoiser"]
        self.mu_nn = saved.get("mu_nn")
        for model in (self.denoiser, self.mu_nn):
            if model is not None:
                model.requires_grad_(False)

        self.two_time_denoiser = build_two_time_denoiser(
            manifest.model.denoiser, self.vocab_size
        )
        load_teacher_weights(self.two_time_denoiser, self.denoiser.state_dict())

        return manifest, saved_tau

    def check_manifest(self, manifest: DictConfig) -> None:
        """Refuse a components directory that does not describe this run."""
        source = self.cfg.distill.teacher
        for key, want, got in (
            ("vocab_size", int(manifest.vocab_size), int(self.vocab_size)),
            ("canvas_len", int(manifest.canvas_len), int(self.canvas_len)),
        ):
            if want != got:
                raise ValueError(
                    f"the weights at {source} were trained with {key}={want} but this run "
                    f"builds {key}={got}. Every token id would be read against a different "
                    f"vocabulary."
                )
        time_input = manifest.flow.get("time_input", "reparametrized")
        if time_input != "reparametrized":
            raise ValueError(
                f"the weights at {source} were trained with "
                f"flow.time_input={time_input!r}, but distillation feeds both times to "
                f"the denoisers as flow time t"
            )

    @property
    def components(self) -> tuple[str, ...]:
        """The student, plus the mu it cannot sample without."""
        return ("two_time_denoiser", "mu_nn")

    def train(self, mode: bool = True) -> "FlowMapping":
        """Keep the teacher in eval whatever Lightning does to the module."""
        super().train(mode)
        for model in (self.denoiser, self.mu_nn):
            if model is not None:
                model.eval()
        return self

    def on_fit_start(self) -> None:
        """Declare the histogram panels' step metric before anything is logged."""
        super().on_fit_start()
        self._define_bin_metrics()

    def on_train_epoch_start(self) -> None:
        """Clear the train accumulators."""
        super().on_train_epoch_start()
        self.reset_metrics("train")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One optimizer step."""
        return self._step(batch, "train")

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        """One forward pass, with each group's record fed to its own metrics."""
        loss, out = self.map_step(batch)
        self._check_finite(loss, out, split)
        self.log_losses(loss, out, split, batch_size=batch["tokens"].size(0))
        self.update_metrics(out, batch["tokens"], None, split)
        return loss

    def map_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """One PSD distillation step."""
        audio = batch["audio"].float()
        tokens = batch["tokens"]
        x1 = F.one_hot(tokens, self.vocab_size).float()

        out = {}
        mu = None
        if self.mu_nn is not None:
            with torch.no_grad():
                mu_logits = self.mu_nn(audio)
                mu = F.softmax(mu_logits, dim=-1)

        iv = self.sample_interval(batch_size=tokens.size(0))
        i_start_point = self.make_interpolant(x1=x1, t=iv.startpoint, mu=mu)

        student_logits = self.two_time_denoiser(
            x_t=i_start_point,
            startpoint=iv.startpoint,
            endpoint=iv.endpoint,
            audio_embeddings=audio,
        )
        student_logprobs = F.log_softmax(student_logits, dim=-1)
        pred_ids = student_logits.argmax(dim=-1)

        is_diagonal = iv.groups == 0
        idx_diag = is_diagonal.nonzero(as_tuple=True)[0]
        idx_rollout = (~is_diagonal).nonzero(as_tuple=True)[0]

        # Each row contributes exactly one term, so the split is also the weighting.
        per_token = torch.zeros_like(tokens, dtype=student_logits.dtype)

        with torch.no_grad():
            teacher_target = F.softmax(
                self.denoiser(
                    x_t=i_start_point[idx_diag],
                    time=iv.startpoint[idx_diag],
                    audio_embeddings=audio[idx_diag],
                ),
                dim=-1,
            )
        per_token[idx_diag] = -torch.sum(
            teacher_target * student_logprobs[idx_diag], dim=-1
        )

        with torch.no_grad():
            x_s = i_start_point[idx_rollout]
            audio_od = audio[idx_rollout]
            mid_logits = self.two_time_denoiser(
                x_t=x_s,
                startpoint=iv.startpoint[idx_rollout],
                endpoint=iv.midpoint[idx_rollout],
                audio_embeddings=audio_od,
            )

            if self.hard:
                d_su = F.one_hot(
                    mid_logits.argmax(dim=-1), self.vocab_size
                ).to(mid_logits.dtype)
            else:
                d_su = F.softmax(mid_logits, dim=-1)

            x_su = iv.state[idx_rollout] * x_s + iv.pred[idx_rollout] * d_su
            d_ut = F.softmax(
                self.two_time_denoiser(
                    x_t=x_su,
                    startpoint=iv.midpoint[idx_rollout],
                    endpoint=iv.endpoint[idx_rollout],
                    audio_embeddings=audio_od,
                ),
                dim=-1,
            )
            w = iv.weighting[idx_rollout]
            rollout_target = w * d_su + (1.0 - w) * d_ut

        per_token[idx_rollout] = -torch.sum(
            rollout_target * student_logprobs[idx_rollout], dim=-1
        )

        # The same weights `FlowMatching.sequence_ce` applies, so the two objectives weigh a
        # pad alike; `flow` is the teacher's, so the student inherits what it was trained with.
        weights = token_weights(
            tokens,
            self.pad_id,
            float(self.pad_loss_weight),
            float(self.cfg.flow.get("pad_decay", 0.0) or 0.0),
        )
        if weights is None:
            per_seq = per_token.mean(dim=1)
        else:
            per_seq = (per_token * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)

        loss = per_seq.mean()

        # Store values for reporting
        for name, idx, bin_value in (
            ("diagonal", idx_diag, iv.startpoint),
            ("boundary", (iv.groups == 1).nonzero(as_tuple=True)[0], iv.startpoint),
            ("offdiag", (iv.groups == 2).nonzero(as_tuple=True)[0], iv.gap),
        ):
            if not idx.numel():
                continue
            out[name] = {
                "loss": per_seq[idx].mean(),
                "loss_per_seq": per_seq[idx],
                "pred_ids": pred_ids[idx],
                "target_ids": tokens[idx],
                "bin_value": bin_value[idx],
                "n": idx.numel(),
            }

        if self.cfg.get("metrics", {}).get("input_perr_bins"):
            out["input"] = {
                "pred_ids": i_start_point.argmax(dim=-1),
                "target_ids": tokens,
                "bin_value": iv.startpoint,
            }

        return loss, out

    def sample_interval(self, batch_size: int) -> MapInterval:
        """Draw an interval in uniform space, then map its three points to flow time."""
        groups = self.assign_groups(batch_size)

        gap = self.draw_u(batch_size)
        u_start = self.draw_u(batch_size) * (1.0 - gap)
        u_end = u_start + gap

        u_diagonal = self.draw_u(batch_size)
        u_start = torch.where(groups == 0, u_diagonal, u_start)
        u_end = torch.where(groups == 0, u_diagonal, u_end)
        u_start = torch.where(groups == 1, torch.zeros_like(u_start), u_start)
        u_end = torch.where(groups == 1, torch.ones_like(u_end), u_end)
        u_mid = (u_start + u_end) / 2.0

        # Recomputed because diagonal and boundary values were replaced above.
        gap = u_end - u_start

        start_point, mid_point, end_point = (
            self.time_reparam.t_of(x) for x in (u_start, u_mid, u_end)
        )

        s, u, t = (x[:, None, None] for x in (start_point, mid_point, end_point))
        remaining = (1.0 - s).clamp_min(EPS)
        state = (1.0 - u) / remaining
        pred = (u - s) / remaining
        weighting = ((1.0 - t) * (u - s)) / ((1.0 - u) * (t - s)).clamp_min(EPS)
        return MapInterval(
            startpoint=start_point,
            midpoint=mid_point,
            endpoint=end_point,
            state=state,
            pred=pred,
            weighting=weighting,
            groups=groups,
            gap=gap,
        )

    def assign_groups(self, batch_size: int) -> torch.Tensor:
        """(B,) group id per row — 0 diagonal, 1 boundary, 2 off-diagonal."""
        counts, prev, cumulative = [], 0, 0.0
        for frac in self.split[:-1]:
            cumulative += frac
            upto = min(round(batch_size * cumulative), batch_size)
            counts.append(max(0, upto - prev))
            prev += counts[-1]
        counts.append(batch_size - prev)

        groups = torch.repeat_interleave(
            torch.arange(len(counts), device=self.device),
            torch.tensor(counts, device=self.device),
        )
        return groups[torch.randperm(batch_size, device=self.device)]

    def draw_u(self, n: int) -> torch.Tensor:
        """(n,) uniform draws in [0, 1), one per row, before tau^-1 maps them to flow time.

        `flow.stratified_t` puts exactly one draw in each slice [i/n, (i+1)/n), shuffled
        across rows — FMLM's `antithetic_sampling` (algo.py:866), which is stratified
        sampling rather than antithetic: antithetic would pair u with 1-u.
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
            i_t = i_t + self.gamma_fn(t)[:, None, None] * mu
        return i_t

    def on_train_epoch_end(self) -> None:
        """Flush the train metrics."""
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
        """Flush the val metrics, then sample from noise if this epoch is due a ladder."""
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
        method: str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate from noise to (B, canvas_len) token ids, asking for each jump directly.

        The two-time analogue of `FlowMatching.sample`, and `sampling.method` picks the same
        two names. Both variants call the model with the ends of the step it is about to
        take, and both decode the last prediction; they differ only in whether the state is
        advanced or rebuilt. At `num_steps = 1` either is one call asking for 0 -> 1, which
        is the number this whole exercise is about.

        The grid is the teacher's, and both model inputs are real flow time — distillation
        has no u conditioning, so the `u` leg of `sampling_grid` is unused here.
        """
        mu = None
        if self.mu_nn is not None:
            mu = F.softmax(self.mu_nn(audio), dim=-1)
        if init_from_mu is None:
            init_from_mu = bool(self.cfg.get("sampling", {}).get("init_from_mu", False))
        if method is None:
            method = str(self.cfg.get("sampling", {}).get("method", "interpolant"))
        if method not in MAP_SAMPLERS:
            raise ValueError(
                f"sampling.method must be one of {sorted(MAP_SAMPLERS)}, got {method!r}"
            )
        grid, _u = self.time_reparam.sampling_grid(num_steps)
        return MAP_SAMPLERS[method](
            self.two_time_denoiser,
            audio,
            grid,
            self.vocab_size,
            self.canvas_len,
            self.alpha_fn,
            self.beta_fn,
            self.gamma_fn,
            mu,
            hard=hard,
            x1_init=mu if init_from_mu else None,
            generator=generator,
        )

    def configure_optimizers(self) -> dict:
        """AdamW with a per-model LR group, plus OneCycleLR stepped every batch.

        `estimated_stepping_batches` already accounts for epochs, accumulation and the
        drop_last on the training loader, so the schedule cannot run past the run.
        """
        train = self.cfg.train
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
