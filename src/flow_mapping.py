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
from src.models import build_two_time_denoiser, load_teacher_weights
from src.reporting import MetricReporting

# Floor on the interval denominators. The drawn step size can be exactly 0, which collapses
# the midpoint onto the start and makes both (t - s) and (u - s) vanish
EPS = 1e-8

# The three regimes a batch is divided into, in the order `distill.split` names them. Each
# row trains exactly one, so these are also the relative weights of the three objectives.
# `train.validate_distill_split` checks the config against this before a module is built.
#
# Also the metric bank's group names, so one vocabulary serves the config key, the code and
# the wandb section — there is no second set of names to map between.
DISTILL_GROUPS = ("diagonal", "boundary", "offdiag")

# What each group's metrics resolve along, for any group that is not measured in flow time.
# The off-diagonal rows are the ones the whole exercise is about and the question asked of
# them is not "where in the flow" but "how far a jump", so they bin on gap = t - s. The name
# reaches the logged key, which is what stops a gap being reported under one that says `t`.
METRIC_AXES = {"offdiag": "gap"}


class MapInterval(NamedTuple):
    """One batch's drawn interval: the three times, the coefficients, and the regime.

    Named rather than a bare tuple because there are eight of them and `map_step` reads
    most: positional unpacking stops being legible, and appending a field would silently
    shift one at every call site.
    """

    # The three flow times the models are conditioned on, (B,) each. s <= u <= t.
    startpoint: torch.Tensor
    midpoint: torch.Tensor
    endpoint: torch.Tensor
    # The Euler map from s to u, x_u = state * x_s + pred * D(x_s, s, u). (B, 1, 1).
    state: torch.Tensor
    pred: torch.Tensor
    # lambda_sut: how much of the distillation target comes from the first sub-step rather
    # than the second (algo.py:1330). (B, 1, 1).
    weighting: torch.Tensor
    # Which regime each row trains, indexing DISTILL_GROUPS. (B,).
    groups: torch.Tensor
    # The jump's size in uniform space, u_end - u_start, i.e. the fraction of the decoding
    # this one evaluation has to cover. (B,).
    #
    # This is *the* gap, and the reason the metric axis is this rather than t - s: the
    # reparametrization only decides which flow time the interpolant is built at, so a gap
    # is measured the same whatever curve is in use, while t - s moves with the curve. It
    # is also the only one that stays uniform — drawn flat on [0, 1] here, where t - s
    # comes out with a median of 0.096 and a 99th percentile of 0.598, which would leave
    # the top of an equal-width histogram empty.
    gap: torch.Tensor

# `sampling.method` selects one, by the same names the one-time path uses so a config
# reads the same for both modules and a teacher/student pair cannot be set up on
# different integrators by accident.
MAP_SAMPLERS = {"interpolant": sample_map_interpolant, "euler": sample_map}


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

        distill = cfg.distill
        variant = distill.get("variant", "best")
        manifest, teacher, teacher_tau = load_components(distill.teacher, variant=variant)
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
        time_input = manifest.flow.get("time_input", "reparametrized")
        if time_input != "reparametrized":
            raise ValueError(
                f"the teacher at {distill.teacher} was trained with "
                f"flow.time_input={time_input!r}, but distillation feeds both times to "
                f"the denoisers as flow time t"
            )

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

        # The rollout's Euler map eliminates the noise from the interpolant to step from s to
        # u. With a mu term that only works when gamma == alpha, because then
        # x_t = alpha (z + mu) + beta x1 and mu rides along inside the noise. For a gamma
        # that is not proportional to alpha the map silently computes the wrong intermediate
        # state — measured at 0.14 absolute on the parabola against 2e-7 here — so it is
        # refused rather than commented.
        gamma_spec = flow.get("gamma")
        if gamma_spec is not None and str(gamma_spec) != str(flow.get("alpha", "linear_down")):
            raise ValueError(
                f"flow.gamma is {gamma_spec!r} and flow.alpha is "
                f"{flow.get('alpha', 'linear_down')!r}. The distillation rollout's Euler map "
                f"is only valid when they match (or gamma is absent), because mu has to fold "
                f"into the noise for the interpolant to be invertible."
            )

        # Validated in `train.validate_distill_split`, so it is a partition by the time it
        # gets here; these fractions are therefore also the weight of each objective.
        self.split = tuple(float(distill.split[k]) for k in DISTILL_GROUPS)

        self.sampling_probe = SamplingProbe(cfg.get("sampling"), cfg.data.batch_size)

        # Always the teacher's, never this run's. A run saves the curve its weights were
        # trained against beside them, so when one is there it is used verbatim. Older
        # checkpoints have none; theirs was the simulated curve, which is the closed form at
        # the accuracy the manifest records, so it rebuilds exactly. Either way the student
        # samples on the teacher's grid, which is what makes the two ladders comparable.
        self.time_reparam = teacher_tau or TimeReparametrization.exact(
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

        # One group per regime, each resolved along the axis its own question is asked in:
        # the diagonal in flow time, where s == t and drift from the teacher is local to a
        # t; the off-diagonal in the gap, which is the size of the jump; the boundary not at
        # all, since its rows are pinned at (0, 1).
        #
        # The teacher's tau, because `sample_interval` draws in uniform space and maps: the
        # diagonal's startpoint is tau^-1 of a uniform draw, distributed exactly like the
        # one-time path's t, so equal-width bins in t would leave the bottom half of them
        # empty and pile the epoch into bins 5-8. `make_metrics` applies these edges only to
        # the groups binned on t — tau's quantiles are quantiles OF t, so they mean nothing
        # on the gap axis, which stays equal-width and uniform by construction.
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

    @property
    def components(self) -> tuple[str, ...]:
        """Only the student is worth saving."""
        return ("two_time_denoiser",)

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
        """Declare the histogram panels' step metric before anything is logged."""
        super().on_fit_start()
        self._define_bin_metrics()

    def on_train_epoch_start(self) -> None:
        """Clear the train accumulators.

        No schedule refresh, and so no re-binning either: tau is the teacher's and does not
        move, and in any case these bins are equal-width rather than cut from it.
        """
        super().on_train_epoch_start()
        self.reset_metrics("train")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One optimizer step."""
        return self._step(batch, "train")

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        """One forward pass, with each group's record fed to its own metrics.

        Every record carries its own rows, targets and axis value, so `update_metrics`
        needs no defaults from here — the two it takes are never the ones used.
        """
        loss, out = self.map_step(batch)
        self._check_finite(loss, out, split)
        self.log_losses(loss, out, split, batch_size=batch["tokens"].size(0))
        self.update_metrics(out, batch["tokens"], None, split)
        return loss

    def map_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """One PSD distillation step (FMLM's `FMLM.loss`, algo.py:1246-1349).

        Every row trains exactly one term, chosen by `distill.split`:

          diagonal (s == t)   against the frozen teacher's own one-step output at s, so the
                              student cannot drift away from being a good denoiser
          otherwise           against the two-step rollout s -> u -> t, which is what makes
                              one evaluation of D(x_s, s, t) agree with two. The boundary
                              rows (s = 0, t = 1) are folded in here, exactly as the
                              reference does at algo.py:1285 — the full jump is just the
                              widest off-diagonal case

        One student forward covers the whole batch; the two losses are slices of it. The
        rollout that builds the off-diagonal target runs entirely under `no_grad`: it is a
        target, and the gradient goes only into the single `student_logits` call.

        The objective has two terms but the records it returns name three groups, because
        the boundary rows are trained with the off-diagonal ones and read separately: their
        jump is 0 -> 1, which is exactly what a one-step sample asks for, so their error is
        a dense estimate of the headline number that the 64-utterance probe only reaches
        every few epochs.
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

        iv = self.sample_interval(batch_size=tokens.size(0))
        i_start_point = self.make_interpolant(x1=x1, t=iv.startpoint, mu=mu)

        # One forward for the whole batch, at each row's own (s, t). On diagonal rows s == t,
        # so the same call is the one-step prediction there (algo.py:1306).
        student_logits = self.two_time_denoiser(
            x_t=i_start_point,
            startpoint=iv.startpoint,
            endpoint=iv.endpoint,
            audio_embeddings=audio,
        )
        student_logprobs = F.log_softmax(student_logits, dim=-1)
        pred_ids = student_logits.argmax(dim=-1)

        # The objective's two slices. `idx_rollout` is every non-diagonal row — boundary
        # and off-diagonal together — because FMLM folds the full jump into the rollout
        # term as its widest case (algo.py:1285).
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

        weights = self.token_weights(tokens)
        if weights is None:
            per_seq = per_token.mean(dim=1)
        else:
            per_seq = (per_token * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)

        loss = per_seq.mean()

        # Reporting only — the objective above is already complete. Three slices rather than
        # the two it used, because the boundary rows are read apart from the off-diagonal
        # ones they train with, and each group is measured along the axis its own question
        # is asked in: the diagonal at the flow time it denoised, the off-diagonal at the
        # size of the jump it was asked for.
        for name, idx, bin_value in (
            ("diagonal", idx_diag, iv.startpoint),
            ("boundary", (iv.groups == 1).nonzero(as_tuple=True)[0], iv.startpoint),
            ("offdiag", (iv.groups == 2).nonzero(as_tuple=True)[0], iv.gap),
        ):
            # An empty group contributes nothing rather than a zero that would pull its
            # mean down. `assign_groups` can produce one whenever a share is small against
            # the batch — boundary is 1.6% of it.
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

        # argmax of the field the student was handed, with no model in it. Read against
        # diagonal_perr it separates what I_s already carried from what the student added,
        # which is the only way to tell a good denoiser from an easy input.
        if self.cfg.get("metrics", {}).get("input_perr_bins"):
            out["input"] = {
                "pred_ids": i_start_point.argmax(dim=-1),
                "target_ids": tokens,
                "bin_value": iv.startpoint,
            }

        return loss, out

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

    def sample_interval(self, batch_size: int) -> MapInterval:
        """Draw an interval in uniform space, then map its three points to flow time.

        The draw is uniform in u and the times come out of tau^-1, which is FMLM's
        `uniform_diff` (algo.py:1260-1297) — they draw `tau_d`, `tau_s`, `tau_t` and only
        then call `_tau_to_t` on each. Doing it the other way round, uniformly in t, sounds
        equivalent and is not: tau is decoding progress, so what a jump *means* is how much
        tau it covers, and a uniform draw in t is wildly non-uniform in tau. Measured on
        this teacher's curve, drawing in t puts 20.5% of rows on jumps covering under 2% of
        the decoding and 43% on jumps covering over 90% — two thirds of the batch is either
        a no-op or almost the whole jump. Drawing in u gives 2% and 10%. A 16-step sample
        takes sixteen steps of exactly 1/16 in u, and the t-space draw almost never
        produced one.

        Rows are assigned to one of three regimes by `distill.split`, and the draw is
        overwritten accordingly — `u_s = u_t` on the diagonal, `u_s = 0, u_t = 1` on the
        boundary, which maps to `tau^-1(0) -> tau^-1(1)`. That is `sampling_grid(1)`, so
        the boundary rows are the one-step jump by construction rather than by coincidence.
        The assignment is a fresh permutation each step, so a given utterance is not pinned
        to one regime for the whole run.

        The three times come back as (B,) because that is what the models take. The
        coefficients come back as (B, 1, 1), because they multiply (B, L, V) fields — a
        (B,) factor would broadcast against the *vocabulary* axis instead of the batch,
        which is silent whenever B happens not to equal V.

        They are also built in real flow time, not in u, which looks like an oversight and
        is not. The interpolant is affine in t, `I_t = alpha(t) x0 + beta(t) x1`, so
        eliminating x0 to step from s to u is only valid there; in u there is no such
        algebra, because tau^-1 is not affine. FMLM is the same — `X_su` and `lambda_sut`
        (algo.py:1315-1322) are built from the mapped `s_od, u_od, t_od`, and only the
        network's conditioning uses tau.

        Every denominator is clamped: the drawn gap can be 0, which puts u on top of s and
        makes both (t - s) and (u - s) vanish.
        """
        groups = self.assign_groups(batch_size)

        gap = torch.rand(batch_size, device=self.device)
        u_start = torch.rand(batch_size, device=self.device) * (1.0 - gap)
        u_end = u_start + gap

        u_diagonal = torch.rand(batch_size, device=self.device)
        u_start = torch.where(groups == 0, u_diagonal, u_start)
        u_end = torch.where(groups == 0, u_diagonal, u_end)
        u_start = torch.where(groups == 1, torch.zeros_like(u_start), u_start)
        u_end = torch.where(groups == 1, torch.ones_like(u_end), u_end)

        # Halfway in uniform space, as FMLM takes it (algo.py:1296). The arithmetic mean in
        # t would be a different intermediate state, since tau^-1 is not affine.
        u_mid = (u_start + u_end) / 2.0

        # Recomputed rather than carried down from the draw above: the diagonal and boundary
        # rows have had their endpoints replaced, so the gap drawn for them is not the gap
        # they ended up with.
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
        """(B,) group id per row — 0 diagonal, 1 boundary, 2 off-diagonal.

        Built by laying out the counts and permuting, as FMLM does (algo.py:1231-1241),
        rather than by sampling each row independently: that way the realised proportions
        match `distill.split` exactly at every batch size instead of only in expectation,
        which matters when the boundary share is 1.6% and the batch is 128.

        The counts are floors of the cumulative fractions, so rounding never accumulates and
        the last group absorbs the remainder. A group can still come out empty on a small
        batch, which `map_step` has to tolerate.
        """
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
        """Flush the val metrics, then sample from noise if this epoch is due one.

        `boundary_perr/val_overall` and `sampled_*_1/perr` measure the same jump — the
        boundary rows sit at (0, 1) and tau's grid starts at t=0 — so when both land in the
        same epoch they are each other's check.
        """
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
