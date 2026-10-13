# Cross-check against FMLM, and notes toward distillation

Two separate things, written to be read slowly:

1. how our flow training compares to the FMLM/DUO reference in
   [src/models/fmlm/algo.py](src/models/fmlm/algo.py), and which differences are
   deliberate versus worth a second look;
2. what their `Distillation` class does, and the specific problems in the current
   [src/distillation.py](src/distillation.py) draft.

A caveat on the first part: their file is the **DUO** codebase (uniform-state discrete
diffusion via a Gaussian duality), not a continuous flow matcher. Several differences are
not bugs on either side — they are different models. I have tried to mark which is which.

---

## Part 1 — our flow-matching training vs theirs

**Correction to my first pass:** I cross-checked against `DUO`, which is a uniform-state
*discrete* diffusion and the wrong comparison. The right one is the
[`FLMBase`](src/models/fmlm/algo.py#L783) / [`FLM`](src/models/fmlm/algo.py#L1034) family
further down the file. Against *those*, our path is not a variant — it is the same thing:

```python
# theirs, corrupt_continuous
x_t = (1 - t) * noise + t * target_data
# ours, make_interpolant
i_t = alpha_fn(t) * x0 + beta_fn(t) * x1 [+ gamma_fn(t) * mu]   # alpha = 1-t, beta = t
```

Identical, and the loss is identical too: their `-(target_data * f).sum(-1)` with `f` a
`log_softmax` is cross-entropy against the one-hot target, which is what
[`sequence_ce`](src/flow_matching.py#L203) computes. Ignore what I wrote about
variance-preserving schedules — that was DUO's, not FLM's.

So the interesting differences are narrower and more specific than I first said.

### 1. They condition the model on τ. We condition it on t.

This is the one I would act on. Their loss, with their own comment:

```python
tau_t = self._sample_t_interval(B, ...)     # uniform in TAU
t     = self._tau_to_t(tau_t)               # convert to real time
x_t, target_data = self.corrupt_continuous(x0, t)
f = self.forward(x_t, tau_t)                # condition on tau_t
```

Ours, at [flow_matching.py#L183](src/flow_matching.py#L183):

```python
u = torch.rand(...)
t = self.sample_t_fn(u)                     # same conversion
i_t = self.make_interpolant(x1, t, mu)
logits = self.denoiser(x_t=i_t, time=t, ...)   # conditioned on t
```

Same corruption, different time signal into the network. The corruption **must** use `t`
— that is what defines the state. The conditioning is a free choice, and they take the
uniform one.

Why it matters, measured on `runs/1w60nmbf`'s schedule over 200k draws:

| conditioning | range | IQR | std |
|---|---|---|---|
| `t` (ours) | 0.187 – 0.983 | **0.721 – 0.800** | 0.076 |
| `u = τ(t)` (theirs) | 0.000 – 1.000 | 0.250 – 0.749 | 0.288 |

**Half of our training samples land in a time window 0.08 wide.** A sinusoidal or Fourier
time embedding has finite resolution; feeding it a variable with std 0.076 means almost
all of its capacity is spent on a region it never sees, and the region it does see is
compressed into a sliver. Conditioning on `u` spreads the same information over the full
input range, with no change to the corruption at all.

Note this is **not** a train/test mismatch — our sampler's grid is `τ⁻¹` of a uniform grid,
so inference feeds the same concentrated distribution. It is a representation-efficiency
problem, and it gets *worse* the better τ gets at concentrating the draw, which is exactly
the direction all our schedule work has been pushing.

The change is small: pass `u` instead of `t` as `time=` in
[`flow_step`](src/flow_matching.py#L183), and at sampling pass the uniform grid rather than
`generation_grid`'s output. The second half needs
[`generation_grid`](src/flow/schedule.py) to return both the `t` it steps on and the `u`
that produced them, since it currently dedups and returns only `t`.

Worth a config flag (`flow.time_conditioning: t | tau`) and a head-to-head, rather than a
silent switch — it changes what every existing checkpoint's time embedding means, so old
checkpoints are not loadable into the new convention.

### 2. They cap the logits before the softmax

[`_process_model_output`](src/models/fmlm/algo.py#L815):

```python
model_output = cap_value * torch.tanh(model_output / cap_value)   # cap_value = 30
return model_output.log_softmax(dim=-1)
```

A soft clamp to ±30 before the log-softmax. We apply none — `flow_step` sends the raw
denoiser output straight into `sequence_ce`. With a 1,193-token vocabulary and an unbounded
head this is mostly a stability guard, and we have not seen the kind of divergence it
prevents on the compact vocabulary. But EXPERIMENTS does record diverged runs at raised
learning rates (`valiant-star-26`, `cool-totem-31`, and row 20's full-vocabulary
divergence at lr 2e-3). If you revisit full-vocabulary training, this is a cheap thing to
try before lowering the learning rate again.

### 3. Their τ is analytic; ours is measured

`_tau_to_t` / `_t_to_tau` go through `utils.build_luts(K=self.vocab_size)` — a lookup
table built from the **vocabulary size alone**, closed form, fixed before training. Ours is
measured from a P_e curve and refreshed during training (`SimulatedTau` / `MeasuredTau`).

We do not have their `utils`, so I cannot check what `alpha_to_gamma` actually computes —
worth pulling if you want to compare their curve against our corruption sweep. If their
analytic τ is close to our simulated one, that is a strong independent check on the sweep;
if it is far, one of us is wrong about where the decoding work is.

### 4. They clamp the sampled time to `[t_min, t_max]`

`_sample_t_interval` draws in `[t_min, t_max]` rather than `[0, 1]`, keeping the model away
from the exact endpoints. Ours is bounded implicitly: `τ⁻¹(0)` is wherever the curve says,
and `τ⁻¹(1)` is the top bin's centre. Different mechanism, similar effect, and ours is
data-driven. No action.

### 5. Antithetic sampling of t

`_sample_t_interval` offsets the draw across the batch so one batch covers the interval
evenly, instead of i.i.d. uniform. Free variance reduction on the loss and on the per-bin
metrics. Low priority, but it is three lines.

### 6. Their sampler ends the way ours does

In [`generate_samples`](src/models/fmlm/algo.py#L1055), the last step is not an Euler step:

```python
if i == num_steps - 1:
    z = x_1_pred_probs
    break
```

They replace the state with the prediction at the final step. That is the same choice our
`interpolant` sampler makes by returning the last prediction, and it is independent
evidence for what we measured yesterday — `interpolant` beat the strict Euler
(`sample_tokens_euler`) by ~3 WER points at every step count. Their grid is also uniform in
τ, like ours.

### What I would do, in order

1. **Try `time=u`.** Biggest expected effect, smallest change, and the measurement above
   says our time embedding is currently being fed a variable with std 0.076.
2. **Pull their `utils.build_luts`** and plot their analytic τ against our corruption
   sweep. Cheap, and either outcome is informative.
3. **Antithetic t sampling**, if you want the metrics quieter.
4. **Logit capping**, only if you return to full vocabulary or raised learning rates.

## Part 2 — their distillation, and ours

### What `Distillation` actually does

[`Distillation(DUO)`](src/models/fmlm/algo.py#L569) is **progressive distillation**: the
student learns to take one step where the teacher takes a smaller one, and the step size
doubles each time the teacher is refreshed.

The four pieces:

**A teacher that is a periodically-frozen copy of the student.**
[`_maybe_update_teacher_weights`](src/models/fmlm/algo.py#L593) copies the student's
weights (or its EMA) into the teacher every `update_teacher_every` steps. There is no
separately-trained teacher — the model bootstraps off its own past self.

**A step size that doubles on each teacher refresh.**
[`_compute_dt`](src/models/fmlm/algo.py#L638) returns `2**n / T` with
`n = global_step // update_teacher_every`. So the student is asked to cover 1 step, then
2, then 4… Each refresh makes the task harder by exactly a factor of two, which is what
makes the curriculum stable.

**A shared-noise trajectory pair.** This is the part most likely to be got wrong.
[`_sample_trajectory`](src/models/fmlm/algo.py#L617) draws **one** `epsilon` and builds
both points from it:

```python
epsilon = torch.randn(x0.shape, ...)
xt = alpha_t * x0 + sigma_t * epsilon
xs = alpha_s * x0 + sigma_s * epsilon
```

`xt` and `xs` are two points on the *same* trajectory. With independent noise they would
be two unrelated samples and the consistency target would be meaningless.

**A distribution-matching loss, teacher detached.** `kl-fwd` is
`Σ p_teacher (log p_teacher − log p_student)`, `kl-bwd` the reverse, and
[`_teacher_logits`](src/models/fmlm/algo.py#L605) is `@torch.no_grad()` with an explicit
`.detach()`.

### Problems in the current draft

Going through [src/distillation.py](src/distillation.py):

**1. Independent noise between the two evaluations — the critical one.**
[`make_interpolant`](src/flow_matching.py#L246) draws a fresh `x₀` on every call
(`x0 = torch.randn_like(x1)`). The draft calls it twice — once for `i_s`, once for
`i_start_point` — so the two states are on *different* trajectories. Any consistency
objective built on them is matching unrelated samples.

Fix: draw `x₀` once and pass it in. That needs `make_interpolant` to accept an optional
`x0`, which is a two-line change and does not affect existing callers.

**2. `middle_logits` are raw logits used where a simplex point is expected.**

```python
map_start_to_middle = (
    (1.0 - mid_point) / (1.0 - start_point) * i_start_point +
    (mid_point - start_point) / (1.0 - start_point) * middle_logits
)
```

The algebra is right — it is exactly one Euler step, and matches our
[`drift`](src/flow/sampler.py#L46):
`x + h(D−x)/(1−t) = x(1−m)/(1−s) + D(m−s)/(1−s)`. But `D` has to be the predicted
**endpoint** on the simplex, i.e. `F.softmax(middle_logits, -1)`, the same thing
[`sample_tokens`](src/flow/sampler.py#L71) uses. Raw logits are unbounded and on a
different scale from `x₁`, so the mixture is not a point on the path.

**3. τ⁻¹ applied to a step size.**

```python
step_size = (1.0 - start_point) * torch.rand(...)
step_size = step_size if self.sample_t_fn is None else self.sample_t_fn(step_size)
```

`sample_t_fn` maps a uniform *quantile* to a *time*. Applying it to a duration is a
category error, and it breaks the bound: before the call `step_size ≤ 1 − start_point`,
after it `step_size ≥ τ⁻¹(0)`, so `end_point` can exceed 1. Either keep the step size in
`t` and drop the τ⁻¹, or — better, and closer to what τ is for — sample both endpoints as
quantiles and convert each:
`s = τ⁻¹(u₁)`, `m = τ⁻¹(u₂)` with `u₂ > u₁`, so the pair is uniform in *progress*.

**4. The "diagonal" term is an entropy, not a distillation loss.**

```python
diagonal_loss = -torch.sum(diagonal_probs.detach() * F.log_softmax(diagonal_logits, -1), -1)
```

With `p` detached and `log p` not, minimising this drives the model toward confident
predictions regardless of the target — it is entropy minimisation, not a match to
anything. If the intent was a self-distillation anchor, the target has to come from a
*different* evaluation (the teacher, or a two-step rollout), not the same forward pass.

**5. No second step and no loss returned.** `map_start_to_middle` is computed and then
dropped. The missing half is: run the model again at `mid_point` from that state, step to
`end_point`, and match the one-step student prediction at `start_point` against that
two-step result, with the two-step branch under `torch.no_grad()`.

**6. The class has no state.** `__init__` sets `denoiser` and `two_time_denoiser` to
`None`, but the body uses `self.vocab_size`, `self.mu_nn`, `self.sample_t_fn`,
`self.alpha_fn`, `self.cfg`, `self.pad_id`. `sequence_ce`, `token_weights` and
`make_interpolant` are also copied verbatim from
[FlowMatching](src/flow_matching.py#L203-L256). Either subclass `FlowMatching` or hold one as
an attribute — the copies will drift from the originals otherwise.

### A suggested shape

Taking their structure but keeping our continuous field:

```
u1, u2 = sorted uniforms          # two quantiles, u2 > u1
s, e   = tau_inv(u1), tau_inv(u2) # start and end times, uniform in progress
m      = (s + e) / 2

x0 = randn_like(x1)               # ONE noise draw, shared by everything below
x_s = interpolant(x1, s, mu, x0=x0)

# teacher: two steps of size (e-s)/2, no grad
with torch.no_grad():
    D_s   = softmax(teacher(x_s, s))
    x_m   = x_s + (m - s) * drift(x_s, D_s, s, mu, waypoint)
    D_m   = softmax(teacher(x_m, m))
    x_e_2 = x_m + (e - m) * drift(x_m, D_m, m, mu, waypoint)

# student: one step of size (e-s)
D_s_student = softmax(student(x_s, s))          # could also take (s, e-s) as input
x_e_1 = x_s + (e - s) * drift(x_s, D_s_student, s, mu, waypoint)

loss = mse(x_e_1, x_e_2)          # or KL between the two endpoint predictions
```

Points to decide:

- **What the student conditions on.** The name `two_time_denoiser` suggests `(t, dt)`,
  which is the shortcut-model formulation and is the right call if you want *one* model
  that works at any step count. DRAX's `Transformer` takes a single `time`, so this needs
  a second embedding — look at how `time` is embedded and add `dt` the same way.
- **Match states or predictions.** Matching `x_e` (states) is what the sketch does.
  Matching `D` (endpoint distributions) via KL is closer to their `kl-fwd`/`kl-bwd` and is
  better behaved, since the states carry the shared `x₀` and so partly cancel anyway.
- **Teacher refresh.** Their doubling curriculum needs a teacher copy and a step counter.
  Simplest first version: teacher = EMA of the student, fixed `dt`, and only add the
  doubling once the fixed-`dt` version trains.
- **μ and the waypoint.** `drift` already takes `waypoint` and handles `three_term`; μ
  should be computed **once** from the audio and reused for every evaluation in the step,
  not recomputed (it is deterministic given audio, so this is about cost, not
  correctness).

### The distillation model: conditioned on (start, end), not (t, dt)

Worth being precise about, because it changes what you build.

`FMLM.loss` calls [`self.forward(x_s, tau_s, tau_t)`](src/models/fmlm/algo.py#L1246) —
**two absolute times, the start and the end of the jump**, both in τ space. Not `(t, dt)`.
The backbone takes them as `sigma` and `sigma_prime`
([`forward`](src/models/fmlm/algo.py#L1128)), and `double_temb: True` is asserted at
config-validation time, so the two-embedding path is mandatory for this class.

`D_st` therefore reads as *"given the state at s, predict the endpoint as if jumping
straight to t"*. The undistilled model is the special case `s == t`.

That special case is not incidental — it is **half the training batch**:

```python
idx_diag, idx_offdiag_bndry, idx_offdiag = self._get_split_indices(
    B, ..., ratios=(diagonal_fraction, (1 - diagonal_fraction) / boundary_prob))
tau_s[idx_diag] = tau_diag[idx_diag];  tau_t[idx_diag] = tau_diag[idx_diag]   # s == t
tau_s[idx_offdiag_bndry] = 0.0;        tau_t[idx_offdiag_bndry] = 1.0         # full jump
tau_s[idx_offdiag] = ...;              tau_t[idx_offdiag] = ...               # s < t
```

Three regimes in every batch:

| rows | `tau_s`, `tau_t` | target |
|---|---|---|
| **diagonal** (`diagonal_fraction`) | `s == t` | the **teacher's** one-step output, `stopgrad` — or the data one-hot if there is no teacher |
| **boundary** | `s = 0`, `t = 1` | the full one-step jump, trained like any off-diagonal row |
| **off-diagonal** | `s < t` | the two-step rollout through the midpoint |

So the diagonal keeps the model honest as a plain denoiser while the off-diagonal teaches
it to jump. **This is what the `diagonal_loss` in the draft was reaching for** — but its
target has to be the teacher's output at `s` (detached), not the model's own detached
softmax from the same forward pass.

The off-diagonal target is the interesting part
([PSD branch](src/models/fmlm/algo.py#L1305)):

```python
D_su = _fwd(x_s, tau_s, tau_u).exp()                       # s -> midpoint
X_su = ((1-u)/(1-s)) * x_s + ((u-s)/(1-s)) * D_su          # Euler step to u
D_ut = _fwd(X_su, tau_u, tau_t).exp()                      # midpoint -> t
lambda_sut = ((1-t)*(u-s)) / ((1-u)*(t-s))
target = stopgrad(lambda_sut * D_su + (1 - lambda_sut) * D_ut)
```

Two observations. First, `X_su` is **exactly the mapping in the draft** — the same
`(1-u)/(1-s) * x + (u-s)/(1-s) * D` — so that algebra was right; what was missing was that
`D` must be `.exp()` of a log-softmax (a simplex point), and that the whole branch is under
`torch.no_grad()`. Second, the target is not the second endpoint prediction alone but a
**convex blend** of the two sub-step predictions, weighted by `lambda_sut`. At the midpoint
`u = (s+t)/2` that weight is not ½ — it is derived from the interpolant geometry.

Note also `tau_u = 0.5*(tau_s + tau_t)` is the midpoint **in τ**, and `s, u, t` are then
each converted to real time. The three times are not equally spaced in `t`.

### How their architecture differs from ours, beyond audio

Ours is DRAX's [`Transformer`](src/models/drax/transformer.py#L279). The conditioning path
is a single line:

```python
c = F.silu(self.time_embedding(time=time))      # transformer.py#L418
```

`c` then drives **adaLN** modulation in every block — `adaLN_modulation` produces 9
parameters per block (shift/scale/gate for self-attention, cross-attention and MLP), zero
initialised. One sinusoidal [`TimestepEmbedder`](src/models/drax/transformer.py#L50),
one conditioning vector.

The differences that matter for distillation:

1. **Two time embeddings, not one.** Theirs needs `double_temb`, i.e. a second embedder for
   `sigma_prime` whose output is combined with the first before the adaLN projection. Ours
   has exactly one `time_embedding` and `forward` takes a single `time`. **This is the
   architectural change you have to make**, and it is localised: a second
   `TimestepEmbedder`, a way to combine (sum, or concat then project to `cond_dim`), and an
   optional `time_end` argument on `forward`. Everything downstream — adaLN, the blocks,
   cross-attention to audio — is untouched.

2. **The combination must degrade to the current model when `s == t`.** If you sum the two
   embeddings, a checkpoint trained with one embedder cannot be loaded; if you initialise
   the second embedder's output projection to zero (the same trick adaLN already uses in
   [transformer.py#L129](src/models/drax/transformer.py#L129)), the two-time model starts
   exactly as the one-time model and learns the dependence on `t` from there. That makes
   `_initialize_student_from_teacher` possible, which is what their `setup` does.

3. **`use_jvp_attn`.** Their ESD (Eulerian) method needs a Jacobian-vector product through
   attention, so the backbone has a JVP-capable attention path. Our DRAX attention has no
   such thing. If you pick PSD (the progressive/semigroup method) you do not need it —
   which is a reason to prefer PSD for a first implementation.

4. **Logit capping before the softmax**, already noted in Part 1: their
   `_process_model_output` does `30 * tanh(out/30)` then `log_softmax`. For distillation
   this matters more than for training, because the targets are `.exp()` of a log-softmax
   and an uncapped teacher can emit near-one-hot targets that the KL cannot move away
   from.

5. **No audio anywhere.** Their cross-attention does not exist; our blocks have a
   cross-attention to `audio_embeddings` with its own adaLN parameters. For distillation,
   audio and μ are *fixed* across every forward in a step — compute them once per batch and
   reuse, since only `x_t` and the two times change.

### What I would do first

1. Make `make_interpolant` accept an optional `x0` — without it no consistency objective
   is testable.
2. Get a **fixed-`dt`, EMA-teacher, state-matching** version training at all, measured
   against the teacher's own WER at the same step count. Distillation that cannot match
   its teacher at equal steps is not yet worth tuning.
3. Only then add the doubling curriculum and the `(t, dt)` conditioning.

The gate for the whole thing: sampled WER at **1–4 steps** against the undistilled model's
ladder, which for `runs/1w60nmbf` is 84.40 / 74.58 / 68.89 at 1 / 2 / 4 steps. Beating
*that* curve at low step counts is the entire point; equalling it at 16 is not.
