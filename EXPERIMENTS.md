# Experiment log

Continuous flow matching for ASR. All results below are the **overfit harness**
(`scripts/sweep_overfit.py`): N utterances, val pointed at the same utterances, dropout
and weight decay off. It answers "can the model memorise?", not "does it generalise".

**SAMPLED WER** = integrate from noise, no target anywhere. The only metric that reflects
inference. The binned metrics score one denoising step from an `x_t` built out of the
target, so at high t they largely measure the input.

## Results

| # | Question | Result | Next |
|---|---|---|---|
| 1 | Does the sampler work at all? | Yes. Toy task where sequences come from a transition matrix, so "is this sequence legal?" is checkable with no reference: **0.0001 (chance) → 0.55** at 32 steps. But with weak audio, more steps made samples *more* legal (0.05→0.54) and *less* like the one the audio encoded (0.34→0.21). | Extra steps buy plausible-looking output, not the right transcript |
| 2 | Can `linear` memorise 64 utts? | **2.56%** sampled WER at lr 3e-4. Gate passed. | Try μ variants |
| 3 | Best LR? | 1e-3 ≈ 3e-4 ≫ 1e-2. Config default was **1e-2**, ~100× too hot (loss 1.86 vs 0.014). | Default → 1e-3 |
| 4 | Can `audio_cond` memorise? | No — 62.8%. Worse as it fit *better*: loss 2.86→0.24 while sampled WER 62.8→77.7. | Anti-correlation ⇒ not a tuning problem |
| 5 | Is μ the problem? | μ was **perfect** (P_e 0.0000) while the denoiser sat at 35% WER. | Log μ's own metrics — they had never been shown |
| 6 | grad-clip or μ loss weight? | 2×2: **μ weight is everything** (41% → **0.00%**), clipping irrelevant. The auxiliary loss was starving the denoiser. | Default `mu_loss_weight` 1.0 → 0.1 |
| 7 | Is the interpolant broken? | No. With a perfect μ the field's recovery triples (t=0.6: 0.044 → 0.230). But below t≈0.8 the noise max (α·√(2 ln V)≈3.0) exceeds the signal peak (~1.0). | — |
| 8 | Is τ right for μ variants? | **No.** `sweep_corruption` measured τ on the *linear* path for every variant. audio_cond n=256: **16.64% → 1.60%** after the fix, in ⅓ the steps. | Largest single win |
| 9 | Does the denoiser use μ or ignore it? | Uses its own audio pathway. At the lowest t bin the input is 94.8% wrong and the denoiser is **99.93% right** — in both variants. | μ is not load-bearing here |
| 10 | Does μ earn its place? | Not at n=256: `linear` **0.15%** vs `audio_cond` **1.60%**. Margin is inside the noise (see 11). | Needs the full dataset, where conditioning is weak |
| 11 | How reproducible is any of this? | `audio_cond` n=256, *same config*, four runs: **1.60 / 39.27 / 0.27 / 1.28%**. Inputs bit-identical, so it is training non-determinism. Usually ≤1.6%, occasionally catastrophic. | Small margins need ≥4 seeds; one run can mislead |
| 12 | Should τ track μ's real accuracy? | **Yes.** Honest prior (`mu_accuracy 0.0`) + refresh every epoch: **0.00 / 0.02%** over 2 seeds. Static optimistic prior, no refresh: **0.27 / 1.28%**. Tighter *and* better. | `refresh_every_n_epochs` on; n=2, so suggestive not settled |

τ is refreshed from μ's measured error (`1 − mu_perr`), re-running the model-free corruption
sweep with that accuracy. Bins move with it, applied at the next epoch's start. Refresh
cannot outpace validation — raise `--val-times` to refresh more often.

## Settled defaults

`denoiser_lr 1e-3` · `mu_loss_weight 0.1` · `weight_decay 0.01` (was silently 0.0) ·
grad clipping ~irrelevant · τ measured on each variant's own path, refreshed from μ.

## Open

- **σ = α/γ untested.** Reparameterises to `x₀′ ~ N(μ, σ²I)`; drift is σ-invariant.
  Controls how much of the answer the field hands over (σ=0.25 → ~95% recovery by t=0.2).
  The μ-sweep figure argues this is the knob that would let μ matter: even a perfect μ
  barely moves P_e(0), because a unit spike cannot beat a noise maximum of ~3.8.
- **More data.** 10.46% is converged on 100 hours. `train.960` is the path to DRAX's ~2%,
  and it needs the feature repack first: 474 GiB untrimmed cannot be cached at any
  `--mem`, 118 GiB pre-pooled fits.
- **Full vocabulary** (row 20) unresolved — lower LR and a real epoch budget untried.
- **Occasional catastrophic runs.** One of four identical overfit runs landed at 39% where
  the others were ≤1.6%; on the cluster, `valiant-star-26` and `cool-totem-31` diverged
  outright. Both were at raised learning rates, so this may simply be LR sensitivity.

## Real runs — full `clean/train.100`, scored by sampling from noise

Best so far: **18.70% sampled WER** on all 2,703 of `clean/validation` at 16 steps
(17.85% at 128) — `audio_cond`, denoiser `small` (40.7M), μ `micro` (1.8M), lr 2e-3, batch
128, canvas 224, τ prior 0.0 refreshed every 5 epochs, **300 epochs**, 3.6 h on one A100
(`skilled-lake-22`).

That run logged **10.46%** during training. The difference is entirely the evaluation
slice: it sampled the first 512 validation utterances, which are 8 points easier than the
split as a whole. Every sampled WER in this file from before `sampling.n_utts` defaulted to
null is on some subset and is not comparable to this number — only to others on the same
slice.

| # | Question | Result | Verdict |
|---|---|---|---|
| 13 | Does μ earn its place on real data? | **Only if it cannot memorise.** All four rescored on 2,703 utterances at 16 steps: `micro` (1.8M) **21.74%**, no μ **31.35%**, `small` (18.5M) **38.74%**, `drax` (5.8M) **51.17%**. μ helps by 9.6 points — or hurts by up to 20 — and which one depends on its train/val accuracy gap: 0.036 for `micro`, 0.510 for `small`. | Row 10 closed. Keep μ small |
| 23 | Why does a large μ hurt so much? | Because the denoiser learns to trust it. `small` reached **1.000 train accuracy against 0.490 val**, so during training it supplied the right answer every time and the denoiser built no audio pathway of its own; at inference it supplies noise, confidently. The samples show it as duplication — `BLIGHT HIS BLIGHT HIS`, `THERE THERE THERE`. | Cap μ's capacity, not its dropout (row 14) |
| 14 | Does μ dropout fix its overfitting? | No. μ's train/val accuracy gap is real (`small`: 1.000 train vs 0.490 val) but dropout 0.1 made `micro` worse, 16.06 → 29.53. | Reject |
| 15 | Is τ's μ prior right? | **No.** The 1.0 default assumed a perfect μ and drew **34%** of each batch below t=0.65; measured μ accuracy is 0.39–0.57. Prior 0.0 puts 8.7% there and matches the no-μ path exactly. | Default → 0.0 |
| 16 | How often should τ refresh? | Doesn't matter much. Every 5 **16.25**, every 1 17.38, never 22.51 — spread ~1 SE. Smoothing μ's accuracy (momentum 0.8) *hurt*: 23.90. | Refresh every 5, momentum 0.0 |
| 17 | Should τ refresh read train or val? | `train`. Reading val lets held-out data steer which t the next epochs train on, and the training field really is built from μ's output on training audio. Costs little: at `micro` the two agree to 0.036. | Default → `train` |
| 18 | Does down-weighting padding speed learning? | On the overfit harness, enormously: 67.38% → **5.13%**, with the LR control ruling out an effective-LR explanation. On real data it **reversed** — three seeds at 21.1 / 26.1 / 27.4 against the control's 16.8. | Memorisation artifact. Reject |
| 19 | What actually moves WER? | **Epochs**, but by less than it first looked. Within one run on a fixed slice: 35.1 @ 44, 22.2 @ 89, 17.4 @ 149, **10.5 @ 299**. That trajectory is valid (constant slice) but the *size* is not transferable — on the full split, 90 epochs with the old prior gives 21.74 and 300 with the new prior gives 18.70, so **≤3.0 points**, not 11.8, and that 3.0 includes the prior fix. | Train longer, but re-measure the gain on the full split |
| 22 | How big was the subset distortion? | On 64 utterances μ's margin read 4.7 points; on 2,703 it is **9.6**. The subset halved the effect it was being used to measure. | Rescore anything load-bearing |
| 21 | How many integration steps? | Gains saturate fast: 49.9 (1 step) → 27.2 (2) → 20.6 (4) → 19.0 (8) → **18.7 (16)** → 17.9 (128). 16 is within 0.85 of 128 at an eighth of the cost. Token error stays flat at ~0.49 the whole way — extra steps fix **alignment**, not per-token accuracy. | Keep 16. Row closed |
| 20 | Does the full Whisper vocabulary work? | Not yet: 90.7% (lr 5e-4), 102.1% (1e-3), diverged (2e-3), at 60 epochs on 8k utterances. μ is fine (acc 0.50) — the denoiser is not. | Needs far lower LR and more epochs |

Rows 13–14 were measured under the old τ prior of 1.0, so they answer "which μ" but not
"how much μ would help with τ built correctly".

**Comparability.** Sampled WER was scored on subsets, which made runs look different when
they were not: two identical configs on the *same* slice differ by **0.6** points
(16.25 / 16.81), two identical configs on *different* slices by **5.4**. `sampling.n_utts`
now defaults to the whole validation split. Anything below a point is noise; the
comparisons above at 2–3 points are suggestive, not settled.

## Throughput

| Fix | Before | After |
|---|---|---|
| Dataloader: pool per-utterance in fp32, not per-batch in fp16 | 106 ms/batch | **47** |
| `BinnedWER`: cache the reference normaliser, one tensor add | 16.9 ms/batch | **9.2** |
| `steps_per_epoch` + `--parallel 2` | GPU 30% | **59%** |

Measured and rejected: `torch.compile` (+45s warmup, no steady-state gain at batch 16),
`--parallel 3` (no gain over 2), fused AdamW (+28% but needs clipping off).

## Cluster

Precompute: **0.166 s/utt**, 6.0 utts/s solo, 3.6–7.5 under 8-way concurrency (GPFS write
contention). Size walltime off the *slowest* shard. `clean/{train.100,validation,test}`
on scratch, untrimmed: 103 + 9.7 + 9.4 GB.
