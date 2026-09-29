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

- **Sampling steps unswept** on real data. Still the cheapest untried win: it needs no
  retraining, only `scripts/sample_eval.py --components runs/<id>/components --steps ...`.
  The toy (row 1) says more steps ≠ better WER, so it could go either way.
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

Best so far: **10.46% sampled WER** on `clean/validation` — `audio_cond`, denoiser `small`
(40.7M), μ `micro` (1.8M), lr 2e-3, batch 128, canvas 224, τ prior 0.0 refreshed every 5
epochs, **300 epochs**, 3.6 h on one A100 (`skilled-lake-22`).

| # | Question | Result | Verdict |
|---|---|---|---|
| 13 | Does μ earn its place on real data? | Yes, but only the smallest. `micro` (1.8M) **16.06%** vs `linear` (no μ) **20.70%**. Bigger μ is worse: `small` 20.32, `drax` 32.04. | μ is a nudge, not a recogniser. Row 10 closed |
| 14 | Does μ dropout fix its overfitting? | No. μ's train/val accuracy gap is real (`small`: 1.000 train vs 0.490 val) but dropout 0.1 made `micro` worse, 16.06 → 29.53. | Reject |
| 15 | Is τ's μ prior right? | **No.** The 1.0 default assumed a perfect μ and drew **34%** of each batch below t=0.65; measured μ accuracy is 0.39–0.57. Prior 0.0 puts 8.7% there and matches the no-μ path exactly. | Default → 0.0 |
| 16 | How often should τ refresh? | Doesn't matter much. Every 5 **16.25**, every 1 17.38, never 22.51 — spread ~1 SE. Smoothing μ's accuracy (momentum 0.8) *hurt*: 23.90. | Refresh every 5, momentum 0.0 |
| 17 | Should τ refresh read train or val? | `train`. Reading val lets held-out data steer which t the next epochs train on, and the training field really is built from μ's output on training audio. Costs little: at `micro` the two agree to 0.036. | Default → `train` |
| 18 | Does down-weighting padding speed learning? | On the overfit harness, enormously: 67.38% → **5.13%**, with the LR control ruling out an effective-LR explanation. On real data it **reversed** — three seeds at 21.1 / 26.1 / 27.4 against the control's 16.8. | Memorisation artifact. Reject |
| 19 | What actually moves WER? | **Epochs.** Same config: 35.1 @ 44, 22.2 @ 89, 17.4 @ 149, 14.0 @ 199, **10.5 @ 299**. −11.8 points from 89 → 299, and converged only at the end. | Train longer before tuning anything |
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
