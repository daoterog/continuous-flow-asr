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
| 10 | Does μ earn its place? | Not at n=256: `linear` **0.15%** vs `audio_cond` **1.60%**. | Needs the full dataset, where conditioning is weak |

## Settled defaults

`denoiser_lr 1e-3` · `mu_loss_weight 0.1` · `weight_decay 0.01` (was silently 0.0) ·
grad clipping ~irrelevant · τ measured on each variant's own path.

## Open

- **τ is static.** `mu_accuracy: 1.0` assumes μ ends perfect; never recomputed.
  `refresh_tau` exists, wired to nothing. Optimistic once μ is imperfect.
- **μ unproven.** Every result above is memorisation, where the denoiser can learn
  audio→text outright. μ's case is weak conditioning = full dataset.
- **σ = α/γ untested.** Reparameterises to `x₀′ ~ N(μ, σ²I)`; drift is σ-invariant.
  Controls how much of the answer the field hands over (σ=0.25 → ~95% recovery by t=0.2).
- **Sampling steps unswept** on real data. The toy (row 1) says more steps ≠ better WER.

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
