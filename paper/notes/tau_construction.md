**Why a schedule is needed.** With V = 1,193 tokens, the noise in x_t has a maximum near
α·√(2 ln V) ≈ 3.8 across the vocabulary, while the target's spike has height β = t. The field's
argmax therefore recovers almost nothing until t ≈ 0.6 and almost everything by t ≈ 0.85
(figure `field_recovery`). Drawing t uniformly spends about 60% of every batch where the state
carries no information and the loss teaches nothing about the path. Removing the schedule costs
~20 WER points (93.1 vs 72.8, table `phase2`). It is the largest single design decision of the
project.

**How the schedule is built.** Let P_e(t) be the per-token error of the field at time t. Define
decoding progress

  τ(t) = 1 − P_e(t) / P_e(0),

which rises from 0 to 1 as the transcript becomes recoverable. Training draws u ~ U[0, 1] and
sets t = τ⁻¹(u). The batch is then spread evenly over *progress* rather than over time. The
density of t is τ′(t), so it concentrates where tokens are actually being decided. The sampler
walks the same map (t_n = τ⁻¹(n/N)), and the denoiser is fed u rather than t (phase 4).

**Where P_e comes from: two constructions.**

- *Simulated.* P_e(t) is the field's own error. The field is a spike plus i.i.d. Gaussians, so
  recovery has a closed form: the probability that β + α·z₀ beats the maximum of V−1 draws of
  α·z. With μ in the path it also depends on μ's accuracy, which training measures each epoch.
  The curve is integrated exactly (Gauss–Hermite) and inverted. It costs milliseconds and has
  no sampling noise. μ's accuracy starts from an honest prior of 0, not an optimistic 1, which
  the phase-3 ablation favours. The curve is re-derived every 5 epochs, so the schedule follows
  μ as it learns.
- *Measured.* P_e(t) is the *denoiser's* error, read by sweeping t uniformly over 2,048 fresh
  training utterances and fitting a monotone (centred isotonic) curve. This follows what the
  model actually finds hard, not what the field alone carries. It is refreshed every 10 epochs:
  refreshing more often moves the schedule faster than the model can follow, and costs up to
  5 points.

**What was decided.** On the 4k-utterance task the two constructions tie: 72.8 vs 72.8 over two
seeds. Simulated was kept for three reasons. It is free, it is deterministic, and it does not
couple the draw to a noisy estimate of the model. Reshaping the draw over τ (a Beta(k, 1) on u)
gave nothing at k = 1.5 and lost 18 points at k = 2.5, so the draw stays uniform in τ. Phase 5
returns to this question with the full recipe and a direct measurement of where the remaining
error is.
