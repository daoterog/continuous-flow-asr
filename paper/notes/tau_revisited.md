With the recipe settled, we asked directly where the model still makes mistakes. A probe of the
stage-3 model fixes t, rebuilds x_t, and measures token recovery with each input taken apart:
the full denoiser; the denoiser with μ replaced by a uniform vector; the field's own argmax; and
μ alone (figure `audio_info`).

**What it shows.**

- **The remaining error is at low and mid t.** Validation recovery rises smoothly from 0.51 at
  t = 0 to 0.81 at t = 0.5 and 0.96 at t = 0.7. The simulated schedule, which follows the
  *field*, puts 66% of each batch in t ∈ [0.7, 0.85], where the model is already 96–98% right,
  and only 5% below t = 0.5, where 19–49% of tokens are still wrong.
- **There is no audio-only plateau.** Recovery climbs from t = 0, so the model gains from being
  trained there.
- **μ is load-bearing.** At t = 0 removing μ costs 11 points on validation and 19 on training.
  This is despite μ being unreadable by argmax under the noise (0.5%). The denoiser reads μ
  through the noise far better than an argmax does.
- **Overfitting sits mid-path.** The train–val gap is 0 at t = 0, 8–9 points at t ≈ 0.4–0.5,
  and 2 points by t = 0.8. More data (train.960) should relieve exactly this region.

**What follows: two ways to move training mass earlier.** Both are previewed in figure
`tau_strategies`, by replaying a finished run's logs through each schedule before training it.

- *A uniform floor.* Mix the schedule's CDF with the identity, u′ = (1−λ)·τ(t) + λ·t. A
  fraction λ of each batch is uniform in t, and the peak keeps its shape.
- *Lower noise.* Scale the noise, α = σ(1−t). Algebraically the path becomes
  x₀′ ~ N(μ, σ²I), so sampling starts *from μ's guess*, and the drift is unchanged. The data and
  μ become readable earlier, so the schedule's own peak moves left and keeps moving as μ
  improves.

Round 1 screens each knob alone (floor 0.3, 0.4; σ 0.7, 0.5, 0.4). Round 2 combines the best σ
with a small floor, so that high t is not left untrained.
