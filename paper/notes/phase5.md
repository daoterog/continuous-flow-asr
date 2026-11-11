Round 1 tests each way of moving training toward low t alone, against the phase-4 recipe
(table `phase5`, curves in `phase5_curves`).

**A uniform floor makes the model worse, by 5–6 points.** With 30–40% of every batch drawn
uniformly over [0, 1], sampled WER goes from 15.0 to 20.6 (floor 0.3) and 21.0 (floor 0.4).
The floor did what it was designed to do: the denoiser's audio-only recovery at t = 0 rose
slightly (0.54 → 0.56–0.57). The cost is elsewhere. Each batch now holds about a third fewer
samples in the transition, t ≈ 0.6–0.85, and that is where a sample's tokens are decided. The
projection behind this arm assumed that training mass should follow the remaining error. The
screen says otherwise: what limits sampled WER is how well the model resolves the transition,
not how well it reads the audio with nothing else to go on. The floor was dropped. In phase 7
even a 0.1 floor on top of the best arm cost 1.6 points.

**A lower noise scale helps, but only once it is low enough.** Scaling the noise to σ(1 − t)
gives x₀ ~ N(μ, σ²I): sampling starts from a noisy copy of μ's guess. The result is not
monotone in σ:

- σ 0.7: 17.3 (+2.3);
- σ 0.5: 17.7 (+2.6);
- σ 0.4: **12.7 (−2.3)**.

Phase 7 confirms σ 0.4 on two seeds. The gain is mostly from training, not from the sampling
grid: σ 0.4 sampled on the control's grid still scores 13.3 against the control's 15.1.

**The noise scale changes what the model relies on.** Removing μ at t = 0 shows how much the
answer depends on it. The control still recovers 0.45 of tokens without μ. At σ 0.5 that falls
to 0.12, and at σ 0.4 to 0.05. One reading of the non-monotone result follows from this. Below
σ ≈ 0.5, μ shows through the noise, and the denoiser learns to start from μ's draft rather than
from the audio alone. That only pays off once μ is readable enough to be a useful draft (σ 0.4);
in between, the denoiser leans on a signal it cannot read reliably. This is single-seed for
σ 0.7 and 0.5, and the explanation is a hypothesis, not a measurement.

Note that each arm's validation loss is computed under its own draw of t, so the loss columns
compare memorisation within a run (train vs val), not across arms.
