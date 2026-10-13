Phase 7 re-tests the winners on the stack with weight decay 0.05 and combines them (table
`phase7`, curves in `phase7_curves`). The control is now run on two seeds, 13.8 and 14.7,
which puts seed noise at about one point.

**The noise scale holds.** σ 0.4 scores 11.7 and 12.5 on two seeds, against the control's 13.8
and 14.7. That is −2.1 on average, and it was ahead at every sampling epoch. σ 0.3 ties
(12.1, one seed). Adding a 0.1 uniform floor to σ 0.4 loses 1.6, which settles the floor.

**The draft corruption stacks on it.** σ 0.4 with 10% of the field resampled from μ scores
10.6, against 11.7 for σ 0.4 alone on the same seed (10.1 on the control's grid). This is the
tuning-phase recipe: the phase-4 recipe + σ 0.4 + weight decay 0.05 + μ-draft corruption.
Over the whole tuning phase, at 100 epochs, sampled WER went from 18.0 to 10.6.

**The model now starts from μ's draft, by design.** Figure `mu_reliance` gives the denoiser,
at t = 0, its own μ, another utterance's μ, or none.

- *At unit noise*, the three readings barely differ (0.56 / 0.51 / 0.51). The model reads the
  audio and treats μ as a hint.
- *At σ 0.4*, removing μ collapses recovery to 0.12. Given another utterance's μ, the model
  copies that μ's tokens 23% of the time, against 4% for the control.

The low-noise path makes μ the starting draft and the denoiser its corrector, which is what
x₀ ~ N(μ, σ²I) intends. It also means the final model's quality is now tied to μ's: μ still
gets 58% of tokens wrong, so a better μ is a direct lever for the final run.

**An open lead: measured τ.** Re-reading the schedule from the denoiser's own training error
every 10 epochs scores 14.5 at unit noise and 11.0 at σ 0.4 on its own sampling grid. Sampled
on the control's grid instead, it scores **12.1 and 9.2**, the best number in the project. Two
things keep it out of the recipe:

- it is one seed;
- the measured curve read off *training* data collapses. The model nearly memorises
  train.100, so its training error is near zero almost everywhere. The draw then piles up near
  t = 0 (median t 0.02 by epoch 99), and the run's own sampling grid follows it there. This is
  also why its validation loss looks large: it is evaluated under that collapsed draw.

What these runs show is that a model trained mostly on low t can still sample very well on the
standard grid. The training draw and the sampling grid do not have to be the same curve. The
next experiment reads the measured curve off validation data, which the model has not
memorised.
