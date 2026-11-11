**Every run nearly memorises train.100.** The control's training loss ends at 0.039 against
0.190 on validation, so phase 6 screens regularisers against it (table `phase6`).

**Generic regularisation barely helps.**

- *Weight decay 0.05* (from 0.01) gained 1.2 points, 15.3 → 14.2, and was ahead at every
  sampling epoch. It was the one adopted.
- *Denoiser dropout 0.2* lost 1.6.
- *Masking the Whisper latents* (SpecAugment-style time and channel masks) lost 17 points,
  and 12–14 in combination. The training loss rose above the validation loss (0.29 vs 0.13),
  and μ, which sees the same masked audio, got worse too. The latents are already a learned,
  robust representation, and masking them removed information the task needs. The model
  underfits rather than generalises.

**Taking the audio away hurts badly.** Removing the audio on a share of rows forces those
rows to be solved from the field alone, which is language modelling. The idea was to teach the
late sampling steps coherence. It lost every time:

- 20% of rows: +7.3;
- 40%·t of rows: +14.8;
- 10%·t of rows, on the phase-7 stack: +8.4.

Even a few percent of rows without audio cost more than any other change gained. The sampler
has the audio at every step, and training the denoiser to do without it is training a
different task.

**What did help: training on a wrong draft.** In training, the field is always built from the
true transcript, so it is never wrong. In sampling it always is: from the first step, the field
is built from the model's own imperfect predictions. Resampling 10% of the real tokens in the
field, with the loss still on the truth, teaches the denoiser to correct a draft instead of
copying it:

- tokens drawn uniformly: 14.2 → 12.6;
- tokens drawn from μ's distribution: 14.2 → **11.9**. These are acoustically plausible errors,
  the kind a real draft makes. This version was adopted.

Of all the changes in this phase, only this one addresses the gap between how the model is
trained and how it is used. It is also the largest gain.
