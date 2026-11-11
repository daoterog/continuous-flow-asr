This phase compares our training loop against a published flow-matching language model (FLM).
Where FLM trains differently from us, the difference was tested as one change against a shared
control, in stages, and every adopted change becomes the next stage's control.

- **Stage 1.** Two changes each gained about 3 points. The first feeds the denoiser the uniform
  draw u instead of t: near t ≈ 0.8, where the schedule concentrates, t barely moves and a time
  embedding cannot tell those states apart, while u spans [0, 1]. The second down-weights the
  padding positions of the canvas to 0.1 under a per-token loss.
- **Stage 2.** A logit soft-cap (30·tanh(x/30)) was neutral and kept, because it is free and
  bounds the logits. Halving or doubling the learning rate both lost, so 2e-3 stands. Restoring
  the stop token to full weight lost 3 points.
- **Stage 3.** An exponential moving average of the weights (decay 0.999) was ahead at every
  sampling epoch from 19 onwards, and 1.3 points ahead at epoch 99.

Together the recipe went from 17.99 to 14.77 sampled WER at 100 epochs. One seed per arm; from
earlier replicates, seed noise at this scale is 1–2 points, so the 0.2 and 1.3 gains are
suggestive and the rest are clear. A 100-epoch screen ranks arms as a 300-epoch run does (gap
correlation 0.97), which is what made screening at a third of the cost sound.
