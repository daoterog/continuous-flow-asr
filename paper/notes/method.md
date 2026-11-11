The model transcribes by flow matching on the probability simplex over the vocabulary. A
transcript is a canvas of 224 one-hot vectors x₁. Training builds an interpolant between Gaussian
noise x₀ and the target,

  x_t = α(t)·x₀ + β(t)·x₁ + γ(t)·μ(a),   α = 1−t, β = t, γ = 1−t,

and a Transformer denoiser, cross-attending to frozen Whisper encoder latents of the audio a,
predicts x₁ from x_t. μ is a small audio decoder that guesses the transcript directly from the
latents. Its softmax sits in the path, so even at t = 0 the state carries a hint of the answer.
Sampling integrates the implied velocity (D(x_t) − x_t)/(1−t) from noise to t = 1 in 16 steps.

Two choices separate the tuning phase from the final model:

- **Vocabulary.** Tuning uses a compact vocabulary: the 1,193 Whisper tokens that occur in
  LibriSpeech train.100. That keeps each experiment cheap, since the field is (B × 224 × V).
  The final model uses Whisper's full 51,866-token vocabulary, so it can be scored on other
  corpora.
- **Data.** Tuning trains on LibriSpeech train.100 (100 h). The final model trains on
  train.960.

Every number before the final model is therefore evidence for a design decision, not a
competitive WER. All WERs are **sampled**: integrated from noise with no reference anywhere,
which is the number inference produces.
