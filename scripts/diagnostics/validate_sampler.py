"""Validate the sampler on the toy problem, where the right answer is checkable.

Sequences are drawn from a random transition matrix that allows only k of V successors
after each token, so a sequence is *valid* or not with no reference needed — and a random
sequence is valid with probability (k/V)^(L-1), which is ~1e-4 at the defaults. Each
sequence also gets a proxy "audio" built from per-token embeddings, so the sample can be
scored against the utterance it was conditioned on.

That gives three independent things to check, none of which a broken integrator can fake:

  1. token accuracy against the conditioning sequence, versus 1/V for chance;
  2. validity under the transition matrix, versus (k/V)^(L-1) for chance;
  3. validity rising with the step count, which is the signature of integration actually
     doing something. Token accuracy need NOT rise with steps: more steps pull the sample
     onto the data manifold, which is what validity measures, while a single step is the
     model's best per-token guess. When the audio is ambiguous the two pull apart — the
     sampler produces *a* valid sequence rather than *the* conditioning one.

It runs the real stack: the DRAX denoiser, the real interpolant, and src.flow.sampler.

    python -m scripts.diagnostics.validate_sampler
    python -m scripts.diagnostics.validate_sampler --variant three_term --epochs 60
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.flow.coefficients import make_schedule  # noqa: E402
from src.flow.sampler import sample_tokens  # noqa: E402
from src.models import build_audio_decoder, build_denoiser  # noqa: E402


def make_transition_matrix(vocab_size: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """(V, V) 0/1 matrix allowing exactly k successors per token."""
    matrix = torch.zeros(vocab_size, vocab_size)
    for row in range(vocab_size):
        picks = torch.randperm(vocab_size, generator=generator)[:k]
        matrix[row, picks] = 1.0
    return matrix


def sample_sequences(
    matrix: torch.Tensor, n: int, length: int, generator: torch.Generator
) -> torch.Tensor:
    """(n, length) sequences that walk the allowed transitions."""
    vocab_size = matrix.size(0)
    probs = matrix / matrix.sum(dim=1, keepdim=True)
    seqs = torch.empty(n, length, dtype=torch.long)
    seqs[:, 0] = torch.randint(0, vocab_size, (n,), generator=generator)
    for i in range(1, length):
        seqs[:, i] = torch.multinomial(probs[seqs[:, i - 1]], 1, generator=generator).squeeze(1)
    return seqs


def validity(seqs: torch.Tensor, matrix: torch.Tensor) -> float:
    """Fraction of sequences whose every adjacent pair is an allowed transition."""
    ok = matrix[seqs[:, :-1].reshape(-1), seqs[:, 1:].reshape(-1)].view(seqs.size(0), -1)
    return ok.all(dim=1).float().mean().item()


def make_embedding_table(
    vocab_size: int, dim: int, confusable_pairs: int, noise: float, generator: torch.Generator
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Per-token embeddings, with some pairs placed almost on top of each other.

    This is the toy's whole point. Tokens in a confusable pair cannot be told apart from
    audio alone, so a single denoising step can only guess between them; resolving them
    needs the sequence prior, which is what iterative refinement supplies. Without any
    confusable pairs the audio determines the transcript outright and one step is enough.
    """
    table = torch.randn(vocab_size, dim, generator=generator) * 2.0
    pairs = []
    for p in range(confusable_pairs):
        a, b = 2 * p, 2 * p + 1
        if b >= vocab_size:
            break
        table[b] = table[a] + noise * torch.randn(dim, generator=generator)
        pairs.append((a, b))
    return table, pairs


def positional_table(length: int, dim: int) -> torch.Tensor:
    """(length, dim) sinusoidal codes added to the proxy audio.

    Without this the proxy carries no order: cross-attention would see which tokens are
    present but not where, because RoPE is applied only inside self-attention. Real
    Whisper latents carry position implicitly, so a proxy without it is the easier
    problem misrepresented as the harder one.
    """
    pos = torch.arange(length).unsqueeze(1)
    i = torch.arange(dim).unsqueeze(0)
    angle = pos / torch.pow(10000, (2 * (i // 2)) / dim)
    code = torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle))
    return code.float()


def make_audio(seqs: torch.Tensor, table: torch.Tensor, sigma: float,
               generator: torch.Generator, positional: bool = True) -> torch.Tensor:
    """Proxy audio: a noisy per-token embedding, one frame per position."""
    clean = table[seqs]
    if positional:
        clean = clean + positional_table(seqs.size(1), table.size(1))
    return clean + sigma * torch.randn(clean.shape, generator=generator)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Toy size, training length and which interpolant variant to validate."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--variant", choices=["linear", "audio_cond", "three_term"], default="linear")
    p.add_argument("--vocab-size", type=int, default=15)
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--transitions", type=int, default=4, help="allowed successors per token")
    p.add_argument("--audio-dim", type=int, default=20)
    p.add_argument("--audio-sigma", type=float, default=0.3)
    p.add_argument("--confusable-pairs", type=int, default=3,
                   help="token pairs that sound alike; only the sequence prior separates them")
    p.add_argument("--confusion-noise", type=float, default=0.05)
    p.add_argument("--n-train", type=int, default=8192)
    p.add_argument("--n-eval", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--hard", action="store_true", help="feed back one-hot instead of softmax")
    p.add_argument("--no-audio-pos", action="store_true",
                   help="drop the positional code from the proxy audio")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Train a toy model on the real stack, then sample and score."""
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    matrix = make_transition_matrix(args.vocab_size, args.transitions, gen)
    train_seqs = sample_sequences(matrix, args.n_train, args.length, gen)
    eval_seqs = sample_sequences(matrix, args.n_eval, args.length, gen)
    table, pairs = make_embedding_table(
        args.vocab_size, args.audio_dim, args.confusable_pairs, args.confusion_noise, gen
    )

    chance_valid = (args.transitions / args.vocab_size) ** (args.length - 1)
    print(f"toy: V={args.vocab_size} L={args.length} k={args.transitions} "
          f"variant={args.variant} device={device}")
    print(f"  training data validity   {validity(train_seqs, matrix):6.3f}  (should be 1.000)")
    print(f"  random-sequence validity {chance_valid:6.4f}  chance token acc "
          f"{1/args.vocab_size:.4f}")
    confusable = sum((eval_seqs == a).sum().item() + (eval_seqs == b).sum().item()
                     for a, b in pairs) / eval_seqs.numel()
    print(f"  confusable pairs {pairs}  covering {confusable:.1%} of eval tokens\n")

    den_cfg = dict(hidden_size=128, cond_dim=64, n_heads=4, n_blocks=3, dropout=0.0,
                   whisper_dim=args.audio_dim)
    denoiser = build_denoiser(den_cfg, args.vocab_size).to(device)
    mu_nn = None
    gamma_fn = None
    if args.variant != "linear":
        mu_nn = build_audio_decoder(
            dict(hidden=128, n_heads=4, n_blocks=2, audio_dim=args.audio_dim),
            args.vocab_size, args.length,
        ).to(device)
        gamma_fn = make_schedule(
            "linear_down" if args.variant == "audio_cond" else {"type": "parabola", "c": 0.5}
        )
    alpha_fn, beta_fn = make_schedule("linear_down"), make_schedule("linear_up")

    params = list(denoiser.parameters()) + (list(mu_nn.parameters()) if mu_nn else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps_total = args.epochs * max(1, args.n_train // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_total)

    for epoch in range(args.epochs):
        perm = torch.randperm(args.n_train, generator=gen)
        total, seen = 0.0, 0
        for start in range(0, args.n_train - args.batch_size + 1, args.batch_size):
            idx = perm[start : start + args.batch_size]
            tokens = train_seqs[idx].to(device)
            audio = make_audio(
                train_seqs[idx], table, args.audio_sigma, gen, not args.no_audio_pos
            ).to(device)
            x1 = F.one_hot(tokens, args.vocab_size).float()
            t = torch.rand(tokens.size(0), device=device)

            mu = F.softmax(mu_nn(audio), dim=-1) if mu_nn is not None else None
            x_t = alpha_fn(t)[:, None, None] * torch.randn_like(x1) + beta_fn(t)[:, None, None] * x1
            if gamma_fn is not None:
                x_t = x_t + gamma_fn(t)[:, None, None] * mu.detach()

            logits = denoiser(x_t=x_t, time=t, audio_embeddings=audio)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), tokens.reshape(-1))
            if mu_nn is not None:
                loss = loss + F.cross_entropy(
                    mu_nn(audio).reshape(-1, args.vocab_size), tokens.reshape(-1)
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * tokens.size(0)
            seen += tokens.size(0)
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"  epoch {epoch+1:3d}/{args.epochs}  loss {total/seen:.4f}")

    denoiser.eval()
    if mu_nn is not None:
        mu_nn.eval()
    tokens = eval_seqs.to(device)
    audio = make_audio(eval_seqs, table, args.audio_sigma, gen, not args.no_audio_pos).to(device)
    with torch.no_grad():
        mu = F.softmax(mu_nn(audio), dim=-1) if mu_nn is not None else None

    print(f"\n{'steps':>6} {'token acc':>10} {'validity':>9}   (chance: "
          f"{1/args.vocab_size:.3f} / {chance_valid:.4f})")
    for n_steps in args.steps:
        grid = torch.linspace(0, 1, n_steps + 1).tolist()
        pred = sample_tokens(
            denoiser, audio, grid, args.vocab_size, args.length,
            alpha_fn, beta_fn, gamma_fn, mu, hard=args.hard,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
        acc = (pred == tokens).float().mean().item()
        print(f"{n_steps:>6} {acc:10.4f} {validity(pred.cpu(), matrix):9.4f}")

    print("\nA working integrator: both far above chance, and validity rising with the "
          "step count.\nAccuracy falling as validity rises is expected when the audio is "
          "ambiguous — more steps\nbuy a sample that is on-manifold, not one that is "
          "closer to this particular reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
