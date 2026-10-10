"""Time the measured-tau probe directly: does chunk matter once the probe gets big?

`jobs/sweeps/probe_chunk.sh` asked this through full training runs and could not answer it.
A 2048-utterance probe is ~2 s against an 81 s epoch, so the thing being measured was 2.4%
of the signal and the 2.3% spread between nodes swallowed it. This runs the probe and
nothing else.

The question is time PER UTTERANCE. `chunk` only sets how many loop iterations the sweep
takes — 64 at probe_n 2048, but 1024 at probe_n 32768 — so if there is a fixed cost per
iteration (the host-to-device copy, the Python loop, the kernel launches) it shows up only
once that count is large. Flat ms/utt across chunk means a bigger probe costs exactly what
it forwards and the chunk can be whatever fits.

It drives the real `MeasuredTau._sweep`, not a copy of it, against a stub that supplies
only the surface the sweep touches.

    python -m scripts.diagnostics.bench_probe
    python -m scripts.diagnostics.bench_probe --probe-n 2048 32768 --chunks 32 256 2048
"""

import argparse
import sys
import time
from collections.abc import Sequence

from src.constants import WHISPER_DIM

import torch
from omegaconf import OmegaConf

from src.flow.coefficients import make_schedule
from src.flow.tau_measured import MeasuredTau
from src.models import build_audio_decoder, build_denoiser

WHISPER_FRAMES = 1500


class StubModel:
    """Only what `MeasuredTau._sweep` reads, with the real models underneath."""

    def __init__(self, vocab_size: int, pad_id: int, device: torch.device,
                 denoiser: torch.nn.Module, mu_nn: torch.nn.Module | None,
                 gamma: str | None) -> None:
        """Hold the models and the coefficient callables the interpolant needs."""
        self.device = device
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.denoiser = denoiser
        self.mu_nn = mu_nn
        self.training = False
        self.alpha_fn = make_schedule("linear_down")
        self.beta_fn = make_schedule("linear_up")
        self.gamma_fn = make_schedule(gamma)

    def eval(self) -> None:
        """Match the nn.Module surface `_sweep` toggles."""
        self.training = False

    def train(self, mode: bool = True) -> None:
        """Match the nn.Module surface `_sweep` restores."""
        self.training = mode

    def make_interpolant(self, x1: torch.Tensor, t: torch.Tensor,
                         mu: torch.Tensor | None = None) -> torch.Tensor:
        """I_t = alpha(t) x0 + beta(t) x1 [+ gamma(t) mu], as `flow_matching` builds it."""
        x0 = torch.randn_like(x1)
        out = self.alpha_fn(t)[:, None, None] * x0 + self.beta_fn(t)[:, None, None] * x1
        if self.gamma_fn is not None and mu is not None:
            out = out + self.gamma_fn(t)[:, None, None] * mu
        return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Probe sizes, chunk sizes, and the model geometry to time them against."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--probe-n", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--chunks", type=int, nargs="+", default=[32, 128, 256, 512, 1024, 2048])
    p.add_argument("--vocab-size", type=int, default=1193)
    p.add_argument("--canvas-len", type=int, default=224)
    p.add_argument("--audio-stride", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--gamma", default="linear_down", help="null for the linear variant")
    p.add_argument("--epoch-seconds", type=float, default=81.0,
                   help="measured epoch, to report the probe as a fraction of one")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Sweep probe_n x chunk, reporting ms/utt and peak memory."""
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = None if args.gamma in ("null", "none", "None") else args.gamma

    denoiser = build_denoiser(
        OmegaConf.create({
            "hidden_size": args.hidden_size, "n_blocks": args.n_blocks,
            "n_heads": args.n_heads, "cond_dim": args.hidden_size,
            "dropout": 0.0, "whisper_dim": WHISPER_DIM,
        }),
        vocab_size=args.vocab_size,
    ).to(device).eval()
    mu_nn = None
    if gamma is not None:
        mu_nn = build_audio_decoder(
            OmegaConf.create({"hidden": 256, "n_heads": 4, "n_blocks": 2,
                              "whisper_dim": WHISPER_DIM}),
            vocab_size=args.vocab_size,
            canvas_len=args.canvas_len,
        ).to(device).eval()

    model = StubModel(args.vocab_size, args.vocab_size - 1, device, denoiser, mu_nn, gamma)
    probe = MeasuredTau(OmegaConf.create({"num_bins": 150}), seed=0)

    print(f"device {device}", end="  ")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"{prop.name}  {prop.total_memory / 2**30:.1f} GiB", end="")
    n_par = sum(p.numel() for p in denoiser.parameters())
    n_par += sum(p.numel() for p in mu_nn.parameters()) if mu_nn else 0
    print(f"\nmodels {n_par / 1e6:.1f}M params   gamma={gamma}   "
          f"canvas {args.canvas_len}   vocab {args.vocab_size:,}")
    print(f"\n{'probe_n':>9}{'chunk':>7}{'iters':>7}{'time_s':>9}{'ms/utt':>9}"
          f"{'peak_GiB':>10}{'% epoch':>9}")

    frames = WHISPER_FRAMES // args.audio_stride
    for n in args.probe_n:
        batch = {
            "tokens": torch.randint(0, args.vocab_size - 1, (n, args.canvas_len)),
            "audio": torch.randn(n, frames, WHISPER_DIM, dtype=torch.float16),
        }
        base = None
        for chunk in args.chunks:
            if chunk > n:
                continue
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                probe._sweep(model, {k: v[: min(2 * chunk, n)] for k, v in batch.items()},
                             chunk=chunk)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                probe._sweep(model, batch, chunk=chunk)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                peak = (torch.cuda.max_memory_allocated() / 2**30
                        if device.type == "cuda" else float("nan"))
                ms = dt / n * 1000
                base = base or ms
                flag = "" if abs(ms / base - 1) < 0.05 else f"  {ms / base:.2f}x"
                print(f"{n:>9}{chunk:>7}{-(-n // chunk):>7}{dt:>9.2f}{ms:>9.3f}"
                      f"{peak:>10.2f}{dt / args.epoch_seconds * 100:>8.1f}%{flag}")
            except torch.OutOfMemoryError:
                print(f"{n:>9}{chunk:>7}{-(-n // chunk):>7}{'OOM':>9}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nFlat ms/utt across chunk means the loop count costs nothing: a bigger probe")
    print("costs exactly what it forwards, at whatever chunk fits.")
    print(f"'% epoch' is against --epoch-seconds {args.epoch_seconds:g}, per refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
