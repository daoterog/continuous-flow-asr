"""Does the per-utterance table curve (src/flow/tau_table.py) match what the field really does?

For a saved run, on real utterances, the per-utterance time curve three ways:

  measured   Monte Carlo: x_t built with each position's sigma and the run's own mu, on a
             grid of t, argmax(x_t) scored against the true tokens. The ground truth.
  table      `FlowMatching.utterance_curves`: each position's preset row, at mu's
             confidence, averaged over mu's predicted real positions. What training and
             sampling use; no targets in it.
  scalar     one simulated curve for everyone, at the batch's mean sigma and mu's measured
             accuracy: what `SimulatedTau` builds.

Twice: with each position's sigma as the run chose it (a fixed run: all one value), and with
each position on a random preset of `--scales`, which is where a learned source puts the
table to the test. Reported as the gap to `measured` in the draw's quantiles — the t
training actually lands on.

    python -m scripts.schedule.validate_tau_table --components runs/<id>/components
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

import train as T
from scripts.evaluation.sample_eval import _config_from_components
from scripts.evaluation.sample_trajectory import load_model
from src.flow.recovery import recovery_curve
from src.flow.source import DEFAULT_SCALES
from src.flow.tau_table import TauTable

QS = (0.1, 0.5, 0.9)


def measured_recovery(model, sigma, mu, tokens, real, ts, draws, gen):  # noqa: ANN001, ANN201
    """(B, T) Monte Carlo field recovery against the true tokens."""
    x1 = F.one_hot(tokens, model.vocab_size).float()
    rows = []
    for t in ts.tolist():
        hit = torch.zeros(tokens.shape, device=tokens.device)
        for _ in range(draws):
            x0 = torch.randn(x1.shape, device=tokens.device, generator=gen)
            x_t = (model.alpha_fn(t) * sigma) * x0 + model.beta_fn(t) * x1 + model.gamma_fn(t) * mu
            hit += (x_t.argmax(-1) == tokens).float()
        rows.append(((hit / draws) * real).sum(1) / real.sum(1))
    return torch.stack(rows, dim=1)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the field's recovery and compare the table's and the scalar curve against it."""
    flags, overrides = T.split_overrides(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--variant", default="best", choices=["best", "final"])
    p.add_argument("--n-utts", type=int, default=64)
    p.add_argument("--draws", type=int, default=4, help="noise draws per utterance and t")
    p.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    p.add_argument("--floor", type=float, default=0.01)
    args = p.parse_args(flags)

    ns = argparse.Namespace(components=args.components, config=None, overrides=overrides)
    cfg = _config_from_components(ns)
    cfg.data.subset_n.val = args.n_utts
    cfg.data.batch_size = cfg.data.eval_batch_size = 16
    cfg.schedule.per_utterance = True
    cfg.schedule.uniform_floor = args.floor
    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    model = load_model(args.components, args.variant, cfg, dm).cuda().eval()
    mixed = TauTable(dm.vocab_size, model.alpha_fn, model.beta_fn, model.gamma_fn,
                     args.scales).to("cuda")
    ts = mixed.ts
    scales = torch.tensor(args.scales, device="cuda")

    res = {"own": {"measured": [], "table": [], "scalar": []},
           "mixed": {"measured": [], "table": [], "scalar": []}}
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for batch in dm.val_dataloader():
            audio, tokens = batch["audio"].cuda().float(), batch["tokens"].cuda()
            real = tokens != dm.pad_id
            mu_logits, noise_logits = model.mu_and_noise(audio)
            mu = F.softmax(mu_logits.float(), dim=-1)
            acc = float(((mu.argmax(-1) == tokens) & real).sum() / real.sum())
            sigma, preset = model.choose_sigma(noise_logits, tokens.shape, training=False)
            pick = torch.randint(len(args.scales), tokens.shape, device="cuda", generator=gen)
            cases = {
                "own": (sigma, model.utterance_curves(mu, preset), model.tau_table),
                "mixed": (scales[pick][..., None], None, mixed),
            }
            for name, (sig, u_rows, table) in cases.items():
                meas = measured_recovery(model, sig, mu, tokens, real, ts, args.draws, gen)
                if u_rows is None:
                    conf, top = mu.max(-1)
                    mreal = (top != dm.pad_id).cumprod(1).bool()
                    mreal[:, 0] = True
                    u_rows = table.curves(table.recovery(pick, conf, mreal), args.floor)
                s_mean = float(sig.float().mean()) if torch.is_tensor(sig) else float(sig)
                scal = recovery_curve(ts.cpu().numpy(), dm.vocab_size,
                                      lambda t, s=s_mean: s * model.alpha_fn(t),
                                      model.beta_fn, model.gamma_fn, acc)
                scal = torch.tensor(scal, dtype=torch.float32, device="cuda").expand_as(meas)
                res[name]["measured"].append(table.curves(meas, args.floor))
                res[name]["table"].append(u_rows)
                res[name]["scalar"].append(table.curves(scal, args.floor))

    want = torch.tensor(QS, device="cuda")
    for name, curves in res.items():
        q = {k: mixed.invert(torch.cat(v), want.expand(torch.cat(v).size(0), -1))
             for k, v in curves.items()}
        print(f"\n== {name} sigma: draw quantiles q10/q50/q90, mean over utterances, and "
              f"mean |error| against measured")
        for k, v in q.items():
            err = "" if k == "measured" else "   |err| " + " ".join(
                f"{float(e):.3f}" for e in (v - q["measured"]).abs().mean(0))
            print(f"  {k:>9}  " + "/".join(f"{float(m):.3f}" for m in v.mean(0)) + err)
        print(f"  measured spread across utterances, std of q50: "
              f"{float(q['measured'][:, 1].std()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
