"""Score a trained model the way inference works: integrate from noise, then compute WER.

This is the number comparable to the leaderboard. The binned metrics logged during
training score a single denoising step from an x_t built out of the target, so at high t
they largely measure the input — the corruption sweep recovers ~75% of tokens at t=0.8
with no model at all.

The default is a sweep over step counts, because the toy validation showed the two things
one might want do not move together: more steps pull the sample onto the data manifold
(fluent, well-formed) while a single step is the model's best per-token guess. Which one
wins on WER is an empirical question per checkpoint, so measure it rather than assume.

    python -m scripts.sample_eval --ckpt runs/.../epoch=49.ckpt
    python -m scripts.sample_eval --ckpt ... --role test --steps 1 4 16 64
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sampler import score_samples  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Checkpoint, which split to score, and how many steps to try."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", type=Path, required=True, help="Lightning checkpoint to score")
    p.add_argument("--role", choices=["train", "val", "test"], default="val")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--n-utts", type=int, default=None, help="cap the split (default: all)")
    p.add_argument("--batch-size", type=int, default=None, help="default: the run's own")
    p.add_argument("--hard", action="store_true", help="feed back one-hot instead of softmax")
    p.add_argument("--show", type=int, default=3, help="example transcripts to print")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild the model from its checkpoint, sample the split, and report WER."""
    import train as T
    from src.metrics import ids_to_text
    from src.pl_module import FlowMatching

    args = parse_args(argv)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = checkpoint["hyper_parameters"]["cfg"]
    if args.n_utts is not None:
        cfg.data.subset_n[args.role] = args.n_utts
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    T.setup_reproducibility(cfg)
    dm = T.build_datamodule(cfg)
    model = FlowMatching.load_from_checkpoint(
        args.ckpt,
        map_location="cpu",
        # The saved hyperparameters include an OmegaConf DictConfig, which torch's
        # weights_only unpickler refuses. These are checkpoints this repo wrote itself.
        weights_only=False,
        tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map,
        sample_t_fn=T.build_sample_t_fn(dm, cfg),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    loader = {
        "train": dm.train_dataloader,
        "val": dm.val_dataloader,
        "test": dm.test_dataloader,
    }[args.role]()
    batches = [
        {"audio": b["audio"].to(device).float(), "tokens": b["tokens"].to(device)}
        for b in loader
    ]
    n_utts = sum(b["tokens"].size(0) for b in batches)
    print(f"{args.ckpt}\n  variant {cfg.get('variant')}  role {args.role}  "
          f"{n_utts:,} utterances  vocab {dm.vocab_size:,}\n")

    print(f"{'steps':>6} {'WER %':>9} {'token err':>10}")
    best = None
    for num_steps in args.steps:
        preds, refs = [], []
        for batch in batches:
            preds.append(model.sample(batch["audio"], num_steps, hard=args.hard))
            refs.append(batch["tokens"])
        pred, ref = torch.cat(preds), torch.cat(refs)
        wer, token_error = score_samples(
            pred, ref, dm.tokenizer, dm.pad_id, dm.vocab_map
        )
        print(f"{num_steps:>6} {wer:9.2f} {token_error:10.4f}")
        if best is None or wer < best[1]:
            best = (num_steps, wer, pred, ref)

    steps, wer, pred, ref = best
    print(f"\nbest: {steps} steps, WER {wer:.2f}%")
    if args.show:
        hyps = ids_to_text(pred[: args.show], dm.tokenizer, dm.pad_id, dm.vocab_map)
        gold = ids_to_text(ref[: args.show], dm.tokenizer, dm.pad_id, dm.vocab_map)
        print()
        for i, (g, h) in enumerate(zip(gold, hyps, strict=True)):
            print(f"  [{i}] ref  {g[:100]}")
            print(f"      hyp  {h[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
