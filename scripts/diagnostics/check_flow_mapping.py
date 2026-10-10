"""Does `FlowMapping` build against a real teacher, and does one step run?

Stage 1 of the distillation work is plumbing: the module could not previously be constructed
at all. This checks the four things that have to hold before the objective is worth arguing
about, and deliberately does not check the loss *value* — the objective is still wrong by
design until stage 2 splits the batch and takes the diagonal target from the teacher.

    python -m scripts.diagnostics.check_flow_mapping
    python -m scripts.diagnostics.check_flow_mapping --teacher runs/<id>/components
"""

import argparse
import sys
from collections.abc import Sequence

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

import train as T
from src.flow_mapping import FlowMapping


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Which teacher to build against, and how small to make the data."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/flow_mapping.yaml")
    p.add_argument("--teacher", default=None, help="override distill.teacher")
    p.add_argument("--n-utts", type=int, default=8)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, then assert parity, freezing, the train() trap, and one step."""
    args = parse_args(argv)
    overrides = [
        f"data.subset_n.train={args.n_utts}",
        f"data.subset_n.val={args.n_utts}",
        "data.num_workers=0",
        "data.batch_size=2",
    ]
    if args.teacher:
        overrides.append(f"distill.teacher={args.teacher}")
    cfg, _ = T.build_config(["--config", args.config, *overrides])

    dm = T.build_datamodule(cfg)
    model = FlowMapping(
        cfg,
        vocab_size=dm.vocab_size,
        pad_id=dm.pad_id,
        canvas_len=cfg.data.canvas_len,
        tokenizer=dm.tokenizer,
        vocab_map=dm.vocab_map,
    )
    print(f"teacher   {cfg.distill.teacher}")
    n_student = sum(p.numel() for p in model.two_time_denoiser.parameters())
    n_teacher = sum(p.numel() for p in model.denoiser.parameters())
    print(f"denoiser  teacher {n_teacher:,}   student {n_student:,}")

    # 1. The student starts AS the teacher. One number covers the build, the weight
    #    transfer and the zero-init of the end branch.
    model.eval()
    torch.manual_seed(0)
    batch = dm.collate_fn([dm.datasets["val"][i] for i in range(2)])
    audio = batch["audio"].float()
    x = torch.randn(2, cfg.data.canvas_len, dm.vocab_size)
    s = torch.rand(2)
    with torch.no_grad():
        ref = model.denoiser(x_t=x, time=s, audio_embeddings=audio)
        got = model.two_time_denoiser(
            x_t=x, startpoint=s, endpoint=s, audio_embeddings=audio
        )
    parity = float((ref - got).abs().max())
    print(f"\n1. student(s, s) vs teacher          max|diff| = {parity:.3e}")
    assert parity == 0.0, "the student does not start as the teacher"

    # 2. The teacher is frozen.
    frozen = not any(p.requires_grad for p in model.denoiser.parameters())
    if model.mu_nn is not None:
        frozen &= not any(p.requires_grad for p in model.mu_nn.parameters())
    print(f"2. teacher frozen                     {frozen}")
    assert frozen

    # 3. The dropout trap: Lightning calls .train() on the module, which recurses. This
    #    teacher has dropout 0.1, so a teacher left in train mode gives a target that moves.
    model.train()
    print(f"3. after model.train(), teacher.training = {model.denoiser.training}")
    assert not model.denoiser.training, "train() let the teacher out of eval"
    if model.mu_nn is not None:
        assert not model.mu_nn.training

    # 3b. The split: every row trains exactly one term, in the configured proportions.
    big = 256
    groups = model.assign_groups(big)
    counts = [int((groups == i).sum()) for i in range(3)]
    want = [float(cfg.distill.split[k]) for k in ("diagonal", "boundary", "offdiag")]
    print("\n3b. split over a batch of 256")
    for nm, got, frac in zip(("diagonal", "boundary", "offdiag"), counts, want, strict=True):
        print(f"    {nm:<10} {got:>4}   {got / big:.3f} vs configured {frac:.3f}")
    assert sum(counts) == big, "rows went missing from the split"
    assert all(abs(c / big - f) < 0.01 for c, f in zip(counts, want, strict=True))

    # 3c. The Euler map is only valid because gamma == alpha on this path. Fed the true
    #     endpoint, the mapped state has to BE the interpolant at the midpoint. Same seed
    #     before each build so both draw the same noise.
    torch.manual_seed(0)
    s_, _u, _t, state, pred, _w, _g = model.sample_interval(batch_size=2)
    x1 = F.one_hot(batch["tokens"], dm.vocab_size).float()
    with torch.no_grad():
        mu = F.softmax(model.mu_nn(audio), dim=-1) if model.mu_nn is not None else None
        torch.manual_seed(1)
        x_s = model.make_interpolant(x1=x1, t=s_, mu=mu)
        torch.manual_seed(1)
        x_u = model.make_interpolant(x1=x1, t=_u, mu=mu)
    err = float((state * x_s + pred * x1 - x_u).abs().max())
    print(f"\n3c. Euler map vs true interpolant at u   max|diff| = {err:.3e}")
    assert err < 1e-4, "the Euler map does not reproduce the interpolant on this path"

    # 4. One step, forward and backward.
    loss = model.training_step(batch, 0)
    print(f"\n4. training_step loss                 {float(loss.detach()):.4f}  "
          f"finite={bool(torch.isfinite(loss))}")
    assert torch.isfinite(loss)
    loss.backward()
    student_grads = sum(
        1 for p in model.two_time_denoiser.parameters() if p.grad is not None
    )
    teacher_grads = sum(1 for p in model.denoiser.parameters() if p.grad is not None)
    print(f"   student params with grad           {student_grads}")
    print(f"   teacher params with grad           {teacher_grads}")
    assert student_grads > 0, "nothing reached the student"
    assert teacher_grads == 0, "gradient leaked into the frozen teacher"

    # 4b. Both terms have to be live. Before stage 2 the diagonal called the frozen teacher
    #     on both sides of its own entropy, so it was a constant: requires_grad False and no
    #     gradient at all. This is the check that would have caught it.
    torch.manual_seed(0)
    _l, out = model.map_step(batch)
    # `out` also carries metric-only records with no loss in them — `input` is the argmax
    # of the interpolant, which no objective scores.
    terms = {name: rec for name, rec in out.items() if "loss" in rec}
    for name, rec in terms.items():
        print(f"   {name:<10} loss {float(rec['loss'].detach()):.4f}  "
              f"n={rec['n']:<3} requires_grad={rec['loss'].requires_grad}")
    live = [n for n, r in terms.items() if r["n"] and not r["loss"].requires_grad]
    assert not live, f"non-empty term(s) with no gradient: {live}"

    model.eval()
    with torch.no_grad():
        val = model.validation_step(batch, 0)
    print(f"   validation_step loss               {float(val):.4f}")
    assert torch.isfinite(val)

    # 5. The whole Lightning path, which is also the only way to reach
    #    `configure_optimizers` — `estimated_stepping_batches` needs a live Trainer.
    model.zero_grad(set_to_none=True)
    trainer = pl.Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    opt = trainer.optimizers[0]
    owned = {id(p) for g in opt.param_groups for p in g["params"]}
    student_ids = {id(p) for p in model.two_time_denoiser.parameters()}
    teacher_ids = {id(p) for p in model.denoiser.parameters()}
    print("\n5. fast_dev_run through Lightning     ok")
    print(f"   optimiser == student params        {owned == student_ids}")
    print(f"   optimiser touches teacher          {bool(owned & teacher_ids)}")
    print(f"   teacher still in eval after fit    {not model.denoiser.training}")
    assert owned == student_ids and not (owned & teacher_ids)
    assert not model.denoiser.training

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
