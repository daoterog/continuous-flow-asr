"""How much compute a run spends: training FLOPs per step and in total, sampling per utterance.

The paper's claim is a continuous-flow recogniser trained from scratch on a small budget, so
the budget has to be a number. FLOPs are counted, not estimated: one real training step —
forward and backward — and one full sampling call go through
`torch.utils.flop_counter.FlopCounterMode`, which tallies every matmul and convolution the
dispatcher sees. Every training step has the same shape (batch x canvas x frames), so one
count times the steps taken is the run's total.

Counted on the uncompiled modules: inductor's fused kernels bypass the dispatcher and would
count as nothing.
"""

from contextlib import contextmanager

import lightning.pytorch as pl
import torch
from torch.utils.flop_counter import FlopCounterMode


@contextmanager
def uncompiled(module: pl.LightningModule):  # noqa: ANN201 - a context manager
    """Swap each compiled component for the module it wraps, and back afterwards."""
    swapped = {}
    for name in getattr(module, "components", ()):
        sub = getattr(module, name, None)
        if sub is not None and hasattr(sub, "_orig_mod"):
            swapped[name] = sub
            setattr(module, name, sub._orig_mod)
    try:
        yield
    finally:
        for name, sub in swapped.items():
            setattr(module, name, sub)


def _train_loss(module: pl.LightningModule, batch: dict) -> torch.Tensor:
    """The training objective on `batch`, without logging or touching the metrics."""
    if hasattr(module, "flow_step"):
        return module.flow_step(batch)[0]
    # MuOnly: its whole objective is mu's cross-entropy.
    return module.sequence_ce(module.mu_nn(batch["audio"].float()), batch["tokens"])[0]


def count_flops(module: pl.LightningModule, batch: dict, sample_steps: int) -> dict:
    """{train_per_step, sample_per_utt} for one training step and one sampling call on `batch`.

    The RNG state is saved and restored, and the gradients cleared, so counting leaves the
    run exactly as it found it — the training step that follows draws the same noise it
    would have drawn.
    """
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = module.training
    try:
        with uncompiled(module):
            module.train()
            with FlopCounterMode(display=False) as counter:
                _train_loss(module, batch).backward()
            train = counter.get_total_flops()
            module.zero_grad(set_to_none=True)

            # FlopCounterMode's module tracker hooks autograd on tensors that require grad,
            # and trips on a forward under no_grad whose parameters still do. Sampling needs
            # no gradients, so the parameters stop asking for them while it is counted.
            module.eval()
            params = [p for p in module.parameters() if p.requires_grad]
            for p in params:
                p.requires_grad_(False)
            try:
                with torch.no_grad(), FlopCounterMode(display=False) as counter:
                    module.sample(batch["audio"].float(), sample_steps)
            finally:
                for p in params:
                    p.requires_grad_(True)
            sample = counter.get_total_flops() / batch["tokens"].size(0)
    finally:
        module.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return {"train_per_step": float(train), "sample_per_utt": float(sample)}


class FlopCount(pl.Callback):
    """Count a run's FLOPs on its first batch; log the running training total every epoch.

    `compute/train_flops_per_step` and `compute/sample_flops_per_utt` go into the run's
    summary once; `compute/train_flops`, steps taken times the per-step count, is logged at
    each epoch end and written to the summary when fit ends.
    """

    def __init__(self, sample_steps: int = 16) -> None:
        """`sample_steps` is the integration the sampling cost is quoted at."""
        super().__init__()
        self.sample_steps = sample_steps
        self.per_step: float | None = None

    def on_train_batch_start(
        self, trainer: pl.Trainer, module: pl.LightningModule, batch: dict, batch_idx: int
    ) -> None:
        """Count once, on the run's first batch, before its real step runs."""
        if self.per_step is not None or trainer.global_step != 0:
            return
        flops = count_flops(module, batch, self.sample_steps)
        self.per_step = flops["train_per_step"]
        print(
            f"compute: {self.per_step:.3e} FLOPs per training step, "
            f"{flops['sample_per_utt']:.3e} per utterance sampled at {self.sample_steps} steps"
        )
        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "summary"):
            experiment.summary["compute/train_flops_per_step"] = self.per_step
            experiment.summary["compute/sample_flops_per_utt"] = flops["sample_per_utt"]
            experiment.summary["compute/sample_steps"] = self.sample_steps

    def on_train_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        """Log the training FLOPs spent so far."""
        if self.per_step is not None:
            module.log("compute/train_flops", self.per_step * trainer.global_step)

    def on_fit_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        """Put the run's total into its summary, where a results table can read it."""
        experiment = getattr(trainer.logger, "experiment", None)
        if self.per_step is not None and experiment is not None and hasattr(experiment, "summary"):
            experiment.summary["compute/train_flops"] = self.per_step * trainer.global_step
