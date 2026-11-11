"""An exponential moving average of the weights, kept out of the model.

`EMAWeights` is a mixin for a LightningModule, like `src.reporting.MetricReporting`: the
module calls `init_ema(decay)` once and gets the rest through Lightning's hooks. It averages
the weights of every model the module's `components` names, swaps the average in for
validation and testing, and hands `save_components` the averaged weights to save.

FLM evaluates and samples with an EMA of its weights rather than the live ones. Averaging
smooths the last stretch of noisy optimizer steps, which is what a sampled transcript, built
from many denoiser calls, is most sensitive to.
"""

import torch
from torch import nn

from src.checkpointing.state_dict import _unwrap


class EMAWeights:
    """Shadow weights updated after every optimizer step, swapped in to evaluate.

    The shadows are float32, one per parameter, built on the first training step. While
    validation or testing runs they sit in the model and the live weights wait in the
    shadows' place; the swap is its own inverse, so `_ema_in_place` is the only state that
    says which way round they are.
    """

    components: tuple[str, ...]

    def init_ema(self, decay: float | None) -> None:
        """Keep an EMA at `decay`, or none at all when it is None. Call from `__init__`."""
        if decay is not None and not 0.0 < float(decay) < 1.0:
            raise ValueError(f"train.ema_decay must be in (0, 1) or null, got {decay}")
        self.ema_decay = None if decay is None else float(decay)
        self._ema: list[torch.Tensor] | None = None
        self._ema_steps = 0
        self._ema_in_place = False

    def _ema_params(self) -> list[nn.Parameter]:
        """Every tracked model's parameters, in a fixed order.

        Read off the unwrapped modules: a compiled module shares its parameters with the one
        it wraps, so this is the same set either way.
        """
        models = [getattr(self, name, None) for name in self.components]
        return [p for m in models if m is not None for p in _unwrap(m).parameters()]

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Fold the weights the optimizer just stepped into the average.

        The decay ramps up as (1 + n) / (10 + n) until it reaches `ema_decay`, so the average
        is not anchored to the random initialisation for its first thousand steps.
        """
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.ema_decay is None:
            return
        live = [p.detach().float() for p in self._ema_params()]
        if self._ema is None:
            self._ema = [p.clone() for p in live]
        self._ema_steps += 1
        decay = min(self.ema_decay, (1 + self._ema_steps) / (10 + self._ema_steps))
        torch._foreach_lerp_(self._ema, live, 1.0 - decay)

    def _swap_ema(self) -> None:
        """Exchange the live weights and the shadows in place; applying it twice undoes it."""
        with torch.no_grad():
            for param, shadow in zip(self._ema_params(), self._ema, strict=True):
                live = param.detach().clone()
                param.copy_(shadow.to(param.dtype))
                shadow.copy_(live.float())
        self._ema_in_place = not self._ema_in_place

    def _ema_in(self) -> None:
        if self._ema is not None and not self._ema_in_place:
            self._swap_ema()

    def _ema_out(self) -> None:
        if self._ema_in_place:
            self._swap_ema()

    # Validation and test are bracketed by these, epoch-end hooks included, so everything
    # they compute — metrics and the sampling probe alike — sees the averaged weights.
    def on_validation_start(self) -> None:
        """Swap the average in before the validation loop."""
        super().on_validation_start()
        self._ema_in()

    def on_validation_end(self) -> None:
        """Put the live weights back once the validation loop is done."""
        super().on_validation_end()
        self._ema_out()

    def on_test_start(self) -> None:
        """Swap the average in before the test loop."""
        super().on_test_start()
        self._ema_in()

    def on_test_end(self) -> None:
        """Put the live weights back after the test loop."""
        super().on_test_end()
        self._ema_out()

    def component_state_dicts(self, name: str) -> tuple[dict, dict | None]:
        """(weights to save as `name`, the live weights when they differ, else None).

        With an EMA the saved component is the averaged model — the one validation and
        sampling measured — and the live weights go alongside, for continuing training.
        Correct whichever way round the weights are at the moment of saving.
        """
        model = _unwrap(getattr(self, name))
        current = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if self._ema is None:
            return current, None
        index = {id(p): i for i, p in enumerate(self._ema_params())}
        other = dict(current)
        for key, param in model.named_parameters():
            other[key] = self._ema[index[id(param)]].to(param.dtype).cpu().clone()
        # `current` is whatever sits in the model now; `other` is what the shadows hold.
        return (current, other) if self._ema_in_place else (other, current)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Keep the EMA in a Lightning checkpoint, so a resumed run continues the average."""
        super().on_save_checkpoint(checkpoint)
        if self._ema is not None:
            checkpoint["ema"] = {
                "shadows": [t.cpu() for t in self._ema],
                "steps": self._ema_steps,
                "in_place": self._ema_in_place,
            }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore the EMA a checkpoint carried; the shadows move to the model's device."""
        super().on_load_checkpoint(checkpoint)
        ema = checkpoint.get("ema")
        if ema is not None:
            device = self._ema_params()[0].device
            self._ema = [t.to(device) for t in ema["shadows"]]
            self._ema_steps = int(ema["steps"])
            self._ema_in_place = bool(ema["in_place"])
