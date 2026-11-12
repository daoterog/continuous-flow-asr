"""Sampling from noise as a measurement, on a fixed slice of a split (val unless told)."""

from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from torch.utils.data import Subset

from src.data.collate import make_loader
from src.metrics import score_samples

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.flow_matching import FlowMatching


class SamplingProbe:
    """Integrates a model from noise on a fixed val slice and scores what comes out.

    Kept apart from the binned metrics: those read tensors the training step has already
    produced, so they need nothing but the batch, while this runs the whole sampler and
    needs the model's inference path, its device and its eval mode. `MetricReporting`
    drives it and logs what comes back.

    The slice is collated once and reused for the life of the probe, so a change in the
    number is the model moving rather than a different draw of utterances.

    `score` expects the model to carry `sample`, `device`, `tokenizer`, `pad_id` and
    `vocab_map` — which is `FlowMatching`'s own surface, read off the model so the
    vocabulary used to decode is always the one the weights were trained over.
    """

    def __init__(self, cfg: DictConfig | None, train_batch_size: int) -> None:
        """`cfg` is the `sampling` config node; the batch size is the chunking fallback."""
        self.cfg = cfg
        self.train_batch_size = train_batch_size
        self._indices: dict[str, list[int]] = {}

    @property
    def enabled(self) -> bool:
        """Whether this run samples at all."""
        return self.cfg is not None and bool(self.cfg.get("enabled", False))

    def is_due(self, epoch: int) -> bool:
        """Whether `epoch` is one of the sampling epochs.

        Sampling integrates the whole canvas several times over, so it runs on a stride
        rather than every validation. `every_n_epochs <= 0` turns it off without having to
        also unset `enabled`.
        """
        if not self.enabled:
            return False
        every = int(self.cfg.get("every_n_epochs", 5))
        return every > 0 and (epoch + 1) % every == 0

    def selection(self, datamodule: "AudioFeatureDataModule", role: str = "val") -> list[int]:
        """The `role` utterances to sample, as indices into the split, chosen once and cached.

        `n_utts` unset means the whole split, which is what makes two runs comparable —
        a WER over a different subset each time is not a number you can put side by side.
        Only the indices are kept: at full latents the whole split is ~10 GB of audio, so
        `score` collates one chunk at a time instead of holding all of it in RAM.
        """
        if role not in self._indices:
            subset = datamodule.datasets[role]
            want = self.cfg.get("n_utts")
            n = len(subset) if want is None else min(int(want), len(subset))
            self._indices[role] = list(range(n))
        return self._indices[role]

    @property
    def methods(self) -> list[str]:
        """The integrators scored each sampling epoch: `log_methods`, or just `method`."""
        listed = self.cfg.get("log_methods")
        if listed:
            return [str(m) for m in listed]
        return [str(self.cfg.get("method", "interpolant"))]

    @property
    def step_counts(self) -> list[int]:
        """The step counts scored each sampling epoch; `num_steps` may be one or a list.

        A list is what makes a flow map legible: it is distilled to answer in one step, and
        a single-valued probe pointed at 16 would only ever measure the regime where an
        integrator is already strong.
        """
        want = self.cfg.get("num_steps", 16)
        counts = [want] if isinstance(want, int | float | str) else list(want)
        return [int(n) for n in counts]

    @property
    def grids(self) -> list[str]:
        """The sampling grids scored each sampling epoch: `grids`, or just the live curve's.

        `reference` walks the control's grid instead (`FlowMatching.reference_reparam`).
        A run whose curve differs from the control's — a uniform floor, a rescaled noise —
        samples on a different grid as well as training on a different draw, and scoring
        it on both is what separates the two.
        """
        return [str(g) for g in (self.cfg.get("grids") or ["live"])]

    @property
    def cfg_strengths(self) -> list[float]:
        """The guidance weights scored each sampling epoch: `cfg_strengths`, else `cfg_strength`.

        1 is no guidance. Several in one list sample the same checkpoint at each, so the
        weight can be picked after training instead of guessed before it.
        """
        listed = self.cfg.get("cfg_strengths")
        if listed:
            return [float(w) for w in listed]
        return [float(self.cfg.get("cfg_strength", 1.0))]

    @staticmethod
    def key(method: str, num_steps: int, grid: str = "live", cfg_strength: float = 1.0) -> str:
        """The section one score is logged under, e.g. `sampled_interpolant_16_cfg2`.

        The live grid and unguided sampling add nothing to the name, so every key from
        before either existed still means what it did.
        """
        key = f"sampled_{method}_{num_steps}"
        if grid != "live":
            key += f"_{grid}"
        if cfg_strength != 1.0:
            key += f"_cfg{cfg_strength:g}"
        return key

    @staticmethod
    def best_key(method: str, num_steps: int, grid: str = "live") -> str:
        """The section the best of several guidance weights is logged under: `..._cfgbest`.

        The lowest WER among the weights scored that epoch, under `/wer`, and the weight
        that reached it under `/cfg`. Picking `_best` on any one weight would choose the
        checkpoint for that weight, when the question is which checkpoint samples best at
        the weight it would be used with.
        """
        return f"{SamplingProbe.key(method, num_steps, grid)}_cfgbest"

    def score(
        self,
        model: "FlowMatching",
        datamodule: "AudioFeatureDataModule",
        method: str | None = None,
        num_steps: int | None = None,
        grid: str = "live",
        role: str = "val",
        cfg_strength: float = 1.0,
    ) -> tuple[float, float, int]:
        """(WER %, token error, utterances scored) for a full integration from noise.

        Chunked because the integrator holds several (B, canvas_len, vocab_size) fields
        alive at once, so the batch that trains is not always one that can be sampled.
        Eval mode is set and restored around the pass: dropout during sampling would make
        the number noisier than the model.

        x0 comes from a generator seeded the same on every call, so two methods scored in
        the same epoch integrate the same noise and differ only by the integrator.
        """
        indices = self.selection(datamodule, role)
        subset = datamodule.datasets[role]
        was_training = model.training
        model.eval()
        try:
            chunk = int(self.cfg.get("batch_size") or self.train_batch_size)
            num_steps = self.step_counts[0] if num_steps is None else int(num_steps)
            hard = bool(self.cfg.get("hard", False))
            generator = torch.Generator(device=model.device)
            generator.manual_seed(int(self.cfg.get("seed", 0)))
            # The slice's chunks, in order, read and collated by the eval loader's workers
            # while the GPU samples the previous one. Collated on the main process, each
            # chunk of full latents left the GPU idle for as long as it took to read.
            loader = make_loader(
                Subset(subset, indices),
                batch_size=chunk,
                shuffle=False,
                collate_fn=datamodule.collate_fn,
                num_workers=datamodule.eval_num_workers,
                pin_memory=datamodule.pin_memory,
            )
            preds, targets = [], []
            for batch in loader:
                preds.append(
                    model.sample(
                        batch["audio"].to(model.device, non_blocking=True).float(),
                        num_steps,
                        hard=hard,
                        method=method,
                        generator=generator,
                        grid=grid,
                        cfg_strength=cfg_strength,
                    )
                )
                targets.append(batch["tokens"].to(model.device))
            pred, tokens = torch.cat(preds), torch.cat(targets)
        finally:
            model.train(was_training)
        wer, token_error = score_samples(
            pred, tokens, model.tokenizer, model.pad_id, model.vocab_map
        )
        return wer, token_error, int(tokens.size(0))
