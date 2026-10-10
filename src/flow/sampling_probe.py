"""Sampling from noise as a measurement, on a fixed slice of the validation split."""

from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig

from src.metrics import score_samples

if TYPE_CHECKING:
    from src.data import AudioFeatureDataModule
    from src.pl_module import FlowMatching


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
        self._batch: dict | None = None

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

    def batch(self, datamodule: "AudioFeatureDataModule") -> dict:
        """The val utterances to sample, collated once and cached.

        `n_utts` unset means the whole split, which is what makes two runs comparable —
        a WER over a different subset each time is not a number you can put side by side.
        """
        if self._batch is None:
            subset = datamodule.datasets["val"]
            want = self.cfg.get("n_utts")
            n = len(subset) if want is None else min(int(want), len(subset))
            self._batch = datamodule.collate_fn([subset[i] for i in range(n)])
        return self._batch

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

    def score(
        self,
        model: "FlowMatching",
        datamodule: "AudioFeatureDataModule",
        method: str | None = None,
        num_steps: int | None = None,
    ) -> tuple[float, float, int]:
        """(WER %, token error, utterances scored) for a full integration from noise.

        Chunked because the integrator holds several (B, canvas_len, vocab_size) fields
        alive at once, so the batch that trains is not always one that can be sampled.
        Eval mode is set and restored around the pass: dropout during sampling would make
        the number noisier than the model.

        x0 comes from a generator seeded the same on every call, so two methods scored in
        the same epoch integrate the same noise and differ only by the integrator.
        """
        batch = self.batch(datamodule)
        was_training = model.training
        model.eval()
        try:
            chunk = int(self.cfg.get("batch_size") or self.train_batch_size)
            num_steps = self.step_counts[0] if num_steps is None else int(num_steps)
            hard = bool(self.cfg.get("hard", False))
            tokens = batch["tokens"].to(model.device)
            generator = torch.Generator(device=model.device)
            generator.manual_seed(int(self.cfg.get("seed", 0)))
            pred = torch.cat(
                [
                    model.sample(
                        batch["audio"][i : i + chunk].to(model.device).float(),
                        num_steps,
                        hard=hard,
                        method=method,
                        generator=generator,
                    )
                    for i in range(0, batch["audio"].size(0), chunk)
                ]
            )
        finally:
            model.train(was_training)
        wer, token_error = score_samples(
            pred, tokens, model.tokenizer, model.pad_id, model.vocab_map
        )
        return wer, token_error, int(tokens.size(0))
