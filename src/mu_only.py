"""mu_psi trained on its own: the baseline with no flow at all.

The audio decoder is already a complete non-autoregressive recogniser — learned canvas
queries cross-attend into the Whisper latents and emit every token in one pass — so trained
alone, at the size of the whole flow model, it answers whether the flow earns its keep:
same audio, same parameter budget, same objective on the same tokens, no denoiser and no
iterative refinement.

Everything around it is shared with `FlowMatching` so the two are scored identically: the
metric bank, the EMA, the components on disk, and the sampling probe, whose "sample" here
is mu's argmax. That keeps `_best` by sampled WER and train.py's end-of-training scoring of
`_best` and `_final` on val and test working unchanged.
"""

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import WhisperTokenizerFast

from src.data import VocabMap
from src.ema import EMAWeights
from src.flow.sampling_probe import SamplingProbe
from src.flow_matching import LOSS_REDUCTIONS, FlowMatching
from src.models import build_audio_decoder
from src.reporting import MetricReporting


class MuOnly(MetricReporting, EMAWeights, pl.LightningModule):
    """The audio decoder alone, trained on the denoiser's objective."""

    def __init__(
        self,
        cfg: DictConfig,
        vocab_size: int,
        pad_id: int,
        canvas_len: int,
        tokenizer: WhisperTokenizerFast,
        vocab_map: VocabMap | None = None,
    ) -> None:
        """Build mu, the probe and the metric bank; the loss reads `flow`'s pad settings."""
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "vocab_map"], logger=False)
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.canvas_len = canvas_len
        self.tokenizer = tokenizer
        self.vocab_map = vocab_map
        self.sampling_probe = SamplingProbe(cfg.get("sampling"), cfg.data.batch_size)

        self.mu_nn = build_audio_decoder(cfg.model.audio_decoder, vocab_size, canvas_len)
        if cfg.train.get("compile"):
            self.mu_nn = torch.compile(self.mu_nn, mode=cfg.train.get("compile_mode", "default"))

        # The same token-level objective the denoiser trains on, pads down-weighted alike,
        # so the comparison is between models and not between losses.
        self.pad_loss_weight = float(cfg.flow.get("pad_loss_weight", 1.0))
        self.loss_reduction = str(cfg.flow.get("loss_reduction", "sequence"))
        if self.loss_reduction not in LOSS_REDUCTIONS:
            raise ValueError(
                f"flow.loss_reduction must be one of {LOSS_REDUCTIONS}, got {self.loss_reduction!r}"
            )

        self.init_ema(cfg.train.get("ema_decay"))
        self.init_metrics(
            ("mu",),
            tokenizer,
            pad_id,
            vocab_map,
            wer_bins=OmegaConf.to_container(cfg.metrics.get("wer_bins", {})) or None,
            input_perr_bins=0,
        )

    # FlowMatching's objective, unchanged: it reads only pad_id, pad_loss_weight,
    # loss_reduction and cfg.flow.pad_decay, all of which this module carries.
    sequence_ce = FlowMatching.sequence_ce

    @property
    def components(self) -> tuple[str, ...]:
        """What `save_components` writes, and what the EMA averages: mu alone."""
        return ("mu_nn",)

    def sample(self, audio: torch.Tensor, num_steps: int = 1, **_: object) -> torch.Tensor:
        """The argmax of mu: the whole transcript in one pass; steps, method and grid are moot.

        The sampling probe calls this exactly as it calls `FlowMatching.sample`, which is
        what puts mu-only's WER under the same key, chosen and scored the same way.
        """
        return self.mu_nn(audio).argmax(dim=-1)

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        audio, tokens = batch["audio"].float(), batch["tokens"]
        logits = self.mu_nn(audio)
        loss, loss_per_seq = self.sequence_ce(logits, tokens)
        out = {"mu": {"loss": loss, "loss_per_seq": loss_per_seq, "pred_ids": logits.argmax(-1)}}
        self.log_losses(loss, out, split, batch_size=tokens.size(0))
        # mu's metrics are unbinned (one bin), so the bin value is never read.
        self.update_metrics(out, tokens, torch.zeros(tokens.size(0), device=self.device), split)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One optimizer step."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One validation batch."""
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One test batch. Only reached through an explicit `trainer.test()`."""
        return self._step(batch, "test")

    def on_train_epoch_start(self) -> None:
        """Clear the train accumulators."""
        super().on_train_epoch_start()
        self.reset_metrics("train")

    def on_train_epoch_end(self) -> None:
        """Flush the train metrics."""
        super().on_train_epoch_end()
        self.log_metrics("train")

    def on_validation_epoch_start(self) -> None:
        """Clear the val accumulators."""
        super().on_validation_epoch_start()
        self.reset_metrics("val")

    def on_validation_epoch_end(self) -> None:
        """Flush the val metrics, then score mu's argmax through the sampling probe."""
        super().on_validation_epoch_end()
        self.log_metrics("val")
        self.log_sampled()

    def on_test_epoch_start(self) -> None:
        """Clear the test accumulators."""
        super().on_test_epoch_start()
        self.reset_metrics("test")

    def on_test_epoch_end(self) -> None:
        """Flush the test metrics."""
        super().on_test_epoch_end()
        self.log_metrics("test")

    def configure_optimizers(self) -> dict:
        """AdamW on mu at `train.audio_decoder_lr`, with OneCycleLR stepped every batch."""
        train = self.cfg.train
        lr = float(train.audio_decoder_lr)
        optimizer = torch.optim.AdamW(
            self.mu_nn.parameters(), lr=lr, weight_decay=train.get("weight_decay", 0.0)
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup = train.get("warmup_steps")
        pct_start = 0.3 if warmup is None else min(max(int(warmup) / total_steps, 1e-3), 0.5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps, pct_start=pct_start
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
