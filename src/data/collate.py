"""Batching: padding a canvas of token ids, and the DataLoader around it."""

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import WhisperTokenizerFast

from .vocab import VocabMap


def collate(
    batch: list[dict],
    tokenizer: WhisperTokenizerFast,
    canvas_len: int,
    vocab_map: VocabMap | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize onto the fixed canvas and stack the (frames, d_model) latents.

    The flow denoises a whole canvas_len-wide field at once, so the target width has to
    be identical every batch — at sampling time there is no reference transcript to size
    it from. Padding is <|endoftext|> (Whisper's pad_token_id) and it is a real target:
    the model learns to emit it, which is how a non-autoregressive sample knows where to
    stop. That is also why nothing here builds an attention mask — every canvas position
    is predicted, so there is nothing to exclude.
    """
    enc = tokenizer(
        [b["text"] for b in batch],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=canvas_len,
    )
    tokens = enc.input_ids
    if vocab_map is not None:
        tokens = vocab_map.encode(tokens)

    # every clip is the same length now that the cache is untrimmed, and the dataset has
    # already pooled along time, so a plain stack does it
    return {"audio": torch.stack([b["audio"] for b in batch]), "tokens": tokens}

def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn: Callable,
    num_workers: int = 4,
    pin_memory: bool = False,
    steps_per_epoch: int | None = None,
) -> DataLoader:
    """One DataLoader, with the settings shared by every split.

    steps_per_epoch decouples the epoch from the dataset size by sampling with
    replacement. On a small subset an epoch is otherwise only a handful of batches, and
    the fixed per-epoch cost (~100 ms of loader restart and hooks) comes to dominate:
    25,600 steps over 64 utterances at batch 16 is 6,400 epoch boundaries. Setting it
    makes an epoch as long as you like without changing what the model sees.
    """
    sampler = None
    if steps_per_epoch:
        sampler = RandomSampler(
            ds, replacement=True, num_samples=steps_per_epoch * batch_size
        )
        shuffle = False  # mutually exclusive with an explicit sampler
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,  # keeps the step count exact for the scheduler
    )


class CudaPrefetcher:
    """A loader whose batches arrive already on the GPU, copied while the previous one trains.

    With the full Whisper latents a batch of 128 is ~490 MB, and the plain path copies it to
    the device on the compute stream right before the step — measured on an A100 at 244 ms
    per step against 172.5 ms with the batch already resident, ~30% of the epoch. Here the
    next batch's copy is issued on a side stream as soon as the current batch is handed out,
    so it overlaps with that batch's forward and backward. Nothing about the batch changes;
    Lightning's own transfer then finds it on the device and does nothing.

    Needs `pin_memory=True` on the wrapped loader, or the non-blocking copy degrades to a
    synchronous one and nothing is gained.
    """

    def __init__(self, loader: DataLoader) -> None:
        """Wrap `loader`; its length, workers and sampler are unchanged."""
        self.loader = loader

    def __len__(self) -> int:
        """Batches per epoch, as the wrapped loader reports them."""
        return len(self.loader)

    def __iter__(self):
        """Yield each batch on the GPU, with the next one already in flight."""
        stream = torch.cuda.Stream()
        batches = iter(self.loader)

        def fetch() -> dict | None:
            batch = next(batches, None)
            if batch is None:
                return None
            with torch.cuda.stream(stream):
                return {
                    k: v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

        upcoming = fetch()
        while upcoming is not None:
            torch.cuda.current_stream().wait_stream(stream)
            batch = upcoming
            for v in batch.values():
                if torch.is_tensor(v):
                    # The tensor was allocated on the side stream but is used on the compute
                    # stream; without this its memory could be reused while still in use.
                    v.record_stream(torch.cuda.current_stream())
            upcoming = fetch()
            yield batch

