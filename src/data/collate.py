"""Batching: padding a canvas of token ids, and the DataLoader around it."""

import queue
import threading
from collections.abc import Callable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, get_worker_info
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
    # already pooled along time, so a plain stack does it. Inside a DataLoader worker the
    # stack goes straight into shared memory, as torch's own default_collate does: a batch
    # is ~490 MB at full latents, and stacking into private memory would cost one more copy
    # when it is handed to the main process.
    audios = [b["audio"] for b in batch]
    out = None
    if get_worker_info() is not None:
        first = audios[0]
        storage = first._typed_storage()._new_shared(len(audios) * first.numel())
        out = first.new(storage).resize_(len(audios), *first.shape)
    return {"audio": torch.stack(audios, out=out), "tokens": tokens}

def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn: Callable,
    num_workers: int = 4,
    pin_memory: bool = False,
    steps_per_epoch: int | None = None,
    prefetch_factor: int | None = None,
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
        # Batches each worker keeps ready; None is DataLoader's own default (2).
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=shuffle,  # keeps the step count exact for the scheduler
    )


class CudaPrefetcher:
    """A loader whose batches arrive already on the GPU, prepared while the step trains.

    With the full Whisper latents a batch of 128 is ~490 MB, and getting it onto the device
    is as much work as the step itself. A background thread takes each batch from the
    loader, copies it into a reused pinned buffer (a multi-threaded copy; the DataLoader's
    own pin thread is limited to one core and capped an H100 run at ~1,040 utts/s), and
    issues the copy to the GPU on a side stream. The training loop only takes finished
    batches from a short queue, so it never waits on the host. H100, train.960 recipe,
    2026-10-20: GPU-only ceiling 1,341 utts/s, the loader alone 1,043, the two together
    741 before this. Nothing about the batch changes; Lightning's own transfer then finds it
    on the device and does nothing.

    The wrapped loader should not pin (`pin_memory=False`): pinning here replaces it.
    """

    def __init__(self, loader: DataLoader, depth: int = 2) -> None:
        """Wrap `loader`; `depth` batches are kept ready on the GPU ahead of the step."""
        self.loader = loader
        self.depth = depth

    def __len__(self) -> int:
        """Batches per epoch, as the wrapped loader reports them."""
        return len(self.loader)

    def __iter__(self) -> Iterator[dict]:
        """Yield each batch on the GPU, with the next ones already prepared."""
        device = torch.cuda.current_device()
        ready: queue.Queue = queue.Queue(maxsize=self.depth)
        stop = threading.Event()
        done = object()

        def put(item: object) -> bool:
            while not stop.is_set():
                try:
                    ready.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            torch.cuda.set_device(device)
            stream = torch.cuda.Stream()
            # depth + 1 slots: one being filled, up to `depth` queued or in use.
            slots: list[tuple[dict, torch.cuda.Event] | None] = [None] * (self.depth + 1)
            try:
                for i, batch in enumerate(self.loader):
                    if stop.is_set():
                        return
                    k = i % len(slots)
                    pinned = slots[k][0] if slots[k] is not None else {}
                    if slots[k] is not None:
                        slots[k][1].synchronize()  # its last copy to the GPU has finished
                    out = {}
                    for name, v in batch.items():
                        if not torch.is_tensor(v):
                            out[name] = v
                            continue
                        buf = pinned.get(name)
                        if buf is None or buf.shape != v.shape or buf.dtype != v.dtype:
                            buf = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                            pinned[name] = buf
                        buf.copy_(v)
                        with torch.cuda.stream(stream):
                            out[name] = buf.to(device, non_blocking=True)
                    copied = torch.cuda.Event()
                    copied.record(stream)
                    slots[k] = (pinned, copied)
                    if not put((out, copied)):
                        return
                put(done)
            except BaseException as err:  # handed to the training loop, raised there
                put(err)

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        try:
            while True:
                item = ready.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                batch, copied = item
                torch.cuda.current_stream().wait_event(copied)
                for v in batch.values():
                    if torch.is_tensor(v):
                        # Allocated on the side stream, used on the compute stream: without
                        # this its memory could be reused while the step still reads it.
                        v.record_stream(torch.cuda.current_stream())
                yield batch
        finally:
            stop.set()
            thread.join(timeout=10)
