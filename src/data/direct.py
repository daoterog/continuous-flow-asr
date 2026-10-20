"""The training loader for features on local disk: reads straight into pinned memory.

A batch of 128 full-latent utterances is ~490 MB. Through a DataLoader each byte is copied
by a worker off the memory map, stacked, moved through shared memory and pinned again by the
main process — on an H100 job's 16 cores that pipeline tops out near 1,000 utts/s against a
GPU that trains 1,341 (2026-10-20), and its workers starve the training process: 741 utts/s
end to end. Here a few threads in the training process `preadv` each utterance's bytes
directly into its row of a reused pinned buffer — one copy, made by the kernel with the GIL
released — and a producer thread sends each finished buffer to the GPU on a side stream
while the previous step trains. The tokens are computed once, up front.

Same batches: the same sampler over the same subset, the same latents byte for byte, the
same tokenization (`collate`'s padding to the canvas is per utterance, so doing it once for
the whole split gives identical ids). Only the full-latent path (`audio_stride` 1) is
supported, and only reading from a local disk makes sense: on GPFS a plain read bypasses
the page cache, so it is meant for features staged to node-local NVMe.
"""

import os
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler, Subset
from transformers import WhisperTokenizerFast

from .features import AudioFeatureDataset
from .vocab import VocabMap


class DirectBatchLoader:
    """Batches of `subset` as {"audio", "tokens"} already on the GPU, `depth` kept ready."""

    def __init__(
        self,
        subset: Subset,
        batch_size: int,
        tokenizer: WhisperTokenizerFast,
        canvas_len: int,
        vocab_map: VocabMap | None = None,
        shuffle: bool = True,
        steps_per_epoch: int | None = None,
        threads: int = 8,
        depth: int = 2,
    ) -> None:
        """Locate every utterance's bytes and tokenize the split once."""
        self.subset = subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps_per_epoch = steps_per_epoch
        self.depth = depth
        self._pool = ThreadPoolExecutor(threads)

        parts: list[AudioFeatureDataset] = list(subset.dataset.datasets)
        starts = np.cumsum([0] + [len(p) for p in parts])
        self._fds: list[int] = []
        fd_of: dict[tuple[int, int], int] = {}
        locs, texts, shapes = [], [], set()
        for i in subset.indices:
            k = int(np.searchsorted(starts, i, side="right")) - 1
            part, j = parts[k], i - int(starts[k])
            if part.audio_stride != 1:
                raise ValueError("DirectBatchLoader reads full latents only (audio_stride 1)")
            shard, offset, length, text = part.utts[j]
            if (k, shard) not in fd_of:
                path = part.index_paths[shard].with_suffix(".npy")
                header = np.load(path, mmap_mode="r")  # parses the header; no data read
                fd = os.open(path, os.O_RDONLY)
                self._fds.append(fd)
                fd_of[(k, shard)] = (fd, header.offset, header.shape[1], header.dtype)
            fd, start, d_model, dtype = fd_of[(k, shard)]
            row = d_model * dtype.itemsize
            locs.append((fd, start + offset * row, length * row))
            shapes.add((length, d_model, dtype))
            texts.append(text)
        if len(shapes) != 1:
            raise ValueError(f"utterances differ in shape {shapes}; the cache must be untrimmed")
        self.frames, self.d_model, dtype = shapes.pop()
        self.dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        self._locs = locs

        ids = []
        for s in range(0, len(texts), 8192):
            enc = tokenizer(
                texts[s : s + 8192], return_tensors="pt", padding="max_length",
                truncation=True, max_length=canvas_len,
            ).input_ids
            ids.append(vocab_map.encode(enc) if vocab_map is not None else enc)
        self.tokens = torch.cat(ids)

    def _batches(self) -> BatchSampler:
        if self.steps_per_epoch:
            sampler = RandomSampler(
                self.subset, replacement=True,
                num_samples=self.steps_per_epoch * self.batch_size,
            )
        elif self.shuffle:
            sampler = RandomSampler(self.subset)
        else:
            sampler = SequentialSampler(self.subset)
        # drop_last with shuffling, as make_loader: the step count stays exact.
        return BatchSampler(sampler, self.batch_size, drop_last=self.shuffle)

    def __len__(self) -> int:
        """Batches per epoch."""
        return len(self._batches())

    def _read_one(self, buf: np.ndarray, row: int, index: int) -> None:
        fd, offset, nbytes = self._locs[index]
        view = memoryview(buf[row]).cast("B")
        done = 0
        while done < nbytes:
            n = os.preadv(fd, [view[done:]], offset + done)
            if n <= 0:
                raise OSError(f"short read: {done} of {nbytes} bytes at {offset}")
            done += n

    def read_batch(self, indices: list[int], out: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fill `out[:len(indices)]` with their latents; the CPU batch for those indices."""
        buf = out.numpy()
        list(self._pool.map(lambda a: self._read_one(buf, *a), enumerate(indices)))
        return {"audio": out[: len(indices)], "tokens": self.tokens[indices]}

    def __iter__(self) -> Iterator[dict]:
        """Yield each batch on the GPU, with the next ones already read and in flight."""
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
            # depth + 1 pinned buffers: one being filled, up to `depth` queued or in use.
            slots: list[tuple[torch.Tensor, torch.cuda.Event | None]] = [
                (torch.empty((self.batch_size, self.frames, self.d_model), dtype=self.dtype,
                             pin_memory=True), None)
                for _ in range(self.depth + 1)
            ]
            try:
                for i, indices in enumerate(self._batches()):
                    if stop.is_set():
                        return
                    buf, copied = slots[i % len(slots)]
                    if copied is not None:
                        copied.synchronize()  # its last copy to the GPU has finished
                    cpu = self.read_batch(indices, buf)
                    with torch.cuda.stream(stream):
                        batch = {k: v.to(device, non_blocking=True) for k, v in cpu.items()}
                    copied = torch.cuda.Event()
                    copied.record(stream)
                    slots[i % len(slots)] = (buf, copied)
                    if not put((batch, copied)):
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
                    # Allocated on the side stream, used on the compute stream: without this
                    # its memory could be reused while the step still reads it.
                    v.record_stream(torch.cuda.current_stream())
                yield batch
        finally:
            stop.set()
            thread.join(timeout=10)
