import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingTokenDataset(IterableDataset):
    """
    Streams text from a HuggingFace dataset, tokenizes on the fly,
    and packs tokens into fixed-length chunks for causal LM training.

    Each yielded sample is a pair (input_ids, targets) of shape (seq_len,)
    where targets = input_ids shifted right by one position.

    Supports both multi-worker DataLoader and multi-GPU DDP sharding.
    Each (rank, worker) pair processes a disjoint slice of the stream.
    """

    def __init__(self, tokenizer, hf_dataset, seq_len: int,
                 rank: int = 0, world_size: int = 1):
        self.tokenizer = tokenizer
        self.hf_dataset = hf_dataset
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1

        # Total shards = num_workers × world_size
        # Each (rank, worker) pair gets a unique shard index
        total_shards = num_workers * self.world_size
        shard_id = self.rank * num_workers + worker_id

        buffer = []
        for idx, sample in enumerate(self.hf_dataset):
            if idx % total_shards != shard_id:
                continue
            text = sample.get("text", "")
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_eot=True)
            buffer.extend(ids)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


class MemmapTokenDataset(IterableDataset):
    """
    Reads pre-tokenized data from a flat memory-mapped uint16 file.

    The file is a contiguous array of token IDs produced by a one-time
    tokenization pass. Each (rank, worker) pair gets a disjoint contiguous
    shard of the file. Yields (input_ids, targets) pairs of shape (seq_len,).

    Supports an optional (start, end) token range to carve out train/eval
    splits from the same file.
    """

    def __init__(self, path: str, seq_len: int,
                 rank: int = 0, world_size: int = 1,
                 start: int = 0, end: int | None = None,
                 shuffle: bool = False, seed: int = 0):
        self.path = path
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.start = start
        self.end = end
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        total_len = len(tokens) if self.end is None else self.end
        region_len = total_len - self.start

        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1

        total_shards = num_workers * self.world_size
        shard_id = self.rank * num_workers + worker_id

        chunk_size = self.seq_len + 1
        shard_tokens = region_len // total_shards
        shard_start = self.start + shard_id * shard_tokens
        n_chunks = shard_tokens // chunk_size

        if n_chunks == 0:
            return

        indices = np.arange(n_chunks)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + shard_id)
            rng.shuffle(indices)

        for idx in indices:
            offset = shard_start + idx * chunk_size
            chunk = torch.from_numpy(tokens[offset : offset + chunk_size].astype(np.int64))
            yield chunk[:-1], chunk[1:]
