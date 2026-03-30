import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingTokenDataset(IterableDataset):
    """
    Streams text from a HuggingFace dataset, tokenizes on the fly,
    and packs tokens into fixed-length chunks for causal LM training.

    Each yielded sample is a pair (input_ids, targets) of shape (seq_len,)
    where targets = input_ids shifted right by one position.

    Multi-worker safe: when num_workers > 0, each worker processes
    a disjoint slice of the stream (round-robin by document index).
    """

    def __init__(self, tokenizer, hf_dataset, seq_len: int):
        self.tokenizer = tokenizer
        self.hf_dataset = hf_dataset
        self.seq_len = seq_len

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1

        buffer = []
        for idx, sample in enumerate(self.hf_dataset):
            if idx % num_workers != worker_id:
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
