from __future__ import annotations

from pathlib import Path
from itertools import islice

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

DEFAULT_SPECIAL_TOKENS = ["<|EOT|>", "<|PAD|>", "<|BOT|>"]


class NetraTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

        self.eot_id = self._tok.token_to_id("<|EOT|>")
        self.pad_id = self._tok.token_to_id("<|PAD|>")
        self.bot_id = self._tok.token_to_id("<|BOT|>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_eot and self.eot_id is not None:
            ids.append(self.eot_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    @classmethod
    def from_file(cls, path: str | Path) -> NetraTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def train(
        cls,
        dataset,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
        num_samples: int = 500_000,
        batch_size: int = 1000,
        save_path: str | Path | None = None,
    ) -> NetraTokenizer:
        """
        Train a BPE tokenizer on a streaming HuggingFace dataset.

        Args:
            dataset: A HuggingFace streaming dataset with a "text" field.
            vocab_size: Target vocabulary size.
            min_frequency: Minimum token frequency to keep.
            special_tokens: List of special tokens. Defaults to EOT, PAD, BOT.
            num_samples: Number of documents to stream for training.
            batch_size: Batch size for the iterator.
            save_path: If provided, save the trained tokenizer to this path.
        """
        if special_tokens is None:
            special_tokens = list(DEFAULT_SPECIAL_TOKENS)

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        trainer_obj = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        def _batch_iterator():
            it = iter(dataset)
            for _ in range(0, num_samples, batch_size):
                batch = list(islice(it, batch_size))
                if not batch:
                    break
                yield [sample["text"] for sample in batch]

        print(f"Training BPE tokenizer on {num_samples:,} documents...")
        tokenizer.train_from_iterator(_batch_iterator(), trainer=trainer_obj)
        print(f"Tokenizer trained — vocab size: {tokenizer.get_vocab_size():,}")

        instance = cls(tokenizer)

        if save_path is not None:
            instance.save(save_path)
            print(f"Tokenizer saved to {save_path}")

        return instance
