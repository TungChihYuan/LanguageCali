"""
src/data/imdb.py
IMDB sentiment dataset loader.

Returns dict with keys: train, validation, test
Each split is a list of dicts: {text: str, label: "positive"/"negative"}
"""

from datasets import load_dataset
from src.data.base import BaseDataset


class IMDB(BaseDataset):
    name = "imdb"

    VAL_SIZE   = 2500   # carved out of training set
    TRAIN_SIZE = 22500  # remaining training examples

    def prepare(self) -> dict:
        print("Downloading IMDB from HuggingFace...")
        raw = load_dataset("imdb")

        def _convert(examples):
            return [
                {"text": ex["text"],
                 "label": "positive" if ex["label"] == 1 else "negative"}
                for ex in examples
            ]

        all_train = _convert(raw["train"])
        data = {
            "train":      all_train[:self.TRAIN_SIZE],
            "validation": all_train[self.TRAIN_SIZE:self.TRAIN_SIZE + self.VAL_SIZE],
            "test":       _convert(raw["test"]),
        }

        for split, items in data.items():
            print(f"  {split}: {len(items):,} examples")

        self._save(data)
        return data
