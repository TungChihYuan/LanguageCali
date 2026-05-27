"""
src/data/conll2003.py
CoNLL-2003 NER dataset loader.

Returns dict with keys: train, validation, test, tag_names
Each split is a list of dicts: {tokens: [...], ner_tags: [...]}
"""

import os
import pickle
from datasets import load_dataset
from src.data.base import BaseDataset, DATA_DIR


class CoNLL2003(BaseDataset):
    name = "conll2003"

    # Map integer IDs → BIO tag strings
    _ID2TAG = [
        "O",
        "B-PER", "I-PER",
        "B-ORG", "I-ORG",
        "B-LOC", "I-LOC",
        "B-MISC","I-MISC",
    ]

    def prepare(self) -> dict:
        print("Downloading CoNLL-2003 from HuggingFace...")
        # datasets 4.x removed loading-script support; use auto-converted parquet
        raw = load_dataset("conll2003", revision="refs/convert/parquet")

        data = {}
        for split in ["train", "validation", "test"]:
            data[split] = [
                {
                    "tokens":   ex["tokens"],
                    "ner_tags": [self._ID2TAG[t] for t in ex["ner_tags"]],
                }
                for ex in raw[split]
            ]
            print(f"  {split}: {len(data[split]):,} sentences")

        data["tag_names"] = self._ID2TAG
        self._save(data)
        return data
