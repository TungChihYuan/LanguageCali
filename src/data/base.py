"""
src/data/base.py
Abstract base class for all datasets.
"""

from abc import ABC, abstractmethod
import os
import pickle

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


class BaseDataset(ABC):
    name: str = ""

    @abstractmethod
    def prepare(self) -> dict:
        """Download, process, and cache data. Returns data dict."""
        ...

    def load(self) -> dict:
        path = os.path.join(DATA_DIR, f"{self.name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        print(f"{self.name} not found — downloading...")
        return self.prepare()

    def _save(self, data: dict) -> None:
        path = os.path.join(DATA_DIR, f"{self.name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved → {path}")
