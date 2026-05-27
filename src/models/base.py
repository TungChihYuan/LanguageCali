"""
src/models/base.py
Abstract base class that every model must implement.

Design principle: all models speak the same language.
Input: raw dataset splits.
Output: List[PredictionRow] — written to a standardized CSV.

Adding a new model = subclass BaseModel + implement train() + predict().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional
import csv
import os


# ── Shared data structure ────────────────────────────────────────────────────

@dataclass
class PredictionRow:
    """
    One prediction record, shared across all tasks and models.

    NER:       all fields used; sample_id = sent_id
    Sentiment: token_idx and token are None
    """
    sample_id:  int
    true_label: str
    pred_label: str
    confidence: float
    token_idx:  Optional[int] = None   # NER only
    token:      Optional[str] = None   # NER only


# ── Abstract interface ───────────────────────────────────────────────────────

class BaseModel(ABC):
    """
    Every NLP model in this project inherits from BaseModel.

    Subclasses must implement:
        train(train_data, val_data)  →  None
        predict(data)                →  List[PredictionRow]
    """

    def __init__(self, name: str, task: str):
        """
        Args:
            name: short identifier used in filenames, e.g. "bert_crf"
            task: "ner" or "sentiment"
        """
        self.name = name
        self.task = task

    @abstractmethod
    def train(self, train_data, val_data=None) -> None:
        """Train the model. val_data used for early stopping / TS."""
        ...

    @abstractmethod
    def predict(self, data) -> list[PredictionRow]:
        """
        Run inference and return standardized predictions.

        Args:
            data: task-specific dataset split (list of dicts)

        Returns:
            List of PredictionRow — one per token (NER) or sentence (sentiment)
        """
        ...

    # ── Optional hooks ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights / artifacts. Override if needed."""
        pass

    def load(self, path: str) -> None:
        """Load model weights / artifacts. Override if needed."""
        pass

    # ── I/O helpers ──────────────────────────────────────────────────────────

    def save_predictions(self, rows: list[PredictionRow], split: str = "test") -> str:
        """
        Write predictions to the standard CSV path.
        Returns the file path.
        """
        from src.utils.io import save_predictions
        return save_predictions(rows, self.name, self.task, split)

    def output_path(self, split: str = "test") -> str:
        """Standard path for this model's prediction CSV."""
        task_dir = os.path.join("outputs", "predictions", self.task)
        return os.path.join(task_dir, f"{self.name}_predictions_{split}.csv")
