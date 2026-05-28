"""
src/utils/io.py
Standardized CSV read / write for all prediction files.

Schema
------
NER (token-level):
    sample_id, token_idx, token, true_label, pred_label, confidence

Sentiment (sequence-level):
    sample_id, true_label, pred_label, confidence
    (token_idx and token are empty strings)
"""

import csv
import os
from dataclasses import asdict
from typing import Optional


NER_FIELDS       = ["sample_id", "token_idx", "token",
                    "true_label", "pred_label", "confidence"]
SENTIMENT_FIELDS = ["sample_id", "true_label", "pred_label", "confidence"]


def save_predictions(rows, model_name: str, task: str,
                     split: str = "test") -> str:
    """
    Write List[PredictionRow] to the standard path.

    outputs/predictions/{task}/{model_name}_predictions_{split}.csv
    """
    out_dir = os.path.join("outputs", "predictions", task)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}_predictions_{split}.csv")

    fields = NER_FIELDS if task == "ner" else SENTIMENT_FIELDS

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            d = asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row)
            d["confidence"] = round(float(d["confidence"]), 6)
            writer.writerow({k: d.get(k, "") for k in fields})

    print(f"  Saved {len(rows):,} rows -> {path}")
    return path


def load_predictions(path: str) -> tuple[list[dict], str]:
    """
    Load a prediction CSV.

    Returns:
        rows: list of dicts with string→typed values
        task: "ner" or "sentiment"
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        task = "ner" if "token_idx" in fields else "sentiment"
        for row in reader:
            row["confidence"] = float(row["confidence"])
            if "token_idx" in row and row["token_idx"] != "":
                row["token_idx"] = int(row["token_idx"])
            rows.append(row)

    return rows, task


def find_prediction_files(base_dir: str = "outputs/predictions") -> list[str]:
    """
    Find all test prediction CSVs (excludes val files).
    Returns sorted list of absolute paths.
    """
    paths = []
    for task in ["ner", "sentiment"]:
        task_dir = os.path.join(base_dir, task)
        if not os.path.isdir(task_dir):
            continue
        for fname in sorted(os.listdir(task_dir)):
            if fname.endswith(".csv") and "_val" not in fname:
                paths.append(os.path.join(task_dir, fname))
    return paths
