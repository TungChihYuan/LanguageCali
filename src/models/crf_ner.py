"""
src/models/crf_ner.py
CRF NER with handcrafted features.

Confidence = token marginal P(y_t | x) from sklearn-crfsuite.
This is the best-calibrated model in the study (ECE ~0.009).
"""

import os
import pickle
import re
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics
from sklearn.metrics import classification_report

from src.models.base import BaseModel, PredictionRow
from src.data.conll2003 import CoNLL2003

DATA_DIR = "data"


# -- Feature engineering -------------------------------------------------------

def _word_features(tokens: list[str], i: int) -> dict:
    w = tokens[i]

    def shape(tok):
        s = re.sub(r"[A-Z]", "X", tok)
        s = re.sub(r"[a-z]", "x", s)
        s = re.sub(r"[0-9]", "d", s)
        return s

    feats = {
        "bias":           1.0,
        "word.lower":     w.lower(),
        "word[-3:]":      w[-3:],
        "word[-2:]":      w[-2:],
        "word[:3]":       w[:3],
        "word[:2]":       w[:2],
        "word.isupper":   w.isupper(),
        "word.istitle":   w.istitle(),
        "word.isdigit":   w.isdigit(),
        "word.shape":     shape(w),
        "word.hasdigit":  any(c.isdigit() for c in w),
        "word.hasupper":  any(c.isupper() for c in w),
        "word.hasdash":   "-" in w,
        "word.len":       len(w),
    }

    if i > 0:
        w1 = tokens[i - 1]
        feats.update({
            "-1:word.lower":   w1.lower(),
            "-1:word.istitle": w1.istitle(),
            "-1:word.isupper": w1.isupper(),
            "-1:word.shape":   shape(w1),
        })
    else:
        feats["BOS"] = True

    if i < len(tokens) - 1:
        w1 = tokens[i + 1]
        feats.update({
            "+1:word.lower":   w1.lower(),
            "+1:word.istitle": w1.istitle(),
            "+1:word.isupper": w1.isupper(),
            "+1:word.shape":   shape(w1),
        })
    else:
        feats["EOS"] = True

    if i > 1:
        w2 = tokens[i - 2]
        feats["-2:word.lower"] = w2.lower()
    if i < len(tokens) - 2:
        w2 = tokens[i + 2]
        feats["+2:word.lower"] = w2.lower()

    return feats


def _sent_to_features(tokens):
    return [_word_features(tokens, i) for i in range(len(tokens))]


def _sent_to_labels(tags):
    return tags


# -- Model ---------------------------------------------------------------------

class CRFNER(BaseModel):
    """
    CRF NER using sklearn-crfsuite.
    Confidence = marginal probability from predict_marginals().
    """

    def __init__(self, c1: float = 0.1, c2: float = 0.1):
        super().__init__(name="crf", task="ner")
        self.c1  = c1
        self.c2  = c2
        self._crf: sklearn_crfsuite.CRF | None = None

    def train(self, train_data, val_data=None) -> None:
        X_train = [_sent_to_features(s["tokens"])   for s in train_data]
        y_train = [_sent_to_labels(s["ner_tags"])   for s in train_data]

        self._crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1, c2=self.c2,
            max_iterations=100,
            all_possible_transitions=True,
        )
        print(f"[CRF] Training on {len(X_train):,} sentences...")
        self._crf.fit(X_train, y_train)

        if val_data:
            X_val = [_sent_to_features(s["tokens"]) for s in val_data]
            y_val = [_sent_to_labels(s["ner_tags"]) for s in val_data]
            y_pred = self._crf.predict(X_val)
            labels = [l for l in self._crf.classes_ if l != "O"]
            f1 = crf_metrics.flat_f1_score(y_val, y_pred,
                                            average="weighted", labels=labels)
            print(f"[CRF] Val F1 (entities): {f1:.4f}")

    def predict(self, data) -> list[PredictionRow]:
        assert self._crf is not None, "Call train() or load() first."
        X = [_sent_to_features(s["tokens"]) for s in data]

        # Marginal probabilities: list of list of dicts {tag: prob}
        marginals = self._crf.predict_marginals(X)
        rows = []

        for sent_idx, (item, sent_marg) in enumerate(zip(data, marginals)):
            tokens = item["tokens"]
            tags   = item["ner_tags"]
            for tok_idx, (token, tok_marg) in enumerate(zip(tokens, sent_marg)):
                pred_tag   = max(tok_marg, key=tok_marg.get)
                confidence = tok_marg[pred_tag]
                rows.append(PredictionRow(
                    sample_id  = sent_idx,
                    token_idx  = tok_idx,
                    token      = token,
                    true_label = tags[tok_idx],
                    pred_label = pred_tag,
                    confidence = round(float(confidence), 6),
                ))

        return rows

    def save(self, path: str = "outputs/saved/crf_ner.pkl") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._crf, f)
        print(f"Saved -> {path}")

    def load(self, path: str = "outputs/saved/crf_ner.pkl") -> None:
        with open(path, "rb") as f:
            self._crf = pickle.load(f)


# -- Entrypoint ----------------------------------------------------------------

if __name__ == "__main__":
    data  = CoNLL2003().load()
    model = CRFNER()

    print("-- Training --")
    model.train(data["train"], data["validation"])
    model.save()

    print("\n-- Predicting test set --")
    test_rows = model.predict(data["test"])
    model.save_predictions(test_rows, split="test")

    print("\n-- Predicting val set --")
    val_rows = model.predict(data["validation"])
    model.save_predictions(val_rows, split="val")

    true  = [r.true_label for r in test_rows]
    pred  = [r.pred_label for r in test_rows]
    elabs = sorted(set(l for l in true + pred if l != "O"))
    acc   = sum(t == p for t, p in zip(true, pred)) / len(true)
    print(f"\nToken accuracy: {acc:.4f}")
    print(classification_report(true, pred, labels=elabs, digits=4, zero_division=0))
