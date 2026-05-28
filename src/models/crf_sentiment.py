"""
src/models/crf_sentiment.py
CRF for sentiment via token-level weak supervision.

Design:
  - Each token in a review is labeled with the document-level sentiment.
  - CRF learns transition patterns (e.g. sentiment-bearing words).
  - Document confidence = mean marginal P(pred_label | token) across all tokens.
  - Mirrors crf_ner.py: same sklearn-crfsuite + marginals approach.

Why this matters for calibration:
  CRF's global normalization over the token sequence should produce
  well-calibrated marginals even on a document-level task.
"""

import os
import pickle
import numpy as np
import sklearn_crfsuite
from sklearn.metrics import classification_report

from src.models.base import BaseModel, PredictionRow
from src.data.imdb import IMDB

MAX_TOKENS = 200   # truncate long reviews for speed


# ── Feature engineering ───────────────────────────────────────────────────────

def _word_features(tokens: list[str], i: int) -> dict:
    w = tokens[i]
    feats = {
        "bias":          1.0,
        "word.lower":    w.lower(),
        "word[-3:]":     w[-3:],
        "word[-2:]":     w[-2:],
        "word.isupper":  w.isupper(),
        "word.istitle":  w.istitle(),
        "word.isdigit":  w.isdigit(),
        "word.len":      min(len(w), 20),
    }
    if i > 0:
        w1 = tokens[i - 1]
        feats.update({"-1:word.lower": w1.lower(), "-1:word.istitle": w1.istitle()})
    else:
        feats["BOS"] = True
    if i < len(tokens) - 1:
        w1 = tokens[i + 1]
        feats.update({"+1:word.lower": w1.lower(), "+1:word.istitle": w1.istitle()})
    else:
        feats["EOS"] = True
    return feats


def _to_features(text: str) -> list[dict]:
    tokens = text.split()[:MAX_TOKENS]
    return [_word_features(tokens, i) for i in range(len(tokens))]


# ── Model ─────────────────────────────────────────────────────────────────────

class CRFSentiment(BaseModel):
    """
    CRF sentiment classifier.
    Token-level weak supervision: every token in a review gets the doc label.
    Document prediction = majority token label; confidence = mean marginal.
    """

    def __init__(self, c1: float = 0.1, c2: float = 0.1):
        super().__init__(name="crf_sentiment", task="sentiment")
        self.c1  = c1
        self.c2  = c2
        self._crf: sklearn_crfsuite.CRF | None = None

    def train(self, train_data, val_data=None) -> None:
        X = [_to_features(d["text"]) for d in train_data]
        # Assign document label to every token (weak supervision)
        y = [[d["label"]] * len(x) for d, x in zip(train_data, X)]

        self._crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1, c2=self.c2,
            max_iterations=100,
            all_possible_transitions=True,
        )
        print(f"[CRF-Sent] Training on {len(X):,} documents...")
        self._crf.fit(X, y)

        if val_data:
            preds = self._predict_labels(val_data)
            acc   = sum(p == d["label"] for p, d in zip(preds, val_data)) / len(val_data)
            print(f"[CRF-Sent] Val accuracy: {acc:.4f}")

    def _predict_labels(self, data) -> list[str]:
        X   = [_to_features(d["text"]) for d in data]
        mgs = self._crf.predict_marginals(X)
        return [self._aggregate(m) for m in mgs]

    def _aggregate(self, sent_marg: list[dict]) -> tuple[str, float]:
        """Average token marginals → document label + confidence."""
        sums: dict[str, float] = {}
        for tok in sent_marg:
            for label, p in tok.items():
                sums[label] = sums.get(label, 0.0) + p
        total     = sum(sums.values()) or 1.0
        pred      = max(sums, key=sums.get)
        confidence = sums[pred] / total
        return pred, float(confidence)

    def predict(self, data) -> list[PredictionRow]:
        assert self._crf is not None, "Call train() or load() first."
        X        = [_to_features(d["text"]) for d in data]
        marginals = self._crf.predict_marginals(X)
        rows     = []

        for idx, (item, sent_marg) in enumerate(zip(data, marginals)):
            pred, conf = self._aggregate(sent_marg)
            rows.append(PredictionRow(
                sample_id  = idx,
                true_label = item["label"],
                pred_label = pred,
                confidence = round(conf, 6),
            ))

        return rows

    def save(self, path: str = "outputs/saved/crf_sentiment.pkl") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._crf, f)
        print(f"Saved -> {path}")

    def load(self, path: str = "outputs/saved/crf_sentiment.pkl") -> None:
        with open(path, "rb") as f:
            self._crf = pickle.load(f)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import classification_report

    data  = IMDB().load()
    model = CRFSentiment()

    print("-- Training --")
    model.train(data["train"], data["validation"])
    model.save()

    print("\n-- Predicting test set --")
    test_rows = model.predict(data["test"])
    model.save_predictions(test_rows, split="test")

    print("\n-- Predicting val set --")
    val_rows = model.predict(data["validation"])
    model.save_predictions(val_rows, split="val")

    true = [r.true_label for r in test_rows]
    pred = [r.pred_label for r in test_rows]
    print(f"\nAccuracy: {sum(t==p for t,p in zip(true,pred))/len(true):.4f}")
    print(classification_report(true, pred, digits=4))
    confs = [r.confidence for r in test_rows]
    print(f"Confidence | mean={np.mean(confs):.4f}  std={np.std(confs):.4f}")
