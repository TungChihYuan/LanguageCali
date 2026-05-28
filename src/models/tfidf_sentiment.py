"""
src/models/tfidf_sentiment.py
TF-IDF + Logistic Regression for sentiment classification.

Confidence = predict_proba() output.
Known behavior: L2 regularization compresses probabilities toward 0.5,
causing systematic underconfidence (T < 1 after calibration).
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.models.base import BaseModel, PredictionRow
from src.data.imdb import IMDB


class TFIDFSentiment(BaseModel):
    """
    TF-IDF unigrams+bigrams -> L2-regularized Logistic Regression.
    Confidence = max class probability from predict_proba().
    """

    def __init__(self, max_features: int = 50_000, C: float = 1.0):
        super().__init__(name="tfidf_logreg", task="sentiment")
        self.max_features = max_features
        self.C            = C
        self._vec: TfidfVectorizer   | None = None
        self._clf: LogisticRegression | None = None

    def train(self, train_data, val_data=None) -> None:
        texts  = [d["text"]  for d in train_data]
        labels = [d["label"] for d in train_data]

        print(f"[TFIDF] Fitting vectorizer on {len(texts):,} examples...")
        self._vec = TfidfVectorizer(
            max_features=self.max_features,
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
        )
        X_train = self._vec.fit_transform(texts)

        print("[TFIDF] Training Logistic Regression...")
        self._clf = LogisticRegression(C=self.C, max_iter=1000,
                                       solver="lbfgs", n_jobs=-1)
        self._clf.fit(X_train, labels)

        if val_data:
            X_val   = self._vec.transform([d["text"]  for d in val_data])
            y_val   = [d["label"] for d in val_data]
            acc     = self._clf.score(X_val, y_val)
            print(f"[TFIDF] Val accuracy: {acc:.4f}")

    def predict(self, data) -> list[PredictionRow]:
        assert self._clf is not None
        texts  = [d["text"]  for d in data]
        labels = [d["label"] for d in data]
        X      = self._vec.transform(texts)
        proba  = self._clf.predict_proba(X)               # (N, 2)
        classes = self._clf.classes_                        # e.g. ["negative","positive"]
        rows = []
        for i, (true_lbl, prob_row) in enumerate(zip(labels, proba)):
            best_k     = prob_row.argmax()
            pred_label = classes[best_k]
            confidence = float(prob_row[best_k])
            rows.append(PredictionRow(
                sample_id  = i,
                true_label = true_lbl,
                pred_label = pred_label,
                confidence = round(confidence, 6),
            ))
        return rows

    def save(self, path: str = "outputs/saved/tfidf_sentiment.pkl") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vec": self._vec, "clf": self._clf}, f)
        print(f"Saved -> {path}")

    def load(self, path: str = "outputs/saved/tfidf_sentiment.pkl") -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._vec = d["vec"]
        self._clf = d["clf"]


if __name__ == "__main__":
    data  = IMDB().load()
    model = TFIDFSentiment()
    model.train(data["train"], data["validation"])
    model.save()

    test_rows = model.predict(data["test"])
    model.save_predictions(test_rows, split="test")

    val_rows = model.predict(data["validation"])
    model.save_predictions(val_rows, split="val")

    true = [r.true_label for r in test_rows]
    pred = [r.pred_label for r in test_rows]
    print(f"\nAccuracy: {sum(t==p for t,p in zip(true,pred))/len(true):.4f}")
    print(classification_report(true, pred, digits=4))
