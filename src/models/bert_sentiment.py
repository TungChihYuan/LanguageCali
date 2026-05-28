"""
src/models/bert_sentiment.py
BERT fine-tuned for binary sentiment classification.

Confidence = softmax probability of predicted class.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report

from src.models.base import BaseModel, PredictionRow
from src.data.imdb import IMDB

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _best_device()
MAX_LEN    = 256
LABELS     = ["negative", "positive"]
LABEL2ID   = {l: i for i, l in enumerate(LABELS)}
ID2LABEL   = {i: l for l, i in LABEL2ID.items()}


class _SentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_LEN):
        self.data    = data
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc  = self.tok(item["text"], max_length=self.max_len,
                        truncation=True, padding="max_length",
                        return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(LABEL2ID[item["label"]], dtype=torch.long),
        }


class BertSentiment(BaseModel):
    def __init__(
        self,
        bert_name:  str   = "bert-base-uncased",
        epochs:     int   = 2,
        batch_size: int   = 16,
        lr:         float = 2e-5,
    ):
        super().__init__(name="bert_sentiment", task="sentiment")
        self.bert_name  = bert_name
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.tokenizer  = BertTokenizerFast.from_pretrained(bert_name)
        self._model     = None

    def train(self, train_data, val_data=None) -> None:
        train_ds = _SentDataset(train_data, self.tokenizer)
        loader   = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self._model = BertForSequenceClassification.from_pretrained(
            self.bert_name, num_labels=2
        ).to(DEVICE)

        optimizer   = AdamW(self._model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(loader) * self.epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.06 * total_steps),
            num_training_steps=total_steps,
        )

        print(f"\n[BertSentiment] Training on {DEVICE}")
        t0 = time.time()
        for epoch in range(self.epochs):
            self._model.train()
            total_loss = 0.0
            for i, batch in enumerate(loader):
                ids   = batch["input_ids"].to(DEVICE)
                mask  = batch["attention_mask"].to(DEVICE)
                lbls  = batch["labels"].to(DEVICE)
                out   = self._model(input_ids=ids, attention_mask=mask, labels=lbls)
                out.loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                total_loss += out.loss.item()
                if (i + 1) % 200 == 0:
                    print(f"  Epoch {epoch+1} | {i+1}/{len(loader)}"
                          f" | loss {total_loss/(i+1):.4f}")
            print(f"Epoch {epoch+1} done | "
                  f"avg loss {total_loss/len(loader):.4f} | "
                  f"{(time.time()-t0)/60:.1f}m")

    def predict(self, data) -> list[PredictionRow]:
        assert self._model is not None
        self._model.eval()

        ds     = _SentDataset(data, self.tokenizer)
        loader = DataLoader(ds, batch_size=32)
        rows   = []

        sample_id = 0
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            with torch.no_grad():
                logits = self._model(input_ids=ids,
                                     attention_mask=mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            true_lbls = [data[sample_id + j]["label"]
                         for j in range(len(probs))]

            for j, (prob_row, true_lbl) in enumerate(zip(probs, true_lbls)):
                best_k = int(prob_row.argmax())
                rows.append(PredictionRow(
                    sample_id  = sample_id + j,
                    true_label = true_lbl,
                    pred_label = ID2LABEL[best_k],
                    confidence = round(float(prob_row[best_k]), 6),
                ))
            sample_id += len(probs)

        return rows

    def save(self, path: str = "outputs/saved/bert_sentiment.pt") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self._model.state_dict(),
                    "bert_name": self.bert_name}, path)
        print(f"Saved -> {path}")

    def load(self, path: str = "outputs/saved/bert_sentiment.pt") -> None:
        ckpt        = torch.load(path, map_location=DEVICE)
        self._model = BertForSequenceClassification.from_pretrained(
            ckpt["bert_name"], num_labels=2
        ).to(DEVICE)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()


if __name__ == "__main__":
    data  = IMDB().load()
    model = BertSentiment()
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
