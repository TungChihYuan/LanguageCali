"""
src/models/bert_ner.py
BERT fine-tuned for NER (token classification).

Confidence = softmax probability of predicted class.
Each token's distribution is computed INDEPENDENTLY — no sequence constraint.
This is the key difference vs BERT-CRF (bert_crf_ner.py).
"""

import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report

from src.models.base import BaseModel, PredictionRow
from src.data.conll2003 import CoNLL2003

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _best_device()
MAX_LEN    = 128
DATA_DIR   = "data"


class _NERDataset(Dataset):
    def __init__(self, data, tokenizer, tag2id, max_len=MAX_LEN):
        self.data   = data
        self.tok    = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        tokens = item["tokens"]
        tags   = item["ner_tags"]

        enc  = self.tok(tokens, is_split_into_words=True,
                        max_length=self.max_len, truncation=True,
                        padding="max_length", return_tensors="pt")
        wids = enc.word_ids()

        label_ids, prev = [], None
        for wid in wids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                label_ids.append(self.tag2id[tags[wid]])
            else:
                label_ids.append(-100)   # ignore non-first subwords
            prev = wid

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(label_ids, dtype=torch.long),
            # word_ids excluded: contains None (special tokens), breaks DataLoader collate
        }


class BertNER(BaseModel):
    """
    BERT softmax NER.
    Confidence = max softmax probability (independent per token).
    """

    def __init__(
        self,
        bert_name:  str   = "bert-base-uncased",
        epochs:     int   = 3,
        batch_size: int   = 16,
        lr:         float = 2e-5,
    ):
        super().__init__(name="bert", task="ner")
        self.bert_name  = bert_name
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.tokenizer  = BertTokenizerFast.from_pretrained(bert_name)
        self._model     = None
        self.tag2id: dict = {}
        self.id2tag: dict = {}

    def train(self, train_data, val_data=None) -> None:
        all_tags = sorted({t for s in train_data for t in s["ner_tags"]})
        self.tag2id = {t: i for i, t in enumerate(all_tags)}
        self.id2tag = {i: t for t, i in self.tag2id.items()}

        train_ds = _NERDataset(train_data, self.tokenizer, self.tag2id)
        loader   = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self._model = BertForTokenClassification.from_pretrained(
            self.bert_name, num_labels=len(self.tag2id)
        ).to(DEVICE)

        optimizer   = AdamW(self._model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(loader) * self.epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        print(f"\n[BertNER] Training on {DEVICE}")
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
        rows = []

        for sent_idx, item in enumerate(data):
            tokens = item["tokens"]
            tags   = item["ner_tags"]

            enc  = self.tokenizer(tokens, is_split_into_words=True,
                                  max_length=MAX_LEN, truncation=True,
                                  padding="max_length", return_tensors="pt")
            wids = enc.word_ids()

            with torch.no_grad():
                out    = self._model(
                    input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE),
                )
                probs = torch.softmax(out.logits, dim=-1).squeeze(0)  # (L, K)

            prev   = None
            word_n = -1
            for sub_i, wid in enumerate(wids):
                if wid is None or wid == prev:
                    prev = wid
                    continue
                word_n += 1
                if word_n >= len(tokens):
                    break
                best_k = probs[sub_i].argmax().item()
                rows.append(PredictionRow(
                    sample_id  = sent_idx,
                    token_idx  = word_n,
                    token      = tokens[word_n],
                    true_label = tags[word_n],
                    pred_label = self.id2tag[best_k],
                    confidence = round(probs[sub_i][best_k].item(), 6),
                ))
                prev = wid

            if (sent_idx + 1) % 500 == 0:
                print(f"  Predicted {sent_idx+1}/{len(data)}")

        return rows

    def save(self, path: str = "outputs/saved/bert_ner.pt") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self._model.state_dict(),
                    "tag2id": self.tag2id, "bert_name": self.bert_name}, path)
        print(f"Saved → {path}")

    def load(self, path: str = "outputs/saved/bert_ner.pt") -> None:
        ckpt        = torch.load(path, map_location=DEVICE)
        self.tag2id = ckpt["tag2id"]
        self.id2tag = {i: t for t, i in self.tag2id.items()}
        self._model = BertForTokenClassification.from_pretrained(
            ckpt["bert_name"], num_labels=len(self.tag2id)
        ).to(DEVICE)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()


if __name__ == "__main__":
    data  = CoNLL2003().load()
    model = BertNER()
    model.train(data["train"], data["validation"])
    model.save()

    test_rows = model.predict(data["test"])
    model.save_predictions(test_rows, split="test")

    val_rows = model.predict(data["validation"])
    model.save_predictions(val_rows, split="val")

    true  = [r.true_label for r in test_rows]
    pred  = [r.pred_label for r in test_rows]
    elabs = sorted(set(l for l in true + pred if l != "O"))
    print(f"\nAccuracy: {sum(t==p for t,p in zip(true,pred))/len(true):.4f}")
    print(classification_report(true, pred, labels=elabs, digits=4, zero_division=0))
