"""
src/models/bert_crf_ner.py
BERT-CRF for NER with forward-backward marginal probabilities.

Key purpose: Test Theory 1.
  - Same BERT encoder as bert_ner.py
  - Different output: CRF layer + marginal P(y_t=k|x)
  - If calibration improves vs BERT softmax → structured marginals matter
  - If not → model complexity is the key variable, not marginals
"""

import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast, BertModel,
    AdamW, get_linear_schedule_with_warmup,
)
from torchcrf import CRF
from sklearn.metrics import classification_report

from src.models.base import BaseModel, PredictionRow

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN  = 128
DATA_DIR = "data"


# ── Neural components ─────────────────────────────────────────────────────────

class _BertCRFModule(nn.Module):
    """
    BERT encoder → linear emission scores → CRF decoder.

    The CRF layer replaces the final softmax. The critical difference:
      - Softmax: each token's distribution is independent
      - CRF: globally normalizes over all possible label sequences
    """

    def __init__(self, bert_name: str, num_tags: int, dropout: float = 0.1):
        super().__init__()
        self.bert       = BertModel.from_pretrained(bert_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf        = CRF(num_tags, batch_first=True)
        self.num_tags   = num_tags

    def emissions(self, input_ids, attention_mask) -> torch.Tensor:
        h = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(h))   # (B, L, num_tags)

    def forward(self, input_ids, attention_mask, labels=None):
        emit = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            return -self.crf(emit, safe_labels, mask=mask, reduction="mean")
        return self.crf.decode(emit, mask=mask)

    def marginals(self, input_ids, attention_mask) -> torch.Tensor:
        """
        Marginal probabilities P(y_t = k | x) via forward-backward.

        Unlike Viterbi (MAP decoding), marginals account for ALL possible
        label sequences, weighted by their probability. This is the
        globally-normalized confidence that Theory 1 is about.

        Returns: (batch, seq_len, num_tags) marginal probabilities
        """
        emit = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        B, L, K = emit.shape

        trans   = self.crf.transitions         # (K, K)
        start_t = self.crf.start_transitions   # (K,)
        end_t   = self.crf.end_transitions     # (K,)

        # ── Forward pass (log space) ──────────────────────────────────────
        # alpha[t] = log P(y_1..y_t, y_t | x)  shape (B, K)
        alphas = []
        alpha  = start_t.unsqueeze(0) + emit[:, 0]          # (B, K)
        alphas.append(alpha)

        for t in range(1, L):
            # logsumexp over previous tags
            a = alpha.unsqueeze(2) + trans.unsqueeze(0)      # (B, K, K)
            alpha = torch.logsumexp(a, dim=1) + emit[:, t]   # (B, K)
            # Mask: padding positions keep the previous alpha
            active = mask[:, t].float().unsqueeze(1)
            alpha  = alpha * active + alphas[-1] * (1 - active)
            alphas.append(alpha)

        # ── Backward pass (log space) ─────────────────────────────────────
        # beta[t] = log P(y_{t+1}..y_n | y_t, x)  shape (B, K)
        betas      = [None] * L
        beta       = end_t.unsqueeze(0).expand(B, -1)        # (B, K)
        betas[L-1] = beta

        for t in range(L - 2, -1, -1):
            # (B, K, K): from tag i at t to tag j at t+1
            b = trans.unsqueeze(0) + emit[:, t+1].unsqueeze(1) + beta.unsqueeze(1)
            beta = torch.logsumexp(b, dim=2)                  # (B, K)
            active   = mask[:, t+1].float().unsqueeze(1)
            beta     = beta * active + end_t.unsqueeze(0) * (1 - active)
            betas[t] = beta

        # ── Partition function Z ──────────────────────────────────────────
        log_Z = torch.logsumexp(
            alphas[-1] + end_t.unsqueeze(0), dim=1
        ).unsqueeze(1)                                        # (B, 1)

        # ── Marginals ─────────────────────────────────────────────────────
        marg_list = []
        for t in range(L):
            log_m = alphas[t] + betas[t] - log_Z             # (B, K)
            marg_list.append(torch.softmax(log_m, dim=-1))

        return torch.stack(marg_list, dim=1)                  # (B, L, K)


# ── Dataset ───────────────────────────────────────────────────────────────────

class _NERDataset(Dataset):
    def __init__(self, data, tokenizer, tag2id, max_len=MAX_LEN):
        self.data    = data
        self.tok     = tokenizer
        self.tag2id  = tag2id
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
                label_ids.append(self.tag2id[tags[wid]])
            prev = wid

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(label_ids, dtype=torch.long),
            "word_ids":       wids,
        }


# ── BertCRFNER model ──────────────────────────────────────────────────────────

class BertCRFNER(BaseModel):
    """
    BERT-CRF NER model.

    Confidence = marginal P(y_t=k|x) from forward-backward algorithm.
    NOT Viterbi path score. NOT independent softmax.
    """

    def __init__(
        self,
        bert_name:  str   = "bert-base-uncased",
        epochs:     int   = 3,
        batch_size: int   = 16,
        lr:         float = 3e-5,
    ):
        super().__init__(name="bert_crf", task="ner")
        self.bert_name  = bert_name
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.tokenizer  = BertTokenizerFast.from_pretrained(bert_name)
        self._module: _BertCRFModule | None = None
        self.tag2id: dict = {}
        self.id2tag: dict = {}

    # ── Train ─────────────────────────────────────────────────────────────

    def train(self, train_data, val_data=None) -> None:
        # Build tag vocabulary from training data
        all_tags = sorted({tag for item in train_data for tag in item["ner_tags"]})
        self.tag2id = {t: i for i, t in enumerate(all_tags)}
        self.id2tag = {i: t for t, i in self.tag2id.items()}

        train_ds = _NERDataset(train_data, self.tokenizer, self.tag2id)
        loader   = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self._module = _BertCRFModule(self.bert_name, len(self.tag2id)).to(DEVICE)
        optimizer    = AdamW(self._module.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps  = len(loader) * self.epochs
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        print(f"\n[BertCRF] Training on {DEVICE} | "
              f"{sum(p.numel() for p in self._module.parameters()):,} params")

        t0 = time.time()
        for epoch in range(self.epochs):
            self._module.train()
            total_loss = 0.0
            for i, batch in enumerate(loader):
                ids   = batch["input_ids"].to(DEVICE)
                mask  = batch["attention_mask"].to(DEVICE)
                lbls  = batch["labels"].to(DEVICE)
                loss  = self._module(ids, mask, lbls)
                loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                total_loss += loss.item()
                if (i + 1) % 200 == 0:
                    print(f"  Epoch {epoch+1} | {i+1}/{len(loader)} "
                          f"| loss {total_loss/(i+1):.4f}")
            print(f"Epoch {epoch+1} done | "
                  f"avg loss {total_loss/len(loader):.4f} | "
                  f"{(time.time()-t0)/60:.1f}m elapsed")

    # ── Predict ───────────────────────────────────────────────────────────

    def predict(self, data) -> list[PredictionRow]:
        assert self._module is not None, "Call train() first (or load())."
        self._module.eval()
        rows = []

        for sent_idx, item in enumerate(data):
            tokens = item["tokens"]
            tags   = item["ner_tags"]

            enc  = self.tokenizer(tokens, is_split_into_words=True,
                                  max_length=MAX_LEN, truncation=True,
                                  padding="max_length", return_tensors="pt")
            wids = enc.word_ids()
            ids  = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)

            with torch.no_grad():
                marg = self._module.marginals(ids, mask).squeeze(0)  # (L, K)

            # Map subword positions → word positions
            prev    = None
            word_n  = -1
            for sub_i, wid in enumerate(wids):
                if wid is None or wid == prev:
                    prev = wid
                    continue
                word_n += 1
                if word_n >= len(tokens):
                    break

                best_k = marg[sub_i].argmax().item()
                rows.append(PredictionRow(
                    sample_id  = sent_idx,
                    token_idx  = word_n,
                    token      = tokens[word_n],
                    true_label = tags[word_n],
                    pred_label = self.id2tag[best_k],
                    confidence = round(marg[sub_i][best_k].item(), 6),
                ))
                prev = wid

            if (sent_idx + 1) % 500 == 0:
                print(f"  Predicted {sent_idx+1}/{len(data)} sentences")

        return rows

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str = "outputs/saved/bert_crf_ner.pt") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self._module.state_dict(),
            "tag2id":     self.tag2id,
            "bert_name":  self.bert_name,
        }, path)
        print(f"Saved → {path}")

    def load(self, path: str = "outputs/saved/bert_crf_ner.pt") -> None:
        ckpt         = torch.load(path, map_location=DEVICE)
        self.tag2id  = ckpt["tag2id"]
        self.id2tag  = {i: t for t, i in self.tag2id.items()}
        self._module = _BertCRFModule(ckpt["bert_name"], len(self.tag2id)).to(DEVICE)
        self._module.load_state_dict(ckpt["state_dict"])
        self._module.eval()
        print(f"Loaded from {path}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    with open(os.path.join(DATA_DIR, "conll2003.pkl"), "rb") as f:
        data = pickle.load(f)

    model = BertCRFNER(epochs=3)

    print("\n── Training ──")
    model.train(data["train"], data["validation"])
    model.save()

    print("\n── Predicting test set ──")
    test_rows = model.predict(data["test"])
    model.save_predictions(test_rows, split="test")

    print("\n── Predicting val set (for temperature scaling) ──")
    val_rows = model.predict(data["validation"])
    model.save_predictions(val_rows, split="val")

    # Quick eval
    true  = [r.true_label for r in test_rows]
    pred  = [r.pred_label for r in test_rows]
    elabs = sorted(set(l for l in true+pred if l != "O"))
    print(f"\nAccuracy: {sum(t==p for t,p in zip(true,pred))/len(true):.4f}")
    print(classification_report(true, pred, labels=elabs, digits=4, zero_division=0))
    confs = [r.confidence for r in test_rows]
    print(f"Confidence | mean={np.mean(confs):.4f} std={np.std(confs):.4f}")
