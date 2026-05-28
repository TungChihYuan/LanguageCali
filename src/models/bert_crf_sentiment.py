"""
src/models/bert_crf_sentiment.py
BERT-CRF for sentiment via token-level weak supervision.

Design:
  - Mirrors bert_crf_ner.py: BERT encoder → linear emission → CRF decoder.
  - Binary labels: "positive" / "negative" per token.
  - Training: document label propagated to every token (weak supervision).
  - Confidence: mean forward-backward marginal P(pred_label | token) across tokens.

Why this matters for calibration (Theory 1 extension):
  If structured marginals explain CRF's good calibration on NER,
  the same effect should appear here — BERT-CRF should be better calibrated
  than plain BERT softmax on sentiment too.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from TorchCRF import CRF
from sklearn.metrics import classification_report

from src.models.base import BaseModel, PredictionRow
from src.data.imdb import IMDB


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE   = _best_device()
MAX_LEN  = 256
LABELS   = ["negative", "positive"]
TAG2ID   = {l: i for i, l in enumerate(LABELS)}
ID2TAG   = {i: l for l, i in TAG2ID.items()}
NUM_TAGS = len(LABELS)


# ── Neural components ─────────────────────────────────────────────────────────

class _BertCRFModule(nn.Module):
    def __init__(self, bert_name: str, dropout: float = 0.1):
        super().__init__()
        self.bert       = BertModel.from_pretrained(bert_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_TAGS)
        self.crf        = CRF(NUM_TAGS, batch_first=True)

    def emissions(self, input_ids, attention_mask) -> torch.Tensor:
        h = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(h))          # (B, L, NUM_TAGS)

    def forward(self, input_ids, attention_mask, labels=None):
        emit = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        if labels is not None:
            safe = labels.clone()
            safe[safe == -100] = 0
            return -self.crf(emit, safe, mask=mask, reduction="mean")
        return self.crf.decode(emit, mask=mask)

    def marginals(self, input_ids, attention_mask) -> torch.Tensor:
        """Forward-backward marginals P(y_t=k | x). Shape: (B, L, NUM_TAGS)."""
        emit = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        B, L, K = emit.shape

        trans   = self.crf.transitions
        start_t = self.crf.start_transitions
        end_t   = self.crf.end_transitions

        # Forward
        alphas = []
        alpha  = start_t.unsqueeze(0) + emit[:, 0]
        alphas.append(alpha)
        for t in range(1, L):
            a     = alpha.unsqueeze(2) + trans.unsqueeze(0)
            alpha = torch.logsumexp(a, dim=1) + emit[:, t]
            act   = mask[:, t].float().unsqueeze(1)
            alpha = alpha * act + alphas[-1] * (1 - act)
            alphas.append(alpha)

        # Backward
        betas      = [None] * L
        beta       = end_t.unsqueeze(0).expand(B, -1)
        betas[L-1] = beta
        for t in range(L - 2, -1, -1):
            b    = trans.unsqueeze(0) + emit[:, t+1].unsqueeze(1) + beta.unsqueeze(1)
            beta = torch.logsumexp(b, dim=2)
            act  = mask[:, t+1].float().unsqueeze(1)
            beta = beta * act + end_t.unsqueeze(0) * (1 - act)
            betas[t] = beta

        log_Z = torch.logsumexp(
            alphas[-1] + end_t.unsqueeze(0), dim=1
        ).unsqueeze(1)

        marg_list = []
        for t in range(L):
            log_m = alphas[t] + betas[t] - log_Z
            marg_list.append(torch.softmax(log_m, dim=-1))
        return torch.stack(marg_list, dim=1)              # (B, L, NUM_TAGS)


# ── Dataset ───────────────────────────────────────────────────────────────────

class _SentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_LEN):
        self.data    = data
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        label = TAG2ID[item["label"]]
        enc   = self.tok(item["text"],
                         max_length=self.max_len, truncation=True,
                         padding="max_length", return_tensors="pt")
        seq_len = enc["attention_mask"].sum().item()
        # Every real token gets the document label; padding → -100
        labels = torch.full((self.max_len,), -100, dtype=torch.long)
        labels[:int(seq_len)] = label

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ── BertCRFSentiment model ────────────────────────────────────────────────────

class BertCRFSentiment(BaseModel):
    """
    BERT-CRF sentiment classifier.
    Confidence = mean forward-backward marginal across tokens.
    """

    def __init__(
        self,
        bert_name:  str   = "bert-base-uncased",
        epochs:     int   = 2,
        batch_size: int   = 16,
        lr:         float = 2e-5,
    ):
        super().__init__(name="bert_crf_sentiment", task="sentiment")
        self.bert_name  = bert_name
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.tokenizer  = BertTokenizerFast.from_pretrained(bert_name)
        self._module: _BertCRFModule | None = None

    def train(self, train_data, val_data=None) -> None:
        ds      = _SentDataset(train_data, self.tokenizer)
        loader  = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._module = _BertCRFModule(self.bert_name).to(DEVICE)
        optimizer    = AdamW(self._module.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps  = len(loader) * self.epochs
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.06 * total_steps),
            num_training_steps=total_steps,
        )

        print(f"\n[BertCRF-Sent] Training on {DEVICE} | "
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

    def predict(self, data) -> list[PredictionRow]:
        assert self._module is not None, "Call train() first."
        self._module.eval()
        rows = []

        for idx, item in enumerate(data):
            enc  = self.tokenizer(item["text"],
                                  max_length=MAX_LEN, truncation=True,
                                  padding="max_length", return_tensors="pt")
            ids  = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)
            seq_len = mask.sum().item()

            with torch.no_grad():
                marg = self._module.marginals(ids, mask).squeeze(0)  # (L, 2)

            # Average marginals over real tokens only
            real_marg  = marg[:int(seq_len)]           # (seq_len, 2)
            mean_marg  = real_marg.mean(dim=0)         # (2,)
            best_k     = mean_marg.argmax().item()
            confidence = mean_marg[best_k].item()

            rows.append(PredictionRow(
                sample_id  = idx,
                true_label = item["label"],
                pred_label = ID2TAG[best_k],
                confidence = round(confidence, 6),
            ))

            if (idx + 1) % 1000 == 0:
                print(f"  Predicted {idx+1}/{len(data)}")

        return rows

    def save(self, path: str = "outputs/saved/bert_crf_sentiment.pt") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self._module.state_dict(),
            "bert_name":  self.bert_name,
        }, path)
        print(f"Saved → {path}")

    def load(self, path: str = "outputs/saved/bert_crf_sentiment.pt") -> None:
        ckpt         = torch.load(path, map_location=DEVICE)
        self._module = _BertCRFModule(ckpt["bert_name"]).to(DEVICE)
        self._module.load_state_dict(ckpt["state_dict"])
        self._module.eval()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data  = IMDB().load()
    model = BertCRFSentiment(epochs=2)

    print("── Training ──")
    model.train(data["train"], data["validation"])
    model.save()

    print("\n── Predicting test set ──")
    test_rows = model.predict(data["test"])
    model.save_predictions(test_rows, split="test")

    print("\n── Predicting val set ──")
    val_rows = model.predict(data["validation"])
    model.save_predictions(val_rows, split="val")

    true = [r.true_label for r in test_rows]
    pred = [r.pred_label for r in test_rows]
    print(f"\nAccuracy: {sum(t==p for t,p in zip(true,pred))/len(true):.4f}")
    print(classification_report(true, pred, digits=4))
    confs = [r.confidence for r in test_rows]
    print(f"Confidence | mean={np.mean(confs):.4f}  std={np.std(confs):.4f}")
