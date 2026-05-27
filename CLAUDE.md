# entity-ece — Claude Code Guide

## Research Question

**Does a stronger NLP model produce less trustworthy confidence scores?**

Systematically compare calibration quality across four model families —
CRF / BERT / BERT-CRF / LLM — on two tasks:
NER (CoNLL-2003) and Sentiment Analysis (IMDB).

Core metric: **ECE (Expected Calibration Error)** — lower is better.

---

## Project Structure

```
entity-ece/
├── CLAUDE.md                        ← You are reading this
├── requirements.txt
├── configs/
│   ├── ner.yaml                     ← NER experiment config
│   └── sentiment.yaml               ← Sentiment analysis experiment config
├── src/
│   ├── data/
│   │   ├── base.py                  ← Abstract dataset interface
│   │   ├── conll2003.py             ← CoNLL-2003 loader
│   │   └── imdb.py                  ← IMDB loader
│   ├── models/
│   │   ├── base.py                  ← Abstract model interface (all models must inherit)
│   │   ├── crf_ner.py               ← CRF + hand-crafted features
│   │   ├── bert_ner.py              ← BERT softmax (independent tokens)
│   │   ├── bert_crf_ner.py          ← BERT-CRF + forward-backward marginals ★
│   │   ├── tfidf_sentiment.py       ← TF-IDF + Logistic Regression
│   │   ├── bert_sentiment.py        ← BERT sentiment classifier
│   │   └── llm.py                   ← GPT-4o-mini (NER + sentiment, unified)
│   ├── calibration/
│   │   ├── metrics.py               ← ECE / entity-ECE / per-type ECE / Brier
│   │   ├── scaling.py               ← Temperature Scaling (logit space, correct version)
│   │   └── viz.py                   ← Reliability diagram / comparison plots
│   └── utils/
│       └── io.py                    ← Standardized CSV read/write
├── experiments/
│   ├── run_calibration.py           ← Main entry: analyze all prediction files
│   └── compare.py                   ← Multi-model comparison + Theory 1 charts
├── outputs/
│   ├── predictions/ner/             ← Per-model prediction CSVs (NER)
│   ├── predictions/sentiment/       ← Per-model prediction CSVs (sentiment)
│   └── analysis/                    ← JSON results + PNG charts
└── scripts/
    ├── train_all.sh                 ← One-command training for all models
    └── evaluate_all.sh              ← One-command evaluation for all models
```

---

## Core Interface

### Unified interface for all models (`src/models/base.py`)

```python
model = SomeModel(config)
model.train(train_data, val_data)
rows = model.predict(test_data)      # → List[PredictionRow]
model.save("outputs/saved/model.pt")
```

### Standardized CSV Format

**NER (token-level):**
```
sent_id, token_idx, token, true_label, pred_label, confidence
0, 0, "Tim", "B-PER", "B-PER", 0.923
```

**Sentiment (sentence-level):**
```
sample_id, true_label, pred_label, confidence
0, "positive", "positive", 0.847
```

---

## Execution Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python -c "from src.data.conll2003 import CoNLL2003; CoNLL2003().prepare()"
python -c "from src.data.imdb import IMDB; IMDB().prepare()"

# 3. Train all models
bash scripts/train_all.sh

# 4. Run calibration analysis
python experiments/run_calibration.py

# 5. Generate comparison plots
python experiments/compare.py
```

---

## Adding a New Model (Extension Guide)

1. Create a new file in `src/models/`, inheriting from `BaseModel`
2. Implement the `train()` and `predict()` methods
3. Add a line to `scripts/train_all.sh`
4. Re-run `experiments/run_calibration.py`

```python
# src/models/my_new_model.py
from src.models.base import BaseModel, PredictionRow

class MyNewModel(BaseModel):
    def train(self, train_data, val_data=None): ...
    def predict(self, data) -> list[PredictionRow]: ...
```

---

## Adding a New Calibration Method (Extension Guide)

Add a new function in `src/calibration/scaling.py` that accepts the same
`(val_conf, val_corr, test_conf)` parameters and returns `(param, scaled_conf)`.

---

## Key Design Decisions

### Temperature Scaling: logit space (fixed)
```python
# Correct: scale in logit space
logit = log(p / (1-p))
scaled = sigmoid(logit / T)

# Wrong (original code): scale in probability space (not mathematically equivalent)
scaled = p**(1/T) / (p**(1/T) + (1-p)**(1/T))
```

### BERT-CRF Confidence: forward-backward marginals
```python
# Not Viterbi score (MAP), but marginal P(y_t=k|x)
# Computed via the forward-backward algorithm — this is the core of Theory 1
marginals = model.compute_marginals(input_ids, attention_mask)
```

### Three Levels of ECE
```python
metrics.compute_ece(confs, corr)            # All tokens (traditional)
metrics.compute_entity_ece(rows)            # Entity tokens only, excluding O
metrics.compute_per_type_ece(rows)          # Per entity type
```

---

## Research Hypothesis

**Theory 1 (to be verified):**
> CRF is well-calibrated because of structured marginals (global normalization),
> not merely because the model is simple.

**Test:**
```
BERT softmax  ECE = ?   (bert_ner.py)
BERT-CRF      ECE = ?   (bert_crf_ner.py) ★

BERT-CRF ≈ CRF  → Theory 1 holds   (marginals are the key)
BERT-CRF ≈ BERT → Theory 1 fails   (model complexity is the reason)
```

---

## Known Bugs (Fixed)

| Original File    | Issue                                      | Fix                          |
|------------------|--------------------------------------------|------------------------------|
| calibration.py   | TS scaled in probability space             | Changed to logit space       |
| comparision.py   | Typo in filename                           | Renamed to comparison.py     |
| calibration.py   | Entity-only ECE existed but wasn't output  | Formally added to pipeline   |

---

## Dependencies

```
torch>=2.0          # BERT / BERT-CRF
transformers>=4.30  # BERT tokenizer + model
datasets            # CoNLL-2003, IMDB
sklearn-crfsuite    # CRF
torchcrf            # CRF layer for BERT-CRF
openai              # LLM
matplotlib seaborn  # Visualization
scipy scikit-learn numpy
```
