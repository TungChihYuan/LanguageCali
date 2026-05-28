# Experiment Methodology

## Overview

Six models across two tasks, evaluated on calibration quality (ECE).
Each model outputs a confidence score per prediction; we measure how well
that confidence correlates with actual accuracy.

---

## Task 1 — NER (CoNLL-2003)

**Dataset:** CoNLL-2003 English NER. Train 14,041 / Val 3,250 / Test 3,684 sentences.
Labels: PER, ORG, LOC, MISC (BIO scheme) + O.

### Model 1 — CRF (`src/models/crf_ner.py`)

- Library: `sklearn-crfsuite`
- Features: handcrafted per-token features — word shape, prefix/suffix (2–4 chars),
  POS-like patterns, is-upper, is-digit, BOS/EOS flags, window ±2 neighbours
- Training: L-BFGS, c1=0.1, c2=0.1, max_iter=100
- **Confidence:** marginal probability `P(y_t | x)` from the forward-backward
  algorithm (sklearn-crfsuite's `predict_marginals`)
- No GPU required

### Model 2 — BERT (`src/models/bert_ner.py`)

- Base: `bert-base-uncased` fine-tuned for token classification
- Tokenizer: `BertTokenizerFast`, max_len=128, subword→word alignment via
  first-subword strategy (subsequent subwords masked with -100)
- Training: AdamW, lr=2e-5, linear warmup (10%), 3 epochs, batch=16
- **Confidence:** softmax probability of the predicted class at each token.
  Tokens are scored *independently* — no sequence constraint.
- GPU: CUDA / MPS / CPU (auto-detected)

### Model 3 — BERT-CRF (`src/models/bert_crf_ner.py`) ★ Theory 1

- Architecture: BERT encoder → linear emission layer → CRF decoder
- Same BERT base and fine-tuning setup as Model 2
- CRF layer: `pytorch-crf` (TorchCRF), `batch_first=True`
- Training: NLL loss from CRF forward pass, same AdamW/schedule as BERT
- **Confidence:** forward-backward marginals `P(y_t=k | x)` computed via the
  forward-backward algorithm over the CRF lattice (NOT the Viterbi score).
  This is the key difference from BERT softmax.
- Purpose: test whether structured marginals improve calibration relative to
  independent softmax (Theory 1)

---

## Task 2 — Sentiment Analysis (IMDB)

**Dataset:** IMDB binary sentiment. Train 22,500 / Val 2,500 / Test 25,000 reviews.
Labels: positive, negative.

### Model 4 — CRF (`src/models/crf_sentiment.py`)

- Same library and feature set as CRF NER
- **Weak supervision design:** every token in a review is assigned the
  document-level sentiment label. The CRF learns transition patterns over
  sentiment-bearing words.
- **Confidence:** mean marginal `P(pred_label | token)` across all tokens in
  the document (truncated to MAX_TOKENS=200)
- This design mirrors CRF NER to keep the comparison fair, but inflates
  confidence because all tokens agree on the same label by construction.

### Model 5 — BERT (`src/models/bert_sentiment.py`)

- Base: `bert-base-uncased` fine-tuned for sequence classification
  (`BertForSequenceClassification`)
- Tokenizer: max_len=128, padding to max_length
- Training: AdamW, lr=2e-5, linear warmup, 2 epochs, batch=16
- **Confidence:** softmax probability of the predicted class ([CLS] token)
- GPU: CUDA / MPS / CPU (auto-detected)

### Model 6 — BERT-CRF (`src/models/bert_crf_sentiment.py`) ★ Theory 1

- Architecture: BERT encoder → linear emission (2 tags) → CRF decoder
- **Weak supervision design:** same as CRF sentiment — each token gets the
  document label, CRF operates over the token sequence
- Training: NLL from CRF, AdamW, lr=2e-5, 2 epochs, batch=16, max_len=128
- **Confidence:** mean forward-backward marginal across all tokens
- Purpose: test whether structured marginals improve calibration on sentiment

---

## Calibration Evaluation (`experiments/run_calibration.py`)

### ECE (Expected Calibration Error)

Predictions are sorted into 15 uniform bins by confidence.
For each bin: `|accuracy - mean_confidence|`, weighted by bin size.

```
ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
```

Three variants for NER:
- **ECE (all):** all tokens including O
- **ECE (entity):** entity tokens only (true label != O)
- **ECE (per-type):** separate ECE per entity type (PER/ORG/LOC/MISC)

Sentiment uses only ECE (all) since there are no entity types.

### Adaptive ECE (AdaECE)

Equal-mass binning (each bin has the same number of samples) instead of
equal-width. More robust for skewed confidence distributions.

### Brier Score

Mean squared error between confidence and correctness:
`Brier = (1/n) * sum (conf_i - corr_i)^2`

Decomposed into reliability + resolution + uncertainty.

### Temperature Scaling (`src/calibration/scaling.py`)

Post-hoc recalibration using a single scalar temperature T, fitted on the
validation set by minimising ECE.

**Correct implementation (logit space):**
```
logit(p)  = log(p / (1-p))
scaled(p) = sigmoid(logit(p) / T)
```

T > 1 means the model is overconfident (predictions are sharpened too much).
T < 1 means underconfident.

The optimal T is found via `scipy.optimize.minimize_scalar` on the validation
set, then applied to the test set.

---

## Reliability Diagrams (`src/calibration/viz.py`)

One diagram per model (before and after temperature scaling).
X-axis: mean confidence per bin. Y-axis: accuracy per bin.
Perfect calibration = diagonal line. Bar height = gap from diagonal.
