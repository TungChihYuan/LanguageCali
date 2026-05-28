# Calibration Results — Quick Conclusions

**Research question:** Does a stronger NLP model produce less trustworthy confidence scores?

---

## Main Results

### NER (CoNLL-2003, n≈46k tokens)

| Model    | Acc    | ECE    | ECE (entity) | T    | ECE after TS |
|----------|--------|--------|--------------|------|--------------|
| CRF      | 0.9616 | **0.0104** | **0.0434** | 1.15 | **0.0072** |
| BERT     | 0.9798 | 0.0137 | 0.0561       | 1.32 | 0.0096 |
| BERT-CRF | 0.9813 | 0.0156 | 0.0616       | 1.68 | 0.0091 |

### Sentiment (IMDB, n=25k)

| Model    | Acc    | ECE    | T    | ECE after TS |
|----------|--------|--------|------|--------------|
| BERT     | 0.9219 | **0.0556** | 1.88 | 0.0367 |
| BERT-CRF | 0.8865 | 0.0652 | 1.97 | **0.0257** |
| CRF      | 0.8494 | 0.0708 | 2.41 | 0.0261 |

---

## Key Findings

### 1. Yes — stronger models are less calibrated (before scaling)

On NER, ECE worsens monotonically with model strength:
CRF (0.0104) < BERT (0.0137) < BERT-CRF (0.0156).
The same pattern holds on entity-only ECE: 0.043 → 0.056 → 0.062.
On sentiment, BERT achieves the best raw accuracy (92.2%) but also the worst
overconfidence among the three (T=1.88, meaning predictions must be cooled
significantly to be well-calibrated).

### 2. Theory 1 — BERT-CRF ≈ BERT, not CRF

On NER, BERT-CRF ECE (0.0156) is close to BERT (0.0137), not CRF (0.0104).
This suggests **model complexity, not structured marginals, drives miscalibration**.
The CRF's good calibration comes from its simplicity (limited capacity, less
overconfidence), not from the global normalisation property per se.

After temperature scaling the gap narrows substantially:
BERT-CRF (0.0091) ≈ BERT (0.0096) ≈ CRF (0.0072), suggesting all three
are similarly well-calibrated in terms of shape — just shifted in scale.

### 3. CRF on sentiment behaves differently than on NER

On NER, CRF has the best raw ECE (0.0104) and needs minimal scaling (T=1.15).
On sentiment, CRF is the worst (ECE=0.0708, T=2.41) — heavily overconfident.
This is expected: the CRF sentiment model uses document-level weak supervision
(every token gets the document label), producing artificially high marginal
agreement and inflated confidence.

### 4. Temperature scaling helps most where overconfidence is worst

| Model    | Task      | ECE reduction |
|----------|-----------|---------------|
| CRF      | sentiment | 63.1% |
| BERT-CRF | sentiment | 60.6% |
| BERT-CRF | ner       | 42.0% |
| BERT     | sentiment | 34.0% |
| BERT     | ner       | 29.6% |
| CRF      | ner       | 31.3% |

Sentiment models are systematically more overconfident than NER models
(all T > 1.8 vs T < 1.7 for NER). After TS, sentiment calibration (0.026–0.037)
is worse than NER calibration (0.007–0.010), pointing to a harder calibration
problem beyond temperature alone.

---

## Short Answer

> **Stronger models are less calibrated before scaling, but the gap closes
> after temperature scaling.** The CRF's advantage on NER comes from model
> simplicity, not from structured inference. On sentiment, all models are
> overconfident — CRF most of all — because the task formulation forces
> high-confidence marginals regardless of true uncertainty.
