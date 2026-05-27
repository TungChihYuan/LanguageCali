# LanguageCali — NLP Model Calibration Study

> **Do stronger NLP models produce less trustworthy confidence scores?**

This project systematically compares **calibration quality** across four model families on two NLP tasks, using ECE (Expected Calibration Error) as the core metric.

---

## Overview

| Task | Dataset | Models |
|------|---------|--------|
| Named Entity Recognition (NER) | CoNLL-2003 | CRF, BERT, BERT-CRF, LLM |
| Sentiment Analysis | IMDB | TF-IDF+LR, BERT, LLM |

**Core metric:** ECE (Expected Calibration Error) — lower is better.

---

## Research Hypothesis

**Theory 1:** CRF is well-calibrated because of *structured marginals* (global normalization), not merely because the model is simpler.

| Outcome | Interpretation |
|---------|----------------|
| BERT-CRF ECE ≈ CRF ECE | Theory 1 holds — marginals are the key |
| BERT-CRF ECE ≈ BERT ECE | Theory 1 fails — model complexity is the reason |

---

## Project Structure

```
LanguageCali/
├── configs/              # Experiment configs (NER, sentiment)
├── src/
│   ├── data/             # Dataset loaders (CoNLL-2003, IMDB)
│   ├── models/           # CRF, BERT, BERT-CRF, LLM, TF-IDF
│   ├── calibration/      # ECE metrics, Temperature Scaling, plots
│   └── utils/            # CSV I/O helpers
├── experiments/
│   ├── run_calibration.py   # Main analysis entry point
│   └── compare.py           # Multi-model comparison + Theory 1 charts
├── outputs/
│   ├── predictions/         # Per-model prediction CSVs
│   └── analysis/            # JSON results + PNG charts
└── scripts/
    ├── train_all.sh          # Train all models
    └── evaluate_all.sh       # Evaluate all models
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare datasets
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

## Calibration Metrics

Three levels of ECE are reported:

| Metric | Description |
|--------|-------------|
| `compute_ece` | All tokens (traditional) |
| `compute_entity_ece` | Entity tokens only (excludes `O` label) |
| `compute_per_type_ece` | Per entity type (PER, ORG, LOC, MISC) |

**Temperature Scaling** is applied in logit space:
```python
logit  = log(p / (1 - p))
scaled = sigmoid(logit / T)
```

**BERT-CRF confidence** uses forward-backward marginals `P(y_t=k | x)`, not Viterbi scores.

---

## Dependencies

```
torch>=2.0
transformers>=4.30
datasets
sklearn-crfsuite
torchcrf
openai
matplotlib
seaborn
scipy
scikit-learn
numpy
```

---

## License

[MIT](LICENSE)
