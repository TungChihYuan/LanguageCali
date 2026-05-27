#!/usr/bin/env bash
# scripts/train_all.sh
# Train all models. Run from project root: bash scripts/train_all.sh

set -e
echo "===== entity-ece: train all models ====="

echo ""
echo "── 1/7  Data preparation ──"
python -c "
from src.data.conll2003 import CoNLL2003
from src.data.imdb import IMDB
CoNLL2003().prepare()
IMDB().prepare()
"

echo ""
echo "── 2/7  CRF NER ──"
python src/models/crf_ner.py

echo ""
echo "── 3/7  BERT NER ──"
python src/models/bert_ner.py

echo ""
echo "── 4/7  BERT-CRF NER (Theory 1 experiment) ──"
python src/models/bert_crf_ner.py

echo ""
echo "── 5/7  TF-IDF + LogReg Sentiment ──"
python src/models/tfidf_sentiment.py

echo ""
echo "── 6/7  BERT Sentiment ──"
python src/models/bert_sentiment.py

echo ""
echo "── 7/7  LLM NER + Sentiment (needs OPENAI_API_KEY) ──"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "  OPENAI_API_KEY not set — skipping LLM."
else
    python src/models/llm.py --task ner
    python src/models/llm.py --task sentiment
fi

echo ""
echo "===== Training complete ====="
echo "Next step:"
echo "  python experiments/run_calibration.py"
echo "  python experiments/compare.py"
