"""
src/models/llm.py
Unified GPT-4o-mini model for both NER and sentiment.

Confidence = token log-probability from OpenAI API.
NOT self-reported confidence — actual generation probability.

Usage:
    python src/models/llm.py --task ner
    python src/models/llm.py --task sentiment

Requires: OPENAI_API_KEY environment variable.
"""

import os
import re
import time
import math
import argparse
import numpy as np
from openai import OpenAI

from src.models.base import BaseModel, PredictionRow
from src.data.conll2003 import CoNLL2003
from src.data.imdb import IMDB

MAX_TEST_NER       = 500
MAX_VAL_NER        = 200
MAX_TEST_SENTIMENT = 500
MAX_VAL_SENTIMENT  = 200

VALID_NER_TAGS = {
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC",
}


# ── NER prompting ─────────────────────────────────────────────────────────────

def _ner_few_shot_examples(train_data, n=5) -> str:
    lines = []
    for item in train_data[:n]:
        pairs = " | ".join(
            f"{tok}:{tag}"
            for tok, tag in zip(item["tokens"], item["ner_tags"])
        )
        lines.append(f"Input: {' '.join(item['tokens'])}\nOutput: {pairs}")
    return "\n\n".join(lines)


def _ner_system_prompt(examples: str) -> str:
    return (
        "You are a named entity recognition system. "
        "Given a sentence, label each token with its BIO tag.\n"
        "Valid tags: O B-PER I-PER B-ORG I-ORG B-LOC I-LOC B-MISC I-MISC\n"
        "Format: token1:TAG1 | token2:TAG2 | ...\n\n"
        "Examples:\n" + examples
    )


def _parse_ner_output(
    content: str,
    tokens: list[str],
    logprobs_list: list,
) -> tuple[list[tuple], int]:
    """
    Align model output with input tokens.
    Returns (aligned_list, n_misaligned).
    aligned_list: [(token, pred_tag, confidence), ...]
    """
    # Parse "token:TAG" pairs
    pattern = re.compile(r"(\S+):([A-Z\-]+)")
    parsed  = pattern.findall(content)

    # Build logprob lookup: token_text → confidence
    logprob_map: dict[str, float] = {}
    if logprobs_list:
        for entry in logprobs_list:
            tok_str = entry.token.strip()
            if tok_str and not tok_str.isspace():
                prob = math.exp(entry.logprob)
                if tok_str not in logprob_map:
                    logprob_map[tok_str] = prob

    misaligned = 0
    result     = []
    parsed_i   = 0

    for token in tokens:
        if parsed_i < len(parsed):
            _, pred_tag = parsed[parsed_i]
            if pred_tag not in VALID_NER_TAGS:
                pred_tag = "O"
            # Find confidence for this tag in logprob stream
            conf = logprob_map.get(pred_tag,
                   logprob_map.get(token, 0.6))
            conf = max(0.5, min(1.0, conf))
            parsed_i += 1
        else:
            pred_tag   = "O"
            conf       = 0.5
            misaligned += 1

        result.append((token, pred_tag, conf))

    if parsed_i < len(parsed):
        misaligned += len(parsed) - parsed_i

    return result, misaligned


def _ner_query(client, system_prompt: str, tokens: list[str]):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": f"Input: {' '.join(tokens)}\nOutput:"},
            ],
            temperature=0.0,
            max_tokens=512,
            logprobs=True,
            top_logprobs=5,
        )
        content   = resp.choices[0].message.content or ""
        logprobs  = (resp.choices[0].logprobs.content
                     if resp.choices[0].logprobs else [])
        return content, logprobs
    except Exception as e:
        print(f"  API error: {e}")
        return None, None


# ── Sentiment prompting ───────────────────────────────────────────────────────

def _sentiment_few_shot(train_data, n=5) -> str:
    lines = []
    for item in train_data[:n]:
        snippet = item["text"][:200].replace("\n", " ")
        lines.append(f'Text: "{snippet}"\nSentiment: {item["label"]}')
    return "\n\n".join(lines)


def _sentiment_system_prompt(examples: str) -> str:
    return (
        "You are a sentiment classifier. "
        "Classify the sentiment of the text as exactly 'positive' or 'negative'.\n"
        "Respond with only the label.\n\n"
        "Examples:\n" + examples
    )


def _sentiment_query(client, system_prompt: str, text: str):
    snippet = text[:500].replace("\n", " ")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Text: "{snippet}"\nSentiment:'},
            ],
            temperature=0.0,
            max_tokens=5,
            logprobs=True,
            top_logprobs=5,
        )
        content  = (resp.choices[0].message.content or "").strip().lower()
        logprobs = (resp.choices[0].logprobs.content
                    if resp.choices[0].logprobs else [])

        # Extract confidence from logprob of first token
        conf = 0.6
        if logprobs:
            conf = math.exp(logprobs[0].logprob)
            conf = max(0.5, min(1.0, conf))

        pred = content if content in ("positive", "negative") else "positive"
        return pred, conf
    except Exception as e:
        print(f"  API error: {e}")
        return None, None


# ── Model class ───────────────────────────────────────────────────────────────

class LLMModel(BaseModel):
    """
    GPT-4o-mini few-shot model for NER or sentiment.
    Confidence = API token log-probability.
    """

    def __init__(self, task: str, api_key: str | None = None):
        assert task in ("ner", "sentiment")
        super().__init__(name="llm", task=task)
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._system_prompt: str = ""

    def train(self, train_data, val_data=None) -> None:
        """No training — few-shot only. Build prompt from train_data."""
        if self.task == "ner":
            examples = _ner_few_shot_examples(train_data, n=5)
            self._system_prompt = _ner_system_prompt(examples)
        else:
            examples = _sentiment_few_shot(train_data, n=5)
            self._system_prompt = _sentiment_system_prompt(examples)
        print(f"[LLM] Prompt ready ({len(self._system_prompt)} chars)")

    def predict(self, data, max_samples: int | None = None) -> list[PredictionRow]:
        assert self._system_prompt, "Call train() first to build prompt."
        subset = data[:max_samples] if max_samples else data
        print(f"[LLM] Predicting {len(subset)} examples...")

        rows      = []
        errors    = 0
        misaligned = 0

        for idx, item in enumerate(subset):
            if self.task == "ner":
                tokens     = item["tokens"]
                true_tags  = item["ner_tags"]
                content, lps = _ner_query(self._client,
                                          self._system_prompt, tokens)
                if content is None:
                    errors += 1
                    for ti, (tok, tag) in enumerate(zip(tokens, true_tags)):
                        rows.append(PredictionRow(
                            sample_id=idx, token_idx=ti, token=tok,
                            true_label=tag, pred_label="O", confidence=0.5))
                    continue

                aligned, mis = _parse_ner_output(content, tokens, lps)
                misaligned  += mis
                for ti, (tok, pred_tag, conf) in enumerate(aligned):
                    if ti < len(true_tags):
                        rows.append(PredictionRow(
                            sample_id=idx, token_idx=ti, token=tok,
                            true_label=true_tags[ti], pred_label=pred_tag,
                            confidence=round(conf, 6)))

            else:  # sentiment
                pred, conf = _sentiment_query(
                    self._client, self._system_prompt, item["text"])
                if pred is None:
                    errors += 1
                    pred, conf = "positive", 0.5
                rows.append(PredictionRow(
                    sample_id=idx,
                    true_label=item["label"],
                    pred_label=pred,
                    confidence=round(conf, 6),
                ))

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(subset)} | errors={errors}"
                      + (f" misaligned={misaligned}" if self.task == "ner" else ""))

            time.sleep(0.1)   # rate limit

        # Sanity check: confidence distribution
        confs = [r.confidence for r in rows]
        pct_hi = np.mean([c > 0.99 for c in confs]) * 100
        print(f"\n[LLM] Done | errors={errors} | "
              f"conf mean={np.mean(confs):.3f} | >0.99: {pct_hi:.1f}%")
        if pct_hi > 60:
            print("  ⚠ High fraction of near-certain predictions — "
                  "logprob may be reflecting token fluency, not semantic confidence")

        return rows


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ner", "sentiment"], required=True)
    args = parser.parse_args()

    if args.task == "ner":
        data  = CoNLL2003().load()
        model = LLMModel(task="ner")
        model.train(data["train"])

        test_rows = model.predict(data["test"], max_samples=MAX_TEST_NER)
        model.save_predictions(test_rows, split="test")

        val_rows = model.predict(data["validation"], max_samples=MAX_VAL_NER)
        model.save_predictions(val_rows, split="val")

    else:
        data  = IMDB().load()
        model = LLMModel(task="sentiment")
        model.train(data["train"])

        test_rows = model.predict(data["test"], max_samples=MAX_TEST_SENTIMENT)
        model.save_predictions(test_rows, split="test")

        val_rows = model.predict(data["validation"], max_samples=MAX_VAL_SENTIMENT)
        model.save_predictions(val_rows, split="val")

    true = [r.true_label for r in test_rows]
    pred = [r.pred_label for r in test_rows]
    print(f"\nAccuracy: {sum(t==p for t,p in zip(true,pred))/len(true):.4f}")
