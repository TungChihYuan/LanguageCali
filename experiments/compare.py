"""
experiments/compare.py
Load all analysis JSONs and generate comparison figures + LaTeX tables.

Figures produced:
  1. ECE before/after TS — NER
  2. ECE before/after TS — Sentiment
  3. Entity-only vs all-token ECE
  4. Theory 1: BERT vs BERT-CRF

Usage:
    python experiments/compare.py
"""

import os
import json
import numpy as np

from src.calibration.viz import (
    plot_ece_bars,
    plot_entity_vs_all,
    plot_theory1,
    DISPLAY_NAMES,
    _model_key,
)

ANALYSIS_DIR = "outputs/analysis"
FIGURES_DIR  = os.path.join(ANALYSIS_DIR, "figures")


def load_all() -> list[dict]:
    results = []
    for fname in sorted(os.listdir(ANALYSIS_DIR)):
        if fname.startswith("results_") and fname.endswith(".json"):
            with open(os.path.join(ANALYSIS_DIR, fname)) as f:
                results.append(json.load(f))
    return results


def print_main_table(results: list[dict]) -> None:
    print(f"\n{'='*105}")
    print("MAIN RESULTS")
    print(f"{'='*105}")
    hdr = (f"{'Model':<30} {'Task':<11} {'Acc':>6} "
           f"{'ECE':>7} {'ECE(ent)':>9} {'AdaECE':>8} "
           f"{'Brier':>7} {'T':>6} {'ECE(T)':>8}")
    print(hdr)
    print("-" * 105)

    order = {"ner": 0, "sentiment": 1}
    prev  = None
    for r in sorted(results, key=lambda x: (order.get(x["task"], 2), x["ece"])):
        if prev and r["task"] != prev: print()
        prev  = r["task"]
        name  = DISPLAY_NAMES.get(_model_key(r["model"]), r["model"])
        T     = f"{r.get('temperature', None):.2f}" if r.get("temperature") else "  —"
        ece_t = f"{r.get('ece_after_scaling'):.4f}" if r.get("ece_after_scaling") else "   —"
        ent   = f"{r.get('entity_ece'):.4f}"        if r.get("entity_ece")        else "   —"
        print(f"{name:<30} {r['task']:<11} {r['accuracy']:>6.4f} "
              f"{r['ece']:>7.4f} {ent:>9} {r['ada_ece']:>8.4f} "
              f"{r['brier']['brier']:>7.4f} {T:>6} {ece_t:>8}")


def print_latex(results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("LATEX TABLE")
    print(f"{'='*60}")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Calibration results. ECE$_\text{ent}$ excludes O tokens.}")
    print(r"\small")
    print(r"\begin{tabular}{llccccccc}")
    print(r"\toprule")
    print(r"Model & Task & Acc & ECE$\downarrow$ & ECE$_\text{ent}\downarrow$"
          r" & AdaECE$\downarrow$ & Brier$\downarrow$ & $T$ & ECE$(T)\downarrow$ \\")
    print(r"\midrule")

    order = {"ner": 0, "sentiment": 1}
    prev  = None
    for r in sorted(results, key=lambda x: (order.get(x["task"], 2), x["ece"])):
        if prev and r["task"] != prev: print(r"\midrule")
        prev  = r["task"]
        name  = DISPLAY_NAMES.get(_model_key(r["model"]), r["model"])
        name  = name.replace("★", r"$\star$").replace("_", r"\_")
        T     = f"{r.get('temperature'):.2f}" if r.get("temperature") else "--"
        ece_t = f"{r.get('ece_after_scaling'):.4f}" if r.get("ece_after_scaling") else "--"
        ent   = f"{r.get('entity_ece'):.4f}" if r.get("entity_ece") else "--"
        print(f"{name} & {r['task']} & {r['accuracy']:.4f} & "
              f"{r['ece']:.4f} & {ent} & {r['ada_ece']:.4f} & "
              f"{r['brier']['brier']:.4f} & {T} & {ece_t} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    results = load_all()
    if not results:
        print("No results found. Run experiments/run_calibration.py first.")
        raise SystemExit(1)

    print(f"Loaded {len(results)} result files.")

    print_main_table(results)
    print_latex(results)

    plot_ece_bars(results, "ner",
                  os.path.join(FIGURES_DIR, "ece_ner.png"))
    plot_ece_bars(results, "sentiment",
                  os.path.join(FIGURES_DIR, "ece_sentiment.png"))
    plot_entity_vs_all(results,
                       os.path.join(FIGURES_DIR, "entity_vs_all_ece.png"))
    plot_theory1(results,
                 os.path.join(FIGURES_DIR, "theory1_bert_vs_bert_crf.png"))

    print(f"\nAll figures → {FIGURES_DIR}/")
