"""
src/calibration/viz.py
All visualization functions. Each saves a PNG and returns nothing.

Bin data is passed as plain dicts:
  {"bin_lower", "bin_upper", "avg_conf", "avg_acc", "count", "gap"}
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = "outputs/analysis/figures"

MODEL_COLORS = {
    "crf":          "#2ECC71",
    "bert":         "#3498DB",
    "bert_crf":     "#9B59B6",
    "tfidf_logreg": "#F39C12",
    "llm":          "#E74C3C",
}
DISPLAY_NAMES = {
    "crf":          "CRF",
    "bert":         "BERT",
    "bert_crf":     "BERT-CRF *",
    "tfidf_logreg": "TF-IDF+LR",
    "llm":          "LLM (GPT-4o-mini)",
}


def _model_key(name: str) -> str:
    n = name.lower()
    for s in ["_predictions", "_ner", "_sentiment", "_val", "_test"]:
        n = n.replace(s, "")
    if "bert_crf" in n: return "bert_crf"
    if "bert"     in n: return "bert"
    if "crf"      in n: return "crf"
    if "tfidf"    in n or "logreg" in n: return "tfidf_logreg"
    if "llm"      in n: return "llm"
    return n


# -- Reliability diagram -------------------------------------------------------

def plot_reliability(
    bins:       list[dict],
    title:      str,
    save_path:  str,
    ece:        float | None = None,
    brier:      float | None = None,
    entity_ece: float | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    non_empty = [b for b in bins if b["count"] > 0]
    if not non_empty:
        return

    fig, (ax, ax_cnt) = plt.subplots(
        2, 1, figsize=(7, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    mids   = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in non_empty]
    widths = [b["bin_upper"] - b["bin_lower"]        for b in non_empty]
    accs   = [b["avg_acc"]  for b in non_empty]
    conts  = [b["count"]    for b in non_empty]
    x_min  = np.floor(min(b["bin_lower"] for b in non_empty) * 10) / 10

    ax.bar(mids, accs, width=widths, color="#2C3E50", alpha=0.85,
           edgecolor="white", linewidth=1.2, label="Empirical Accuracy", zorder=2)
    ax.plot([x_min, 1.0], [x_min, 1.0], "--", color="#E74C3C",
            linewidth=2.0, label="Perfect Calibration", zorder=3)
    ax.set_xlim(x_min, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    lines = []
    if ece        is not None: lines.append(f"ECE:        {ece:.4f}")
    if entity_ece is not None: lines.append(f"ECE (ent):  {entity_ece:.4f}")
    if brier      is not None: lines.append(f"Brier:      {brier:.4f}")
    if lines:
        side = "right" if (mids and mids[-1] < 0.8) else "left"
        ax.text(0.97 if side == "right" else 0.03, 0.04,
                "\n".join(lines), transform=ax.transAxes,
                fontsize=9, va="bottom",
                ha="right" if side == "right" else "left",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white", edgecolor="#CCC", alpha=0.92))

    ax_cnt.bar(mids, conts, width=widths, color="#7F8C8D",
               alpha=0.7, edgecolor="white")
    ax_cnt.set_yscale("log")
    ax_cnt.set_xlim(x_min, 1.0)
    ax_cnt.set_xlabel("Predicted Confidence", fontsize=11)
    ax_cnt.set_ylabel("Count", fontsize=9)
    ax_cnt.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {save_path}")


# -- ECE bar chart -------------------------------------------------------------

def plot_ece_bars(results: list[dict], task: str, save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    data = sorted([r for r in results if r["task"] == task],
                  key=lambda r: r["ece"], reverse=True)
    if not data:
        return

    labels = [DISPLAY_NAMES.get(_model_key(r["model"]), r["model"]) for r in data]
    before = [r["ece"]                                for r in data]
    after  = [r.get("ece_after_scaling") or r["ece"] for r in data]
    colors = [MODEL_COLORS.get(_model_key(r["model"]), "#95A5A6") for r in data]
    x = np.arange(len(labels)); w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, before, w, color=colors, alpha=0.9, label="Before T-Scaling")
    b2 = ax.bar(x + w/2, after,  w, color=colors, alpha=0.4,
                edgecolor=colors, linewidth=1.5, label="After T-Scaling")
    for bars, vals in [(b1, before), (b2, after)]:
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_height() + max(vals)*0.02,
                    f"{v:.4f}", ha="center", fontsize=8)

    task_label = "NER (CoNLL-2003)" if task == "ner" else "Sentiment (IMDB)"
    ax.set_title(f"ECE Before / After Temperature Scaling -- {task_label}",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("ECE (lower = better)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, max(before) * 1.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {save_path}")


# -- Entity-only vs all-token --------------------------------------------------

def plot_entity_vs_all(results: list[dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    data = [r for r in results
            if r["task"] == "ner" and r.get("entity_ece") is not None]
    if not data:
        print("No entity_ece data yet -- skipping entity plot.")
        return
    data = sorted(data, key=lambda r: r["ece"], reverse=True)

    labels     = [DISPLAY_NAMES.get(_model_key(r["model"]), r["model"]) for r in data]
    ece_all    = [r["ece"]        for r in data]
    ece_entity = [r["entity_ece"] for r in data]
    colors     = [MODEL_COLORS.get(_model_key(r["model"]), "#95A5A6") for r in data]
    x = np.arange(len(labels)); w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, ece_all,    w, color=colors, alpha=0.9, label="All tokens (incl. O)")
    ax.bar(x + w/2, ece_entity, w, color=colors, alpha=0.4,
           edgecolor=colors, linewidth=1.5, label="Entity tokens only (excl. O)")

    for i, (ea, ee) in enumerate(zip(ece_all, ece_entity)):
        if ee > ea * 1.05:
            ax.annotate(f"+{(ee-ea):.4f}", xy=(x[i]+w/2, ee),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=8, color="darkred", fontweight="bold")

    ax.set_title("All-token ECE vs Entity-only ECE\n"
                 "O-class (~80%) artificially improves overall ECE",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("ECE (lower = better)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {save_path}")


# -- Theory 1 -----------------------------------------------------------------

def plot_theory1(results: list[dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    ner      = [r for r in results if r["task"] == "ner"]
    crf      = next((r for r in ner if _model_key(r["model"]) == "crf"),      None)
    bert     = next((r for r in ner if _model_key(r["model"]) == "bert"),     None)
    bert_crf = next((r for r in ner if _model_key(r["model"]) == "bert_crf"), None)

    if bert_crf is None:
        print("No BERT-CRF results yet. Run src/models/bert_crf_ner.py first.")
        return

    models = [m for m in [crf, bert, bert_crf] if m is not None]
    labels = [DISPLAY_NAMES.get(_model_key(r["model"]), r["model"]) for r in models]
    colors = [MODEL_COLORS.get(_model_key(r["model"]), "#95A5A6")   for r in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Theory 1: Do Structured Marginals Improve Calibration?\n"
                 "(BERT encoder fixed -- only output layer changes)",
                 fontsize=13, fontweight="bold")

    ece_all = [r["ece"]                for r in models]
    ece_ent = [r.get("entity_ece") or 0 for r in models]
    x = np.arange(len(labels)); w = 0.35

    ax1.bar(x - w/2, ece_all, w, color=colors, alpha=0.9, label="All tokens")
    ax1.bar(x + w/2, ece_ent, w, color=colors, alpha=0.4,
            edgecolor=colors, linewidth=1.5, label="Entity only")
    for i, (ea, ee) in enumerate(zip(ece_all, ece_ent)):
        ax1.text(x[i]-w/2, ea+0.0005, f"{ea:.4f}", ha="center", fontsize=9, fontweight="bold")
        ax1.text(x[i]+w/2, ee+0.0005, f"{ee:.4f}", ha="center", fontsize=9)
    ax1.set_title("ECE Comparison")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("ECE (lower = better)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.2, axis="y")

    for r, lbl, col in zip(models, labels, colors):
        ax2.scatter(r["accuracy"], r["ece"], s=220, color=col, zorder=3,
                    edgecolors="black", linewidth=0.8)
        ax2.annotate(lbl, (r["accuracy"], r["ece"]),
                     xytext=(8, 4), textcoords="offset points", fontsize=10)
    ax2.set_xlabel("Accuracy ^", fontsize=11)
    ax2.set_ylabel("ECE v (better calibrated)", fontsize=11)
    ax2.set_title("Accuracy vs Calibration Tradeoff")
    ax2.grid(True, alpha=0.3)

    if bert is not None and bert_crf is not None:
        delta   = bert["ece"] - bert_crf["ece"]
        verdict = "OK Supports Theory 1" if delta > 0.001 else "X Theory 1 not supported"
        ax2.text(0.03, 0.97,
                 f"BERT-CRF vs BERT: DeltaECE = {delta:+.4f}\n{verdict}",
                 transform=ax2.transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round", facecolor="lightyellow",
                           edgecolor="orange", alpha=0.95))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {save_path}")
