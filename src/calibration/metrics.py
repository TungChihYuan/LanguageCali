"""
src/calibration/metrics.py
All calibration metrics.

Three ECE variants for NER:
  compute_ece()          — all tokens (standard)
  compute_entity_ece()   — entity tokens only, excluding O
  compute_per_type_ece() — per entity type (PER/ORG/LOC/MISC)

Bin data is returned as plain dicts for JSON compatibility:
  {"bin_lower", "bin_upper", "avg_conf", "avg_acc", "count", "gap"}
"""

from __future__ import annotations
import numpy as np


# ── ECE ───────────────────────────────────────────────────────────────────────

def compute_ece(
    confidences: list[float] | np.ndarray,
    correctness: list[int]   | np.ndarray,
    n_bins:   int = 15,
    strategy: str = "uniform",
) -> tuple[float, list[dict]]:
    """
    Expected Calibration Error.

    Returns:
        ece:      scalar ECE value
        bin_data: list of bin dicts for plotting
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=float)
    n    = len(conf)
    assert n > 0, "Empty input"

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges   = np.unique(np.percentile(conf, np.linspace(0, 100, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
        n_bins  = len(edges) - 1

    bins = []
    ece  = 0.0

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask   = (conf >= lo) & (conf <= hi) if i == n_bins - 1 \
                 else (conf >= lo) & (conf < hi)
        cnt    = int(mask.sum())

        if cnt == 0:
            bins.append({"bin_lower": float(lo), "bin_upper": float(hi),
                         "avg_conf": 0.0, "avg_acc": 0.0,
                         "count": 0, "gap": 0.0})
            continue

        bin_conf = float(conf[mask].mean())
        bin_acc  = float(corr[mask].mean())
        gap      = abs(bin_acc - bin_conf)
        ece     += (cnt / n) * gap

        bins.append({"bin_lower": float(lo), "bin_upper": float(hi),
                     "avg_conf": bin_conf, "avg_acc": bin_acc,
                     "count": cnt, "gap": float(gap)})

    return float(ece), bins


# ── NER-specific ECE variants ─────────────────────────────────────────────────

def compute_entity_ece(
    rows:   list[dict],
    n_bins: int = 15,
) -> float | None:
    """ECE on entity tokens only (B-* and I-*), excluding O."""
    entity = [r for r in rows if r.get("true_label", "O") != "O"]
    if not entity:
        return None
    conf = [float(r["confidence"]) for r in entity]
    corr = [1 if r["true_label"] == r["pred_label"] else 0 for r in entity]
    ece, _ = compute_ece(conf, corr, n_bins=n_bins)
    return ece


def compute_per_type_ece(rows: list[dict]) -> dict[str, dict]:
    """ECE per entity type: PER, ORG, LOC, MISC."""
    types: set[str] = set()
    for r in rows:
        lbl = r.get("true_label", "O")
        if lbl != "O" and "-" in lbl:
            types.add(lbl.split("-", 1)[1])

    out = {}
    for et in sorted(types):
        subset = [r for r in rows
                  if r.get("true_label", "O") != "O"
                  and r.get("true_label", "").endswith(et)]
        if len(subset) < 5:
            continue
        conf = [float(r["confidence"]) for r in subset]
        corr = [1 if r["true_label"] == r["pred_label"] else 0 for r in subset]
        ece, _ = compute_ece(conf, corr, n_bins=10)
        out[et] = {
            "ece":            float(ece),
            "accuracy":       float(np.mean(corr)),
            "mean_confidence":float(np.mean(conf)),
            "n":              len(subset),
        }
    return out


# ── Brier Score ───────────────────────────────────────────────────────────────

def compute_brier(
    confidences: list[float] | np.ndarray,
    correctness: list[int]   | np.ndarray,
) -> dict:
    """Brier Score with Murphy decomposition."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correctness, dtype=float)

    brier = float(np.mean((c - y) ** 2))
    y_bar = float(y.mean())
    uncertainty = float(y_bar * (1.0 - y_bar))

    edges = np.linspace(0.0, 1.0, 11)
    reliability = resolution = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        mask   = (c >= lo) & (c <= hi) if i == 9 else (c >= lo) & (c < hi)
        cnt    = mask.sum()
        if cnt == 0: continue
        w = cnt / len(c)
        reliability += w * (float(c[mask].mean()) - float(y[mask].mean())) ** 2
        resolution  += w * (float(y[mask].mean()) - y_bar) ** 2

    return {
        "brier":       brier,
        "reliability": float(reliability),
        "resolution":  float(resolution),
        "uncertainty": float(uncertainty),
    }


# ── Convenience ───────────────────────────────────────────────────────────────

def full_metrics(rows: list[dict], task: str) -> dict:
    """All metrics for a prediction set. Returns flat dict (JSON-serializable)."""
    conf = [float(r["confidence"]) for r in rows]
    corr = [1 if r["true_label"] == r["pred_label"] else 0 for r in rows]

    ece,     bin_data = compute_ece(conf, corr, n_bins=15, strategy="uniform")
    ada_ece, _        = compute_ece(conf, corr, n_bins=15, strategy="quantile")
    brier             = compute_brier(conf, corr)

    entity_ece   = None
    per_type_ece = {}
    if task == "ner":
        entity_ece   = compute_entity_ece(rows)
        per_type_ece = compute_per_type_ece(rows)

    return {
        "n_samples":    len(rows),
        "accuracy":     float(np.mean(corr)),
        "ece":          ece,
        "ada_ece":      ada_ece,
        "entity_ece":   entity_ece,
        "per_type_ece": per_type_ece,
        "brier":        brier,
        "_bin_data":    bin_data,   # for reliability diagram (excluded from JSON save)
    }
