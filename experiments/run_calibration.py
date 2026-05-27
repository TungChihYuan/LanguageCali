"""
experiments/run_calibration.py
Main calibration analysis pipeline.

For each prediction CSV in outputs/predictions/:
  1. ECE (all / entity-only / per-type)
  2. Brier Score
  3. Temperature Scaling (logit space, correct implementation)
  4. Save results JSON + reliability diagram PNGs

Usage:
    python experiments/run_calibration.py              # all files
    python experiments/run_calibration.py path/to.csv  # single file
"""

import os
import sys
import json
import numpy as np

from src.utils.io          import load_predictions, find_prediction_files
from src.calibration.metrics import full_metrics, compute_ece
from src.calibration.scaling import SCALING_METHODS
from src.calibration.viz     import plot_reliability

ANALYSIS_DIR = "outputs/analysis"
FIGURES_DIR  = os.path.join(ANALYSIS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def analyze(filepath: str) -> dict:
    basename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n{'='*60}\n  {basename}\n{'='*60}")

    rows, task = load_predictions(filepath)
    print(f"  Task: {task} | n={len(rows):,}")

    # ── Base metrics ──────────────────────────────────────────────────────
    results = {"model": basename, "task": task}
    results.update(full_metrics(rows, task))

    conf = [float(r["confidence"]) for r in rows]
    corr = [1 if r["true_label"] == r["pred_label"] else 0 for r in rows]

    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  ECE:       {results['ece']:.4f}")
    print(f"  AdaECE:    {results['ada_ece']:.4f}")
    if results.get("entity_ece") is not None:
        n_entity = sum(1 for r in rows if r.get("true_label", "O") != "O")
        print(f"  ECE(ent):  {results['entity_ece']:.4f}  [n={n_entity:,}]")

    pct_hi = np.mean([c > 0.99 for c in conf]) * 100
    if pct_hi > 50:
        print(f"  ⚠ {pct_hi:.1f}% of predictions have conf > 0.99 — likely overconfident")

    if task == "ner" and results.get("per_type_ece"):
        print("  Per-type ECE:")
        for et, v in results["per_type_ece"].items():
            print(f"    {et:<6}  ECE={v['ece']:.4f}  Acc={v['accuracy']:.4f}  n={v['n']}")

    # ── Reliability diagram ───────────────────────────────────────────────
    plot_reliability(
        bins       = results["_bin_data"],
        title      = f"Reliability: {basename}",
        save_path  = os.path.join(FIGURES_DIR, f"reliability_{basename}.png"),
        ece        = results["ece"],
        brier      = results["brier"]["brier"],
        entity_ece = results.get("entity_ece"),
    )

    # ── Temperature Scaling ───────────────────────────────────────────────
    # Try several naming conventions for val file
    val_path = None
    for candidate in [
        filepath.replace("_test.csv", "_val.csv"),
        filepath.replace("predictions_test.csv", "predictions_val.csv"),
        filepath.replace(".csv", "_val.csv"),
    ]:
        if os.path.exists(candidate):
            val_path = candidate
            break

    results["scaling"]          = {}
    results["temperature"]      = None
    results["ece_after_scaling"] = None

    if val_path:
        val_rows, _ = load_predictions(val_path)
        val_conf = [float(r["confidence"]) for r in val_rows]
        val_corr = [1 if r["true_label"] == r["pred_label"] else 0
                    for r in val_rows]
        print(f"\n  Temperature Scaling (val n={len(val_rows):,})")

        for method_name, scale_fn in SCALING_METHODS.items():
            param, scaled = scale_fn(val_conf, val_corr, conf)
            ece_s, bins_s = compute_ece(list(scaled), corr)
            reduction     = (results["ece"] - ece_s) / results["ece"] * 100

            direction = "overconfident" if param > 1 else "underconfident"
            print(f"    [{method_name}] T={param:.3f} ({direction}) | "
                  f"ECE: {results['ece']:.4f} → {ece_s:.4f} ({reduction:.1f}%↓)")

            plot_reliability(
                bins      = bins_s,
                title     = f"After {method_name} (T={param:.2f}): {basename}",
                save_path = os.path.join(
                    FIGURES_DIR, f"reliability_{basename}_{method_name}.png"),
                ece       = ece_s,
            )

            results["scaling"][method_name] = {
                "param": param, "ece": ece_s, "reduction_pct": reduction,
            }

        ts = results["scaling"].get("temperature", {})
        results["temperature"]       = ts.get("param")
        results["ece_after_scaling"] = ts.get("ece")
    else:
        print(f"  No val file found — skipping Temperature Scaling.")

    # ── Save JSON ─────────────────────────────────────────────────────────
    save_r = {k: v for k, v in results.items() if k != "_bin_data"}
    out    = os.path.join(ANALYSIS_DIR, f"results_{basename}.json")
    with open(out, "w") as f:
        json.dump(save_r, f, indent=2)
    print(f"\n  Saved → {out}")
    return results


def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*100}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*100}")
    hdr = (f"{'Model':<32} {'Task':<11} {'Acc':>6} "
           f"{'ECE':>7} {'ECE(ent)':>9} {'AdaECE':>8} "
           f"{'Brier':>7} {'T':>6} {'ECE(T)':>8}")
    print(hdr); print("-" * 100)

    order = {"ner": 0, "sentiment": 1}
    prev  = None
    for r in sorted(results, key=lambda x: (order.get(x["task"], 2), x["ece"])):
        if prev and r["task"] != prev: print()
        prev  = r["task"]
        T     = f"{r['temperature']:.2f}"       if r.get("temperature")       else "  —"
        ece_t = f"{r['ece_after_scaling']:.4f}" if r.get("ece_after_scaling") else "   —"
        ent   = f"{r['entity_ece']:.4f}"        if r.get("entity_ece")        else "   —"
        print(f"{r['model']:<32} {r['task']:<11} {r['accuracy']:>6.4f} "
              f"{r['ece']:>7.4f} {ent:>9} {r['ada_ece']:>8.4f} "
              f"{r['brier']['brier']:>7.4f} {T:>6} {ece_t:>8}")
    print(f"\nFigures → {FIGURES_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        all_results = [analyze(sys.argv[1])]
    else:
        files = find_prediction_files()
        if not files:
            print("No prediction CSVs in outputs/predictions/")
            print("Run: bash scripts/train_all.sh")
            sys.exit(1)
        print(f"Found {len(files)} prediction files.")
        all_results = [analyze(fp) for fp in files]
    print_summary(all_results)
