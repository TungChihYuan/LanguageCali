"""
src/calibration/scaling.py
Post-hoc calibration methods.

Currently implemented:
  - Temperature Scaling (Guo et al., ICML 2017)

Design: each method is a standalone function with the same signature:
    fit_and_apply(val_conf, val_corr, test_conf) → (param, scaled_conf)

To add a new calibration method (e.g., Platt Scaling, Isotonic Regression):
  1. Write a new function with the same signature
  2. Register it in SCALING_METHODS dict at the bottom

The runner (experiments/run_calibration.py) iterates SCALING_METHODS
and reports results for each.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid   # numerically stable sigmoid


# ── Temperature Scaling ───────────────────────────────────────────────────────

def temperature_scaling(
    val_conf: list[float] | np.ndarray,
    val_corr: list[int]   | np.ndarray,
    test_conf: list[float] | np.ndarray,
    T_bounds: tuple[float, float] = (0.05, 20.0),
) -> tuple[float, np.ndarray]:
    """
    Find optimal temperature T on validation set, apply to test set.

    CORRECT implementation: scaling in logit space.

        logit(p)  = log(p / (1-p))
        scaled(p) = sigmoid(logit(p) / T)

    Why logit space?
        Temperature scaling was defined as dividing logits by T
        before softmax (Guo et al. 2017). For binary predictions,
        this is equivalent to the logit-space formula above.
        The original code used p^(1/T) / (p^(1/T) + (1-p)^(1/T))
        which is a different, incorrect approximation.

    Args:
        val_conf:  validation confidences for fitting T
        val_corr:  validation correctness labels (0 or 1)
        test_conf: test confidences to rescale
        T_bounds:  search range for T

    Returns:
        best_T:       optimal temperature (T>1 overconfident, T<1 underconfident)
        scaled_conf:  rescaled test confidences
    """
    EPS = 1e-7

    val_c  = np.clip(np.asarray(val_conf,  dtype=float), EPS, 1 - EPS)
    val_y  = np.asarray(val_corr,  dtype=float)
    test_c = np.clip(np.asarray(test_conf, dtype=float), EPS, 1 - EPS)

    # Convert to logit space
    val_logits  = np.log(val_c  / (1 - val_c))
    test_logits = np.log(test_c / (1 - test_c))

    def neg_log_likelihood(T: float) -> float:
        scaled = sigmoid(val_logits / T)
        scaled = np.clip(scaled, EPS, 1 - EPS)
        return -float(np.mean(
            val_y * np.log(scaled) + (1 - val_y) * np.log(1 - scaled)
        ))

    result = minimize_scalar(neg_log_likelihood,
                             bounds=T_bounds, method="bounded")
    best_T = float(result.x)

    scaled_test = sigmoid(test_logits / best_T)
    return best_T, scaled_test


# ── Registry ──────────────────────────────────────────────────────────────────
# Add new calibration methods here.
# Each value must be a callable with signature:
#   fn(val_conf, val_corr, test_conf) → (param, scaled_conf)

SCALING_METHODS: dict[str, callable] = {
    "temperature": temperature_scaling,
    # "platt":     platt_scaling,       # future
    # "isotonic":  isotonic_scaling,    # future
    # "structured_ts": structured_temperature_scaling,  # Theory 1 variant
}
