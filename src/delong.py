"""DeLong test utilities for paired ROC AUC comparison."""

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for DeLong's ROC AUC variance estimator."""
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """Fast DeLong implementation for one or more correlated ROC curves."""
    predictions_sorted_transposed = np.asarray(predictions_sorted_transposed, dtype=float)
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m

    if m <= 0 or n <= 0:
        raise ValueError("Both positive and negative classes are required for DeLong test.")

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def paired_delong_test(y_true, prob_model_1, prob_model_2) -> dict:
    """Compare two correlated AUCs evaluated on the same cases."""
    y_true = np.asarray(y_true).astype(int)
    prob_model_1 = np.asarray(prob_model_1, dtype=float)
    prob_model_2 = np.asarray(prob_model_2, dtype=float)

    if not (len(y_true) == len(prob_model_1) == len(prob_model_2)):
        raise ValueError("y_true and prediction arrays must have the same length.")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("DeLong test requires binary labels encoded as 0 and 1.")

    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    predictions = np.vstack([prob_model_1[order], prob_model_2[order]])

    aucs, cov = fast_delong(predictions, int(np.sum(y_sorted)))
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])

    if var <= 0:
        z_score = np.nan
        p_value = np.nan
    else:
        z_score = abs(diff) / np.sqrt(var)
        p_value = 2.0 * (1.0 - stats.norm.cdf(z_score))

    return {
        "auc_model_1": float(aucs[0]),
        "auc_model_2": float(aucs[1]),
        "auc_difference_model_1_minus_model_2": diff,
        "z_score": float(z_score) if np.isfinite(z_score) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
    }
