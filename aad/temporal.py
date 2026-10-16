"""
Causal temporal integration of window decisions with a two-state HMM.

Attention is sustained for minutes, so per-window evidence can be accumulated.
Window scores are first calibrated into log-likelihood ratios (logistic fit on
training-subject windows), then a forward filter with switch probability p_switch
yields the posterior log-odds after each window. Only past windows are used, so
this runs online; the price is a delay after an attention switch (reported as
switch latency).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def fit_llr_scale(scores: np.ndarray, y: np.ndarray, iters: int = 50) -> float:
    """Slope a of p(right | s) = sigmoid(a * s + c); a * s is the calibrated LLR."""
    a, c = 1.0, 0.0
    for _ in range(iters):
        p = expit(a * scores + c)
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum((p - y) * scores), np.sum(p - y)])
        H = np.array([[np.sum(w * scores ** 2), np.sum(w * scores)],
                      [np.sum(w * scores), np.sum(w)]]) + 1e-6 * np.eye(2)
        a, c = np.array([a, c]) - np.linalg.solve(H, g)
    return float(a)


def hmm_forward(llr: np.ndarray, p_switch: float) -> np.ndarray:
    """Posterior log-odds (right vs left) after each window."""
    lp, lq = np.log(p_switch), np.log1p(-p_switch)
    L, out = 0.0, np.empty(len(llr))
    for t, e in enumerate(llr):
        L = np.logaddexp(lq + L, lp) - np.logaddexp(lp + L, lq) + e
        out[t] = L
    return out


def switch_latency(decisions: np.ndarray, y: np.ndarray) -> float:
    """Mean number of windows after a label change until the decision first matches."""
    changes = np.flatnonzero(np.diff(y)) + 1
    lat = []
    for i, c in enumerate(changes):
        end = changes[i + 1] if i + 1 < len(changes) else len(y)
        hit = np.flatnonzero(decisions[c:end] == y[c])
        lat.append(hit[0] if len(hit) else end - c)
    return float(np.mean(lat)) if lat else float("nan")
