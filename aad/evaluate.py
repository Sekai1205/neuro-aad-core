"""
Leave-one-subject-out evaluation with a short calibration recording.

Protocol (identical to the legacy benchmark so numbers are comparable):
  * test subject: trials concatenated in recording order; the first CALIB_S seconds
    are calibration (labels known), the rest is cut into non-overlapping WIN_S-second
    decision windows labelled by the attended ear at the window centre;
  * the other 15 subjects are training data;
  * every hyper-parameter / classifier fitted on training subjects only. The LDA and
    LLR calibration use *nested* scores: training subject j is scored by a model fitted
    without j (and without the test subject).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from joblib import Parallel, delayed

from .data import FS
from .features import FeatureConfig, SubjectStream, Standardizer
from . import linear as lin

CALIB_S = 120
WIN_S = 10
CALIB = CALIB_S * FS
WIN = WIN_S * FS


def window_starts(n: int, start: int, win: int = WIN) -> np.ndarray:
    return start + np.arange((n - start) // win) * win


def window_labels(st: SubjectStream, starts: np.ndarray, win: int = WIN) -> np.ndarray:
    return st.label[starts + win // 2]


# ---------------------------------------------------------------- linear CCA/LDA
def _subject_cov(st, fcfg):
    X, Yl, Yr = lin.design(st, fcfg, Standardizer.fit(st.eeg))
    return lin.covariances(X, Yl, Yr, st.label)


def _nested_features(j, st, fcfg, pair_models, subjects):
    """Window features of training subject j under every model that excludes j."""
    X, Yl, Yr = lin.design(st, fcfg, Standardizer.fit(st.eeg))
    starts = window_starts(len(st.label), 0)
    y = window_labels(st, starts)
    return j, {s: (lin.window_features(X, Yl, Yr, pair_models[frozenset((s, j))], starts, WIN), y)
               for s in subjects if s != j}


def _test_features(s, st, fcfg, ccfg, covs, subjects):
    X, Yl, Yr = lin.design(st, fcfg, Standardizer.fit(st.eeg[:CALIB]))
    cov_o = lin.mean_cov([covs[o] for o in subjects if o != s])
    cov_c = lin.covariances(X, Yl, Yr, st.label, slice(0, CALIB))
    starts = window_starts(len(st.label), CALIB)
    out = {"y": window_labels(st, starts)}
    for name, beta in (("cca_cross_subject", 0.0), ("cca_adapted", ccfg.beta_calib)):
        W = lin.fit_cca(lin.blend(cov_o, cov_c, beta), ccfg)
        out[name] = lin.window_features(X, Yl, Yr, W, starts, WIN)
    # calibration-only decoder (first canonical component, 120 s of data) as the reference
    out["cca_calib_only"] = lin.window_features(
        X, Yl, Yr, lin.fit_cca(cov_c, lin.CCAConfig(n_components=1, lam_eeg=ccfg.lam_eeg)), starts, WIN)
    return s, out


def linear_scores(streams: dict[int, SubjectStream], fcfg: FeatureConfig,
                  ccfg: lin.CCAConfig, n_jobs: int = 4) -> dict:
    """
    Returns {s: {"y": labels, model: test scores, "train": (nested scores, labels)}}.
    Scores are LDA outputs (> 0 -> right ear attended).
    """
    subjects = sorted(streams)
    covs = dict(zip(subjects, Parallel(n_jobs)(
        delayed(_subject_cov)(streams[s], fcfg) for s in subjects)))
    pair_models = {frozenset(p): lin.fit_cca(lin.mean_cov([covs[o] for o in subjects if o not in p]), ccfg)
                   for p in combinations(subjects, 2)}
    nested = dict(Parallel(n_jobs)(
        delayed(_nested_features)(j, streams[j], fcfg, pair_models, subjects) for j in subjects))
    tests = dict(Parallel(n_jobs)(
        delayed(_test_features)(s, streams[s], fcfg, ccfg, covs, subjects) for s in subjects))

    out = {}
    for s in subjects:
        F = np.concatenate([nested[j][s][0] for j in subjects if j != s])
        y = np.concatenate([nested[j][s][1] for j in subjects if j != s])
        lda = lin.LDA.fit(F, y, ccfg.lda_shrink)
        t = tests[s]
        out[s] = {
            "y": t["y"],
            "cca_cross_subject": lda(t["cca_cross_subject"]),
            "cca_adapted": lda(t["cca_adapted"]),
            "cca_calib_only": t["cca_calib_only"][:, 0],
            # per training subject, in time order (for HMM tuning / LLR calibration)
            "train": [(lda(nested[j][s][0]), nested[j][s][1]) for j in subjects if j != s],
            "train_ids": [j for j in subjects if j != s],
        }
    return out
