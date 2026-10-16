"""
Linear stimulus-reconstruction AAD: cross-subject CCA + LDA.

  EEG (lagged)  --Wx-->  K canonical EEG components
  env (lagged)  --Wy-->  K canonical stimulus components

For a decision window, each component k gives a correlation with the left and the
right stream; d_k = rho_right,k - rho_left,k. An LDA trained on (nested) training-
subject windows turns d into a score s (> 0 -> right ear attended).

Covariances are sufficient statistics, so leave-one-subject-out models are just
averages of per-subject covariance matrices, and the 120 s calibration of the test
subject is blended in with weight `beta`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .features import FeatureConfig, SubjectStream, Standardizer, lagged_by_trial, lag_range

COV_KEYS = ("Cxx", "Cyy", "Cxy")


@dataclass(frozen=True)
class CCAConfig:
    n_components: int = 10
    lam_eeg: float = 0.1          # ridge, relative to mean eigenvalue
    lam_env: float = 0.01
    beta_calib: float = 0.1       # weight of the test subject's calibration covariance
    lda_shrink: float = 0.05


def env_standardizer(st: SubjectStream) -> Standardizer:
    # Stimulus-only statistics (no label information).
    return Standardizer.fit(np.concatenate([st.env_l, st.env_r]))


def design(st: SubjectStream, fcfg: FeatureConfig, eeg_std: Standardizer):
    estd = env_standardizer(st)
    lx, ly = lag_range(fcfg.eeg_lags_ms), lag_range(fcfg.env_lags_ms)
    X = lagged_by_trial(eeg_std(st.eeg), st.trial_id, lx)
    Yl = lagged_by_trial(estd(st.env_l), st.trial_id, ly)
    Yr = lagged_by_trial(estd(st.env_r), st.trial_id, ly)
    return X, Yl, Yr


def covariances(X, Yl, Yr, label, sl=slice(None)) -> dict:
    X, Yl, Yr, label = X[sl], Yl[sl], Yr[sl], label[sl]
    Ya = np.where(label[:, None] == 1, Yr, Yl).astype(np.float64)
    X = X.astype(np.float64)
    n = len(X)
    return dict(Cxx=X.T @ X / n, Cyy=Ya.T @ Ya / n, Cxy=X.T @ Ya / n)


def mean_cov(covs: list[dict]) -> dict:
    return {k: np.mean([c[k] for c in covs], axis=0) for k in COV_KEYS}


def blend(a: dict, b: dict, beta: float) -> dict:
    return {k: (1 - beta) * a[k] + beta * b[k] for k in COV_KEYS}


def _ridge(C: np.ndarray, lam: float) -> np.ndarray:
    return C + lam * np.trace(C) / len(C) * np.eye(len(C))


def fit_cca(cov: dict, cfg: CCAConfig):
    """Regularised CCA. Returns (Wx, Wy) with components sorted by canonical corr."""
    Lx = linalg.cholesky(_ridge(cov["Cxx"], cfg.lam_eeg), lower=True)
    Ly = linalg.cholesky(_ridge(cov["Cyy"], cfg.lam_env), lower=True)
    M = linalg.solve_triangular(
        Lx, linalg.solve_triangular(Ly, cov["Cxy"].T, lower=True).T, lower=True)
    U, _, Vt = linalg.svd(M, full_matrices=False)
    K = min(cfg.n_components, U.shape[1])
    Wx = linalg.solve_triangular(Lx.T, U[:, :K], lower=False)
    Wy = linalg.solve_triangular(Ly.T, Vt.T[:, :K], lower=False)
    return Wx.astype(np.float32), Wy.astype(np.float32)


def _window_corr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def window_features(X, Yl, Yr, W, starts: np.ndarray, win: int) -> np.ndarray:
    """(n_windows, K) correlation differences rho_right - rho_left."""
    Wx, Wy = W
    zx, zl, zr = X @ Wx, Yl @ Wy, Yr @ Wy
    seg = lambda z: np.stack([z[s:s + win] for s in starts])
    ex = seg(zx)
    return _window_corr(ex, seg(zr)) - _window_corr(ex, seg(zl))


@dataclass
class LDA:
    w: np.ndarray
    b: float

    @classmethod
    def fit(cls, F: np.ndarray, y: np.ndarray, shrink: float) -> "LDA":
        m1, m0 = F[y == 1].mean(0), F[y == 0].mean(0)
        R = np.concatenate([F[y == 1] - m1, F[y == 0] - m0])
        S = R.T @ R / len(R)
        S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S))
        w = linalg.solve(S, m1 - m0, assume_a="pos")
        return cls(w, float(-w @ (m1 + m0) / 2))

    def __call__(self, F: np.ndarray) -> np.ndarray:
        return F @ self.w + self.b
