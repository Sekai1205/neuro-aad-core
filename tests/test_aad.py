"""Fast unit tests for the aad package (no dataset needed).  Run: pytest tests -q"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aad.features import lagged, lagged_by_trial
from aad import linear as lin
from aad.temporal import fit_llr_scale, hmm_forward, switch_latency


def test_lagged_shifts_and_zero_pads():
    x = np.arange(5.0)[:, None]
    out = lagged(x, np.array([0, 2, -1]))
    np.testing.assert_array_equal(out[:, 0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(out[:, 1], [2, 3, 4, 0, 0])
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1, 2, 3])


def test_lags_do_not_cross_trial_boundaries():
    x = np.arange(6.0)[:, None]
    out = lagged_by_trial(x, np.array([0, 0, 0, 1, 1, 1]), np.array([1]))
    np.testing.assert_array_equal(out[:, 0], [1, 2, 0, 4, 5, 0])


def test_cca_recovers_planted_response_and_discriminates():
    rng = np.random.default_rng(0)
    T, win = 20000, 640
    s_att, s_un = rng.standard_normal(T), rng.standard_normal(T)
    eeg = np.stack([np.roll(s_att, 5) + 2 * rng.standard_normal(T) for _ in range(4)], 1)
    X = lagged(eeg, np.arange(0, 10))
    Ya, Yu = s_att[:, None], s_un[:, None]
    cov = lin.covariances(X, Yu, Ya, np.ones(T, int))       # right = attended
    W = lin.fit_cca(cov, lin.CCAConfig(n_components=1))
    starts = np.arange(0, T - win, win)
    d = lin.window_features(X, Yu, Ya, W, starts, win)
    assert np.mean(d[:, 0] > 0) > 0.95


def test_hmm_accumulates_and_recovers_after_switch():
    llr = np.r_[np.full(10, 0.5), np.full(10, -0.5)]
    post = hmm_forward(llr, 0.05)
    assert post[9] > post[0] > 0
    assert post[-1] < 0
    y = np.r_[np.ones(10), np.zeros(10)].astype(int)
    assert 0 < switch_latency((post > 0).astype(int), y) < 10


def test_llr_scale_is_positive_for_informative_scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    s = (2 * y - 1) * 0.3 + rng.standard_normal(2000)
    assert fit_llr_scale(s, y) > 0
