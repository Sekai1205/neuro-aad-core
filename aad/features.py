"""
Signal conditioning and lagged design matrices.

EEG side : 4 electrodes against the amplifier reference (the stored KU Leuven data
           is NOT average-referenced, so this is what a 4-electrode + reference
           device records) -> filterbank (delta + theta) -> per-subject z-scoring
           -> time lags.
Envelope : band-pass to the same range -> z-scoring -> time lags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .data import FS, CH_IDX_4, Trial


@dataclass(frozen=True)
class FeatureConfig:
    bands: tuple = ((0.5, 4.0), (4.0, 8.0))   # EEG filterbank (Hz)
    env_band: tuple = (0.5, 8.0)               # envelope band (Hz)
    eeg_lags_ms: tuple = (0, 400)              # EEG samples after the stimulus
    env_lags_ms: tuple = (-250, 0)             # stimulus history
    channels: tuple = CH_IDX_4
    reference: str = "hardware"                # 'hardware' | 'local' (mean of the 4)
    onset: bool = True                         # add half-wave rectified envelope derivative
    filter_order: int = 4


def lag_range(lags_ms: tuple) -> np.ndarray:
    lo, hi = (int(round(v * FS / 1000)) for v in lags_ms)
    return np.arange(lo, hi + 1)


def lagged(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """out[t, l*C + c] = x[t + lags[l], c] (zero outside the signal)."""
    if x.ndim == 1:
        x = x[:, None]
    T, C = x.shape
    out = np.zeros((T, len(lags) * C), dtype=np.float64)
    for i, lag in enumerate(lags):
        if lag >= 0:
            out[: T - lag, i * C:(i + 1) * C] = x[lag:]
        else:
            out[-lag:, i * C:(i + 1) * C] = x[: T + lag]
    return out


def _bandpass(x: np.ndarray, band: tuple, order: int) -> np.ndarray:
    sos = signal.butter(order, band, btype="band", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def eeg_channels(trial: Trial, cfg: FeatureConfig) -> np.ndarray:
    x = trial.eeg[:, list(cfg.channels)].astype(np.float64)
    if cfg.reference == "local":
        x = x - x.mean(axis=1, keepdims=True)
    return np.concatenate([_bandpass(x, b, cfg.filter_order) for b in cfg.bands], axis=1)


def envelope(env: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """(T, F) stimulus features: envelope [+ onset envelope], band-passed."""
    env = env.astype(np.float64)
    feats = [env]
    if cfg.onset:
        feats.append(np.maximum(np.diff(env, prepend=env[0]), 0.0))
    return _bandpass(np.stack(feats, axis=1), cfg.env_band, cfg.filter_order)


@dataclass
class SubjectStream:
    """A subject's trials, conditioned and concatenated in recording order."""
    eeg: np.ndarray        # (T, C)  un-normalised conditioned EEG
    env_l: np.ndarray      # (T, F)
    env_r: np.ndarray      # (T, F)
    label: np.ndarray      # (T,) 0 = left attended, 1 = right attended
    trial_id: np.ndarray   # (T,)

    @property
    def env_att(self) -> np.ndarray:
        return np.where(self.label[:, None] == 1, self.env_r, self.env_l)


def subject_stream(trials: list[Trial], cfg: FeatureConfig) -> SubjectStream:
    eeg, el, er, lab, tid = [], [], [], [], []
    for i, t in enumerate(trials):
        eeg.append(eeg_channels(t, cfg))
        el.append(envelope(t.env_left, cfg))
        er.append(envelope(t.env_right, cfg))
        lab.append(np.full(len(t.env_left), t.label))
        tid.append(np.full(len(t.env_left), i))
    return SubjectStream(np.concatenate(eeg), np.concatenate(el), np.concatenate(er),
                         np.concatenate(lab), np.concatenate(tid))


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        s = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(s < 1e-12, 1.0, s))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def lagged_by_trial(x: np.ndarray, trial_id: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Lag within each trial so that no lag reaches across a trial boundary."""
    out = np.empty((len(x), len(lags) * (x.shape[1] if x.ndim > 1 else 1)), dtype=np.float32)
    for t in np.unique(trial_id):
        m = trial_id == t
        out[m] = lagged(x[m], lags)
    return out
