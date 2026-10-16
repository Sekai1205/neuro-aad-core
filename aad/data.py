"""
KU Leuven AAD dataset: loading, envelope extraction and caching.

Each subject's .mat file is read once and reduced to a compact cache
(data/cache_v3/S{n}.npz) holding, per trial:
  - 64-channel EEG resampled 128 Hz -> FS (float32)
  - the left-ear and right-ear speech envelopes at FS
  - the attended ear (label) and trial metadata

Envelopes are always computed from the *dry* (non-HRTF) stimulus, as recommended
by the dataset authors, with a gammatone filterbank + power-law compression
(Biesmans et al. 2016).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.io
import soundfile as sf
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "KULeuven data set"
CACHE_DIR = PROJECT_ROOT / "data" / "cache_v3"

FS = 64                      # working sample rate (Hz)
FS_EEG_RAW = 128
N_SUBJECTS = 16

# BioSemi-64 indices (0-based) of the 4-electrode hardware montage
CH_NAMES_4 = ("TP7", "TP8", "Fz", "Cz")
CH_IDX_4 = (15, 52, 37, 47)

# Envelope extraction (Biesmans et al. 2016)
AUDIO_FS = 16000
GT_FMIN, GT_FMAX, GT_NBANDS = 150.0, 4000.0, 15
POWER_LAW = 0.6


@dataclass
class Trial:
    eeg: np.ndarray          # (T, 64) float32 at FS
    env_left: np.ndarray     # (T,) envelope of the left-ear stream
    env_right: np.ndarray    # (T,) envelope of the right-ear stream
    label: int               # 0 = attended left, 1 = attended right
    condition: str           # 'dry' | 'hrtf'
    experiment: int
    part: int
    repetition: bool

    @property
    def env_attended(self) -> np.ndarray:
        return self.env_right if self.label == 1 else self.env_left

    @property
    def env_unattended(self) -> np.ndarray:
        return self.env_left if self.label == 1 else self.env_right


def _erb_space(fmin: float, fmax: float, n: int) -> np.ndarray:
    """Centre frequencies equally spaced on the ERB-rate scale (Glasberg & Moore)."""
    erb = lambda f: 21.4 * np.log10(1 + 0.00437 * f)
    inv = lambda e: (10 ** (e / 21.4) - 1) / 0.00437
    return inv(np.linspace(erb(fmin), erb(fmax), n))


@lru_cache(maxsize=None)
def stimulus_envelope(filename: str) -> np.ndarray:
    """Gammatone-subband, power-law compressed broadband envelope at FS."""
    path = DATA_DIR / "stimuli" / filename
    audio, sr = sf.read(str(path), dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    g = np.gcd(int(sr), AUDIO_FS)
    audio = signal.resample_poly(audio, AUDIO_FS // g, int(sr) // g)

    env = np.zeros_like(audio)
    for fc in _erb_space(GT_FMIN, GT_FMAX, GT_NBANDS):
        b, a = signal.gammatone(fc, "iir", fs=AUDIO_FS)
        env += np.abs(signal.lfilter(b, a, audio)) ** POWER_LAW

    # AUDIO_FS -> FS with anti-alias filtering
    return signal.resample_poly(env, FS, AUDIO_FS).astype(np.float32)


def _dry_name(name: str) -> str:
    return name.replace("_hrtf", "_dry")


def _build_subject_cache(subject: int) -> Path:
    out = CACHE_DIR / f"S{subject}.npz"
    if out.exists():
        return out
    mat = scipy.io.loadmat(str(DATA_DIR / f"S{subject}.mat"),
                           squeeze_me=True, struct_as_record=False)
    arrays, meta = {}, []
    for i, t in enumerate(mat["trials"]):
        assert int(t.FileHeader.SampleRate) == FS_EEG_RAW
        eeg = signal.resample_poly(t.RawData.EegData.astype(np.float64),
                                   FS, FS_EEG_RAW, axis=0)
        env_l = stimulus_envelope(_dry_name(str(t.stimuli[0])))
        env_r = stimulus_envelope(_dry_name(str(t.stimuli[1])))
        n = min(len(eeg), len(env_l), len(env_r))
        arrays[f"eeg{i}"] = eeg[:n].astype(np.float32)
        arrays[f"envl{i}"] = env_l[:n]
        arrays[f"envr{i}"] = env_r[:n]
        meta.append((1 if str(t.attended_ear) == "R" else 0, str(t.condition),
                     int(t.experiment), int(t.part), int(t.repetition)))
    arrays["meta_label"] = np.array([m[0] for m in meta])
    arrays["meta_condition"] = np.array([m[1] for m in meta])
    arrays["meta_experiment"] = np.array([m[2] for m in meta])
    arrays["meta_part"] = np.array([m[3] for m in meta])
    arrays["meta_repetition"] = np.array([m[4] for m in meta])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(out)
    return out


def build_cache(n_workers: int = 4) -> None:
    todo = [s for s in range(1, N_SUBJECTS + 1) if not (CACHE_DIR / f"S{s}.npz").exists()]
    if not todo:
        return
    with ProcessPoolExecutor(n_workers) as ex:
        for path in ex.map(_build_subject_cache, todo):
            print(f"  cached {path.name}")


def load_subject(subject: int) -> list[Trial]:
    z = np.load(_build_subject_cache(subject))
    n = len(z["meta_label"])
    return [Trial(eeg=z[f"eeg{i}"], env_left=z[f"envl{i}"], env_right=z[f"envr{i}"],
                  label=int(z["meta_label"][i]), condition=str(z["meta_condition"][i]),
                  experiment=int(z["meta_experiment"][i]), part=int(z["meta_part"][i]),
                  repetition=bool(z["meta_repetition"][i]))
            for i in range(n)]


def load_all() -> dict[int, list[Trial]]:
    build_cache()
    return {s: load_subject(s) for s in range(1, N_SUBJECTS + 1)}


if __name__ == "__main__":
    build_cache()
