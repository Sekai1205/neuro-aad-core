"""
Channel Selector for Biosemi 64-channel EEG

Maps standard 10-20 electrode names to 0-based column indices
in the KULeuven dataset (Biosemi ActiveTwo 64-channel cap).

Note: The KULeuven dataset uses the standard Biosemi 64-channel layout
(A1-A32 left hemisphere, B1-B32 right hemisphere). TP9/TP10 are NOT
part of the standard 64-channel cap (they are external electrodes).
We substitute TP7 (idx 15) and TP8 (idx 52) as the nearest
temporal-parietal electrodes available.
"""

import numpy as np
from typing import List

# ── Biosemi 64-channel → 10-20 name mapping (0-based index) ──────────
# Source: Biosemi ActiveTwo 64-channel cap layout
# A1-A32 (indices 0-31), B1-B32 (indices 32-63)
BIOSEMI64_CHANNEL_MAP = {
    # A-row (left hemisphere + midline posterior)
    'Fp1': 0,   'AF7': 1,   'AF3': 2,   'F1': 3,    'F3': 4,
    'F5': 5,    'F7': 6,    'FT7': 7,   'FC5': 8,   'FC3': 9,
    'FC1': 10,  'C1': 11,   'C3': 12,   'C5': 13,   'T7': 14,
    'TP7': 15,  'CP5': 16,  'CP3': 17,  'CP1': 18,  'P1': 19,
    'P3': 20,   'P5': 21,   'P7': 22,   'P9': 23,   'PO7': 24,
    'PO3': 25,  'O1': 26,   'Iz': 27,   'Oz': 28,   'POz': 29,
    'Pz': 30,   'CPz': 31,
    # B-row (right hemisphere + midline anterior)
    'Fpz': 32,  'Fp2': 33,  'AF8': 34,  'AF4': 35,  'AFz': 36,
    'Fz': 37,   'F2': 38,   'F4': 39,   'F6': 40,   'F8': 41,
    'FT8': 42,  'FC6': 43,  'FC4': 44,  'FC2': 45,  'FCz': 46,
    'Cz': 47,   'C2': 48,   'C4': 49,   'C6': 50,   'T8': 51,
    'TP8': 52,  'CP6': 53,  'CP4': 54,  'CP2': 55,  'P2': 56,
    'P4': 57,   'P6': 58,   'P8': 59,   'P10': 60,  'PO8': 61,
    'PO4': 62,  'O2': 63,
}

# Reverse map for display
INDEX_TO_NAME = {v: k for k, v in BIOSEMI64_CHANNEL_MAP.items()}


# ── Electrode configurations ──────────────────────────────────────────

# CONFIG_AUDIO: Temporal + central electrodes for auditory cortical tracking
# TP9/TP10 are external electrodes not in the standard 64-ch cap.
# Substituted with TP7 (idx 15) and TP8 (idx 52), the nearest
# temporal-parietal electrodes in the Biosemi 64-channel layout.
CONFIG_AUDIO = [
    BIOSEMI64_CHANNEL_MAP['TP7'],   # 15 — substitute for TP9
    BIOSEMI64_CHANNEL_MAP['TP8'],   # 52 — substitute for TP10
    BIOSEMI64_CHANNEL_MAP['Fz'],    # 37
    BIOSEMI64_CHANNEL_MAP['Cz'],    # 47
]
CONFIG_AUDIO_NAMES = ['TP7 (≈TP9)', 'TP8 (≈TP10)', 'Fz', 'Cz']

# CONFIG_VISUAL: Occipital electrodes (negative control — no auditory signal expected)
CONFIG_VISUAL = [
    BIOSEMI64_CHANNEL_MAP['O1'],    # 26
    BIOSEMI64_CHANNEL_MAP['Oz'],    # 28
    BIOSEMI64_CHANNEL_MAP['O2'],    # 63
    BIOSEMI64_CHANNEL_MAP['POz'],   # 29
]
CONFIG_VISUAL_NAMES = ['O1', 'Oz', 'O2', 'POz']

# BASELINE_64CH: All 64 channels
BASELINE_64CH = list(range(64))


def channel_selector(eeg_data: np.ndarray, channel_indices: List[int]) -> np.ndarray:
    """
    Select a subset of EEG channels by index.

    Pure array slicing — does not alter any other preprocessing step.

    Args:
        eeg_data: EEG array of shape (n_samples, n_channels)
        channel_indices: List of 0-based column indices to keep

    Returns:
        EEG array of shape (n_samples, len(channel_indices))
    """
    if eeg_data.ndim != 2:
        raise ValueError(f"Expected 2D array (samples, channels), got {eeg_data.ndim}D")

    max_idx = eeg_data.shape[1] - 1
    for idx in channel_indices:
        if idx < 0 or idx > max_idx:
            raise IndexError(
                f"Channel index {idx} out of range [0, {max_idx}]"
            )

    return eeg_data[:, channel_indices]


def get_config_description(config_name: str) -> str:
    """Return a human-readable description of a channel configuration."""
    descriptions = {
        'CONFIG_AUDIO': (
            f"4-ch auditory: {CONFIG_AUDIO_NAMES} (indices {CONFIG_AUDIO})"
        ),
        'CONFIG_VISUAL': (
            f"4-ch occipital: {CONFIG_VISUAL_NAMES} (indices {CONFIG_VISUAL})"
        ),
        'BASELINE_64CH': "64-ch baseline (all channels)",
    }
    return descriptions.get(config_name, f"Unknown config: {config_name}")
