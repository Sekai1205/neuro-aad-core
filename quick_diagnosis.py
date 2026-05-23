#!/usr/bin/env python3
"""Quick diagnosis: Check if EEG and envelope are properly aligned"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data.kuleuven_loader import load_trial_data
from mtrf.envelope import downsample_eeg
from scipy import signal

print("="*70)
print("QUICK DIAGNOSIS: Check basic data properties")
print("="*70)

# Load first few trials
print("\n[1] Loading EEG data from trials...")
all_eeg = []
all_labels = []

for trial_idx in range(5):
    td = load_trial_data(1, trial_idx)
    eeg = td['eeg']
    eeg_ds, _ = downsample_eeg(eeg, 128, 100)
    all_eeg.append(eeg_ds)
    all_labels.append(td['label'])
    print(f"  Trial {trial_idx+1}: EEG shape {eeg_ds.shape}, label={td['label']}")

# Check correlation structure
print("\n[2] Check EEG auto-correlation and signal structure...")
eeg_sample = all_eeg[0]

# Check channel correlation
ch_corr = np.corrcoef(eeg_sample[:, :5].T)
print(f"  EEG channel correlation (first 5 channels):")
print(f"    Mean |r|: {np.mean(np.abs(ch_corr[np.triu_indices_from(ch_corr, k=1)])):.4f}")

# Check temporal auto-correlation
print(f"\n[3] Temporal structure...")
lags = [1, 10, 100]
for lag in lags:
    for ch in [0, 10, 30]:
        if lag < eeg_sample.shape[0] // 2:
            r_lag = np.corrcoef(eeg_sample[:-lag, ch], eeg_sample[lag:, ch])[0,1]
            print(f"    Channel {ch:2d}, lag {lag:3d} samples: r={r_lag:.4f}")

# Check spectral properties
print(f"\n[4] EEG spectrum (channel 0, first 10 sec)...")
eeg_win = eeg_sample[:1000, 0]  # 10 sec @ 100Hz
freqs, pxx = signal.periodogram(eeg_win, fs=100)
top_idx = np.argsort(pxx)[-5:]
print(f"    Top 5 frequencies: {freqs[top_idx]}")
print(f"    Top 5 powers: {pxx[top_idx]}")

# Check for data issues
print(f"\n[5] Data quality checks...")
for i, eeg in enumerate(all_eeg[:3]):
    n_zeros = np.sum(eeg == 0)
    n_nans = np.sum(np.isnan(eeg))
    n_infs = np.sum(np.isinf(eeg))
    print(f"    Trial {i+1}: zeros={n_zeros}, NaNs={n_nans}, Infs={n_infs}")
    
    # Check if EEG is just noise
    mean_val = np.mean(eeg)
    std_val = np.std(eeg)
    print(f"            mean={mean_val:.6f}, std={std_val:.6f}")

print("\n" + "="*70)
print("KEY QUESTIONS:")
print("="*70)
print("1. Is EEG mean ~0 and std large (>10)? YES=good signal, NO=preprocessed/noise")
print("2. Are channel correlations <0.3? YES=good (independent), NO=high correlation (problem)")
print("3. Is there strong temporal structure? YES=signals have meaning, NO=white noise")
print("4. Do trials have similar properties? YES=consistent, NO=highly variable")
print("="*70)
