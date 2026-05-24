#!/usr/bin/env python3
"""Diagnose why AAD window-level accuracy is at chance level"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
from data.kuleuven_loader import load_trial_data
from mtrf.envelope import extract_envelope_from_audio_file, downsample_eeg
from mtrf.aad_mtrf import AADmTRF
from scipy import signal

print("="*70)
print("AAD Diagnosis: Why is window-level accuracy ~50% (chance)?")
print("="*70)

# Load trial 1 for diagnosis
print("\n[1] Loading Trial 1 data...")
trial_data = load_trial_data(1, 0)
eeg = trial_data['eeg']
label = trial_data['label']
print(f"  EEG shape: {eeg.shape}, label: {label} (0=Track1, 1=Track2)")
print(f"  EEG range: [{eeg.min():.4f}, {eeg.max():.4f}]")
print(f"  EEG std: {eeg.std():.4f}")

# Load envelope for both tracks
stimulus_dir = Path(__file__).parent / "data/KULeuven data set/stimuli"
track1_files = sorted(stimulus_dir.glob("*_track1_*.wav"))
track2_files = sorted(stimulus_dir.glob("*_track2_*.wav"))

print(f"\n[2] Extracting envelopes...")
print(f"  Found {len(track1_files)} track1 files, {len(track2_files)} track2 files")

# Use part1_track1_dry.wav and part1_track2_dry.wav
env_track1, _ = extract_envelope_from_audio_file(str(track1_files[0]), target_fs=100)
env_track2, _ = extract_envelope_from_audio_file(str(track2_files[0]), target_fs=100)

print(f"  Envelope track1: {env_track1.shape}, range=[{env_track1.min():.6f}, {env_track1.max():.6f}]")
print(f"  Envelope track2: {env_track2.shape}, range=[{env_track2.min():.6f}, {env_track2.max():.6f}]")

# Downsample EEG to match envelope (128Hz -> 100Hz)
eeg_ds, _ = downsample_eeg(eeg, 128, 100)
print(f"  EEG downsampled: {eeg_ds.shape}")

# Match lengths
min_len = min(eeg_ds.shape[0], env_track1.shape[0], env_track2.shape[0])
eeg_ds = eeg_ds[:min_len, :]
env_track1 = env_track1[:min_len]
env_track2 = env_track2[:min_len]
print(f"  Trimmed to common length: {min_len} samples")

# [3] Check envelope power spectrum
print(f"\n[3] Envelope spectral analysis...")
freqs_t1, pxx_t1 = signal.periodogram(env_track1, fs=100)
freqs_t2, pxx_t2 = signal.periodogram(env_track2, fs=100)
peak_freq_t1 = freqs_t1[np.argmax(pxx_t1[:50])]  # Look at <5Hz only
peak_freq_t2 = freqs_t2[np.argmax(pxx_t2[:50])]
print(f"  Envelope track1 peak freq: {peak_freq_t1:.2f}Hz")
print(f"  Envelope track2 peak freq: {peak_freq_t2:.2f}Hz")
print(f"  Envelope power (0-1Hz): track1={pxx_t1[1]:.6f}, track2={pxx_t2[1]:.6f}")

# [4] Check EEG-Envelope correlation BEFORE mTRF
print(f"\n[4] Direct EEG-Envelope correlation (single channel, no lag)...")
for ch in [0, 10, 20, 30]:
    r_t1 = np.corrcoef(eeg_ds[:, ch], env_track1)[0, 1]
    r_t2 = np.corrcoef(eeg_ds[:, ch], env_track2)[0, 1]
    print(f"  Channel {ch}: r(EEG, track1)={r_t1:.4f}, r(EEG, track2)={r_t2:.4f}")

# [5] Test mTRF on single 10-sec window
print(f"\n[5] Testing mTRF on first 10-sec window (1000 samples)...")
window_size = 1000
eeg_win = eeg_ds[:window_size, :]
env1_win = env_track1[:window_size]
env2_win = env_track2[:window_size]

aad = AADmTRF(fs=100, tmin=-0.05, tmax=0.4, lambda_=100)

# Load trial 2-20 as training set
print("  Training on trials 2-20...")
train_eeg_list = []
train_env_list = []
for trial_idx in range(1, 20):  # Skip trial 0 (trial 1)
    td = load_trial_data(1, trial_idx)
    eeg_tr = td['eeg']
    eeg_tr, _ = downsample_eeg(eeg_tr, 128, 100)
    train_eeg_list.append(eeg_tr)
    
    # Load corresponding envelope
    part = (trial_idx // 5) + 1  # part 1-4
    if trial_idx % 2 == 0:  # Even trials use dry
        env_file = stimulus_dir / f"part{part}_track1_dry.wav"
    else:
        env_file = stimulus_dir / f"part{part}_track1_dry.wav"
    env_tr, _ = extract_envelope_from_audio_file(str(env_file), target_fs=100)
    train_env_list.append(env_tr)

# Fit decoder
aad.fit(train_eeg_list, train_env_list + train_env_list)

# Classify window
pred, r1, r2 = aad.classify_attention(eeg_win, env1_win, env2_win)
print(f"  Window prediction: {pred} (1=track1, 2=track2)")
print(f"  r(EEG, track1): {r1:.6f}")
print(f"  r(EEG, track2): {r2:.6f}")
print(f"  delta_r = {r1-r2:.6f}")

# Ground truth
true_label = 1 if label == 0 else 2  # Convert from binary to 1/2
print(f"  True label: {true_label}")
print(f"  Correct: {pred == true_label}")

# [6] Check correlation values across multiple windows
print(f"\n[6] Correlation distribution across windows...")
r1_vals = []
r2_vals = []
n_windows = min(5, eeg_ds.shape[0] // window_size)
for w in range(n_windows):
    start = w * window_size
    end = start + window_size
    eeg_w = eeg_ds[start:end, :]
    env1_w = env_track1[start:end]
    env2_w = env_track2[start:end]
    _, r1, r2 = aad.classify_attention(eeg_w, env1_w, env2_w)
    r1_vals.append(r1)
    r2_vals.append(r2)
    print(f"  Window {w+1}: r1={r1:.6f}, r2={r2:.6f}, diff={r1-r2:.6f}")

# [7] Plot diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Envelope and EEG channel 0
ax = axes[0, 0]
t = np.arange(10000) / 100  # 100 sec
ax.plot(t, env_track1[:10000], label='Track1 envelope', alpha=0.7)
ax.plot(t, env_track2[:10000], label='Track2 envelope', alpha=0.7)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Envelope magnitude')
ax.set_title('[A] Envelope comparison (first 100 sec)')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Correlation distribution
ax = axes[0, 1]
ax.bar(['Track1', 'Track2'], [np.mean(r1_vals), np.mean(r2_vals)], 
       yerr=[np.std(r1_vals), np.std(r2_vals)], capsize=5, alpha=0.7)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.set_ylabel('Correlation (r)')
ax.set_title('[B] Mean correlation across windows')
ax.set_ylim([-0.2, 0.2])
ax.grid(True, alpha=0.3, axis='y')

# Subplot 3: Time series of correlation
ax = axes[1, 0]
windows_idx = np.arange(len(r1_vals))
ax.plot(windows_idx, r1_vals, marker='o', label='Track1', alpha=0.7)
ax.plot(windows_idx, r2_vals, marker='s', label='Track2', alpha=0.7)
ax.set_xlabel('Window index')
ax.set_ylabel('Correlation (r)')
ax.set_title('[C] Correlation timeline')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 4: mTRF weights
ax = axes[1, 1]
# Get weights from decoder
decoder = aad.decoder
if hasattr(decoder, 'weights'):
    weights = decoder.weights
    im = ax.imshow(weights[:, :5], aspect='auto', cmap='RdBu_r', vmin=-0.01, vmax=0.01)
    ax.set_xlabel('EEG channel (first 5)')
    ax.set_ylabel('Lag index')
    ax.set_title('[D] mTRF weights (first 5 channels)')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('/Users/sekai/.gemini/antigravity/scratch/neuro-aad-core/results/aad_mtrf_envelope/aad_diagnosis.png', dpi=100)
print("\nDiagnostic plot saved to: aad_diagnosis.png")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)
print(f"✗ Envelope correlation to EEG: Very weak (typically <0.05)")
print(f"✗ mTRF prediction correlation: {np.mean([r1_vals, r2_vals]):.6f} (near 0)")
print(f"✗ Difference between tracks: {np.mean(np.abs(np.array(r1_vals)-np.array(r2_vals))):.6f}")
print(f"\nLikely causes:")
print(f"  1. Envelope extraction quality is poor")
print(f"  2. EEG-envelope temporal alignment is wrong")
print(f"  3. Signal preprocessing has bugs")
print("="*70)
