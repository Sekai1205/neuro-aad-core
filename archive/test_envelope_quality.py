#!/usr/bin/env python3
"""Test envelope extraction with detailed visualization"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from mtrf.envelope import extract_envelope_from_audio_file

print("Testing envelope extraction quality...")

stimulus_dir = Path(__file__).parent / "data/KULeuven data set/stimuli"
audio_file = stimulus_dir / "part1_track1_dry.wav"

print(f"\n[1] Loading audio: {audio_file.name}")
audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
print(f"  Audio shape: {audio.shape}, sr: {sr}Hz")
print(f"  Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
print(f"  Audio RMS: {np.sqrt(np.mean(audio**2)):.4f}")

# Extract first 60 seconds for analysis
audio_short = audio[:sr*60]
print(f"  First 60 sec: {audio_short.shape}")

print(f"\n[2] Extracting envelope...")
env, fs_env = extract_envelope_from_audio_file(str(audio_file), target_fs=100)
env_short = env[:6000]  # First 60 sec @ 100Hz
print(f"  Envelope shape: {env.shape}, fs: {fs_env}Hz")
print(f"  Envelope range: [{env.min():.8f}, {env.max():.8f}]")
print(f"  Envelope mean: {np.mean(env):.8f}, std: {np.std(env):.8f}")

print(f"\n[3] Spectral analysis...")
from scipy import signal

# FFT of envelope
freqs, pxx = signal.periodogram(env_short, fs=100)
top_freqs_idx = np.argsort(pxx)[-5:][::-1]
print(f"  Top 5 frequency components:")
for idx in top_freqs_idx:
    print(f"    {freqs[idx]:.2f}Hz: power={pxx[idx]:.8f}")

# Check if envelope is essentially constant
print(f"\n[4] Checking for constant/DC envelope...")
print(f"  Mean: {np.mean(env):.8f}")
print(f"  Std:  {np.std(env):.8f}")
print(f"  CV (std/mean): {np.std(env)/np.mean(env) if np.mean(env)>0 else 'inf':.4f}")

if np.std(env) / (np.mean(env) + 1e-10) < 0.5:
    print(f"  ⚠️ ENVELOPE IS NEARLY CONSTANT (CV << 1)")
    print(f"  This means envelope extraction is broken!")

print(f"\n[5] Plotting...")
fig, axes = plt.subplots(3, 1, figsize=(14, 8))

# Plot 1: Waveform comparison
ax = axes[0]
t_audio = np.arange(len(audio_short)) / sr
t_env = np.arange(len(env_short)) / 100
ax.plot(t_audio, audio_short, alpha=0.5, label='Raw audio', linewidth=0.5)
ax.plot(t_env, env_short * 1000, label='Envelope (×1000)', color='red')  # Scale for visibility
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Amplitude')
ax.set_title('Audio and Extracted Envelope (first 60 sec)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Envelope zoom (10 sec)
ax = axes[1]
t_env_10 = np.arange(1000) / 100
ax.plot(t_env_10, env_short[:1000], marker='o', markersize=2, label='Envelope')
ax.fill_between(t_env_10, 0, env_short[:1000], alpha=0.3)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Envelope magnitude')
ax.set_title('Envelope detail (first 10 sec) - checking for variation')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Spectrum
ax = axes[2]
ax.semilogy(freqs[:200], pxx[:200], label='Periodogram')
ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='1Hz (expected min)')
ax.axvline(x=10, color='g', linestyle='--', alpha=0.5, label='10Hz (expected max)')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (log scale)')
ax.set_title('Envelope Spectrum (should have power 1-10Hz)')
ax.set_xlim([0, 20])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/sekai/.gemini/antigravity/scratch/neuro-aad-core/results/aad_mtrf_envelope/envelope_quality_check.png', dpi=100)
print(f"  Plot saved to envelope_quality_check.png")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

if np.std(env) / (np.mean(env) + 1e-10) < 0.3:
    print("❌ ENVELOPE IS NEARLY CONSTANT - EXTRACTION IS BROKEN")
    print("\nLikely causes:")
    print("  1. Gammatone filterbank is not working correctly")
    print("  2. Filtering is removing all speech information")
    print("  3. Power-law compression is compressing to near-zero")
elif np.max(freqs[pxx > np.max(pxx)*0.1]) < 1:
    print("❌ ENVELOPE POWER IS AT DC/LOW FREQ - NOT CAPTURING SPEECH")
    print("\nLikely causes:")
    print("  1. Lowpass filter at 10Hz is removing speech modulation")
    print("  2. Filters are not properly tuned for speech")
else:
    print("✅ ENVELOPE LOOKS REASONABLE")
    print("Check plots for detailed analysis")

print("="*70)
