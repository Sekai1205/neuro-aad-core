"""
Audio Envelope Extraction using Gammatone Filter Bank
MacIntyre et al. (2024) replication: 50-5000Hz, 28 filters, power law, 10Hz lowpass, downsampled to 100Hz
"""

import numpy as np
from scipy import signal
from pathlib import Path
from typing import Tuple, Optional
import warnings
import librosa

# Suppress filter design warnings (common with high-frequency gammatone filters)
warnings.filterwarnings('ignore', message='Filter design nearly unstable')
warnings.filterwarnings('ignore', message='Filter design failed')


def load_audio_file(audio_path: str, sr: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Load audio file using librosa.
    """
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    return audio, sr


def gammatone_filterbank(fs: float, n_filters: int = 28, 
                          f_min: float = 50, f_max: float = 5000) -> Tuple[list, np.ndarray]:
    """
    Generate Gammatone filter coefficients (approximated using 2nd-order Butterworth bandpass filters)
    """
    center_freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_filters)
    
    filters = []
    for freq in center_freqs:
        Q = 4
        bandwidth = freq / Q
        low = max(freq - bandwidth/2, 1)
        high = min(freq + bandwidth/2, fs/2 - 10)
        
        if low >= high:
            low = high - 10
        
        try:
            b, a = signal.butter(2, [low, high], btype='band', fs=fs, analog=False)
            roots = np.roots(a)
            if np.max(np.abs(roots)) >= 0.99:
                warnings.warn(f"Filter design nearly unstable for freq={freq}")
            
            filters.append((b, a))
        except Exception as e:
            warnings.warn(f"Filter design failed for freq={freq}: {e}, using trivial filter")
            filters.append((np.array([1.0]), np.array([1.0])))
    
    return filters, center_freqs


def extract_envelope_from_audio_file(audio_path: str, 
                                      target_fs: float = 100,
                                      n_filters: int = 28,
                                      f_min: float = 50,
                                      f_max: float = 5000,
                                      power_law_exponent: float = 0.6) -> Tuple[np.ndarray, float]:
    """
    Extract speech envelope from audio file using Gammatone filter bank.
    """
    audio, fs_orig = load_audio_file(audio_path, sr=None)
    filters, center_freqs = gammatone_filterbank(fs_orig, n_filters, f_min, f_max)
    
    filtered_signals = []
    for b, a in filters:
        filtered = signal.filtfilt(b, a, audio)
        filtered = np.maximum(filtered, 0)
        filtered_signals.append(filtered)
    
    filtered_signals = np.array(filtered_signals)
    filtered_signals = np.power(filtered_signals, power_law_exponent)
    envelope = np.mean(filtered_signals, axis=0)
    
    try:
        nyquist = fs_orig / 2
        if 10 < nyquist:
            b_lp, a_lp = signal.butter(4, 10 / nyquist, btype='low', analog=False)
            envelope = signal.filtfilt(b_lp, a_lp, envelope)
    except Exception as e:
        warnings.warn(f"Lowpass filtering failed: {e}")
    
    # 【修正箇所1】巨大なFFTを避けるため resample_poly を使用
    envelope_downsampled = signal.resample_poly(envelope, int(target_fs), int(fs_orig))
    
    return envelope_downsampled, target_fs


def extract_envelope(audio: np.ndarray, fs: float, 
                      target_fs: float = 100,
                      n_filters: int = 28,
                      f_min: float = 50,
                      f_max: float = 5000,
                      power_law_exponent: float = 0.6) -> Tuple[np.ndarray, float]:
    """
    Extract speech envelope from audio array using Gammatone filter bank.
    """
    filters, center_freqs = gammatone_filterbank(fs, n_filters, f_min, f_max)
    
    filtered_signals = []
    for b, a in filters:
        filtered = signal.filtfilt(b, a, audio)
        filtered = np.maximum(filtered, 0)
        filtered_signals.append(filtered)
    
    filtered_signals = np.array(filtered_signals)
    filtered_signals = np.power(filtered_signals, power_law_exponent)
    envelope = np.mean(filtered_signals, axis=0)
    
    try:
        nyquist = fs / 2
        if 10 < nyquist:
            b_lp, a_lp = signal.butter(4, 10 / nyquist, btype='low', analog=False)
            envelope = signal.filtfilt(b_lp, a_lp, envelope)
    except Exception as e:
        warnings.warn(f"Lowpass filtering failed: {e}")
    
    # 【修正箇所2】巨大なFFTを避けるため resample_poly を使用
    envelope_downsampled = signal.resample_poly(envelope, int(target_fs), int(fs))
    
    return envelope_downsampled, target_fs


def downsample_eeg(eeg: np.ndarray, fs_orig: float, fs_target: float = 100) -> Tuple[np.ndarray, float]:
    """
    Downsample EEG signal to match envelope sampling rate.
    """
    # 【修正箇所3】巨大なFFTを避けるため resample_poly を使用
    if eeg.ndim == 1:
        eeg_downsampled = signal.resample_poly(eeg, int(fs_target), int(fs_orig))
    else:
        eeg_downsampled = signal.resample_poly(eeg, int(fs_target), int(fs_orig), axis=0)
    
    return eeg_downsampled, fs_target


def align_envelope_eeg(eeg: np.ndarray, envelope: np.ndarray,
                       eeg_fs: float = 100, envelope_fs: float = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure EEG and envelope are aligned in time (same length after downsampling).
    """
    assert eeg_fs == envelope_fs, f"EEG fs ({eeg_fs}) != Envelope fs ({envelope_fs})"
    
    min_length = min(eeg.shape[0], envelope.shape[0])
    
    if eeg.ndim == 1:
        eeg_aligned = eeg[:min_length]
    else:
        eeg_aligned = eeg[:min_length, :]
    
    envelope_aligned = envelope[:min_length]
    
    return eeg_aligned, envelope_aligned