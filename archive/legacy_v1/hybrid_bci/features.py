import numpy as np
from scipy import signal

class GaborFeatureExtractor:
    """
    Layer 2: Gabor Wavelet Time-Frequency and Scattering Modulus Feature Extractor.
    Extracts stable time-frequency envelopes in the Auditory EEG band (1-8 Hz) 
    and EMG artifact band (20-90 Hz) matching translation-invariant scattering principles.
    """
    def __init__(self, fs: float = 100, fs_target: float = 64):
        self.fs = fs
        self.fs_target = fs_target
        
        # Design Butterworth bandpass filters
        # 1-8 Hz Auditory Cortical Band
        self.b_aud, self.a_aud = signal.butter(3, [1.0, 8.0], btype='band', fs=fs_target)
        # 20-90 Hz EMG Artifact Band (restricted for Nyquist safety at fs_target=250Hz/Ganglion)
        # Note: If target sample rate is 64 Hz, the Nyquist frequency is 32 Hz.
        # If we downsample to 64 Hz, the 20-90 Hz band is above the Nyquist limit!
        # To handle this robustly and adapt to different target rates, we automatically
        # cap the EMG high-pass filter below the target Nyquist frequency (e.g. 0.9 * Nyquist).
        nyquist = fs_target / 2.0
        emg_high = min(90.0, nyquist - 2.0)
        emg_low = min(20.0, emg_high - 2.0)
        
        if emg_low > 1.0:
            self.b_emg, self.a_emg = signal.butter(3, [emg_low, emg_high], btype='band', fs=fs_target)
            self.has_emg = True
        else:
            self.has_emg = False
            print("Warning: Target sample rate too low to extract 20-90Hz EMG. Capping EMG band.")

    def downsample_signal(self, X: np.ndarray) -> np.ndarray:
        """
        Downsample input signal from native fs (e.g. 100Hz) to target hardware rate (e.g. 64Hz).
        X: (n_samples, n_channels)
        """
        if self.fs == self.fs_target:
            return X
            
        n_samples, n_channels = X.shape
        n_samples_target = int(np.round(n_samples * self.fs_target / self.fs))
        
        # Resample along axis 0
        return signal.resample(X, n_samples_target, axis=0)

    def extract_features(self, X_aligned: np.ndarray) -> np.ndarray:
        """
        Extract time-frequency scattering envelopes from aligned EEG.
        Preserves the raw low-frequency phase-locked waveform for auditory tracking (1-8 Hz),
        while taking the absolute envelope of the EMG band (20-90 Hz) for artifact gating.
        X_aligned: (n_samples, n_channels) at fs_target rate.
        Returns: feature matrix of shape (n_samples, 2 * n_channels)
        """
        n_samples, n_channels = X_aligned.shape
        
        # 1. Filter auditory band 1-8 Hz (Preserve raw phase-locked waveform!)
        X_aud = signal.filtfilt(self.b_aud, self.a_aud, X_aligned, axis=0)
        
        # 2. Filter EMG band if Nyquist limits allow
        if self.has_emg:
            X_emg = signal.filtfilt(self.b_emg, self.a_emg, X_aligned, axis=0)
            # EMG features use the absolute magnitude (envelope) of muscle noise
            env_emg = np.abs(X_emg)
            # Concatenate phase-locked auditory signal and EMG noise magnitude
            features = np.hstack([X_aud, env_emg])
        else:
            # Fallback: replicate auditory signal to keep feature dimension consistent
            features = np.hstack([X_aud, X_aud * 0.1])
            
        return features
