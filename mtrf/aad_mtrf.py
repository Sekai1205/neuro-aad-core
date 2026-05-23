"""
AAD Classifier using mTRF Envelope Reconstruction
Standard approach: Biesmans et al. (2016)

The classifier trains a backward decoder (EEG → envelope) on attended
speech envelopes ONLY, then compares reconstruction correlations for
each speaker's envelope. The speaker with higher correlation is deemed
the attended speaker.

    r_track1 = corr(decoder(EEG), envelope_track1)
    r_track2 = corr(decoder(EEG), envelope_track2)

    If r_track1 > r_track2 → predict Track1 (attended)
    Otherwise             → predict Track2 (attended)

Key design principle:
    Training on attended envelopes only is critical. If we train on both
    speakers' envelopes, the decoder learns to reconstruct ANY speech from EEG,
    and the correlation difference (r_attended - r_unattended) collapses to zero.
"""

import numpy as np
from mtrf.decoder import MTRFDecoder
from typing import Tuple


class AADmTRF:
    """
    AAD Classifier using mTRF backward decoder.

    Training:
        Train backward decoder on (EEG → attended envelope) only.
        The decoder learns weights optimized for attended speech reconstruction.

    Inference:
        For each test window:
        1. Reconstruct envelope from EEG using learned decoder weights
        2. Correlate the single reconstruction with each speaker's envelope
        3. Higher correlation → attended speaker
    """

    def __init__(self, fs: float = 100, tmin: float = 0.0, tmax: float = 0.25,
                 lambda_: float = 0.001):
        """
        Initialize AAD classifier.

        Args:
            fs: Sampling rate (Hz)
            tmin: Minimum time lag (seconds)
            tmax: Maximum time lag (seconds)
            lambda_: Ridge regularization parameter
        """
        self.fs = fs
        self.tmin = tmin
        self.tmax = tmax
        self.lambda_ = lambda_
        self.decoder = None
        self.is_fitted = False

    def fit(self, eeg_list: list, attended_envelope_list: list) -> 'AADmTRF':
        """
        Train mTRF backward decoder on EEG and ATTENDED envelopes only.

        Each EEG trial is paired 1:1 with its corresponding attended
        speaker's envelope. The decoder learns to reconstruct the attended
        speech signal from EEG.

        Args:
            eeg_list: List of EEG arrays (n_trials,), each (n_samples, n_channels)
            attended_envelope_list: List of attended envelope arrays (n_trials,),
                                    each (n_samples,). Must be 1:1 with eeg_list.

        Returns:
            self
        """
        assert len(eeg_list) == len(attended_envelope_list), \
            (f"EEG trials ({len(eeg_list)}) must match attended envelope trials "
             f"({len(attended_envelope_list)}). Pass only attended envelopes.")

        # Concatenate all training data
        eeg_concat = np.concatenate(eeg_list, axis=0)
        env_concat = np.concatenate(attended_envelope_list, axis=0)

        # Train backward decoder: EEG → attended envelope
        self.decoder = MTRFDecoder(
            fs=self.fs, tmin=self.tmin, tmax=self.tmax, lambda_=self.lambda_
        )
        self.decoder.fit(eeg_concat, env_concat)

        self.is_fitted = True
        return self

    def classify_attention(self, eeg: np.ndarray, env_track1: np.ndarray,
                           env_track2: np.ndarray) -> Tuple[int, float, float]:
        """
        Classify attended speaker via envelope reconstruction correlation.

        The learned decoder produces a SINGLE reconstruction from the EEG
        (optimized for attended speech). We then correlate this reconstruction
        with each speaker's actual envelope. The speaker yielding higher
        correlation is predicted as attended.

        Args:
            eeg: EEG signal (n_samples, n_channels)
            env_track1: Track 1 envelope (n_samples,)
            env_track2: Track 2 envelope (n_samples,)

        Returns:
            predicted_track: 1 if r_track1 > r_track2, else 2
            r_track1: Pearson r between decoder output and Track 1
            r_track2: Pearson r between decoder output and Track 2
        """
        if not self.is_fitted:
            raise ValueError("Decoder not fitted. Call fit() first.")

        # score() uses learned weights only — no re-training
        r_track1 = self.decoder.score(eeg, env_track1)
        r_track2 = self.decoder.score(eeg, env_track2)

        # Higher correlation → attended speaker
        predicted_track = 1 if r_track1 > r_track2 else 2

        return predicted_track, r_track1, r_track2
