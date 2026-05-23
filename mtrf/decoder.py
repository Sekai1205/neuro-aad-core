"""
Multivariate Temporal Response Function (mTRF) Backward Decoder

Reconstructs speech envelope from multi-channel EEG using ridge regression
with time-lagged features.

References:
    - Biesmans et al. (2016): Auditory-Inspired Speech Envelope Extraction
    - Crosse et al. (2016): The Multivariate Temporal Response Function (mTRF) Toolbox

Model (Backward / Decoder):
    ŝ(t) = Σ_τ Σ_c  w(c,τ) · eeg(c, t+τ)

    where:
        ŝ(t)       = reconstructed envelope at time t
        eeg(c,t+τ) = EEG channel c at time t+τ
        w(c,τ)     = decoder weight for channel c at lag τ
        τ           = time lag index from tmin to tmax

    The brain's neural response to audio at time t occurs with a delay,
    so to reconstruct the stimulus at time t, we use EEG from t+tmin to t+tmax
    (typically 0–400ms post-stimulus).
"""

import numpy as np
from scipy import stats
from typing import Tuple


class MTRFDecoder:
    """
    mTRF Backward Decoder: EEG → Speech Envelope

    Reconstructs the attended speech envelope from multi-channel EEG
    using ridge regression with time-lagged features.

    Key design decisions:
        1. BACKWARD model (EEG→Envelope), not forward (Envelope→EEG)
        2. Z-score standardization of EEG (per-channel) and envelope
           → ensures Ridge penalty λ||w||² operates on a uniform scale
        3. Memory-efficient solver: computes X^T X block-by-block
           without forming the full design matrix (saves ~2.5 GB for 64ch × 46 lags)
        4. score() uses learned weights ONLY — no re-training, no data leakage
    """

    def __init__(self, fs: float = 100, tmin: float = -0.05, tmax: float = 0.4,
                 lambda_: float = 100, verbose: bool = False):
        """
        Initialize mTRF backward decoder.

        Args:
            fs: Sampling rate (Hz), should match downsampled EEG/envelope rate
            tmin: Minimum time lag (seconds).
                  Negative values allow acausal component (EEG before stimulus).
                  Default -0.05s (= -50ms)
            tmax: Maximum time lag (seconds).
                  Positive values capture the post-stimulus neural response.
                  Default 0.4s (= 400ms)
            lambda_: Ridge regularization parameter.
                     Controls bias-variance tradeoff. With Z-scored data,
                     this value has consistent meaning regardless of data scale.
            verbose: If True, print diagnostic information during fit/predict.
        """
        self.fs = fs
        self.tmin = tmin
        self.tmax = tmax
        self.lambda_ = lambda_
        self.verbose = verbose

        # Compute lag indices
        self.lag_samples_min = int(np.round(tmin * fs))
        self.lag_samples_max = int(np.round(tmax * fs))
        self.n_lags = self.lag_samples_max - self.lag_samples_min + 1

        # Model state (populated during fit)
        self.weights_ = None        # (n_channels * n_lags,)
        self.mean_eeg_ = None       # (n_channels,)
        self.std_eeg_ = None        # (n_channels,)
        self.mean_env_ = None       # scalar
        self.std_env_ = None        # scalar
        self.n_channels_ = None
        self.is_fitted = False

        if verbose:
            print(f"mTRF Backward Decoder initialized:")
            print(f"  Sampling rate: {fs} Hz")
            print(f"  Time range: [{tmin:.3f}s, {tmax:.3f}s]")
            print(f"  Lag samples: [{self.lag_samples_min}, {self.lag_samples_max}] "
                  f"({self.n_lags} lags)")
            print(f"  Regularization (λ): {lambda_}")

    def _compute_valid_range(self, n_samples: int) -> Tuple[int, int]:
        """
        Compute the valid output time range for a given input length.

        Due to time-lagging, edge samples cannot be used. This method computes
        the start and end indices of the valid range.

        Args:
            n_samples: Number of input time samples

        Returns:
            valid_start: First valid output index (inclusive)
            valid_end: Last valid output index (exclusive)
        """
        valid_start = max(0, -self.lag_samples_min)
        valid_end = n_samples - max(0, self.lag_samples_max)

        if valid_end <= valid_start:
            raise ValueError(
                f"Input too short ({n_samples} samples) for lag range "
                f"[{self.lag_samples_min}, {self.lag_samples_max}]. "
                f"Need at least "
                f"{self.lag_samples_max - min(0, self.lag_samples_min) + 1} samples."
            )
        return valid_start, valid_end

    def fit(self, eeg: np.ndarray, envelope: np.ndarray) -> 'MTRFDecoder':
        """
        Fit backward mTRF decoder using ridge regression: EEG → Envelope.

        Solves the regularized normal equation:
            (X^T X + λI) w = X^T y

        where X is the time-lagged, Z-scored EEG feature matrix and
        y is the Z-scored envelope.

        Memory efficiency:
            For 64 channels × 46 lags × 114k samples, the full design matrix
            would require ~2.7 GB. Instead, we compute X^T X and X^T y
            block-by-block using views of the original EEG array (~60 MB total).

        Args:
            eeg: EEG signal (n_samples, n_channels) or (n_samples,)
            envelope: Speech envelope (n_samples,)

        Returns:
            self (fitted decoder)
        """
        if eeg.ndim == 1:
            eeg = eeg.reshape(-1, 1)

        n_samples, n_channels = eeg.shape
        assert len(envelope) == n_samples, \
            f"Length mismatch: EEG {n_samples} vs envelope {len(envelope)}"

        # ── Z-score standardization (training statistics) ──────────────
        self.mean_eeg_ = np.mean(eeg, axis=0)        # (n_channels,)
        self.std_eeg_ = np.std(eeg, axis=0)           # (n_channels,)
        self.std_eeg_[self.std_eeg_ < 1e-12] = 1.0   # avoid division by zero

        self.mean_env_ = float(np.mean(envelope))
        self.std_env_ = float(np.std(envelope))
        if self.std_env_ < 1e-12:
            self.std_env_ = 1.0

        eeg_z = (eeg - self.mean_eeg_) / self.std_eeg_
        env_z = (envelope - self.mean_env_) / self.std_env_

        # ── Compute valid time range ──────────────────────────────────
        valid_start, valid_end = self._compute_valid_range(n_samples)
        n_valid = valid_end - valid_start
        y = env_z[valid_start:valid_end]  # (n_valid,)

        # ── Build normal equation block-by-block ──────────────────────
        # X^T X is a (p × p) matrix where p = n_channels × n_lags
        # Each block (i,j) is X_i^T @ X_j of size (n_channels × n_channels)
        # X_i = eeg_z[valid_start+lag_i : valid_end+lag_i, :] is a VIEW (no copy)
        p = n_channels * self.n_lags
        XtX = np.zeros((p, p))
        Xty = np.zeros(p)

        # Pre-extract lag blocks as views of eeg_z (minimal memory overhead)
        lag_blocks = []
        for lag in range(self.lag_samples_min, self.lag_samples_max + 1):
            lag_blocks.append(eeg_z[valid_start + lag: valid_end + lag, :])

        # Accumulate X^T X (symmetric) and X^T y
        for i in range(self.n_lags):
            ci = slice(i * n_channels, (i + 1) * n_channels)
            Xty[ci] = lag_blocks[i].T @ y  # (n_channels,)

            for j in range(i + 1):
                cj = slice(j * n_channels, (j + 1) * n_channels)
                block = lag_blocks[i].T @ lag_blocks[j]  # (n_channels, n_channels)
                XtX[ci, cj] = block
                if i != j:
                    XtX[cj, ci] = block.T
        
        # Normalize by number of samples to make lambda invariant to dataset size
        XtX /= n_valid
        Xty /= n_valid

        # ── Solve ridge regression ────────────────────────────────────
        A = XtX + self.lambda_ * np.eye(p)
        self.weights_ = np.linalg.solve(A, Xty)  # (p,)

        self.n_channels_ = n_channels
        self.is_fitted = True

        if self.verbose:
            # Compute training correlation for diagnostics
            env_z_pred = np.zeros(n_valid)
            for i in range(self.n_lags):
                w_block = self.weights_[i * n_channels: (i + 1) * n_channels]
                env_z_pred += lag_blocks[i] @ w_block
            r_train, _ = stats.pearsonr(y, env_z_pred)
            print(f"  Backward decoder fitted: {n_samples} samples, "
                  f"{n_channels} ch, {self.n_lags} lags → p={p}")
            print(f"  Training r = {r_train:.4f}")

        return self

    def predict(self, eeg: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Predict speech envelope from EEG using fitted decoder weights.

        Pipeline:
            1. Z-score normalize EEG using TRAINING statistics
            2. Extract time-lagged features (views, no full matrix)
            3. Compute weighted sum: ŝ_z(t) = Σ_τ Σ_c w(c,τ) · eeg_z(c, t+τ)
            4. De-standardize to original envelope scale

        Args:
            eeg: EEG signal (n_samples, n_channels) or (n_samples,)

        Returns:
            env_pred: Predicted envelope (n_valid,), in original scale
            valid_start: Start index of valid range in original time axis
            valid_end: End index of valid range in original time axis
        """
        assert self.is_fitted, "Decoder must be fitted first. Call fit()."

        if eeg.ndim == 1:
            eeg = eeg.reshape(-1, 1)

        assert eeg.shape[1] == self.n_channels_, \
            f"Channel mismatch: expected {self.n_channels_}, got {eeg.shape[1]}"

        # Standardize using TRAINING statistics (no data leakage)
        eeg_z = (eeg - self.mean_eeg_) / self.std_eeg_

        # Compute valid range
        n_samples = eeg_z.shape[0]
        valid_start, valid_end = self._compute_valid_range(n_samples)
        n_valid = valid_end - valid_start

        # Predict standardized envelope (memory-efficient: no full design matrix)
        env_z_pred = np.zeros(n_valid)
        for lag_idx, lag in enumerate(range(self.lag_samples_min,
                                            self.lag_samples_max + 1)):
            block = eeg_z[valid_start + lag: valid_end + lag, :]
            w_block = self.weights_[lag_idx * self.n_channels_:
                                    (lag_idx + 1) * self.n_channels_]
            env_z_pred += block @ w_block

        # De-standardize to original envelope scale
        env_pred = env_z_pred * self.std_env_ + self.mean_env_

        return env_pred, valid_start, valid_end

    def score(self, eeg: np.ndarray, envelope: np.ndarray) -> float:
        """
        Evaluate decoder: Pearson r between true and predicted envelope.

        CRITICAL: This method uses ONLY the weights learned during fit().
        No new model is trained on the test data. No data leakage.

        Args:
            eeg: EEG signal (n_samples, n_channels) or (n_samples,)
            envelope: True speech envelope (n_samples,)

        Returns:
            r: Pearson correlation coefficient between true and predicted envelope.
               Higher r indicates better reconstruction quality.
        """
        assert self.is_fitted, "Decoder must be fitted first. Call fit()."

        # Predict using learned weights (no re-training)
        env_pred, valid_start, valid_end = self.predict(eeg)

        # Trim true envelope to the same valid range
        env_true = envelope[valid_start:valid_end]

        # Edge cases: insufficient data or zero-variance signals
        if len(env_pred) < 2:
            return 0.0
        if np.std(env_pred) < 1e-12 or np.std(env_true) < 1e-12:
            return 0.0

        r, _ = stats.pearsonr(env_true, env_pred)

        return float(r)

    def cross_validate_loocv(self, eeg_list: list,
                             envelope_list: list) -> Tuple[np.ndarray, float, float]:
        """
        Leave-one-out cross-validation (LOOCV) for trial-based data.

        For each fold:
            1. Concatenate N-1 trials as training data
            2. fit() computes Z-score stats from training data only
            3. score() evaluates on held-out trial using learned weights

        Args:
            eeg_list: List of EEG arrays, one per trial
            envelope_list: List of envelope arrays, one per trial

        Returns:
            r_scores: Correlation scores per trial (n_trials,)
            mean_r: Mean correlation across folds
            std_r: Standard deviation of correlations
        """
        n_trials = len(eeg_list)
        r_scores = np.zeros(n_trials)

        for test_idx in range(n_trials):
            # Training set: all except test trial
            train_eeg = np.concatenate(
                [eeg_list[i] for i in range(n_trials) if i != test_idx], axis=0)
            train_env = np.concatenate(
                [envelope_list[i] for i in range(n_trials) if i != test_idx], axis=0)

            # Fit on training data (Z-score stats from training only)
            self.fit(train_eeg, train_env)

            # Score on test data (uses learned weights only, no re-training)
            r_scores[test_idx] = self.score(eeg_list[test_idx],
                                            envelope_list[test_idx])

        mean_r = float(np.mean(r_scores))
        std_r = float(np.std(r_scores))

        return r_scores, mean_r, std_r
