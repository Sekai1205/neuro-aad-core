"""
PyTorch implementation of the mTRF Backward Decoder.

Mathematically equivalent to the Ridge regression in mtrf/decoder.py,
but implemented as a differentiable nn.Module for use with gradient-based
meta-learning (e.g. FOMAML).

Architecture:
    ŝ(t) = W @ x(t)     where x(t) = [eeg(c, t+τ) for τ, c]
    Single linear layer, no bias, L2 via optimizer weight_decay.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from typing import Tuple, Optional


def build_lag_matrix(eeg: np.ndarray, tmin: float, tmax: float,
                     fs: float) -> Tuple[np.ndarray, int, int]:
    """
    Construct the time-lagged feature matrix from EEG data.

    Identical logic to MTRFDecoder: for each output time t, stack
    eeg[:, t + lag] for lag in [lag_min .. lag_max].

    Args:
        eeg: EEG array (n_samples, n_channels), already Z-scored
        tmin: Minimum lag in seconds
        tmax: Maximum lag in seconds
        fs: Sampling rate in Hz

    Returns:
        X: Lag matrix (n_valid, n_channels * n_lags)
        valid_start: First valid index
        valid_end: Last valid index (exclusive)
    """
    lag_min = int(np.round(tmin * fs))
    lag_max = int(np.round(tmax * fs))
    n_lags = lag_max - lag_min + 1

    n_samples, n_channels = eeg.shape
    valid_start = max(0, -lag_min)
    valid_end = n_samples - max(0, lag_max)

    if valid_end <= valid_start:
        raise ValueError(
            f"Input too short ({n_samples} samples) for lag range "
            f"[{lag_min}, {lag_max}]."
        )

    n_valid = valid_end - valid_start
    X = np.zeros((n_valid, n_channels * n_lags), dtype=np.float32)

    for i, lag in enumerate(range(lag_min, lag_max + 1)):
        block = eeg[valid_start + lag: valid_end + lag, :]
        X[:, i * n_channels: (i + 1) * n_channels] = block

    return X, valid_start, valid_end


class LinearmTRF(nn.Module):
    """
    Differentiable mTRF backward decoder: EEG → Envelope.

    Single linear layer (no bias) mapping the time-lagged EEG feature
    vector to a scalar envelope value.

    L2 regularization is applied via optimizer weight_decay, NOT as an
    explicit loss term.

    Usage:
        model = LinearmTRF(n_features=n_channels * n_lags)
        model.fit(eeg_train, env_train, tmin, tmax, fs)
        r = model.score(eeg_test, env_test, tmin, tmax, fs)
    """

    def __init__(self, n_features: int):
        """
        Args:
            n_features: Input dimension = n_channels * n_lags
        """
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)
        self.n_features = n_features

        # Z-score statistics (populated during fit)
        self.mean_eeg_ = None     # (n_channels,)
        self.std_eeg_ = None      # (n_channels,)
        self.mean_env_ = None     # scalar
        self.std_env_ = None      # scalar
        self.n_channels_ = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x (batch, n_features) → envelope (batch, 1)
        """
        return self.linear(x)

    def _zscore_eeg(self, eeg: np.ndarray,
                    fit_stats: bool = False) -> np.ndarray:
        """Z-score EEG per channel, matching MTRFDecoder exactly."""
        if fit_stats:
            self.mean_eeg_ = np.mean(eeg, axis=0)
            self.std_eeg_ = np.std(eeg, axis=0)
            self.std_eeg_[self.std_eeg_ < 1e-12] = 1.0
        return (eeg - self.mean_eeg_) / self.std_eeg_

    def _zscore_env(self, envelope: np.ndarray,
                    fit_stats: bool = False) -> np.ndarray:
        """Z-score envelope, matching MTRFDecoder exactly."""
        if fit_stats:
            self.mean_env_ = float(np.mean(envelope))
            self.std_env_ = float(np.std(envelope))
            if self.std_env_ < 1e-12:
                self.std_env_ = 1.0
        return (envelope - self.mean_env_) / self.std_env_

    def fit(self, eeg: np.ndarray, envelope: np.ndarray,
            tmin: float, tmax: float, fs: float,
            lr: float = 1e-3, weight_decay: float = 0.001,
            max_steps: int = 5000, patience: int = 100,
            delta: float = 1e-6,
            val_split: float = 0.1,
            verbose: bool = False) -> 'LinearmTRF':
        """
        Train the model on EEG → envelope via Adam + MSE + weight_decay.

        Args:
            eeg: (n_samples, n_channels)
            envelope: (n_samples,)
            tmin, tmax, fs: Lag window parameters
            lr: Learning rate for Adam
            weight_decay: L2 penalty coefficient
            max_steps: Maximum training iterations
            patience: Early stopping patience
            delta: Minimum improvement threshold for early stopping
            val_split: Fraction of data for validation (for early stopping)
            verbose: Print training progress

        Returns:
            self
        """
        if eeg.ndim == 1:
            eeg = eeg.reshape(-1, 1)

        self.n_channels_ = eeg.shape[1]

        # Z-score (compute training stats)
        eeg_z = self._zscore_eeg(eeg, fit_stats=True)
        env_z = self._zscore_env(envelope, fit_stats=True)

        # Build lag matrix
        X, vs, ve = build_lag_matrix(eeg_z, tmin, tmax, fs)
        y = env_z[vs:ve].astype(np.float32)

        # Verify feature dimension
        assert X.shape[1] == self.n_features, (
            f"Feature mismatch: model expects {self.n_features}, "
            f"got {X.shape[1]}"
        )

        # Train/val split for early stopping
        n = X.shape[0]
        n_val = max(1, int(n * val_split))
        n_train = n - n_val

        X_train = torch.from_numpy(X[:n_train])
        y_train = torch.from_numpy(y[:n_train]).unsqueeze(1)
        X_val = torch.from_numpy(X[n_train:])
        y_val = torch.from_numpy(y[n_train:]).unsqueeze(1)

        # Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.train()
        for step in range(max_steps):
            # Forward
            y_pred = self.forward(X_train)
            loss = criterion(y_pred, y_train)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            if step % 10 == 0:
                self.eval()
                with torch.no_grad():
                    val_pred = self.forward(X_val)
                    val_loss = criterion(val_pred, y_val).item()
                self.train()

                if val_loss < best_val_loss - delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1

                if verbose and step % 500 == 0:
                    print(f"  Step {step:5d}: train_loss={loss.item():.6f}, "
                          f"val_loss={val_loss:.6f}, patience={patience_counter}")

                if patience_counter >= patience // 10:
                    # Check every 10 steps, so effective patience
                    if verbose:
                        print(f"  Early stopping at step {step}")
                    break

        # Restore best weights
        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        return self

    def predict(self, eeg: np.ndarray,
                tmin: float, tmax: float, fs: float
                ) -> Tuple[np.ndarray, int, int]:
        """
        Predict envelope from EEG using learned weights.

        Uses training Z-score statistics (no data leakage).

        Args:
            eeg: (n_samples, n_channels)
            tmin, tmax, fs: Lag window parameters

        Returns:
            env_pred: Predicted envelope (n_valid,) in original scale
            valid_start, valid_end: Valid range indices
        """
        if eeg.ndim == 1:
            eeg = eeg.reshape(-1, 1)

        # Z-score using training stats
        eeg_z = self._zscore_eeg(eeg, fit_stats=False)

        # Build lag matrix
        X, vs, ve = build_lag_matrix(eeg_z, tmin, tmax, fs)
        X_t = torch.from_numpy(X)

        self.eval()
        with torch.no_grad():
            env_z_pred = self.forward(X_t).squeeze(1).numpy()

        # De-standardize
        env_pred = env_z_pred * self.std_env_ + self.mean_env_
        return env_pred, vs, ve

    def score(self, eeg: np.ndarray, envelope: np.ndarray,
              tmin: float, tmax: float, fs: float) -> float:
        """
        Evaluate: Pearson r between true and predicted envelope.

        Args:
            eeg: (n_samples, n_channels)
            envelope: (n_samples,)
            tmin, tmax, fs: Lag window parameters

        Returns:
            r: Pearson correlation coefficient
        """
        env_pred, vs, ve = self.predict(eeg, tmin, tmax, fs)
        env_true = envelope[vs:ve]

        if len(env_pred) < 2:
            return 0.0
        if np.std(env_pred) < 1e-12 or np.std(env_true) < 1e-12:
            return 0.0

        r, _ = stats.pearsonr(env_true, env_pred)
        return float(r)
