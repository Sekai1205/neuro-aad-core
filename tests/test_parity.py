"""
Parity test: sklearn Ridge vs PyTorch LinearmTRF on Subject S1.

Verifies that the PyTorch reimplementation produces Pearson r values
within 0.005 of the sklearn Ridge solution on identical data.

Run:  pytest tests/test_parity.py -v
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy import signal, stats
from sklearn.linear_model import Ridge

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.kuleuven_loader import load_trial_data
from mtrf.envelope import extract_envelope_from_audio_file, downsample_eeg
from models.mtrf_torch import LinearmTRF, build_lag_matrix
from config import DATA_DIR


# ── Constants matching existing pipeline ──────────────────────────────
TMIN = 0.0
TMAX = 0.25
FS_TARGET = 100
LAMBDA = 0.001


def _load_subject1_data():
    """Load and preprocess S1 data: 20 trials of (eeg, attended_envelope)."""
    import scipy.io

    envelope_cache = {}
    trials = []

    for trial_idx in range(20):
        trial_data = load_trial_data(1, trial_idx)
        eeg = trial_data['eeg']
        metadata = trial_data['metadata']

        # Access stimulus info
        mat_path = DATA_DIR / "S1.mat"
        mat = scipy.io.loadmat(str(mat_path))
        trial_meta = mat['trials'][0][trial_idx]

        stimuli = trial_meta['stimuli'][0][0]
        attended_ear = str(trial_meta['attended_ear'][0][0][0])
        attended_track_idx = 0 if attended_ear == 'L' else 1

        # Extract attended envelope
        stimuli_element = stimuli[attended_track_idx][0]
        if hasattr(stimuli_element, 'item'):
            filename = stimuli_element.item()
        elif isinstance(stimuli_element, np.ndarray):
            filename = str(stimuli_element[0]) if len(stimuli_element) > 0 else str(stimuli_element)
        else:
            filename = str(stimuli_element)

        if filename in envelope_cache:
            envelope = envelope_cache[filename]
        else:
            audio_path = DATA_DIR / "stimuli" / filename
            envelope, _ = extract_envelope_from_audio_file(
                str(audio_path), target_fs=FS_TARGET
            )
            envelope_cache[filename] = envelope

        # Downsample EEG
        eeg_ds, _ = downsample_eeg(eeg, metadata['sample_rate'],
                                    fs_target=FS_TARGET)

        # Align
        min_len = min(eeg_ds.shape[0], envelope.shape[0])
        trials.append({
            'eeg': eeg_ds[:min_len],
            'envelope': envelope[:min_len],
        })

    return trials


def _bandpass_filter(data, fs=100, low=1.0, high=8.0, order=3):
    """1-8 Hz bandpass filter matching existing pipeline."""
    b, a = signal.butter(order, [low, high], btype='band', fs=fs)
    return signal.filtfilt(b, a, data, axis=0)


def test_parity():
    """
    Fit sklearn Ridge and PyTorch LinearmTRF on identical S1 data.
    Assert Pearson r difference < 0.005.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    print("\n=== Parity Test: sklearn Ridge vs PyTorch LinearmTRF ===")
    print(f"Parameters: tmin={TMIN}, tmax={TMAX}, fs={FS_TARGET}, λ={LAMBDA}")

    # Load data
    print("Loading Subject S1 data (20 trials)...")
    trials = _load_subject1_data()

    # Apply bandpass filter
    print("Applying 1-8 Hz bandpass filter...")
    for t in trials:
        t['eeg'] = _bandpass_filter(t['eeg'])
        t['envelope'] = _bandpass_filter(t['envelope'])

    # Split: first 16 trials (80%) as train, last 4 trials (20%) as test
    n_train_trials = 16
    train_eeg = np.concatenate([t['eeg'] for t in trials[:n_train_trials]])
    train_env = np.concatenate([t['envelope'] for t in trials[:n_train_trials]])
    test_eeg = np.concatenate([t['eeg'] for t in trials[n_train_trials:]])
    test_env = np.concatenate([t['envelope'] for t in trials[n_train_trials:]])

    print(f"Train: {train_eeg.shape[0]} samples, Test: {test_eeg.shape[0]} samples")
    n_channels = train_eeg.shape[1]

    # ── Z-score normalization (from training data) ────────────────────
    mean_eeg = np.mean(train_eeg, axis=0)
    std_eeg = np.std(train_eeg, axis=0)
    std_eeg[std_eeg < 1e-12] = 1.0
    mean_env = float(np.mean(train_env))
    std_env = float(np.std(train_env))
    if std_env < 1e-12:
        std_env = 1.0

    train_eeg_z = (train_eeg - mean_eeg) / std_eeg
    train_env_z = (train_env - mean_env) / std_env
    test_eeg_z = (test_eeg - mean_eeg) / std_eeg
    test_env_z = (test_env - mean_env) / std_env

    # ── Build lag matrices ────────────────────────────────────────────
    X_train, vs_tr, ve_tr = build_lag_matrix(train_eeg_z, TMIN, TMAX, FS_TARGET)
    y_train = train_env_z[vs_tr:ve_tr]

    X_test, vs_te, ve_te = build_lag_matrix(test_eeg_z, TMIN, TMAX, FS_TARGET)
    y_test = test_env_z[vs_te:ve_te]
    env_test_original = test_env[vs_te:ve_te]

    n_valid_train = X_train.shape[0]
    print(f"Lag matrix: {X_train.shape[1]} features "
          f"({n_channels} ch × {X_train.shape[1] // n_channels} lags)")
    print(f"Train valid samples: {n_valid_train}")

    # ── sklearn Ridge ─────────────────────────────────────────────────
    # Our custom solver: (X^T X / N + λ I) w = X^T y / N
    # sklearn Ridge(α): (X^T X + α I) w = X^T y
    # Equivalence: α_sklearn = N * λ_custom
    alpha_sklearn = n_valid_train * LAMBDA

    print(f"\nsklearn Ridge: alpha={alpha_sklearn:.2f} "
          f"(= N_train × λ = {n_valid_train} × {LAMBDA})")

    ridge = Ridge(alpha=alpha_sklearn, fit_intercept=False)
    ridge.fit(X_train, y_train)

    y_pred_sklearn_z = ridge.predict(X_test)
    y_pred_sklearn = y_pred_sklearn_z * std_env + mean_env

    r_sklearn, _ = stats.pearsonr(env_test_original, y_pred_sklearn)
    print(f"  r_sklearn = {r_sklearn:.6f}")

    # ── PyTorch LinearmTRF ────────────────────────────────────────────
    n_features = X_train.shape[1]
    model = LinearmTRF(n_features=n_features)

    # Copy Z-score stats so model.predict/score can use them
    model.mean_eeg_ = mean_eeg
    model.std_eeg_ = std_eeg
    model.mean_env_ = mean_env
    model.std_env_ = std_env
    model.n_channels_ = n_channels

    # Train on same data
    X_tr_t = torch.from_numpy(X_train)
    y_tr_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    X_te_t = torch.from_numpy(X_test)

    # Match regularization: weight_decay in Adam adds wd*w to gradient.
    # At convergence with MSE(mean): (X^T X/N + wd/2 * I) w ≈ X^T y/N
    # We want (X^T X/N + λ I): so wd = 2λ
    wd_torch = 2.0 * LAMBDA

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=wd_torch
    )
    criterion = nn.MSELoss()

    # Train with early stopping
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    print(f"\nPyTorch LinearmTRF: Adam(lr=1e-3, weight_decay={wd_torch:.4f}), "
          f"max 5000 steps")

    for step in range(5000):
        y_pred = model(X_tr_t)
        loss = criterion(y_pred, y_tr_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            current_loss = loss.item()
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= 10:  # effective patience = 100 steps
                print(f"  Early stopping at step {step}, loss={best_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred_torch_z = model(X_te_t).squeeze(1).numpy()

    y_pred_torch = y_pred_torch_z * std_env + mean_env
    r_torch, _ = stats.pearsonr(env_test_original, y_pred_torch)
    print(f"  r_torch   = {r_torch:.6f}")

    # ── Assertion ─────────────────────────────────────────────────────
    delta = abs(r_sklearn - r_torch)
    print(f"\n  |r_sklearn - r_torch| = {delta:.6f}")
    print(f"  Threshold: 0.005")

    assert delta < 0.005, (
        f"Parity check FAILED: |{r_sklearn:.6f} - {r_torch:.6f}| = "
        f"{delta:.6f} >= 0.005"
    )

    print("  ✓ PARITY CHECK PASSED")


if __name__ == "__main__":
    test_parity()
