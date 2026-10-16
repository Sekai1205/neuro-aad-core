"""
Task D: Bayesian Linear Regression Meta-Learning for AAD.
CCA-Purged Edition with Leaky SPRT.

Evaluates four models under a LOOCV fold setup:
1. bayesian_raw: Bayesian mTRF Decoder with empirical prior, raw window-by-window argmax.
2. bayesian_sprt: Same decoder, with Leaky SPRT gating for smoothed decisions.
3. direct_ridge_calib: Standard Ridge regression decoder trained on target calibration data.
4. subject_specific_mtrf: 4ch upper bound accuracy from baseline Ridge decoder.
"""

import sys
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
import gc

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mtrf.experiment_aad_mtrf import AADmTRFExperiment
from mtrf.decoder import MTRFDecoder
from mtrf.aad_mtrf import AADmTRF
from models.mtrf_torch import build_lag_matrix

# Set random seeds
np.random.seed(42)

# Quick mode config
QUICK_MODE = False  # Set to True for S1-only validation run, False for full 16-subj benchmark

# 4-Channel Electrode Configuration
AUDIO_4CH_INDICES = [15, 52, 37, 47]  # TP7, TP8, Fz, Cz

# Constants
TMIN = 0.0
TMAX = 0.25
FS = 100
LAMBDA = 0.001
CALIB_TIMES = [10, 30, 60, 120]  # seconds
WINDOW_SIZE = 1000  # 10 seconds @ 100 Hz

# sigma_noise cancels out of the posterior (spherical prior: precision*scale == lambda;
# empirical prior: annealing schedule overrides it), so it is fixed rather than grid-searched.
SIGMA_NOISE = 1.0

# Leaky SPRT constants (Discordance-driven Dynamic Bound & Leakage Control, DBC)
SPRT_BOUND = np.log(19.0)      # ~2.9444, 95% confidence Wald boundary (B_0)
SPRT_BOUND_MIN = np.log(3.0)   # ~1.0986, lowest bound under full discordance
SPRT_KAPPA = 2.0               # Bound shrink rate vs discordance index
SPRT_ALPHA_MIN = 0.05          # Leakage under full concordance
SPRT_ALPHA_MAX = 0.40          # Leakage under full discordance
SPRT_BETA_S = 0.05             # Discordance index low-pass coefficient
SPRT_BUFFER_LEN = 6            # d(t) moving-statistics buffer (6 windows = 60s)
SPRT_MIN_STATS_LEN = 3         # Below this, use cross-subject prior (mu, sigma_sq)
SPRT_MU_FLOOR = 0.005
SPRT_SIGMA_SQ_FLOOR = 0.0001


def compress_envelope(env: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Stevens' Power-Law Loudness Compression.
    Models cochlear logarithmic scaling in the primary auditory cortex.
    """
    return np.sign(env) * (np.abs(env) ** alpha)


def preprocess_differential_channels(eeg_4ch: np.ndarray) -> np.ndarray:
    """
    Software Differential Virtual Channel Synthesis.
    Input eeg_4ch columns: 0: TP9 (TP7), 1: TP10 (TP8), 2: Fz, 3: Cz.
    Returns:
        np.ndarray: Re-synthesized 4-channel differential space.
        - Column 0: TP9 - TP10 (Left-Right temporal differential)
        - Column 1: Fz - Cz (Antero-posterior midline differential)
        - Column 2: TP9 - Cz (Left mastoid referenced to vertex)
        - Column 3: TP10 - Cz (Right mastoid referenced to vertex)
    """
    tp9 = eeg_4ch[:, 0]
    tp10 = eeg_4ch[:, 1]
    fz = eeg_4ch[:, 2]
    cz = eeg_4ch[:, 3]
    
    ch1 = tp9 - tp10
    ch2 = fz - cz
    ch3 = tp9 - cz
    ch4 = tp10 - cz
    
    return np.stack([ch1, ch2, ch3, ch4], axis=1)


def compute_closed_loop_gain(lambda_llr: float, bound_B: float = SPRT_BOUND) -> tuple:
    """
    Clinical-Safe Closed-Loop DRC Speaker Gains.
    Considers the dynamic range recruitment constraint. Clamps speaker volumes strictly in [0.8, 1.2].
    DBC-compliant single-pass tanh DRC.
    """
    # 1. DBC-compliant single-pass tanh DRC soft-limiter
    g_limited = 1.0 + 0.20 * np.tanh(lambda_llr / bound_B)
    
    # Left and Right headphone gains
    g_left = float(g_limited)
    g_right = float(2.0 - g_limited)
    
    return g_left, g_right


def estimate_sprt_params_v2(subjects_data: dict, train_ids: list, prior_weights: np.ndarray) -> tuple:
    """
    Estimate empirical SPRT parameters (mu, sigma_sq) from other subjects' correlation
    differences in full P=104 space. Uses HPF on d(t) to remove slow drift bias.

    Returns:
        mu: mean of attended correlation difference (safety floor 0.005)
        sigma_sq: variance of correlation difference (safety floor 0.0001)
    """
    diffs = []
    
    for i, sub_id in enumerate(train_ids):
        sub_trials = subjects_data[sub_id]
        
        # Build prior weights for this sub from all OTHER training subjects
        sub_prior_weights = np.delete(prior_weights, i, axis=0)
        if sub_prior_weights.shape[0] == 0:
            continue
        
        # Concatenate all trials
        test_eeg_concat = np.concatenate([t['eeg'] for t in sub_trials], axis=0)
        test_attended_env_concat = np.concatenate([
            t['envelope_track1'] if t['ground_truth_track'] == 1 else t['envelope_track2']
            for t in sub_trials
        ], axis=0)
        test_env1_concat = np.concatenate([t['envelope_track1'] for t in sub_trials])
        test_env2_concat = np.concatenate([t['envelope_track2'] for t in sub_trials])
        
        # Use 120s calibration for SPRT parameter estimation
        calib_samples = int(120 * FS)
        if test_eeg_concat.shape[0] <= calib_samples:
            continue
        
        calib_eeg = test_eeg_concat[:calib_samples]
        calib_env = test_attended_env_concat[:calib_samples]
        
        # Fit decoder with prior (no CCA, full P=104)
        decoder = BayesianMTRFDecoder(fs=FS, tmin=TMIN, tmax=TMAX)
        decoder.set_prior_from_others(sub_prior_weights)
        decoder.adapt_online(calib_eeg, calib_env, lambda_=LAMBDA)
        
        # Score remaining windows
        rem_eeg = test_eeg_concat[calib_samples:]
        rem_env1 = test_env1_concat[calib_samples:]
        rem_env2 = test_env2_concat[calib_samples:]
        
        sample_gts = []
        for t in sub_trials:
            sample_gts.extend([t['ground_truth_track']] * t['eeg'].shape[0])
        sample_gts = np.array(sample_gts)[calib_samples:]
        
        n_windows = rem_eeg.shape[0] // WINDOW_SIZE
        for w in range(n_windows):
            start, end = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
            r1 = decoder.score(rem_eeg[start:end], rem_env1[start:end])
            r2 = decoder.score(rem_eeg[start:end], rem_env2[start:end])
            gt = sample_gts[start + WINDOW_SIZE // 2]
            
            # d(t) is always "attended minus unattended"
            d = r1 - r2 if gt == 1 else r2 - r1
            diffs.append(d)
    
    if len(diffs) > 5:
        diffs_arr = np.array(diffs)
        
        # HPF: Remove slow drift by subtracting a simple running mean (boxcar detrend)
        # This shields the variance estimate from non-stationary attention switches
        kernel_size = min(5, len(diffs_arr))
        if kernel_size > 1:
            running_mean = np.convolve(diffs_arr, np.ones(kernel_size) / kernel_size, mode='same')
            diffs_hpf = diffs_arr - running_mean + np.mean(diffs_arr)
        else:
            diffs_hpf = diffs_arr
        
        mu = float(np.mean(diffs_arr))
        sigma_sq = float(np.var(diffs_hpf))
    else:
        mu = 0.02
        sigma_sq = 0.01
    
    # Safety floors
    mu = max(mu, SPRT_MU_FLOOR)
    sigma_sq = max(sigma_sq, SPRT_SIGMA_SQ_FLOOR)

    return mu, sigma_sq


class BayesianMTRFDecoder:
    """
    Bayesian Linear Regression Decoder for mTRF.
    Uses empirical priors constructed from other subjects' decoders.
    Full P=104 dimensional space (no CCA compression).
    """
    def __init__(self, fs: float = 100, tmin: float = 0.0, tmax: float = 0.25, 
                 sigma_noise: float = 1.0, epsilon: float = 1e-6, use_noise_annealing: bool = True):
        self.fs = fs
        self.tmin = tmin
        self.tmax = tmax
        self.sigma_noise = sigma_noise
        self.epsilon = epsilon
        self.use_noise_annealing = use_noise_annealing
        
        self.lag_samples_min = int(np.round(tmin * fs))
        self.lag_samples_max = int(np.round(tmax * fs))
        self.n_lags = self.lag_samples_max - self.lag_samples_min + 1
        
        self.mu_0 = None      # Prior mean (p,)
        self.var_0 = None     # Prior variance (p,)
        self.weights_ = None  # Posterior mean (p,)
        
        self.mean_eeg_ = None
        self.std_eeg_ = None
        self.mean_env_ = None
        self.std_env_ = None
        self.n_channels_ = None
        self.is_fitted = False

    def set_prior_from_others(self, other_weights: np.ndarray):
        """
        Set diagonal prior mean and variance from other subjects' weights.
        other_weights: (S, p) where S is the S training subjects and p is features.
        """
        self.mu_0 = np.mean(other_weights, axis=0)
        self.var_0 = np.var(other_weights, axis=0)

    def adapt_online(self, eeg: np.ndarray, envelope: np.ndarray, lambda_=0.001):
        """
        Closed-form Bayesian adaptation on calibration data.
        Operates in full P=104 dimensional space.
        """
        if eeg.ndim == 1:
            eeg = eeg.reshape(-1, 1)
            
        n_samples, n_channels = eeg.shape
        
        # Z-score normalization of calibration data
        self.mean_eeg_ = np.mean(eeg, axis=0)
        self.std_eeg_ = np.std(eeg, axis=0)
        self.std_eeg_[self.std_eeg_ < 1e-12] = 1.0
        
        self.mean_env_ = float(np.mean(envelope))
        self.std_env_ = float(np.std(envelope))
        if self.std_env_ < 1e-12:
            self.std_env_ = 1.0
            
        eeg_z = (eeg - self.mean_eeg_) / self.std_eeg_
        env_z = (envelope - self.mean_env_) / self.std_env_
        
        # Build lag feature matrix
        X, vs, ve = build_lag_matrix(eeg_z, self.tmin, self.tmax, self.fs)
        y = env_z[vs:ve]
        
        n_valid, p = X.shape
        
        # Normalize XtX and Xty by number of samples matching standard Ridge
        XtX = (X.T @ X) / n_valid
        Xty = (X.T @ y) / n_valid
        
        # Prior parameters
        if self.mu_0 is None:
            # Spherical prior: equivalent to Ridge regression with lambda
            prior_var = np.ones(p) * (self.sigma_noise**2 / lambda_)
            prior_mean = np.zeros(p)
        else:
            prior_mean = self.mu_0
            # Add small epsilon to stabilize division
            prior_var = self.var_0 + self.epsilon
            
        # Inverse prior variance (prior precision)
        prior_precision = 1.0 / prior_var
        
        # Dynamic observation noise variance (sigma_noise_sq) annealing schedule
        if self.use_noise_annealing and self.mu_0 is not None:
            tau = 10.0 * self.fs
            sigma_base_sq = 1.0
            sigma_prior_sq = 100.0
            sigma_noise_sq = sigma_base_sq + (sigma_prior_sq - sigma_base_sq) * np.exp(-n_valid / tau)
        else:
            sigma_noise_sq = self.sigma_noise**2
            
        # Solve Posterior Mean
        scale = sigma_noise_sq
        A = np.diag(prior_precision * scale) + XtX
        b = (prior_precision * scale) * prior_mean + Xty
        
        self.weights_ = np.linalg.solve(A, b)
        self.n_channels_ = n_channels
        self.is_fitted = True
        
        return self

    def predict(self, eeg: np.ndarray) -> tuple:
        """Predict reconstructed envelope using posterior mean weights."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        if eeg.ndim == 1:
            eeg = eeg.reshape(-1, 1)
            
        eeg_z = (eeg - self.mean_eeg_) / self.std_eeg_
        
        # Build lag matrix
        X, vs, ve = build_lag_matrix(eeg_z, self.tmin, self.tmax, self.fs)
        
        env_z_pred = (X @ self.weights_).flatten()
        env_pred = env_z_pred * self.std_env_ + self.mean_env_
        return env_pred, vs, ve

    def score(self, eeg: np.ndarray, envelope: np.ndarray) -> float:
        """Pearson correlation coefficient between true and predicted envelope."""
        env_pred, vs, ve = self.predict(eeg)
        env_true = envelope[vs:ve]
        
        if len(env_pred) < 2 or np.std(env_pred) < 1e-12 or np.std(env_true) < 1e-12:
            return 0.0
            
        r, _ = stats.pearsonr(env_true, env_pred)
        return float(r)


class AADBayesian:
    """
    AAD Attention Classifier wrapper for Bayesian Linear Regression.
    """
    def __init__(self, fs: float = 100, tmin: float = 0.0, tmax: float = 0.25, 
                 sigma_noise: float = 1.0, epsilon: float = 1e-6):
        self.fs = fs
        self.tmin = tmin
        self.tmax = tmax
        self.sigma_noise = sigma_noise
        self.epsilon = epsilon
        self.decoder = BayesianMTRFDecoder(fs=fs, tmin=tmin, tmax=tmax, 
                                            sigma_noise=sigma_noise, epsilon=epsilon)
        self.is_fitted = False
        
    def fit_with_prior(self, eeg: np.ndarray, env: np.ndarray, prior_weights: np.ndarray, lambda_=0.001):
        self.decoder.set_prior_from_others(prior_weights)
        self.decoder.adapt_online(eeg, env, lambda_=lambda_)
        self.is_fitted = True
        return self
        
    def fit_no_prior(self, eeg: np.ndarray, env: np.ndarray, lambda_=0.001):
        self.decoder.mu_0 = None
        self.decoder.var_0 = None
        self.decoder.adapt_online(eeg, env, lambda_=lambda_)
        self.is_fitted = True
        return self
        
    def classify_attention(self, eeg: np.ndarray, env1: np.ndarray, env2: np.ndarray) -> tuple:
        if not self.is_fitted:
            raise ValueError("Decoder not fitted. Call fit first.")
            
        r1 = self.decoder.score(eeg, env1)
        r2 = self.decoder.score(eeg, env2)
        pred = 1 if r1 > r2 else 2
        return pred, r1, r2


def load_subject_specific_accuracies(csv_path: Path) -> dict:
    """Loads baseline 4ch auditory accuracies from 4ch_comparison.csv."""
    accuracies = {}
    if not csv_path.exists():
        print(f"Warning: 4ch baseline CSV not found at {csv_path}. Using fallbacks.")
        return {'S1': 56.19, 'S5': 63.94, 'S10': 61.28, 'S14': 65.93}
    try:
        df = pd.read_csv(csv_path)
        df_filt = df[(df['config'] == 'CONFIG_AUDIO') & (df['window_s'] == 10)]
        for _, row in df_filt.iterrows():
            sub = str(row['subject']).strip()
            acc = float(row['accuracy_pct'])
            accuracies[sub] = acc
        print(f"Successfully loaded {len(accuracies)} baseline accuracies dynamically.")
    except Exception as e:
        print(f"Error loading baseline accuracies: {e}. Using fallbacks.")
        return {'S1': 56.19, 'S5': 63.94, 'S10': 61.28, 'S14': 65.93}
    return accuracies


def bandpass_filter(data, fs=100, low=1.0, high=8.0, order=3):
    """1-8 Hz bandpass filter matching baseline pipelines."""
    b, a = signal.butter(order, [low, high], btype='band', fs=fs)
    return signal.filtfilt(b, a, data, axis=0)


def train_subject_decoders(subjects_data: dict, train_ids: list) -> np.ndarray:
    """
    Fits BayesianMTRFDecoders on all training subjects to gather prior weights.
    Full P=104 dimensional space.
    """
    weights = []
    for sub_id in train_ids:
        sub_trials = subjects_data[sub_id]
        eeg_concat = np.concatenate([t['eeg'] for t in sub_trials], axis=0)
        env_concat = np.concatenate([
            t['envelope_track1'] if t['ground_truth_track'] == 1 else t['envelope_track2']
            for t in sub_trials
        ], axis=0)
        
        # Spherical prior adapt equivalent to direct ridge fit
        decoder = BayesianMTRFDecoder(fs=FS, tmin=TMIN, tmax=TMAX)
        decoder.adapt_online(eeg_concat, env_concat, lambda_=LAMBDA)
        weights.append(decoder.weights_)
    return np.array(weights)


def evaluate_bayesian_on_subject(test_sub_id: int, train_ids: list, subjects_data: dict,
                                 sigma_noise: float, baseline_acc: float,
                                 subject_weights: dict = None) -> list:
    """
    Evaluates Bayesian decoders (raw and SPRT-gated) and direct Ridge on a test subject.
    All decoders operate in full P=104 dimensional space.

    subject_weights: optional {sub_id: weights} cache of per-subject decoders. Each subject's
    decoder is fold-independent, so passing this avoids refitting them in every LOOCV fold.
    """
    results = []

    # 1. Train prior weights from training subjects (full P=104)
    if subject_weights is not None:
        prior_weights = np.array([subject_weights[s] for s in train_ids])
    else:
        prior_weights = train_subject_decoders(subjects_data, train_ids)
    
    # 2. Estimate SPRT params from other subjects (full P=104, HPF-shielded)
    sprt_mu, sprt_sigma_sq = estimate_sprt_params_v2(subjects_data, train_ids, prior_weights)
    print(f"  SPRT params: mu = {sprt_mu:.4f}, sigma_sq = {sprt_sigma_sq:.6f}")
    
    # 3. Extract test subject concatenated data
    test_trials = subjects_data[test_sub_id]
    test_eeg_concat = np.concatenate([t['eeg'] for t in test_trials], axis=0)
    test_env1_concat = np.concatenate([t['envelope_track1'] for t in test_trials])
    test_env2_concat = np.concatenate([t['envelope_track2'] for t in test_trials])
    
    test_attended_env_concat = np.concatenate([
        t['envelope_track1'] if t['ground_truth_track'] == 1 else t['envelope_track2']
        for t in test_trials
    ], axis=0)
    
    sample_gts = []
    for t in test_trials:
        sample_gts.extend([t['ground_truth_track']] * t['eeg'].shape[0])
    sample_gts = np.array(sample_gts)
    
    for calib_s in CALIB_TIMES:
        calib_samples = int(calib_s * FS)
        calib_eeg = test_eeg_concat[:calib_samples]
        calib_env = test_attended_env_concat[:calib_samples]
        
        rem_eeg = test_eeg_concat[calib_samples:]
        rem_env1 = test_env1_concat[calib_samples:]
        rem_env2 = test_env2_concat[calib_samples:]
        rem_gts = sample_gts[calib_samples:]
        
        n_windows_rem = rem_eeg.shape[0] // WINDOW_SIZE
        if n_windows_rem == 0:
            continue
            
        # Extract ground truth per window matching evaluation logic
        window_gts = [rem_gts[w * WINDOW_SIZE + WINDOW_SIZE // 2] for w in range(n_windows_rem)]
        window_gts = np.array(window_gts)
        
        # --- A. Bayesian With Prior: Raw argmax AND Leaky SPRT ---
        bayesian_prior = AADBayesian(fs=FS, tmin=TMIN, tmax=TMAX, sigma_noise=sigma_noise)
        bayesian_prior.fit_with_prior(calib_eeg, calib_env, prior_weights, lambda_=LAMBDA)
        
        # SPRT state with Discordance-driven Dynamic Bound & Leakage Control (DBC)
        lambda_llr = 0.0
        prev_decision = None    # No SPRT decision yet: fall back to raw argmax while holding
        s_t = 0.0               # Discordance index initialized to 0

        preds_raw = []
        preds_sprt = []

        # Dynamic d(t) buffer for moving statistics
        d_t_buffer = deque(maxlen=SPRT_BUFFER_LEN)

        for w in range(n_windows_rem):
            start, end = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
            pred_raw, r1, r2 = bayesian_prior.classify_attention(
                rem_eeg[start:end], rem_env1[start:end], rem_env2[start:end]
            )
            preds_raw.append(pred_raw)

            d_t = r1 - r2
            d_t_buffer.append(d_t)

            # Online statistics; need >= 2 first differences for a meaningful variance,
            # otherwise use the cross-subject prior estimates.
            if len(d_t_buffer) >= SPRT_MIN_STATS_LEN:
                d_arr = np.array(d_t_buffer)
                mu_t = float(np.mean(np.abs(d_arr)))
                # 1st-order difference detrending; Var(x[n] - x[n-1]) = 2*sigma^2 for iid noise
                sigma_t_sq = float(np.var(np.diff(d_arr))) / 2.0
            else:
                mu_t = sprt_mu
                sigma_t_sq = sprt_sigma_sq

            # Safety floors to prevent division by zero or numerical explosion
            mu_t = max(mu_t, SPRT_MU_FLOOR)
            sigma_t_sq = max(sigma_t_sq, SPRT_SIGMA_SQ_FLOOR)

            # DBC: Discordance updating using Low-Pass Filter
            is_discordant = np.sign(d_t) * np.sign(lambda_llr) < 0
            s_t = (1.0 - SPRT_BETA_S) * s_t + SPRT_BETA_S * float(is_discordant)

            # DBC Parameters Dynamic Modulation
            alpha_t = SPRT_ALPHA_MIN + (SPRT_ALPHA_MAX - SPRT_ALPHA_MIN) * s_t
            B_t = SPRT_BOUND_MIN + (SPRT_BOUND - SPRT_BOUND_MIN) * np.exp(-SPRT_KAPPA * s_t)

            # LLR Accumulation with dynamic leakage alpha_t
            lambda_llr = (1.0 - alpha_t) * lambda_llr + (2.0 * mu_t / sigma_t_sq) * d_t

            # 3-way decision logic with dynamic bound B_t
            if abs(lambda_llr) >= B_t:
                decision = 1 if lambda_llr > 0 else 2
                prev_decision = decision
                lambda_llr = 0.0
                s_t = 0.0  # Reset discordance index on decision trigger
            else:
                # HOLD zone: retain last triggered decision (raw argmax before the first trigger)
                decision = prev_decision if prev_decision is not None else pred_raw

            preds_sprt.append(decision)
            
        acc_raw = np.mean(np.array(preds_raw) == window_gts)
        acc_sprt = np.mean(np.array(preds_sprt) == window_gts)
        
        # --- B. Bayesian No Prior ---
        bayesian_noprior = AADBayesian(fs=FS, tmin=TMIN, tmax=TMAX, sigma_noise=sigma_noise)
        bayesian_noprior.fit_no_prior(calib_eeg, calib_env, lambda_=LAMBDA)
        
        preds_noprior = []
        for w in range(n_windows_rem):
            start, end = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
            pred, _, _ = bayesian_noprior.classify_attention(rem_eeg[start:end], rem_env1[start:end], rem_env2[start:end])
            preds_noprior.append(pred)
        acc_noprior = np.mean(np.array(preds_noprior) == window_gts)
        
        # --- C. Direct Ridge Calibration (Standard P=104 Dimensional Baseline) ---
        direct_ridge = AADmTRF(fs=FS, tmin=TMIN, tmax=TMAX, lambda_=LAMBDA)
        direct_ridge.fit([calib_eeg], [calib_env])
        
        preds_ridge = []
        for w in range(n_windows_rem):
            start, end = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
            pred, _, _ = direct_ridge.classify_attention(rem_eeg[start:end], rem_env1[start:end], rem_env2[start:end])
            preds_ridge.append(pred)
        acc_ridge = np.mean(np.array(preds_ridge) == window_gts)
        
        results.append({
            'subject': f"S{test_sub_id}",
            'calib_s': calib_s,
            'bayesian_raw': acc_raw * 100,
            'bayesian_sprt': acc_sprt * 100,
            'bayesian_no_prior': acc_noprior * 100,
            'direct_ridge_calib': acc_ridge * 100,
            'subject_specific_mtrf': baseline_acc
        })
        
    return results


def main():
    print("=" * 80)
    print("RUNNING TASK D: BAYESIAN LINEAR REGRESSION (CCA-PURGED + LEAKY SPRT)")
    print("=" * 80)
    
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize AAD Experiment and load baseline accuracies
    experiment = AADmTRFExperiment(quick_mode=False)
    baseline_csv = results_dir / "4ch_comparison.csv"
    baseline_accuracies = load_subject_specific_accuracies(baseline_csv)
    
    # Pre-load, pre-filter, and cache all 16 subjects' data with differential channels + Stevens compression
    # New cache directory to avoid stale CCA-contaminated caches
    cache_dir = PROJECT_ROOT / "data" / "cache_v2_nocca"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    subjects_data = {}
    for subject_id in range(1, 17):
        cache_file = cache_dir / f"S{subject_id}_4ch_differential.npy"
        if cache_file.exists():
            print(f"Loading preprocessed subject S{subject_id} from cache (v2_nocca)...")
            subject_trials = np.load(cache_file, allow_pickle=True).tolist()
        else:
            print(f"Loading and pre-filtering Subject S{subject_id} data (4ch differential + Stevens)...")
            subject_trials = []
            for trial_idx in range(20):
                trial_data = experiment.load_trial_data_with_envelopes(subject_id, trial_idx)
                if trial_data is not None:
                    # 4ch spatial constraint (TP9, TP10, Fz, Cz mapped on standard grid)
                    raw_eeg_4ch = trial_data['eeg'][:, AUDIO_4CH_INDICES]
                    
                    # Software Differential Virtual Channel Synthesis
                    diff_eeg_4ch = preprocess_differential_channels(raw_eeg_4ch)
                    
                    # Stevens' Power-Law cochlear compression (alpha = 0.6)
                    env1_comp = compress_envelope(trial_data['envelope_track1'], alpha=0.6)
                    env2_comp = compress_envelope(trial_data['envelope_track2'], alpha=0.6)
                    
                    # Apply standard 1-8 Hz filter to differential EEG and compressed envelopes
                    trial_data['eeg'] = bandpass_filter(diff_eeg_4ch, fs=100, low=1.0, high=8.0, order=3)
                    trial_data['envelope_track1'] = bandpass_filter(env1_comp, fs=100, low=1.0, high=8.0, order=3)
                    trial_data['envelope_track2'] = bandpass_filter(env2_comp, fs=100, low=1.0, high=8.0, order=3)
                    
                    subject_trials.append(trial_data)
            np.save(cache_file, subject_trials, allow_pickle=True)
            print(f"  Saved preprocessed S{subject_id} to cache (v2_nocca).")
            
        subjects_data[subject_id] = subject_trials
        
    # --- PHASE 1: Per-subject decoders (fold-independent, fitted once) ---
    print("\n--- Phase 1: Fitting per-subject decoders for empirical priors ---")
    subject_ids = list(range(1, 17))
    subject_weights = dict(zip(subject_ids, train_subject_decoders(subjects_data, subject_ids)))

    if QUICK_MODE:
        print("\nQUICK_MODE is ACTIVE: Evaluating Subject S1 only.")
        sub_res = evaluate_bayesian_on_subject(
            test_sub_id=1,
            train_ids=subject_ids[1:],
            subjects_data=subjects_data,
            sigma_noise=SIGMA_NOISE,
            baseline_acc=baseline_accuracies.get('S1', np.nan),
            subject_weights=subject_weights
        )
        print("\n--- S1 Quick Results ---")
        for r in sub_res:
            print(f"  Calib {r['calib_s']:3d}s | raw={r['bayesian_raw']:.2f}% sprt={r['bayesian_sprt']:.2f}% "
                  f"noprior={r['bayesian_no_prior']:.2f}% ridge={r['direct_ridge_calib']:.2f}% "
                  f"upper={r['subject_specific_mtrf']:.2f}%")
        sys.exit(0)

    # --- PHASE 2: Full 16-Subject LOOCV Benchmark ---
    print(f"\n--- Phase 2: Full 16-Subject LOOCV Benchmark (sigma_noise = {SIGMA_NOISE}) ---")
    all_results = []

    for test_sub_id in subject_ids:
        sub_name = f"S{test_sub_id}"
        print(f"Processing fold: Test Subject {sub_name}...")
        train_ids = [s for s in subject_ids if s != test_sub_id]

        # Missing baseline -> NaN (excluded from means) instead of a fake 50% "upper bound"
        sub_res = evaluate_bayesian_on_subject(
            test_sub_id=test_sub_id,
            train_ids=train_ids,
            subjects_data=subjects_data,
            sigma_noise=SIGMA_NOISE,
            baseline_acc=baseline_accuracies.get(sub_name, np.nan),
            subject_weights=subject_weights
        )
        all_results.extend(sub_res)
        
        # Clear local structures to save memory
        del sub_res
        gc.collect()
        
    # Save raw results
    df_raw = pd.DataFrame(all_results)
    
    # Save in standard long format for raw results
    long_rows = []
    for _, row in df_raw.iterrows():
        sub = row['subject']
        calib = row['calib_s']
        long_rows.append({'subject': sub, 'model': 'bayesian_raw', 'calib_s': calib, 'accuracy_pct': row['bayesian_raw']})
        long_rows.append({'subject': sub, 'model': 'bayesian_sprt', 'calib_s': calib, 'accuracy_pct': row['bayesian_sprt']})
        long_rows.append({'subject': sub, 'model': 'bayesian_no_prior', 'calib_s': calib, 'accuracy_pct': row['bayesian_no_prior']})
        long_rows.append({'subject': sub, 'model': 'direct_ridge_calib', 'calib_s': calib, 'accuracy_pct': row['direct_ridge_calib']})
        long_rows.append({'subject': sub, 'model': 'subject_specific_mtrf', 'calib_s': calib, 'accuracy_pct': row['subject_specific_mtrf']})
        
    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(results_dir / "bayesian_results.csv", index=False)
    print(f"\nSaved raw long-format results to {results_dir / 'bayesian_results.csv'}")
    
    # --- PHASE 3: Generate Leaderboard Report for Professor ---
    print("\n" + "=" * 80)
    print("GENERATING LEADERBOARD REPORT FOR PROFESSOR")
    print("=" * 80)
    
    # === Sub-Report A: Group-level summary across all calibration times ===
    print("\n--- Group-Level Summary Across All Calibration Times ---")
    for calib_s in CALIB_TIMES:
        df_c = df_raw[df_raw['calib_s'] == calib_s]
        print(f"  {calib_s:3d}s | raw={df_c['bayesian_raw'].mean():.2f}% "
              f"sprt={df_c['bayesian_sprt'].mean():.2f}% "
              f"noprior={df_c['bayesian_no_prior'].mean():.2f}% "
              f"ridge={df_c['direct_ridge_calib'].mean():.2f}% "
              f"upper={df_c['subject_specific_mtrf'].mean():.2f}%")
    
    # === Sub-Report B: Per-subject table at 120s ===
    df_120 = df_raw[df_raw['calib_s'] == 120]
    
    report_df = pd.DataFrame({
        '被験者': df_120['subject'].values,
        '4ch上限(mTRF)': df_120['subject_specific_mtrf'].values,
        'bayesian_raw_120s': df_120['bayesian_raw'].values,
        'bayesian_sprt_120s': df_120['bayesian_sprt'].values,
        'bayesian_no_prior_120s': df_120['bayesian_no_prior'].values,
        'direct_ridge_120s': df_120['direct_ridge_calib'].values,
        'Delta(raw vs ridge)': df_120['bayesian_raw'].values - df_120['direct_ridge_calib'].values,
        'Delta(sprt vs ridge)': df_120['bayesian_sprt'].values - df_120['direct_ridge_calib'].values,
    })
    
    # Mean and Std calculations
    numeric_cols = ['4ch上限(mTRF)', 'bayesian_raw_120s', 'bayesian_sprt_120s', 
                    'bayesian_no_prior_120s', 'direct_ridge_120s', 
                    'Delta(raw vs ridge)', 'Delta(sprt vs ridge)']
    
    mean_row = {'被験者': '平均 (Mean)'}
    std_row = {'被験者': '標準偏差 (Std)'}
    
    for col in numeric_cols:
        mean_row[col] = report_df[col].mean()
        std_row[col] = report_df[col].std()
        
    report_df = pd.concat([report_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save wide comparison report (hybrid recovery report)
    report_csv = results_dir / "hybrid_comparison_report.csv"
    report_df.to_csv(report_csv, index=False)
    print(f"Comparison report successfully saved to {report_csv}")
    
    # Print formatted console report
    fmt_df = report_df.copy()
    for col in numeric_cols:
        if 'Delta' in col:
            fmt_df[col] = report_df[col].apply(
                lambda v: f"+{v:.2f}%" if pd.notna(v) and v > 0 else (f"{v:.2f}%" if pd.notna(v) else "N/A")
            )
        else:
            fmt_df[col] = fmt_df[col].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
    print("\n--- Formatted Leaderboard Table (120s) ---")
    print(fmt_df.to_string(index=False))
    print("=" * 80 + "\n")
    
    # --- PHASE 4: Plot Bayesian Learning Curve ---
    summary = df_long.groupby(['model', 'calib_s'])['accuracy_pct'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(12, 7))
    colors = {
        'bayesian_raw': '#2A9D8F',
        'bayesian_sprt': '#E76F51',
        'bayesian_no_prior': '#264653',
        'direct_ridge_calib': '#E9C46A',
        'subject_specific_mtrf': '#8338EC'
    }
    markers = {
        'bayesian_raw': 'o',
        'bayesian_sprt': 'P',
        'bayesian_no_prior': 's',
        'direct_ridge_calib': 'D',
        'subject_specific_mtrf': '^'
    }
    labels = {
        'bayesian_raw': 'Bayesian w/ Prior (Raw Argmax)',
        'bayesian_sprt': 'Bayesian w/ Prior (Leaky SPRT)',
        'bayesian_no_prior': 'Bayesian No Prior (≡ Ridge)',
        'direct_ridge_calib': 'Direct Ridge Calibration',
        'subject_specific_mtrf': 'Subject-Specific (4ch Upper Bound)'
    }
    
    plot_order = ['bayesian_raw', 'bayesian_sprt', 'bayesian_no_prior', 'direct_ridge_calib', 'subject_specific_mtrf']
    for m in plot_order:
        m_data = summary[summary['model'] == m].sort_values('calib_s')
        if m_data.empty:
            continue
        plt.plot(m_data['calib_s'], m_data['mean'], label=labels[m],
                 color=colors[m], marker=markers[m], linewidth=2.5, markersize=8)
        plt.fill_between(m_data['calib_s'], m_data['mean'] - m_data['std'], m_data['mean'] + m_data['std'],
                         color=colors[m], alpha=0.15)
                         
    plt.axhline(50.0, color='black', linestyle='--', alpha=0.5, label='Chance (50%)')
    plt.xscale('log')
    plt.xticks(CALIB_TIMES, [f"{t}s" for t in CALIB_TIMES], fontsize=11)
    plt.xlabel("Calibration Data Duration (log scale)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(40, 100)
    plt.title("Bayesian Few-Shot Adaptation: Raw vs Leaky SPRT", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "bayesian_learning_curve.png", dpi=150)
    plt.close()
    print("Bayesian Learning Curve figure successfully saved!")


if __name__ == "__main__":
    main()
