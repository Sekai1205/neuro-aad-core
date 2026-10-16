"""
Task E: PC-based Virtual Real-Time Attention Tracking Simulator.

Simulates a real-time BCI hearing aid system:
1. Chronologically concatenates all 20 trials of subject S1 to form a continuous 45-minute BCI session.
2. The first 120 seconds are used as the "Calibration Phase" to adapt the Bayesian Decoder with empirical priors from S2-S16.
3. For the remaining session, the system decodes attention direction in 10-second windows.
4. Evidences are integrated using a stateful Leaky Integrator to track real-time attention switches.
5. Saves a beautiful time-series tracking plot to figures/realtime_simulation_S1.png.
"""

import sys
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
from pipelines.run_bayesian import BayesianMTRFDecoder, bandpass_filter, train_subject_decoders, AUDIO_4CH_INDICES

# Constants
TMIN = 0.0
TMAX = 0.25
FS = 100
LAMBDA = 0.001
SIGMA_NOISE = 10.0
WINDOW_SIZE = 1000  # 10s window @ 100Hz
CALIB_S = 120       # 120s calibration phase


class LeakyIntegrator:
    """Stateful Leaky Integrator representing the integration cup."""
    def __init__(self, leak_rate: float = 0.15):
        self.leak_rate = leak_rate
        self.score = 0.0  # Current level in the cup

    def update(self, input_val: float) -> float:
        """S_t = (1 - alpha) * S_{t-1} + alpha * input_val"""
        self.score = (1.0 - self.leak_rate) * self.score + self.leak_rate * input_val
        return self.score


def main():
    print("=" * 80)
    print("RUNNING VIRTUAL REAL-TIME BCI ATTENTION TRACKING SIMULATOR")
    print("=" * 80)
    
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize AAD experiment to load KULeuven data
    experiment = AADmTRFExperiment(quick_mode=False)
    
    # 1. Load Subject S1 (test subject) and S2-S16 (train subjects)
    subjects_data = {}
    for subject_id in range(1, 17):
        print(f"Loading Subject S{subject_id} trials...")
        subject_trials = []
        for trial_idx in range(20):
            trial_data = experiment.load_trial_data_with_envelopes(subject_id, trial_idx)
            if trial_data is not None:
                trial_data['eeg'] = trial_data['eeg'][:, AUDIO_4CH_INDICES]
                trial_data['eeg'] = bandpass_filter(trial_data['eeg'])
                trial_data['envelope_track1'] = bandpass_filter(trial_data['envelope_track1'])
                trial_data['envelope_track2'] = bandpass_filter(trial_data['envelope_track2'])
                subject_trials.append(trial_data)
        subjects_data[subject_id] = subject_trials
        
    print("\n--- Phase 1: Constructing Prior weights from S2-S16 ---")
    train_ids = list(range(2, 17))
    prior_weights = train_subject_decoders(subjects_data, train_ids)
    
    # 2. Concatenate Subject S1 trials in chronological order to form a continuous session
    s1_trials = subjects_data[1]
    eeg_concat = np.concatenate([t['eeg'] for t in s1_trials], axis=0)
    env1_concat = np.concatenate([t['envelope_track1'] for t in s1_trials], axis=0)
    env2_concat = np.concatenate([t['envelope_track2'] for t in s1_trials], axis=0)
    
    # Construct a continuous ground truth array (1 or 2)
    # Also attended envelope sequence for calibration
    attended_env_concat = []
    ground_truth_seq = []
    
    for t in s1_trials:
        n_samples = t['eeg'].shape[0]
        gt = t['ground_truth_track']
        ground_truth_seq.extend([gt] * n_samples)
        attended_env = t['envelope_track1'] if gt == 1 else t['envelope_track2']
        attended_env_concat.extend(attended_env)
        
    ground_truth_seq = np.array(ground_truth_seq)
    attended_env_concat = np.array(attended_env_concat)
    
    total_samples = eeg_concat.shape[0]
    total_minutes = total_samples / (FS * 60)
    print(f"Continuous BCI Session constructed: {total_samples} samples (~{total_minutes:.1f} minutes)")
    
    # 3. Calibration Phase: Adapt Bayesian Decoder on first 120 seconds
    print(f"\n--- Phase 2: Calibration Phase (First {CALIB_S}s of data) ---")
    calib_samples = CALIB_S * FS
    calib_eeg = eeg_concat[:calib_samples]
    calib_env = attended_env_concat[:calib_samples]
    
    bayes_decoder = BayesianMTRFDecoder(fs=FS, tmin=TMIN, tmax=TMAX, sigma_noise=SIGMA_NOISE)
    bayes_decoder.set_prior_from_others(prior_weights)
    bayes_decoder.adapt_online(calib_eeg, calib_env, lambda_=LAMBDA)
    print("Bayesian decoder successfully adapted and calibrated!")
    
    # 4. Online Decoding Simulation Loop (every 10 seconds)
    print("\n--- Phase 3: Online Real-Time Decoding Simulation ---")
    integrator = LeakyIntegrator(leak_rate=0.15)  # 15% leak rate
    
    time_history = []
    raw_evidences = []
    smoothed_scores = []
    gt_history = []
    predictions = []
    
    # Step forward by 10s windows
    num_windows = (total_samples - calib_samples) // WINDOW_SIZE
    print(f"Decoding session: {num_windows} windows (10-second steps)")
    
    for w in range(num_windows):
        start = calib_samples + w * WINDOW_SIZE
        end = start + WINDOW_SIZE
        current_time = CALIB_S + w * 10.0  # seconds
        
        win_eeg = eeg_concat[start:end]
        win_env1 = env1_concat[start:end]
        win_env2 = env2_concat[start:end]
        win_gt = ground_truth_seq[start + WINDOW_SIZE // 2]
        
        # Pearson correlations with both speakers
        r1 = bayes_decoder.score(win_eeg, win_env1)
        r2 = bayes_decoder.score(win_eeg, win_env2)
        
        # Evidence: attention to Track 1 (positive) vs Track 2 (negative)
        raw_evidence = r1 - r2
        
        # Stateful integration
        smoothed_score = integrator.update(raw_evidence)
        
        # Decision rule: positive score -> track 1, negative score -> track 2
        prediction = 1 if smoothed_score > 0 else 2
        
        # Ground truth scale adjusted for overlay plotting (-0.05 or +0.05)
        gt_scaled = 0.05 if win_gt == 1 else -0.05
        
        time_history.append(current_time)
        raw_evidences.append(raw_evidence)
        smoothed_scores.append(smoothed_score)
        gt_history.append(gt_scaled)
        predictions.append(prediction)
        
    time_history = np.array(time_history)
    raw_evidences = np.array(raw_evidences)
    smoothed_scores = np.array(smoothed_scores)
    gt_history = np.array(gt_history)
    predictions = np.array(predictions)
    
    # Calculate real-time tracking accuracy
    tracking_accuracy = np.mean(predictions == ground_truth_seq[calib_samples + WINDOW_SIZE//2::WINDOW_SIZE][:len(predictions)])
    print(f"\nReal-Time Tracking Complete! Overall Session Accuracy: {tracking_accuracy:.2%}")
    
    # 5. Save Simulation CSV
    sim_df = pd.DataFrame({
        'time_s': time_history,
        'raw_evidence': raw_evidences,
        'smoothed_score': smoothed_scores,
        'ground_truth': (gt_history > 0).astype(int) + 1,
        'prediction': predictions
    })
    sim_csv = results_dir / "realtime_simulation_results.csv"
    sim_df.to_csv(sim_csv, index=False)
    print(f"Simulation results successfully saved to {sim_csv}")
    
    # 6. Plot the majestic continuous tracking curve
    plt.figure(figsize=(14, 6))
    
    # Plot raw evidences in light gray
    plt.plot(time_history / 60.0, raw_evidences, color='#CCCCCC', alpha=0.6, label='Raw Differential Correlation (r1 - r2)')
    
    # Plot smoothed integrated scores in professional ocean-blue
    plt.plot(time_history / 60.0, smoothed_scores, color='#2A9D8F', linewidth=2.5, label='Leaky Integrated Attention Score (Alpha=0.15)')
    
    # Plot ground truth shifts in bright orange
    plt.step(time_history / 60.0, gt_history, where='mid', color='#E76F51', linewidth=2.0, linestyle='--', label='Ground Truth Attended Speaker')
    
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.xlabel("Session Duration (minutes)", fontsize=12)
    plt.ylabel("Attention Evidence Score\n(<- Track 2 (Right)  |  Track 1 (Left) ->)", fontsize=12)
    plt.title(f"Virtual Real-Time Attention Tracking Simulation on S1\n(10s Decision Window, Calibration: 120s, Overall Accuracy: {tracking_accuracy:.2%})", 
              fontsize=14, fontweight='bold')
    plt.ylim(-0.15, 0.15)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plot_path = figures_dir / "realtime_simulation_S1.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Continuous attention tracking plot successfully saved to {plot_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
