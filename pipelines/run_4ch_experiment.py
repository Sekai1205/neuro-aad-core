"""
Task A: 4-Channel Spatial Constraint Simulation vs 64-Channel Baseline

Evaluates AAD accuracy with 3 channel configurations × 2 window sizes across 16 subjects.
1. BASELINE_64CH (all 64 channels)
2. CONFIG_AUDIO (4 channels: TP7, TP8, Fz, Cz)
3. CONFIG_VISUAL (4 channels: O1, Oz, O2, POz — negative control)

Evaluates at 10-second and 30-second windows under Leave-One-Trial-Out CV.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import gc

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mtrf.experiment_aad_mtrf import AADmTRFExperiment
from pipelines.channel_selector import (
    channel_selector, CONFIG_AUDIO, CONFIG_VISUAL, BASELINE_64CH,
    CONFIG_AUDIO_NAMES, CONFIG_VISUAL_NAMES
)
from mtrf.aad_mtrf import AADmTRF

# Set random seeds
np.random.seed(42)

def bandpass_filter(data, fs=100, low=1.0, high=8.0, order=3):
    """1-8 Hz bandpass filter matching the baseline pipeline."""
    b, a = signal.butter(order, [low, high], btype='band', fs=fs)
    return signal.filtfilt(b, a, data, axis=0)

def main():
    print("=" * 80)
    print("RUNNING TASK A: 4-CHANNEL SPATIAL CONSTRAINT SIMULATION")
    print("=" * 80)

    # Output directories
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "4ch_comparison.csv"

    # Initialize experiment class (shares the envelope cache)
    experiment = AADmTRFExperiment(quick_mode=False)

    # Settings
    configs = {
        'BASELINE_64CH': BASELINE_64CH,
        'CONFIG_AUDIO': CONFIG_AUDIO,
        'CONFIG_VISUAL': CONFIG_VISUAL
    }
    window_sizes = [10, 30]  # seconds

    # Check for existing results (Checkpoint recovery)
    completed_subjects = set()
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            if 'subject' in df_existing.columns:
                completed_subjects = set(df_existing['subject'].tolist())
                print(f"Found existing CSV {csv_path}. Completed subjects: {completed_subjects}")
        except Exception as e:
            print(f"Could not read existing CSV: {e}. Will create new or overwrite.")

    # Run subjects 1 to 16
    for subject_id in range(1, 17):
        sub_name = f"S{subject_id}"
        if sub_name in completed_subjects:
            print(f"\n>>> Subject {sub_name} already completed. Skipping...")
            continue
            
        print(f"\n--- Processing Subject S{subject_id} ---")
        
        # Load all 20 trials once to avoid repeated loading
        subject_trials = []
        for trial_idx in range(20):
            trial_data = experiment.load_trial_data_with_envelopes(subject_id, trial_idx)
            if trial_data is not None:
                subject_trials.append(trial_data)
        
        if len(subject_trials) < 20:
            print(f"Error: Could only load {len(subject_trials)}/20 trials for S{subject_id}")
            # Memory cleanup
            del subject_trials
            gc.collect()
            continue

        subject_results = []

        for config_name, config_indices in configs.items():
            print(f"  Configuration: {config_name}")
            
            # Apply channel selector after downsampling but before bandpass filtering
            preprocessed_trials = []
            for trial in subject_trials:
                eeg_sel = channel_selector(trial['eeg'], config_indices)
                eeg_filt = bandpass_filter(eeg_sel)
                env1_filt = bandpass_filter(trial['envelope_track1'])
                env2_filt = bandpass_filter(trial['envelope_track2'])
                preprocessed_trials.append({
                    'eeg': eeg_filt,
                    'envelope_track1': env1_filt,
                    'envelope_track2': env2_filt,
                    'ground_truth_track': trial['ground_truth_track']
                })
            
            for win_s in window_sizes:
                samples_per_window = win_s * 100  # 100 Hz sampling rate
                
                # Leave-One-Trial-Out CV
                all_predictions = []
                all_ground_truths = []
                all_r_attended = []
                
                for test_idx in range(20):
                    train_indices = [i for i in range(20) if i != test_idx]
                    
                    # Prepare training data
                    train_eeg_list = [preprocessed_trials[i]['eeg'] for i in train_indices]
                    train_attended_env_list = []
                    for i in train_indices:
                        gt = preprocessed_trials[i]['ground_truth_track']
                        env = preprocessed_trials[i]['envelope_track1'] if gt == 1 else preprocessed_trials[i]['envelope_track2']
                        train_attended_env_list.append(env)
                    
                    # Train model
                    aad = AADmTRF(fs=100, tmin=-0.05, tmax=0.4, lambda_=0.1)
                    aad.fit(train_eeg_list, train_attended_env_list)
                    
                    # Test trial windows
                    test_trial = preprocessed_trials[test_idx]
                    test_eeg = test_trial['eeg']
                    test_env1 = test_trial['envelope_track1']
                    test_env2 = test_trial['envelope_track2']
                    test_gt = test_trial['ground_truth_track']
                    
                    n_windows = test_eeg.shape[0] // samples_per_window
                    for w in range(n_windows):
                        start_idx = w * samples_per_window
                        end_idx = (w + 1) * samples_per_window
                        
                        win_eeg = test_eeg[start_idx:end_idx, :]
                        win_env1 = test_env1[start_idx:end_idx]
                        win_env2 = test_env2[start_idx:end_idx]
                        
                        pred, r1, r2 = aad.classify_attention(win_eeg, win_env1, win_env2)
                        all_predictions.append(pred)
                        all_ground_truths.append(test_gt)
                        
                        r_att = r1 if test_gt == 1 else r2
                        all_r_attended.append(r_att)
                    
                    # Memory cleanup inside LOOCV loop
                    del aad
                    gc.collect()
                
                accuracy = np.mean(np.array(all_predictions) == np.array(all_ground_truths))
                mean_r = np.mean(all_r_attended)
                
                print(f"    Window {win_s}s: Accuracy = {accuracy:.2%}, Mean Pearson r = {mean_r:.4f}")
                
                subject_results.append({
                    'subject': f"S{subject_id}",
                    'config': config_name,
                    'window_s': win_s,
                    'r': mean_r,
                    'accuracy_pct': accuracy * 100
                })
            
            # Explicit cleanup of preprocessed arrays
            del preprocessed_trials
            gc.collect()

        # Save this subject's results to CSV progressively
        if subject_results:
            df_sub = pd.DataFrame(subject_results)
            header = not csv_path.exists() or csv_path.stat().st_size == 0
            df_sub.to_csv(csv_path, mode='a', index=False, header=header)
            print(f"Saved progressive results for {sub_name} to {csv_path}")
            
        # Clean up subject level variables
        del subject_trials
        del subject_results
        gc.collect()

    # Verify CSV file and load full results for figures
    if not csv_path.exists():
        print("Error: No results file generated.")
        return
        
    df_full = pd.read_csv(csv_path)
    print(f"\nTotal loaded rows from {csv_path}: {len(df_full)}")

    # Generate Figures
    generate_figures(df_full, figures_dir)
    print("Figures successfully generated!")

def generate_figures(df, figures_dir):
    """Generate high-quality plots for analysis."""
    # Compute group level summary (mean + SD)
    summary = df.groupby(['config', 'window_s'])['accuracy_pct'].agg(['mean', 'std']).reset_index()
    print("\n=== Group-Level Accuracy Summary (%) ===")
    print(summary.to_string(index=False))

    # Success criteria check
    audio_10s_mean = summary[(summary['config'] == 'CONFIG_AUDIO') & (summary['window_s'] == 10)]['mean'].values[0]
    print(f"\nCONFIG_AUDIO at 10s group mean accuracy: {audio_10s_mean:.2f}%")
    if audio_10s_mean >= 55.0:
        print("✓ SUCCESS CRITERION MET: CONFIG_AUDIO at 10s >= 55%")
    else:
        print("✗ SUCCESS CRITERION FAILED: CONFIG_AUDIO at 10s < 55%")

    # Figure 1: Grouped bar chart (configs x windows)
    plt.figure(figsize=(10, 6))
    
    configs_order = ['BASELINE_64CH', 'CONFIG_AUDIO', 'CONFIG_VISUAL']
    windows = [10, 30]
    colors = {
        'BASELINE_64CH': '#2A9D8F',
        'CONFIG_AUDIO': '#E76F51',
        'CONFIG_VISUAL': '#E9C46A'
    }
    
    x = np.arange(len(windows))
    width = 0.25
    
    for idx, conf in enumerate(configs_order):
        means = [summary[(summary['config'] == conf) & (summary['window_s'] == w)]['mean'].values[0] for w in windows]
        stds = [summary[(summary['config'] == conf) & (summary['window_s'] == w)]['std'].values[0] for w in windows]
        
        plt.bar(x + (idx - 1) * width, means, width, yerr=stds, label=conf,
                color=colors[conf], capsize=5, edgecolor='black', alpha=0.9)
        
    plt.axhline(50.0, color='black', linestyle='--', alpha=0.5, label='Chance (50%)')
    plt.xticks(x, [f"{w}s Window" for w in windows], fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(30, 100)
    plt.title("AAD Accuracy: Electrode Configuration and Decision Window Effects", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "4ch_bar.png", dpi=150)
    plt.close()

    # Figure 2: Per-subject line plot comparing configs at 10s
    plt.figure(figsize=(12, 6))
    
    df_10s = df[df['window_s'] == 10]
    subjects = [f"S{i}" for i in range(1, 17)]
    
    markers = {
        'BASELINE_64CH': 'o',
        'CONFIG_AUDIO': 's',
        'CONFIG_VISUAL': '^'
    }
    
    for conf in configs_order:
        conf_data = df_10s[df_10s['config'] == conf]
        # Align subject order
        conf_data = conf_data.set_index('subject').reindex(subjects).reset_index()
        
        plt.plot(conf_data['subject'], conf_data['accuracy_pct'], label=conf,
                 color=colors[conf], marker=markers[conf], linewidth=2, markersize=8)
        
    plt.axhline(50.0, color='black', linestyle='--', alpha=0.5, label='Chance (50%)')
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Subject", fontsize=12)
    plt.ylim(30, 100)
    plt.title("Individual Subject AAD Performance at 10s Window", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "4ch_subject.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
