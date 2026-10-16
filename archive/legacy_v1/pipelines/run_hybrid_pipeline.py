"""
Task C: Full Benchmark and Continuous Tracking Simulation for the Integrated Hybrid BCI Pipeline.
Runs a 16-subject LOOCV simulation using the 5-layer Hybrid AAD pipeline (EA-ST-ESN-Stiefel-HMM) 
under hardware constraints (fs_target=64Hz, calibration_time=10 seconds!).
Saves tracking plots to figures/hybrid_realtime_tracking.png.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import gc

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mtrf.experiment_aad_mtrf import AADmTRFExperiment
from pipelines.run_bayesian import load_subject_specific_accuracies, bandpass_filter
from hybrid_bci.simulator import HybridBCISimulator

# Random seeds
np.random.seed(42)

# Quick mode config
QUICK_MODE = False  # Set to True for S1-only validation run, False for full 16-subj benchmark

def main():
    print("=" * 80)
    print("INTEGRATED HYBRID BCI ARCHITECTURE: BENCHMARK & SIMULATION")
    print("=" * 80)
    
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dynamic baseline accuracies (4ch CONFIG_AUDIO 10s baseline)
    baseline_csv = results_dir / "4ch_comparison.csv"
    baseline_accuracies = load_subject_specific_accuracies(baseline_csv)
    
    # Pre-load and pre-filter all 16 subjects' EEG data using 4ch electrode constraints
    # TP7, TP8, Fz, Cz. To match Ganglion/Cyton, we downsample and filter EMG in the pipeline.
    AUDIO_4CH_INDICES = [15, 52, 37, 47]
    
    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Shared experiment instance to reuse envelopes_cache across all subjects
    experiment = AADmTRFExperiment(quick_mode=False)
    
    subjects_data = {}
    for subject_id in range(1, 17):
        cache_file = cache_dir / f"S{subject_id}_4ch_preprocessed.npy"
        if cache_file.exists():
            print(f"Loading preprocessed S{subject_id} from cache...")
            subject_trials = np.load(cache_file, allow_pickle=True).tolist()
        else:
            print(f"Loading and pre-filtering Subject S{subject_id} data (4ch constraint)...")
            subject_trials = []
            for trial_idx in range(20):
                trial_data = experiment.load_trial_data_with_envelopes(subject_id, trial_idx)
                if trial_data is not None:
                    # Pre-apply 4-channel spatial constraint
                    trial_data['eeg'] = trial_data['eeg'][:, AUDIO_4CH_INDICES]
                    trial_data['eeg'] = bandpass_filter(trial_data['eeg'])
                    trial_data['envelope_track1'] = bandpass_filter(trial_data['envelope_track1'])
                    trial_data['envelope_track2'] = bandpass_filter(trial_data['envelope_track2'])
                    subject_trials.append(trial_data)
            np.save(cache_file, subject_trials, allow_pickle=True)
            print(f"  Saved preprocessed S{subject_id} to cache.")
            
        subjects_data[subject_id] = subject_trials
        
    # Set up simulator
    # 10s ultra-fast rocket start calibration, sliding with 2s step size for smooth tracking curve
    simulator = HybridBCISimulator(
        fs=100, fs_target=64, n_reservoir=200, lambda_=0.001, 
        transition_prob=0.02, calib_s=10.0, step_s=2.0
    )
    
    if QUICK_MODE:
        print("\n--- QUICK_MODE: Simulating Subject S1 ---")
        sim_df = simulator.run_simulation(subjects_data, test_sub_id=1)
        
        # Calculate accuracy
        accuracy = np.mean(sim_df['prediction'] == sim_df['ground_truth'])
        print(f"\nS1 Virtual Real-Time Simulation Complete!")
        print(f"S1 Overall Session Tracking Accuracy: {accuracy:.2%}")
        
        # Plot continuous tracking curve
        plot_realtime_tracking(sim_df, figures_dir / "hybrid_realtime_tracking.png", 1, accuracy)
        print(f"Tracking plot saved to {figures_dir / 'hybrid_realtime_tracking.png'}")
        
        # Save S1 simulation results
        sim_df.to_csv(results_dir / "hybrid_results_quick.csv", index=False)
        print("QUICK_MODE execution successfully completed!")
        return
        
    # --- Full LOOCV Benchmark ---
    print("\n--- Running Full 16-Subject LOOCV Benchmark (10s Rocket-Start Calibration) ---")
    loocv_accuracies = {}
    s1_sim_df = None
    
    for subject_id in range(1, 17):
        sub_name = f"S{subject_id}"
        print(f"\nProcessing Fold: Test Subject {sub_name}...")
        
        # Run continuous session simulation
        sim_df = simulator.run_simulation(subjects_data, subject_id)
        
        # Calculate tracking accuracy
        accuracy = np.mean(sim_df['prediction'] == sim_df['ground_truth'])
        loocv_accuracies[sub_name] = accuracy * 100
        print(f"Fold {sub_name} Complete! Session Tracking Accuracy = {accuracy:.2%}")
        
        # Store S1 simulation for plotting
        if subject_id == 1:
            s1_sim_df = sim_df
            
        # Clean memory progressively
        del sim_df
        gc.collect()
        
    print("\n" + "=" * 80)
    print("LOOCV BENCHMARK COMPLETED: GENERATING COMPARISON REPORT")
    print("=" * 80)
    
    # Save S1 plot
    if s1_sim_df is not None:
        plot_realtime_tracking(s1_sim_df, figures_dir / "hybrid_realtime_tracking.png", 1, loocv_accuracies['S1']/100.0)
        print(f"Continuous tracking plot saved to {figures_dir / 'hybrid_realtime_tracking.png'}")
        
    # Build Leaderboard Comparison Report matching run_bayesian/fomaml structure
    report_rows = []
    for subject_id in range(1, 17):
        sub_name = f"S{subject_id}"
        upper_bound = baseline_accuracies.get(sub_name, 50.0)
        hybrid_acc = loocv_accuracies[sub_name]
        report_rows.append({
            '被験者': sub_name,
            '4ch上限(mTRF)': upper_bound,
            'Hybrid_10s_Calib': hybrid_acc,
            'Delta(Hybrid vs UpperBound)': hybrid_acc - upper_bound
        })
        
    report_df = pd.DataFrame(report_rows)
    
    # Add Mean and Std rows
    mean_row = {
        '被験者': '平均 (Mean)',
        '4ch上限(mTRF)': report_df['4ch上限(mTRF)'].mean(),
        'Hybrid_10s_Calib': report_df['Hybrid_10s_Calib'].mean(),
        'Delta(Hybrid vs UpperBound)': report_df['Delta(Hybrid vs UpperBound)'].mean()
    }
    std_row = {
        '被験者': '標準偏差 (Std)',
        '4ch上限(mTRF)': report_df['4ch上限(mTRF)'].std(),
        'Hybrid_10s_Calib': report_df['Hybrid_10s_Calib'].std(),
        'Delta(Hybrid vs UpperBound)': report_df['Delta(Hybrid vs UpperBound)'].std()
    }
    
    report_df = pd.concat([report_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save report
    report_csv = results_dir / "hybrid_comparison_report.csv"
    report_df.to_csv(report_csv, index=False)
    print(f"Comparison report saved to {report_csv}")
    
    # Print formatted Markdown table
    print("\n--- Formatted Leaderboard Table ---")
    fmt_df = report_df.copy()
    for col in ['4ch上限(mTRF)', 'Hybrid_10s_Calib']:
        fmt_df[col] = fmt_df[col].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
    fmt_df['Delta(Hybrid vs UpperBound)'] = report_df['Delta(Hybrid vs UpperBound)'].apply(
        lambda v: f"+{v:.2f}%" if pd.notna(v) and v > 0 else (f"{v:.2f}%" if pd.notna(v) else "N/A")
    )
    print(fmt_df.to_string(index=False))
    print("=" * 80 + "\n")


def plot_realtime_tracking(df: pd.DataFrame, save_path: Path, subject_id: int, accuracy: float):
    """
    Plots a professional, beautiful continuous time-series tracking plot of HMM posteriors.
    """
    plt.figure(figsize=(14, 6))
    
    # Minutes timeline
    time_min = df['time_s'] / 60.0
    
    # Raw Pearson difference in light gray
    plt.plot(time_min, df['raw_evidence'], color='#DDDDDD', alpha=0.5, label='Raw ESN Reconstruction Evidence (r1 - r2)')
    
    # Stateful HMM posterior belief (Track 1) in rich ocean-green
    plt.plot(time_min, df['hmm_posterior_track1'], color='#1A5F7A', linewidth=2.5, 
             label='Layer 5: HMM Stateful Attention Belief P(z_t = Left | Evidence)')
    
    # Step labels for ground truth: scale to fit the 0-1 probability axis
    # Attended Track 1 = 0.9 (Top), Track 2 = 0.1 (Bottom)
    gt_scaled = np.where(df['ground_truth'] == 1, 0.9, 0.1)
    plt.step(time_min, gt_scaled, where='mid', color='#E76F51', linewidth=2.0, linestyle='--', 
             label='Ground Truth Attended Speaker (Left vs Right)')
    
    plt.axhline(0.5, color='black', linestyle=':', alpha=0.5, label='Decision Boundary (0.5)')
    
    plt.xlabel("Session Duration (minutes)", fontsize=12)
    plt.ylabel("HMM Stateful Attention Probability\n(<- Attend Right (Track 2) | Attend Left (Track 1) ->)", fontsize=12)
    plt.ylim(-0.2, 1.2)
    
    plt.title(f"Continuous Real-Time Attention Tracking on Subject S{subject_id}\n"
              f"(Integrated Hybrid BCI Pipeline: 10s Calibration, 64Hz Stream, Accuracy: {accuracy:.2%})", 
              fontsize=14, fontweight='bold')
    
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
