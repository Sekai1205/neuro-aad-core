"""
AAD Experiment using mTRF Envelope Reconstruction
Biesmans et al. (2016) approach: Attended speaker detection via correlation comparison

Procedure:
- LOOCV (Leave-One-Trial-Out)
- For each trial:
  1. Train mTRF on 19 remaining trials
  2. For test trial, compute r(EEG, envelope_track1) and r(EEG, envelope_track2)
  3. Predict: argmax(r1, r2)
  4. Compare to true attended track
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import FS, DATA_DIR, RESULTS_DIR
from data.kuleuven_loader import load_trial_data
from mtrf.envelope import extract_envelope_from_audio_file, downsample_eeg, align_envelope_eeg
from mtrf.aad_mtrf import AADmTRF

warnings.filterwarnings('ignore')


class AADmTRFExperiment:
    """
    AAD experiment using mTRF-based attended speaker detection.
    """
    
    def __init__(self, quick_mode: bool = True):
        """
        Initialize experiment.
        
        Args:
            quick_mode: If True, run only on S1. If False, all 16 subjects.
        """
        self.quick_mode = quick_mode
        self.envelopes_cache = {}
        self.results_dir = Path(RESULTS_DIR) / "aad_mtrf_envelope"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("AAD Experiment: mTRF Envelope Reconstruction (Biesmans et al. 2016)")
        print("="*70)
        print(f"Quick mode (S1 only): {quick_mode}")
        print(f"Results directory: {self.results_dir}")
    
    def load_trial_data_with_envelopes(self, subject_id: int, trial_idx: int, 
                                        target_fs: float = 100) -> dict:
        """
        Load EEG and extract both speaker envelopes for a trial.
        
        Args:
            subject_id: Subject ID (1-16)
            trial_idx: Trial index (0-19)
            target_fs: Downsampling rate (Hz)
        
        Returns:
            dict: 'eeg', 'envelope_track1', 'envelope_track2', 'ground_truth_track'
        """
        try:
            import scipy.io
            
            # Load trial data
            trial_data = load_trial_data(subject_id, trial_idx)
            eeg = trial_data['eeg']  # (samples, 64)
            metadata = trial_data['metadata']
            
            # Get mat file to access stimulus info
            mat_path = DATA_DIR / f"S{subject_id}.mat"
            mat = scipy.io.loadmat(str(mat_path))
            trials = mat['trials']
            trial_meta = trials[0][trial_idx]
            
            stimuli = trial_meta['stimuli'][0][0]
            attended_ear = str(trial_meta['attended_ear'][0][0][0])  # 'L' or 'R'
            
            # stimuli[0] is Left ear, stimuli[1] is Right ear
            # We map 'L' -> 1 (envelope_track1), 'R' -> 2 (envelope_track2)
            attended_track_idx = 1 if attended_ear == 'L' else 2
            
            # Get both speaker filenames
            envelopes_dict = {}
            for track_num in [1, 2]:
                stimuli_element = stimuli[track_num - 1][0]
                if hasattr(stimuli_element, 'item'):
                    filename = stimuli_element.item()
                elif isinstance(stimuli_element, np.ndarray):
                    filename = str(stimuli_element[0]) if len(stimuli_element) > 0 else str(stimuli_element)
                else:
                    filename = str(stimuli_element)
                
                # Load audio and extract envelope
                if filename in self.envelopes_cache:
                    envelope = self.envelopes_cache[filename]
                else:
                    audio_path = DATA_DIR / "stimuli" / filename
                    if not audio_path.exists():
                        raise FileNotFoundError(f"Audio file not found: {audio_path}")
                    
                    envelope, _ = extract_envelope_from_audio_file(str(audio_path), target_fs=target_fs)
                    self.envelopes_cache[filename] = envelope
                    
                envelopes_dict[f'envelope_track{track_num}'] = envelope
            
            # Downsample EEG
            eeg_ds, _ = downsample_eeg(eeg, metadata['sample_rate'], fs_target=target_fs)
            
            # Align all signals to same length
            min_length = min(eeg_ds.shape[0], 
                           envelopes_dict['envelope_track1'].shape[0],
                           envelopes_dict['envelope_track2'].shape[0])
            
            result = {
                'eeg': eeg_ds[:min_length, :],
                'envelope_track1': envelopes_dict['envelope_track1'][:min_length],
                'envelope_track2': envelopes_dict['envelope_track2'][:min_length],
                'ground_truth_track': attended_track_idx
            }
            
            return result
        
        except Exception as e:
            print(f"  Error loading trial S{subject_id} trial {trial_idx}: {e}")
            return None
    
    def run_subject_loocv(self, subject_id: int) -> dict:
        """
        Run LOOCV for a subject using WINDOW-LEVEL evaluation.
        
        Standard AAD evaluation: Window-based (not trial-based)
        - Split each trial into 10-second windows (Geirnaert 2021)
        - For each window: compare r_attended vs r_unattended
        - Compute accuracy across all windows in test trial
        
        LOOCV: trial-level (train on 19 trials, test on 1 trial)
        Evaluation: window-level (compute accuracy per window)
        
        Args:
            subject_id: Subject ID (1-16)
        
        Returns:
            Results dict with window-level metrics
        """
        print(f"\n{'='*70}")
        print(f"Subject S{subject_id}: LOOCV with Window-Level Evaluation")
        print(f"Window size: 10 seconds (1000 samples @ 100Hz)")
        print(f"LOOCV: trial-level (train=19, test=1)")
        print(f"Evaluation: window-level (per 10-sec window)")
        print(f"{'='*70}")
        
        # Load all trials
        trials_data = []
        print(f"Loading 20 trials...")
        for trial_idx in range(20):
            print(f"  Trial {trial_idx+1}/20...", end='', flush=True)
            data = self.load_trial_data_with_envelopes(subject_id, trial_idx)
            if data is not None:
                trials_data.append(data)
                print(" ✓")
            else:
                print(" ✗")
        
        if len(trials_data) < 20:
            print(f"ERROR: Only loaded {len(trials_data)}/20 trials")
            return None
        
        print(f"Loaded {len(trials_data)} trials successfully")
        
        # Extract ground truth
        ground_truth = np.array([d['ground_truth_track'] for d in trials_data])
        print(f"Label distribution: Track1={np.sum(ground_truth==1)}, Track2={np.sum(ground_truth==2)}")
        
        # Window parameters
        window_size = 1000  # 10 seconds @ 100 Hz
        
        # Run LOOCV
        print(f"\nRunning Leave-One-Trial-Out CV (window-level evaluation)...")
        
        from scipy import signal
        def bandpass_filter(data, fs=100, low=1.0, high=8.0, order=3):
            b, a = signal.butter(order, [low, high], btype='band', fs=fs)
            return signal.filtfilt(b, a, data, axis=0)
            
        # Pre-filter all trials to save time
        print("Applying 1-8Hz bandpass filter to EEG and envelopes...")
        for i in range(len(trials_data)):
            trials_data[i]['eeg'] = bandpass_filter(trials_data[i]['eeg'])
            trials_data[i]['envelope_track1'] = bandpass_filter(trials_data[i]['envelope_track1'])
            trials_data[i]['envelope_track2'] = bandpass_filter(trials_data[i]['envelope_track2'])
            
        all_window_predictions = []
        all_window_ground_truth = []
        trial_accuracies = []
        
        for test_idx in range(20):
            # Training set (all except test_idx)
            train_indices = [i for i in range(20) if i != test_idx]
            train_eeg_list = [trials_data[i]['eeg'] for i in train_indices]
            
            # Extract ONLY attended envelope per trial based on ground truth
            train_attended_env_list = []
            for i in train_indices:
                gt = trials_data[i]['ground_truth_track']
                if gt == 1:
                    env = trials_data[i]['envelope_track1']
                else:
                    env = trials_data[i]['envelope_track2']
                train_attended_env_list.append(env)
            
            # Test trial
            test_eeg = trials_data[test_idx]['eeg']
            test_env1 = trials_data[test_idx]['envelope_track1']
            test_env2 = trials_data[test_idx]['envelope_track2']
            test_ground_truth = trials_data[test_idx]['ground_truth_track']
            
            # Train AAD backward decoder on attended envelopes only
            aad = AADmTRF(fs=100, tmin=-0.05, tmax=0.4, lambda_=0.1)
            aad.fit(train_eeg_list, train_attended_env_list)
            # Evaluate on test trial at WINDOW level
            n_windows = test_eeg.shape[0] // window_size
            window_predictions = []
            
            # Also track 30s windows
            window_size_30s = 3000
            n_windows_30s = test_eeg.shape[0] // window_size_30s
            window_predictions_30s = []
            
            for win_idx in range(n_windows):
                start_idx = win_idx * window_size
                end_idx = (win_idx + 1) * window_size
                
                # Extract window data
                win_eeg = test_eeg[start_idx:end_idx, :]
                win_env1 = test_env1[start_idx:end_idx]
                win_env2 = test_env2[start_idx:end_idx]
                
                # Classify this window
                pred, r1, r2 = aad.classify_attention(win_eeg, win_env1, win_env2)
                window_predictions.append(pred)
                all_window_predictions.append(pred)
                all_window_ground_truth.append(test_ground_truth)
            
            for win_idx in range(n_windows_30s):
                start_idx = win_idx * window_size_30s
                end_idx = (win_idx + 1) * window_size_30s
                
                # Extract window data
                win_eeg = test_eeg[start_idx:end_idx, :]
                win_env1 = test_env1[start_idx:end_idx]
                win_env2 = test_env2[start_idx:end_idx]
                
                # Classify this window
                pred, r1, r2 = aad.classify_attention(win_eeg, win_env1, win_env2)
                window_predictions_30s.append(pred)
            
            # Compute window accuracy for this trial
            window_preds = np.array(window_predictions)
            trial_acc = np.mean(window_preds == test_ground_truth)
            trial_accuracies.append(trial_acc)
            
            window_preds_30s = np.array(window_predictions_30s)
            trial_acc_30s = np.mean(window_preds_30s == test_ground_truth) if n_windows_30s > 0 else 0
            
            correct = "✓" if trial_acc >= 0.5 else "✗"
            print(f"  Trial {test_idx+1:02d}/20: 10s acc={trial_acc:.2%} {correct}, 30s acc={trial_acc_30s:.2%}")
        
        # Convert to numpy
        all_window_predictions = np.array(all_window_predictions)
        all_window_ground_truth = np.array(all_window_ground_truth)
        trial_accuracies = np.array(trial_accuracies)
        
        # Compute overall metrics at WINDOW level
        accuracy = np.mean(all_window_predictions == all_window_ground_truth)
        cm = confusion_matrix(all_window_ground_truth, all_window_predictions, labels=[1, 2])
        
        # Sensitivity/Specificity
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Results for Subject S{subject_id} (WINDOW-LEVEL):")
        print(f"  Total windows: {len(all_window_predictions)}")
        print(f"  Window accuracy: {accuracy:.4f} ({np.sum(all_window_predictions==all_window_ground_truth)}/{len(all_window_predictions)})")
        print(f"  Sensitivity (Track1): {sensitivity:.4f}")
        print(f"  Specificity (Track2): {specificity:.4f}")
        print(f"  Mean trial accuracy: {np.mean(trial_accuracies):.4f} ± {np.std(trial_accuracies):.4f}")
        print(f"  Confusion Matrix (windows):")
        print(f"    Track1: {tn} correct, {fp} false positives")
        print(f"    Track2: {fn} false negatives, {tp} correct")
        print(f"{'='*70}")
        
        return {
            'subject_id': subject_id,
            'window_predictions': all_window_predictions,
            'window_ground_truth': all_window_ground_truth,
            'trial_accuracies': trial_accuracies,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm,
            'n_windows': len(all_window_predictions),
            'n_trials': 20,
            'window_size': window_size
        }
    
    def save_results(self, results: dict):
        """
        Save results to CSV and create visualizations.
        Window-level evaluation.
        
        Args:
            results: Results dictionary
        """
        subject_id = results['subject_id']
        window_predictions = results['window_predictions']
        window_ground_truth = results['window_ground_truth']
        accuracy = results['accuracy']
        cm = results['confusion_matrix']
        
        # Save CSV (window-level)
        csv_path = self.results_dir / f"aad_mtrf_s{subject_id}_windows.csv"
        df = pd.DataFrame({
            'window': np.arange(1, len(window_predictions) + 1),
            'ground_truth': window_ground_truth,
            'prediction': window_predictions,
            'correct': (window_predictions == window_ground_truth).astype(int)
        })
        df.to_csv(csv_path, index=False)
        print(f"\nWindow-level results saved to: {csv_path}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        
        # 1. Confusion Matrix
        ax = axes[0, 0]
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(cm.flatten()))
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Track1', 'Track2'])
        ax.set_yticklabels(['Track1', 'Track2'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title('Confusion Matrix (Window-Level)', fontweight='bold')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                       color='white', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        # 2. Prediction timeline
        ax = axes[0, 1]
        windows = np.arange(1, len(window_predictions) + 1)
        correct = (window_predictions == window_ground_truth)
        colors = ['green' if c else 'red' for c in correct]
        ax.scatter(windows, window_predictions, c=colors, alpha=0.6, s=50)
        ax.scatter(windows, window_ground_truth, marker='x', color='black', s=100, linewidths=2)
        ax.set_xlabel('Window')
        ax.set_ylabel('Predicted Track')
        ax.set_ylim([0.5, 2.5])
        ax.set_yticks([1, 2])
        ax.set_title('Predictions vs Ground Truth (Green=Correct, Red=Error)', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 3. Accuracy by trial
        ax = axes[1, 0]
        trial_accs = results['trial_accuracies']
        trials = np.arange(1, len(trial_accs) + 1)
        colors_trial = ['green' if acc >= 0.5 else 'red' for acc in trial_accs]
        ax.bar(trials, trial_accs, color=colors_trial, alpha=0.7)
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Chance (50%)')
        ax.axhline(accuracy, color='blue', linestyle='--', alpha=0.5, label=f'Mean ({accuracy:.1%})')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Window Accuracy')
        ax.set_ylim([0, 1])
        ax.set_title('Accuracy per Trial (Window-Level)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = f"""
        Subject S{subject_id} - AAD Results (Window-Level Evaluation)
        
        Total Windows:    {len(window_predictions)}
        Correct Windows:  {np.sum(correct)}/{len(window_predictions)}
        Accuracy:         {accuracy:.4f} ({accuracy:.1%})
        
        Sensitivity (Track1 recall): {results['sensitivity']:.4f}
        Specificity (Track2 recall): {results['specificity']:.4f}
        
        Confusion Matrix (windows):
        ┌─────────────────────────┐
        │ TN={cm[0,0]:4d}  FP={cm[0,1]:4d} │
        │ FN={cm[1,0]:4d}  TP={cm[1,1]:4d} │
        └─────────────────────────┘
        
        Method: mTRF Envelope Reconstruction
        Window size: 10 seconds (1000 samples @ 100Hz)
        LOOCV: trial-level (train=19, test=1)
        Evaluation: window-level
        Reference: Geirnaert et al. (2021)
        Baseline: 75% (Go condition)
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plot_path = self.results_dir / f"aad_mtrf_s{subject_id}_windows.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        plt.close()
    
    def run(self):
        """
        Run complete experiment.
        """
        subjects = [1] if self.quick_mode else list(range(1, 17))
        all_results = []
        
        for subject_id in subjects:
            results = self.run_subject_loocv(subject_id)
            if results is not None:
                all_results.append(results)
                self.save_results(results)
        
        # Summary
        print(f"\n=== AAD Classification: All Subjects ===")
        
        if all_results:
            accuracies = [res['accuracy'] for res in all_results]
            
            for res in all_results:
                correct = int(res['n_windows']*res['accuracy'])
                total = res['n_windows']
                print(f"S{res['subject_id']}:  Accuracy = {res['accuracy']:.4f} ({correct}/{total} windows)")
            
            if len(all_results) > 1:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                print(f"\nGroup Mean: {mean_acc:.4f} ± {std_acc:.4f}")
            else:
                print(f"\nGroup Mean: {accuracies[0]:.4f} ± 0.0000")
            print("Chance level: 0.50")
            
            # Save summary to CSV
            import pandas as pd
            summary_data = []
            for res in all_results:
                summary_data.append({
                    'Subject': f"S{res['subject_id']}",
                    'Accuracy': res['accuracy'],
                    'CorrectWindows': int(res['n_windows']*res['accuracy']),
                    'TotalWindows': res['n_windows']
                })
            df = pd.DataFrame(summary_data)
            csv_path = self.results_dir.parent / "aad_mtrf" / "all_subjects_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)


def main():
    """
    Main entry point.
    """
    experiment = AADmTRFExperiment(quick_mode=False)
    experiment.run()


if __name__ == "__main__":
    main()
