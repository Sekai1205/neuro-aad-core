"""
mTRF Replication Experiment using KULeuven Dataset
Reproduces MacIntyre et al. (2024) speech envelope decoding from EEG
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import FS, DATA_DIR, RESULTS_DIR
from data.kuleuven_loader import load_trial_data
from mtrf.envelope import extract_envelope, downsample_eeg, align_envelope_eeg
from mtrf.decoder import MTRFDecoder

warnings.filterwarnings('ignore')


class mTRFExperiment:
    """
    Experiment runner for mTRF speech envelope decoding
    Leave-one-trial-out cross-validation on KULeuven dataset
    """
    
    def __init__(self, quick_mode: bool = True):
        """
        Initialize experiment.
        
        Args:
            quick_mode: If True, run only on subject S1. If False, all subjects.
        """
        self.quick_mode = quick_mode
        self.results_dir = Path(RESULTS_DIR) / "mtrf_replication"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("mTRF Replication Experiment (MacIntyre et al., 2024)")
        print("="*60)
        print(f"Quick mode (S1 only): {quick_mode}")
        print(f"Results directory: {self.results_dir}")
    
    def extract_trial_envelope_eeg(self, subject_id: int, trial_idx: int, 
                                    target_fs: float = 100) -> dict:
        """
        Load trial data and extract envelope from actual audio file and downsampled EEG.
        
        Args:
            subject_id: Subject number (1-16)
            trial_idx: Trial index (0-19)
            target_fs: Target downsampling rate (Hz)
        
        Returns:
            dict with 'envelope', 'eeg', 'duration', 'metadata', 'audio_file'
        """
        try:
            import scipy.io
            from pathlib import Path
            
            # Load trial metadata to get audio file info
            mat_path = Path(PROJECT_ROOT) / "data" / "KULeuven data set" / f"S{subject_id}.mat"
            mat = scipy.io.loadmat(str(mat_path))
            trials = mat['trials']
            trial_meta = trials[0][trial_idx]
            
            # Extract attended audio file
            stimuli = trial_meta['stimuli'][0][0]
            attended_track = int(trial_meta['attended_track'][0][0][0][0])
            condition = trial_meta['condition'][0][0][0]
            
            # stimuli is (2, 1) array: [[track2], [track1]] or similar
            # attended_track is 1-indexed (1 or 2)
            # Get the attended track file - handle numpy array extraction
            stimuli_element = stimuli[attended_track - 1][0]
            if hasattr(stimuli_element, 'item'):
                attended_filename = stimuli_element.item()  # Extract scalar from numpy array
            elif isinstance(stimuli_element, np.ndarray):
                attended_filename = str(stimuli_element[0]) if len(stimuli_element) > 0 else str(stimuli_element)
            else:
                attended_filename = str(stimuli_element)
            
            # Build full path to audio file
            audio_path = Path(PROJECT_ROOT) / "data" / "KULeuven data set" / "stimuli" / attended_filename
            
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load trial EEG data
            trial_data = load_trial_data(subject_id, trial_idx)
            eeg = trial_data['eeg']  # (samples, 64)
            
            # Extract envelope from actual audio file
            from mtrf.envelope import extract_envelope_from_audio_file
            envelope, fs_envelope = extract_envelope_from_audio_file(
                str(audio_path), target_fs=target_fs
            )
            
            # Downsample EEG to 100 Hz
            eeg_ds, fs_ds = downsample_eeg(eeg, FS, fs_target=target_fs)
            
            # Align envelope and EEG (trim to same length)
            min_length = min(len(envelope), eeg_ds.shape[0])
            envelope_aligned = envelope[:min_length]
            eeg_aligned = eeg_ds[:min_length, :]
            
            return {
                'envelope': envelope_aligned,
                'eeg': eeg_aligned,
                'duration': min_length / target_fs,
                'metadata': trial_data['metadata'],
                'audio_file': str(audio_path),
                'attended_track': attended_track
            }
        
        except Exception as e:
            print(f"Error loading trial S{subject_id} trial {trial_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_subject_loocv(self, subject_id: int) -> dict:
        """
        Run leave-one-trial-out cross-validation for a subject.
        
        Procedure:
        - For each of 20 trials:
          - Train mTRF on 19 remaining trials
          - Evaluate on held-out trial
          - Record Pearson's r correlation
        
        Args:
            subject_id: Subject number (1-16)
        
        Returns:
            dict with cross-validation results
        """
        print(f"\n{'='*60}")
        print(f"Subject S{subject_id}: Leave-One-Trial-Out CV")
        print(f"{'='*60}")
        
        # Load and preprocess all trials for subject
        eeg_list = []
        envelope_list = []
        metadata_list = []
        
        print(f"Loading {20} trials...")
        for trial_idx in range(20):
            data = self.extract_trial_envelope_eeg(subject_id, trial_idx, target_fs=100)
            if data is not None:
                eeg_list.append(data['eeg'])
                envelope_list.append(data['envelope'])
                metadata_list.append(data['metadata'])
            else:
                print(f"  WARNING: Failed to load trial {trial_idx}")
        
        if len(eeg_list) < 20:
            print(f"ERROR: Only loaded {len(eeg_list)}/20 trials")
            return None
        
        print(f"Loaded {len(eeg_list)} trials successfully")
        
        # Initialize decoder
        decoder = MTRFDecoder(fs=100, tmin=-0.05, tmax=0.4, lambda_=100)
        
        # Run LOOCV
        print("\nRunning Leave-One-Trial-Out CV...")
        r_scores = []
        
        for test_idx in range(20):
            # Create training set (all except test_idx)
            train_eeg_list = [eeg_list[i] for i in range(20) if i != test_idx]
            train_envelope_list = [envelope_list[i] for i in range(20) if i != test_idx]
            
            # Concatenate training data
            train_eeg = np.concatenate(train_eeg_list, axis=0)
            train_envelope = np.concatenate(train_envelope_list, axis=0)
            
            # Test data
            test_eeg = eeg_list[test_idx]
            test_envelope = envelope_list[test_idx]
            
            # Fit decoder on training data
            decoder.fit(train_eeg, train_envelope)
            
            # Evaluate on test data
            r = decoder.score(test_eeg, test_envelope)
            r_scores.append(r)
            
            print(f"  Trial {test_idx+1:02d}/20: r = {r:7.4f}")
        
        # Aggregate results
        r_scores = np.array(r_scores)
        mean_r = np.mean(r_scores)
        std_r = np.std(r_scores)
        
        print(f"\n{'='*60}")
        print(f"Results for Subject S{subject_id}:")
        print(f"  Mean r = {mean_r:.4f}")
        print(f"  Std  r = {std_r:.4f}")
        print(f"  Min  r = {np.min(r_scores):.4f}")
        print(f"  Max  r = {np.max(r_scores):.4f}")
        print(f"{'='*60}")
        
        print(f"\nMacIntyre et al. (2024) reported:")
        print(f"  Speech envelope -> EEG decoding (English unprocessed)")
        print(f"  r ≈ 0.15 (cross-subject, naive model)")
        print(f"  r ≈ 0.30-0.40 (single subject, adapted model)")
        print(f"\nNote: Our synthetic envelope gives upper-bound performance.")
        print(f"      Real audio envelopes would yield lower correlations.")
        
        return {
            'subject_id': subject_id,
            'r_scores': r_scores,
            'mean_r': mean_r,
            'std_r': std_r,
            'n_trials': len(r_scores)
        }
    
    def save_results(self, results: dict):
        """
        Save results to CSV and create boxplot.
        
        Args:
            results: Dictionary with cross-validation results
        """
        subject_id = results['subject_id']
        r_scores = results['r_scores']
        
        # Save CSV
        csv_path = self.results_dir / f"mtrf_loocv_s{subject_id}.csv"
        df = pd.DataFrame({
            'trial': np.arange(1, len(r_scores) + 1),
            'pearson_r': r_scores
        })
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Create boxplot
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot([r_scores], labels=[f'S{subject_id}'], patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_ylabel('Pearson Correlation (r)', fontsize=12)
        ax.set_xlabel('Subject', fontsize=12)
        ax.set_title('mTRF Decoding Performance\nLeave-One-Trial-Out CV', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add individual points
        ax.scatter([1]*len(r_scores), r_scores, alpha=0.5, s=50, color='navy')
        
        # Add mean line
        ax.axhline(y=np.mean(r_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r_scores):.3f}')
        ax.legend()
        
        plot_path = self.results_dir / f"mtrf_loocv_s{subject_id}_boxplot.png"
        plt.tight_layout()
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
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        
        if all_results:
            print(f"Processed {len(all_results)} subject(s)")
            for res in all_results:
                print(f"  S{res['subject_id']}: mean_r={res['mean_r']:.4f} ± {res['std_r']:.4f}")


def main():
    """
    Main entry point for experiment.
    """
    # Run on all 16 subjects
    experiment = mTRFExperiment(quick_mode=False)
    experiment.run()


if __name__ == "__main__":
    main()
