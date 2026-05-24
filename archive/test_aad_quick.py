"""Quick test of AAD mTRF on S1 single trial"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, FS
from data.kuleuven_loader import load_trial_data
from mtrf.envelope import extract_envelope_from_audio_file, downsample_eeg
from mtrf.aad_mtrf import AADmTRF
import scipy.io

print("=" * 70)
print("AAD mTRF Quick Test - Single Trial Classification")
print("=" * 70)

# Load 2 trials for quick test
subject_id = 1
trials_data = []

for trial_idx in range(2):
    print(f"\nLoading S1 Trial {trial_idx}...")
    
    # Load EEG
    trial_data = load_trial_data(subject_id, trial_idx)
    eeg = trial_data['eeg']
    print(f"  EEG shape: {eeg.shape}")
    
    # Load mat file for metadata
    mat_path = DATA_DIR / f"S{subject_id}.mat"
    mat = scipy.io.loadmat(str(mat_path))
    trials = mat['trials']
    trial_meta = trials[0][trial_idx]
    
    stimuli = trial_meta['stimuli'][0][0]
    attended_track = int(trial_meta['attended_track'][0][0][0][0])
    print(f"  Attended track: {attended_track}")
    
    # Load envelopes
    envelopes_dict = {}
    for track_num in [1, 2]:
        stimuli_element = stimuli[track_num - 1][0]
        if isinstance(stimuli_element, np.ndarray):
            filename = str(stimuli_element[0]) if len(stimuli_element) > 0 else str(stimuli_element)
        else:
            filename = str(stimuli_element)
        
        audio_path = DATA_DIR / "stimuli" / filename
        print(f"  Loading {filename}...")
        envelope, _ = extract_envelope_from_audio_file(str(audio_path), target_fs=100)
        envelopes_dict[f'envelope_track{track_num}'] = envelope
    
    # Downsample EEG
    eeg_ds, _ = downsample_eeg(eeg, FS, fs_target=100)
    
    # Align signals
    min_length = min(eeg_ds.shape[0], 
                    envelopes_dict['envelope_track1'].shape[0],
                    envelopes_dict['envelope_track2'].shape[0])
    
    trials_data.append({
        'eeg': eeg_ds[:min_length, :],
        'envelope_track1': envelopes_dict['envelope_track1'][:min_length],
        'envelope_track2': envelopes_dict['envelope_track2'][:min_length],
        'ground_truth_track': attended_track
    })
    print(f"  ✓ Trial loaded (aligned length: {min_length})")

# Test: Train on trial 0, test on trial 1
print("\n" + "=" * 70)
print("Testing AAD Classifier")
print("=" * 70)

aad = AADmTRF(fs=100, tmin=-0.05, tmax=0.4, lambda_=100)
print("Training on trial 0...")
aad.fit([trials_data[0]['eeg']], 
        [trials_data[0]['envelope_track1'], trials_data[0]['envelope_track2']])
print("✓ Training complete")

print("\nClassifying trial 1...")
pred, r1, r2 = aad.classify_attention(
    trials_data[1]['eeg'],
    trials_data[1]['envelope_track1'],
    trials_data[1]['envelope_track2']
)

ground_truth = trials_data[1]['ground_truth_track']
correct = "✓ CORRECT" if pred == ground_truth else "✗ WRONG"

print(f"\nResults:")
print(f"  Ground truth: Track {ground_truth}")
print(f"  Prediction: Track {pred} {correct}")
print(f"  r(Track1): {r1:.6f}")
print(f"  r(Track2): {r2:.6f}")
print(f"  Delta: {r1 - r2:+.6f}")
print("\n✓ Test complete!")
