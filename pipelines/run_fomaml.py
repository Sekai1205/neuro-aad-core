"""
Task C: FOMAML Meta-Learning for AAD using PyTorch mTRF Decoder.

Evaluates three models under 4 LOOCV folds (subjects S1, S5, S10, S14):
1. FOMAML_adapted: Meta-initialized LinearmTRF fine-tuned with N-second calibration data.
2. generic_torch: LinearmTRF trained on pooled training subjects' data (no fine-tuning).
3. subject_specific_mtrf: Upper bound accuracy from the baseline Ridge decoder.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mtrf.experiment_aad_mtrf import AADmTRFExperiment
from models.mtrf_torch import LinearmTRF, build_lag_matrix
from mtrf.aad_mtrf import AADmTRF

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Constants
TMIN = 0.0
TMAX = 0.25
FS = 100
LAMBDA = 0.001
INNER_LR = 0.01
INNER_STEPS_META = 5
INNER_STEPS_ADAPT = 10
OUTER_LR = 1e-3
META_EPOCHS = 500
CALIB_TIMES = [10, 30, 60, 120]  # seconds
WINDOW_SIZE = 1000  # 10 seconds @ 100 Hz

# Hardcoded subject-specific upper bound accuracies (from results/aad_mtrf/all_subjects_summary.csv)
SUBJECT_SPECIFIC_ACCURACIES = {
    'S1': 60.18,
    'S5': 77.88,
    'S10': 69.91,
    'S14': 80.31
}

def bandpass_filter(data, fs=100, low=1.0, high=8.0, order=3):
    """1-8 Hz bandpass filter matching the baseline pipeline."""
    b, a = signal.butter(order, [low, high], btype='band', fs=fs)
    return signal.filtfilt(b, a, data, axis=0)

def main():
    print("=" * 80)
    print("RUNNING TASK C: FOMAML META-LEARNING FOR AAD")
    print("=" * 80)

    # Output directories
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment class (shares the envelope cache)
    experiment = AADmTRFExperiment(quick_mode=False)

    # Load all subjects data
    subjects_data = {}
    for subject_id in range(1, 17):
        print(f"Loading and pre-filtering Subject S{subject_id} data...")
        subject_trials = []
        for trial_idx in range(20):
            trial_data = experiment.load_trial_data_with_envelopes(subject_id, trial_idx)
            if trial_data is not None:
                # Pre-apply bandpass filter to save time during training
                trial_data['eeg'] = bandpass_filter(trial_data['eeg'])
                trial_data['envelope_track1'] = bandpass_filter(trial_data['envelope_track1'])
                trial_data['envelope_track2'] = bandpass_filter(trial_data['envelope_track2'])
                subject_trials.append(trial_data)
        subjects_data[subject_id] = subject_trials

    # We evaluate 4 LOOCV folds: held-out subjects S1, S5, S10, S14
    test_subject_ids = [1, 5, 10, 14]
    n_features = 64 * 26  # 64 channels * 26 lags

    results = []

    for test_sub_id in test_subject_ids:
        print(f"\n==================================================")
        print(f"LOOCV Fold: Test Subject S{test_sub_id}")
        print(f"==================================================")
        
        train_subjects = [s for s in range(1, 17) if s != test_sub_id]
        
        # 1. FOMAML Meta-Training
        print("\n--- Starting FOMAML Meta-Training ---")
        meta_model = LinearmTRF(n_features=n_features)
        outer_opt = torch.optim.Adam(meta_model.parameters(), lr=OUTER_LR)
        
        for epoch in range(1, META_EPOCHS + 1):
            # Sample a training subject (task)
            task_sub_id = np.random.choice(train_subjects)
            task_trials = subjects_data[task_sub_id]
            
            # Split into support (first 10 trials) and query (last 10 trials)
            sup_trials = task_trials[:10]
            qry_trials = task_trials[10:]
            
            # Concatenate support set
            sup_eeg = np.concatenate([t['eeg'] for t in sup_trials], axis=0)
            sup_env = np.concatenate([
                t['envelope_track1'] if t['ground_truth_track'] == 1 else t['envelope_track2']
                for t in sup_trials
            ], axis=0)
            
            # Concatenate query set
            qry_eeg = np.concatenate([t['eeg'] for t in qry_trials], axis=0)
            qry_env = np.concatenate([
                t['envelope_track1'] if t['ground_truth_track'] == 1 else t['envelope_track2']
                for t in qry_trials
            ], axis=0)
            
            # Z-score support set and populate stats
            sup_eeg_z = meta_model._zscore_eeg(sup_eeg, fit_stats=True)
            sup_env_z = meta_model._zscore_env(sup_env, fit_stats=True)
            
            # Z-score query set using support set stats
            qry_eeg_z = meta_model._zscore_eeg(qry_eeg, fit_stats=False)
            qry_env_z = meta_model._zscore_env(qry_env, fit_stats=False)
            
            # Build lag matrices
            X_sup, vs_s, ve_s = build_lag_matrix(sup_eeg_z, TMIN, TMAX, FS)
            y_sup = sup_env_z[vs_s:ve_s]
            
            X_qry, vs_q, ve_q = build_lag_matrix(qry_eeg_z, TMIN, TMAX, FS)
            y_qry = qry_env_z[vs_q:ve_q]
            
            # Convert to torch Tensors
            X_sup_t = torch.from_numpy(X_sup)
            y_sup_t = torch.from_numpy(y_sup).unsqueeze(1).float()
            X_qry_t = torch.from_numpy(X_qry)
            y_qry_t = torch.from_numpy(y_qry).unsqueeze(1).float()
            
            # FOMAML First-Order Update
            # Save original weights
            orig_state = {k: v.clone() for k, v in meta_model.state_dict().items()}
            
            # Inner loop SGD steps on support set
            inner_opt = torch.optim.SGD(meta_model.parameters(), lr=INNER_LR)
            meta_model.train()
            for _ in range(INNER_STEPS_META):
                pred = meta_model(X_sup_t)
                loss = F.mse_loss(pred, y_sup_t)
                inner_opt.zero_grad()
                loss.backward()
                inner_opt.step()
                
            # Query loss with adapted weights
            pred_query = meta_model(X_qry_t)
            query_loss = F.mse_loss(pred_query, y_qry_t)
            
            # Backprop query loss to current parameters
            outer_opt.zero_grad()
            query_loss.backward()
            
            # Save query loss gradients w.r.t. adapted parameters
            adapted_grads = {k: p.grad.clone() for k, p in meta_model.named_parameters() if p.grad is not None}
            
            # Restore original weights
            meta_model.load_state_dict(orig_state)
            
            # Apply saved gradients directly as meta-initialization gradients
            for name, param in meta_model.named_parameters():
                if name in adapted_grads:
                    param.grad = adapted_grads[name]
            outer_opt.step()
            
            if epoch % 50 == 0:
                print(f"  Epoch {epoch:3d}/{META_EPOCHS}: Query Loss = {query_loss.item():.6f}")
                
        # Save meta-initialized model checkpoint
        ckpt_path = results_dir / f"fomaml_meta_init_S{test_sub_id}.pt"
        torch.save(meta_model.state_dict(), ckpt_path)
        print(f"Meta-initialized weights saved to {ckpt_path}")

        # 2. Train generic_torch model (pooled training subjects, no fine-tuning)
        print("\n--- Training Generic Torch Model ---")
        train_eeg_all = np.concatenate([t['eeg'] for s in train_subjects for t in subjects_data[s]], axis=0)
        train_env_all = np.concatenate([
            t['envelope_track1'] if t['ground_truth_track'] == 1 else t['envelope_track2']
            for s in train_subjects for t in subjects_data[s]
        ], axis=0)
        
        # Stride by 5 to make generic model training extremely fast and low-memory
        train_eeg_strided = train_eeg_all[::5]
        train_env_strided = train_env_all[::5]
        
        generic_model = LinearmTRF(n_features=n_features)
        # Match matches Adam weight decay with Ridge LAMBDA
        wd_torch = 2.0 * LAMBDA
        generic_model.fit(train_eeg_strided, train_env_strided, TMIN, TMAX, FS,
                          weight_decay=wd_torch, verbose=False)
        print("Generic Torch model trained successfully!")

        # 3. Meta-Testing and Adaptation on Test Subject
        print("\n--- Meta-Testing on Test Subject S{} ---".format(test_sub_id))
        test_trials = subjects_data[test_sub_id]
        
        # Concatenate test subject data
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
        
        # A. Evaluate generic_torch (no fine-tuning, constant across calibration times)
        generic_preds = []
        generic_gts = []
        n_windows_gen = test_eeg_concat.shape[0] // WINDOW_SIZE
        
        for w in range(n_windows_gen):
            start = w * WINDOW_SIZE
            end = (w + 1) * WINDOW_SIZE
            win_eeg = test_eeg_concat[start:end]
            win_env1 = test_env1_concat[start:end]
            win_env2 = test_env2_concat[start:end]
            win_gt = sample_gts[start + WINDOW_SIZE // 2]
            
            env_pred, vs, ve = generic_model.predict(win_eeg, TMIN, TMAX, FS)
            r1, _ = stats.pearsonr(win_env1[vs:ve], env_pred)
            r2, _ = stats.pearsonr(win_env2[vs:ve], env_pred)
            pred = 1 if r1 > r2 else 2
            generic_preds.append(pred)
            generic_gts.append(win_gt)
            
        generic_acc = np.mean(np.array(generic_preds) == np.array(generic_gts))
        print(f"Generic Torch Accuracy: {generic_acc:.2%}")

        # B. Evaluate FOMAML adapted for each calibration time N
        for calib_s in CALIB_TIMES:
            # Clone from meta-init
            adapted_model = LinearmTRF(n_features=n_features)
            adapted_model.load_state_dict(meta_model.state_dict())
            
            # Slice calibration data
            calib_samples = int(calib_s * FS)
            calib_eeg = test_eeg_concat[:calib_samples]
            calib_env = test_attended_env_concat[:calib_samples]
            
            # Z-score and construct lag matrix for calibration
            calib_eeg_z = adapted_model._zscore_eeg(calib_eeg, fit_stats=True)
            calib_env_z = adapted_model._zscore_env(calib_env, fit_stats=True)
            
            X_cal, vs_c, ve_c = build_lag_matrix(calib_eeg_z, TMIN, TMAX, FS)
            y_cal = calib_env_z[vs_c:ve_c]
            
            X_cal_t = torch.from_numpy(X_cal)
            y_cal_t = torch.from_numpy(y_cal).unsqueeze(1).float()
            
            # Fine-tune adaptation
            adapt_opt = torch.optim.SGD(adapted_model.parameters(), lr=INNER_LR)
            adapted_model.train()
            for _ in range(INNER_STEPS_ADAPT):
                pred = adapted_model(X_cal_t)
                loss = F.mse_loss(pred, y_cal_t)
                adapt_opt.zero_grad()
                loss.backward()
                adapt_opt.step()
                
            # Evaluate on remaining test data
            rem_eeg = test_eeg_concat[calib_samples:]
            rem_env1 = test_env1_concat[calib_samples:]
            rem_env2 = test_env2_concat[calib_samples:]
            rem_gts = sample_gts[calib_samples:]
            
            n_windows_rem = rem_eeg.shape[0] // WINDOW_SIZE
            adapted_preds = []
            adapted_gts = []
            
            for w in range(n_windows_rem):
                start = w * WINDOW_SIZE
                end = (w + 1) * WINDOW_SIZE
                win_eeg = rem_eeg[start:end]
                win_env1 = rem_env1[start:end]
                win_env2 = rem_env2[start:end]
                win_gt = rem_gts[start + WINDOW_SIZE // 2]
                
                env_pred, vs, ve = adapted_model.predict(win_eeg, TMIN, TMAX, FS)
                r1, _ = stats.pearsonr(win_env1[vs:ve], env_pred)
                r2, _ = stats.pearsonr(win_env2[vs:ve], env_pred)
                pred = 1 if r1 > r2 else 2
                adapted_preds.append(pred)
                adapted_gts.append(win_gt)
                
            adapted_acc = np.mean(np.array(adapted_preds) == np.array(adapted_gts)) if n_windows_rem > 0 else 0.5
            print(f"FOMAML Adapted ({calib_s}s calib) Accuracy: {adapted_acc:.2%}")
            
            results.append({
                'subject': f"S{test_sub_id}",
                'model': 'FOMAML_adapted',
                'calib_s': calib_s,
                'accuracy_pct': adapted_acc * 100
            })
            
            # Record generic and subject specific for this subject x calib_s combination
            results.append({
                'subject': f"S{test_sub_id}",
                'model': 'generic_torch',
                'calib_s': calib_s,
                'accuracy_pct': generic_acc * 100
            })
            
            results.append({
                'subject': f"S{test_sub_id}",
                'model': 'subject_specific_mtrf',
                'calib_s': calib_s,
                'accuracy_pct': SUBJECT_SPECIFIC_ACCURACIES[f"S{test_sub_id}"]
            })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = results_dir / "fomaml_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFOMAML results saved to {csv_path}")

    # Generate Learning Curve
    generate_figures(df, figures_dir)
    print("FOMAML Learning Curve figure successfully generated!")

def generate_figures(df, figures_dir):
    """Generate learning curve figure for FOMAML comparison."""
    # Compute group level summary
    summary = df.groupby(['model', 'calib_s'])['accuracy_pct'].agg(['mean', 'std']).reset_index()
    print("\n=== FOMAML Group-Level Summary (%) ===")
    print(summary.to_string(index=False))

    plt.figure(figsize=(10, 6))
    
    models = ['FOMAML_adapted', 'generic_torch', 'subject_specific_mtrf']
    colors = {
        'FOMAML_adapted': '#E76F51',
        'generic_torch': '#264653',
        'subject_specific_mtrf': '#2A9D8F'
    }
    markers = {
        'FOMAML_adapted': 'o',
        'generic_torch': 's',
        'subject_specific_mtrf': '^'
    }
    labels = {
        'FOMAML_adapted': 'FOMAML (Meta-Init + Adaptation)',
        'generic_torch': 'Generic PyTorch Decoder (No Adaptation)',
        'subject_specific_mtrf': 'Subject-Specific Decoder (Upper Bound)'
    }

    for m in models:
        m_data = summary[summary['model'] == m].sort_values('calib_s')
        
        plt.plot(m_data['calib_s'], m_data['mean'], label=labels[m],
                 color=colors[m], marker=markers[m], linewidth=2.5, markersize=8)
        
        # Shade error band (±1 SD)
        plt.fill_between(m_data['calib_s'], m_data['mean'] - m_data['std'], m_data['mean'] + m_data['std'],
                         color=colors[m], alpha=0.15)
        
    plt.axhline(50.0, color='black', linestyle='--', alpha=0.5, label='Chance (50%)')
    plt.xscale('log')
    plt.xticks(CALIB_TIMES, [f"{t}s" for t in CALIB_TIMES], fontsize=11)
    plt.xlabel("Calibration Data Duration (log scale)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(40, 100)
    plt.title("FOMAML Meta-Learning Adaptation Speed vs Baseline Models", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "fomaml_learning_curve.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
