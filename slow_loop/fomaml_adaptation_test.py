"""
FOMAML Adaptation Test with KULeuven 4-Channel Data
Tests accuracy improvement via FOMAML few-shot adaptation on the 4-channel subset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / "slow_loop"))
sys.path.insert(0, str(Path(__file__).parent / "data"))

from config import (
    FS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    RESULTS_DIR,
    PLOTS_SAVE_DIR,
)
from kuleuven_loader import load_all_subject_trials, reduce_channels
from models.eeg_1d_cnn import EEG1DCNN


# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class EEGDataset(Dataset):
    """EEG dataset for AAD task"""
    def __init__(self, eeg_list, labels, window_size=512, stride=128):
        self.windows = []
        self.window_labels = []
        
        for eeg_data, label in zip(eeg_list, labels):
            n_samples, n_channels = eeg_data.shape
            n_windows = (n_samples - window_size) // stride + 1
            
            for i in range(n_windows):
                start = i * stride
                end = start + window_size
                window = eeg_data[start:end, :].astype(np.float32)
                self.windows.append(window)
                self.window_labels.append(label)
        
        self.windows = np.array(self.windows)
        self.window_labels = np.array(self.window_labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx].T  # (channels, window_size)
        label = self.window_labels[idx]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def clone_model(model, num_channels, device):
    """Create a clone of the model"""
    clone = EEG1DCNN(num_channels=num_channels).to(device)
    clone.load_state_dict(model.state_dict())
    return clone


def fomaml_adaptation(model, support_loader, test_loader, 
                     num_adaptation_steps=5, inner_lr=0.01,
                     device=DEVICE):
    """
    Apply FOMAML adaptation (First-Order MAML).
    
    Args:
        model: Base model
        support_loader: DataLoader for support set (few-shot adaptation data)
        test_loader: DataLoader for query set (test data)
        num_adaptation_steps: Number of inner loop optimization steps
        inner_lr: Learning rate for inner loop
        device: Device to run on
    
    Returns:
        tuple: (accuracy_before, accuracy_after)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate before adaptation
    correct_before = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct_before += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    acc_before = 100 * correct_before / total if total > 0 else 0
    
    # Create adapted model
    adapted_model = clone_model(model, model.spatial_filter[0].in_channels, device)
    adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)
    
    # Inner loop: adapt on support set
    adapted_model.train()
    for step in range(num_adaptation_steps):
        for inputs, labels in support_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            adapted_optimizer.zero_grad()
            outputs = adapted_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            adapted_optimizer.step()
    
    # Evaluate adapted model
    adapted_model.eval()
    correct_after = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = adapted_model(inputs)
            _, predicted = outputs.max(1)
            correct_after += predicted.eq(labels).sum().item()
    
    acc_after = 100 * correct_after / total if total > 0 else 0
    
    return acc_before, acc_after, adapted_model


def run_fomaml_experiment(subject_id=1, num_support_trials=3):
    """
    Run FOMAML adaptation experiment for a single subject.
    
    Uses 4-channel reduced data (expected lower bound from Experiment A).
    
    Args:
        subject_id (int): Subject number (1-16)
        num_support_trials (int): Number of trials for adaptation (support set)
    
    Returns:
        dict: Results with accuracies before/after adaptation
    """
    print(f"\n{'='*70}")
    print(f"FOMAML Adaptation Test - Subject {subject_id} (4 channels)")
    print(f"{'='*70}")
    print(f"Support set: {num_support_trials} trials for adaptation")
    
    # Load all trials for this subject
    all_trials = load_all_subject_trials(subject_id)
    eeg_list = all_trials['eeg_list']
    labels = all_trials['labels']
    
    # Reduce to 4 channels
    eeg_4ch = [reduce_channels(eeg, 4) for eeg in eeg_list]
    
    results = {
        'before': [],
        'after': [],
        'improvement': []
    }
    
    # Leave-one-trial-out with FOMAML adaptation
    num_trials = len(eeg_list)
    
    for test_trial_idx in range(num_trials):
        # Create support set: first N trials (excluding test trial if possible)
        support_indices = []
        for i in range(num_trials):
            if i != test_trial_idx and len(support_indices) < num_support_trials:
                support_indices.append(i)
        
        if len(support_indices) < num_support_trials:
            # If not enough trials, use some trials multiple times
            while len(support_indices) < num_support_trials:
                support_indices.append(support_indices[0])
        
        # Create test set (single trial)
        test_eeg = [eeg_4ch[test_trial_idx]]
        test_labels = np.array([labels[test_trial_idx]])
        
        # Create train/support sets
        support_eeg = [eeg_4ch[i] for i in support_indices]
        support_labels = np.array([labels[i] for i in support_indices])
        
        # Create datasets
        support_dataset = EEGDataset(support_eeg, support_labels, window_size=512, stride=256)
        test_dataset = EEGDataset(test_eeg, test_labels, window_size=512, stride=256)
        
        support_loader = DataLoader(support_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Create base model
        model = EEG1DCNN(num_channels=4).to(DEVICE)
        
        # Perform FOMAML adaptation
        acc_before, acc_after, _ = fomaml_adaptation(
            model, support_loader, test_loader,
            num_adaptation_steps=5,
            inner_lr=0.01,
            device=DEVICE
        )
        
        improvement = acc_after - acc_before
        results['before'].append(acc_before)
        results['after'].append(acc_after)
        results['improvement'].append(improvement)
        
        print(f"  Trial {test_trial_idx:2d}: Before={acc_before:6.2f}% -> After={acc_after:6.2f}% " +
              f"(+{improvement:+6.2f}%)")
    
    # Aggregate results
    results['mean_before'] = np.mean(results['before'])
    results['std_before'] = np.std(results['before'])
    results['mean_after'] = np.mean(results['after'])
    results['std_after'] = np.std(results['after'])
    results['mean_improvement'] = np.mean(results['improvement'])
    results['std_improvement'] = np.std(results['improvement'])
    
    print(f"\n  Summary:")
    print(f"    Before: {results['mean_before']:.2f}% ± {results['std_before']:.2f}%")
    print(f"    After:  {results['mean_after']:.2f}% ± {results['std_after']:.2f}%")
    print(f"    Improvement: {results['mean_improvement']:.2f}% ± {results['std_improvement']:.2f}%")
    
    return results


def plot_fomaml_results(all_results, output_path):
    """Plot FOMAML results across subjects"""
    subjects = sorted(all_results.keys())
    before_means = [all_results[s]['mean_before'] for s in subjects]
    after_means = [all_results[s]['mean_after'] for s in subjects]
    before_stds = [all_results[s]['std_before'] for s in subjects]
    after_stds = [all_results[s]['std_after'] for s in subjects]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Before/After comparison
    x = np.arange(len(subjects))
    width = 0.35
    ax1.bar(x - width/2, before_means, width, yerr=before_stds, label='Before', 
            alpha=0.8, capsize=5)
    ax1.bar(x + width/2, after_means, width, yerr=after_stds, label='After', 
            alpha=0.8, capsize=5)
    ax1.set_xlabel('Subject', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('FOMAML Adaptation: Before vs After', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{s}' for s in subjects])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Improvements
    improvements = [all_results[s]['mean_improvement'] for s in subjects]
    improvement_stds = [all_results[s]['std_improvement'] for s in subjects]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(x, improvements, color=colors, alpha=0.7, yerr=improvement_stds, capsize=5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Subject', fontsize=12)
    ax2.set_ylabel('Accuracy Improvement (%)', fontsize=12)
    ax2.set_title('FOMAML Adaptation Improvement', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'S{s}' for s in subjects])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path}")
    plt.close()


def save_fomaml_csv(all_results, output_path):
    """Save FOMAML results to CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject', 'Before_Mean_%', 'Before_Std_%', 
                        'After_Mean_%', 'After_Std_%', 'Improvement_%', 'Improvement_Std_%'])
        
        for subject_id in sorted(all_results.keys()):
            results = all_results[subject_id]
            writer.writerow([
                subject_id,
                f"{results['mean_before']:.2f}",
                f"{results['std_before']:.2f}",
                f"{results['mean_after']:.2f}",
                f"{results['std_after']:.2f}",
                f"{results['mean_improvement']:.2f}",
                f"{results['std_improvement']:.2f}",
            ])
    
    print(f"Results saved to CSV: {output_path}")


def main():
    """Run FOMAML experiment for all subjects"""
    # Create output directories
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOTS_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run experiment for all subjects
    all_results = {}
    
    for subject_id in range(1, 17):  # 16 subjects
        results = run_fomaml_experiment(subject_id, num_support_trials=3)
        all_results[subject_id] = results
    
    # Save results
    csv_path = Path(RESULTS_DIR) / f"fomaml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_fomaml_csv(all_results, csv_path)
    
    plot_path = Path(PLOTS_SAVE_DIR) / f"fomaml_adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_fomaml_results(all_results, plot_path)
    
    # Print summary
    print(f"\n{'='*70}")
    print("FOMAML EXPERIMENT SUMMARY (4-channel data)")
    print(f"{'='*70}")
    
    all_before = [all_results[s]['mean_before'] for s in range(1, 17)]
    all_after = [all_results[s]['mean_after'] for s in range(1, 17)]
    all_improvement = [all_results[s]['mean_improvement'] for s in range(1, 17)]
    
    print(f"  Average Before:      {np.mean(all_before):.2f}% ± {np.std(all_before):.2f}%")
    print(f"  Average After:       {np.mean(all_after):.2f}% ± {np.std(all_after):.2f}%")
    print(f"  Average Improvement: {np.mean(all_improvement):.2f}% ± {np.std(all_improvement):.2f}%")


if __name__ == "__main__":
    main()
