"""
Experiment A: Channel Reduction Lower Bound
Test accuracy across different channel counts: 64 → 16 → 8 → 4
Uses trial-aware cross-validation (leave-one-trial-out) per KULeuven authors' recommendation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))  # ルートのみ

from config import (
    FS, N_CHANNELS_FULL, N_SUBJECTS, N_TRIALS_PER_SUBJECT,
    WINDOW_SAMPLES, STRIDE_SAMPLES, CHANNEL_STEPS,
    BASELINE_ACCURACY, GO_THRESHOLD, DATA_DIR, RESULTS_DIR,
    BATCH_SIZE, LEARNING_RATE, RANDOM_SEED
)
from data.kuleuven_loader import load_all_subject_trials, reduce_channels
from slow_loop.models.eeg_1d_cnn import EEG1DCNN

# === 実行モード設定 ===
# Trueで被験者3名・5epochsの高速検証、Falseで本実験
QUICK_MODE = False

if QUICK_MODE:
    SUBJECTS_LIST = [1, 2, 3]
    N_EPOCHS = 5
else:
    SUBJECTS_LIST = list(range(1, N_SUBJECTS + 1))
    N_EPOCHS = 50


# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class EEGDataset(Dataset):
    """
    EEG dataset for AAD task
    """
    def __init__(self, eeg_list, labels, window_size=WINDOW_SAMPLES, stride=STRIDE_SAMPLES):
        """
        Args:
            eeg_list (list): List of EEG data arrays, each (samples, channels)
            labels (array): Binary labels for each trial
            window_size (int): Size of sliding windows
            stride (int): Stride for sliding windows
        """
        self.windows = []
        self.window_labels = []
        
        for eeg_data, label in zip(eeg_list, labels):
            # Create sliding windows
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
        # Return shape (channels, window_size) for Conv1d
        window = self.windows[idx].T  # (channels, window_size)
        label = self.window_labels[idx]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def run_channel_reduction_experiment(subject_id=1):
    """
    Run channel reduction experiment for a single subject.
    Uses leave-one-trial-out cross-validation.
    
    Args:
        subject_id (int): Subject number (1-16)
    
    Returns:
        dict: Results for each channel level
    """
    print(f"\n{'='*70}")
    print(f"Channel Reduction Experiment - Subject {subject_id}")
    print(f"{'='*70}")
    
    # Load all trials for this subject
    all_trials = load_all_subject_trials(subject_id)
    eeg_list = all_trials['eeg_list']
    labels = all_trials['labels']
    
    results_by_channels = {}
    
    for target_channels in CHANNEL_STEPS:
        print(f"\n--- Testing with {target_channels} channels ---")
        
        # Reduce channels for all trials
        eeg_reduced = [reduce_channels(eeg, target_channels) for eeg in eeg_list]
        
        # Leave-one-trial-out cross-validation
        accuracies = []
        losses = []
        
        for test_trial_idx in range(len(eeg_list)):
            # Split data
            train_eeg = [eeg_reduced[i] for i in range(len(eeg_list)) if i != test_trial_idx]
            train_labels = np.array([labels[i] for i in range(len(labels)) if i != test_trial_idx])
            
            test_eeg = [eeg_reduced[test_trial_idx]]
            test_labels = np.array([labels[test_trial_idx]])
            
            # Create datasets and dataloaders
            train_dataset = EEGDataset(train_eeg, train_labels)
            test_dataset = EEGDataset(test_eeg, test_labels)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # Create and train model
            model = EEG1DCNN(num_channels=target_channels).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            
            # Train
            for epoch in range(N_EPOCHS):
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
                if epoch % max(1, N_EPOCHS // 5) == 0:
                    print(f"  Trial {test_trial_idx}: Epoch {epoch}/{N_EPOCHS}, Loss={train_loss:.4f}, Acc={train_acc:.2f}%", end='\r')
            
            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
            accuracies.append(test_acc)
            losses.append(test_loss)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        results_by_channels[target_channels] = {
            'accuracies': accuracies,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'losses': losses,
            'mean_loss': np.mean(losses),
        }
    
    return results_by_channels


def plot_results(results, subject_id, output_path):
    """Plot channel reduction results"""
    channels = sorted(results.keys())
    means = [results[ch]['mean_accuracy'] for ch in channels]
    stds = [results[ch]['std_accuracy'] for ch in channels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(channels, means, yerr=stds, fmt='o-', capsize=5, capthick=2, 
                markersize=8, linewidth=2, label='Mean Accuracy')
    ax.set_xlabel('Number of Channels', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Channel Reduction Lower Bound - Subject {subject_id}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(channels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path}")
    plt.close()


def save_results_csv(results_all_subjects, output_path):
    """Save results to CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject', 'Channels', 'Mean_Accuracy_%', 'Std_Accuracy_%', 'Mean_Loss'])
        
        for subject_id in sorted(results_all_subjects.keys()):
            results = results_all_subjects[subject_id]
            for channels in sorted(results.keys()):
                writer.writerow([
                    subject_id,
                    channels,
                    f"{results[channels]['mean_accuracy']:.2f}",
                    f"{results[channels]['std_accuracy']:.2f}",
                    f"{results[channels]['mean_loss']:.4f}",
                ])
    
    print(f"Results saved to CSV: {output_path}")


def main():
    # Create output directories
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    PLOTS_SAVE_DIR = RESULTS_DIR / "plots"
    PLOTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Print execution mode
    print(f"\nExecution mode: {'QUICK_MODE (3 subjects, 5 epochs)' if QUICK_MODE else 'FULL_MODE (16 subjects, 50 epochs)'}")
    
    # Run experiment for selected subjects
    all_results = {}
    
    for subject_id in SUBJECTS_LIST:
        results = run_channel_reduction_experiment(subject_id)
        all_results[subject_id] = results
        
        # Plot individual subject results
        plot_path = Path(PLOTS_SAVE_DIR) / f"channel_reduction_subject_{subject_id}.png"
        plot_results(results, subject_id, plot_path)
    
    # Save aggregate results
    csv_path = Path(RESULTS_DIR) / f"channel_reduction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results_csv(all_results, csv_path)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for channels in CHANNEL_STEPS:
        accs = [all_results[s][channels]['mean_accuracy'] for s in SUBJECTS_LIST]
        print(f"  {channels:2d} channels: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")


if __name__ == "__main__":
    main()
