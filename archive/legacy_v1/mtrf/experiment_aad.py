"""
Auditory Attention Detection (AAD) Experiment
Evaluates AAD performance using Leave-One-Trial-Out cross-validation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR
from data.kuleuven_loader import load_trial_data
from mtrf.aad_classifier import AADClassifier

warnings.filterwarnings('ignore')


class AADExperiment:
    """
    Experiment runner for AAD classification.
    Leave-one-trial-out cross-validation on KULeuven dataset.
    """
    
    def __init__(self, quick_mode: bool = True):
        """
        Initialize experiment.
        
        Args:
            quick_mode: If True, run only on subject S1. If False, all subjects.
        """
        self.quick_mode = quick_mode
        self.results_dir = Path(PROJECT_ROOT) / "results" / "aad_mtrf"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("AAD Classification Experiment (EEG-based)")
        print("="*60)
        print(f"Quick mode (S1 only): {quick_mode}")
        print(f"Results directory: {self.results_dir}")
    
    def run_subject_loocv(self, subject_id: int, downsample_to: float = None) -> dict:
        """
        Run leave-one-trial-out cross-validation for a subject.
        
        Args:
            subject_id: Subject number (1-16)
            downsample_to: Downsample EEG to this rate (Hz), or None for native
        
        Returns:
            dict with cross-validation results
        """
        print(f"\n{'='*60}")
        print(f"Subject S{subject_id}: Leave-One-Trial-Out CV (AAD Classification)")
        print(f"{'='*60}")
        
        # Load all trials for subject
        eeg_list = []
        labels = []
        
        print(f"Loading {20} trials...")
        for trial_idx in range(20):
            try:
                trial_data = load_trial_data(subject_id, trial_idx)
                eeg = trial_data['eeg']  # (samples, 64)
                label = trial_data['label']  # 0=Track1, 1=Track2
                
                eeg_list.append(eeg)
                labels.append(label)
            except Exception as e:
                print(f"  WARNING: Failed to load trial {trial_idx}: {e}")
        
        if len(eeg_list) < 20:
            print(f"ERROR: Only loaded {len(eeg_list)}/20 trials")
            return None
        
        labels = np.array(labels)
        print(f"Loaded {len(eeg_list)} trials successfully")
        print(f"Label distribution: Track1={np.sum(labels==0)}, Track2={np.sum(labels==1)}")
        
        # Run LOOCV
        print("\nRunning Leave-One-Trial-Out CV...")
        predictions = []
        probabilities = []
        
        for test_idx in range(20):
            # Create training set (all except test_idx)
            train_eeg_list = [eeg_list[i] for i in range(20) if i != test_idx]
            train_labels = np.array([labels[i] for i in range(20) if i != test_idx])
            
            # Test data
            test_eeg = eeg_list[test_idx]
            test_label = labels[test_idx]
            
            # Initialize and fit classifier
            classifier = AADClassifier(random_seed=42)
            classifier.fit(train_eeg_list, train_labels)
            
            # Predict on test trial
            pred, conf = classifier.predict(test_eeg)
            proba = classifier.predict_proba(test_eeg)
            
            predictions.append(pred)
            probabilities.append(proba[1])  # Probability for Track2
            
            correct = "✓" if pred == test_label else "✗"
            print(f"  Trial {test_idx+1:02d}/20: Pred={pred}, True={test_label} {correct} (conf={conf:.3f})")
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Aggregate results
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
        
        accuracy = accuracy_score(labels, predictions)
        auc_score = roc_auc_score(labels, probabilities)
        cm = confusion_matrix(labels, predictions)
        
        print(f"\n{'='*60}")
        print(f"Results for Subject S{subject_id}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC:      {auc_score:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    Predicted Track1: {cm[0,0]} correct, {cm[0,1]} false positives")
        print(f"    Predicted Track2: {cm[1,0]} false negatives, {cm[1,1]} correct")
        print(f"{'='*60}")
        
        return {
            'subject_id': subject_id,
            'predictions': predictions,
            'probabilities': probabilities,
            'ground_truth': labels,
            'accuracy': accuracy,
            'auc': auc_score,
            'confusion_matrix': cm,
            'n_trials': len(predictions)
        }
    
    def save_results(self, results: dict):
        """
        Save results to CSV and create visualizations.
        
        Args:
            results: Dictionary with cross-validation results
        """
        subject_id = results['subject_id']
        predictions = results['predictions']
        probabilities = results['probabilities']
        ground_truth = results['ground_truth']
        accuracy = results['accuracy']
        auc_score = results['auc']
        cm = results['confusion_matrix']
        
        # Save CSV
        csv_path = self.results_dir / f"aad_loocv_s{subject_id}.csv"
        df = pd.DataFrame({
            'trial': np.arange(1, len(predictions) + 1),
            'ground_truth': ground_truth,
            'prediction': predictions,
            'probability_track2': probabilities
        })
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Confusion Matrix
        ax = axes[0, 0]
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(cm.flatten()))
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Track1', 'Track2'])
        ax.set_yticklabels(['Track1', 'Track2'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title('Confusion Matrix', fontweight='bold')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        # 2. ROC Curve
        ax = axes[0, 1]
        fpr, tpr, _ = roc_curve(ground_truth, probabilities)
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={auc_score:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Trial-by-trial predictions
        ax = axes[1, 0]
        trials = np.arange(1, len(predictions) + 1)
        colors = ['green' if predictions[i] == ground_truth[i] else 'red' for i in range(len(predictions))]
        ax.scatter(trials, probabilities, c=colors, alpha=0.6, s=100)
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Probability (Track2)')
        ax.set_ylim([-0.05, 1.05])
        ax.set_title('Trial-by-Trial Predictions (Green=Correct, Red=Error)', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 4. Accuracy metrics
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = f"""
        Subject S{subject_id} - AAD Classification Results
        
        Accuracy:  {accuracy:.4f} ({int(np.sum(predictions==ground_truth))}/{len(predictions)})
        AUC:       {auc_score:.4f}
        
        Confusion Matrix:
        ┌────────────────────┐
        │ TP={cm[1,1]:2d}  FP={cm[0,1]:2d} │
        │ FN={cm[1,0]:2d}  TN={cm[0,0]:2d} │
        └────────────────────┘
        
        Sensitivity: {cm[1,1]/(cm[1,1]+cm[1,0]):.4f}  (Track2 recall)
        Specificity: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}  (Track1 recall)
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace', 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plot_path = self.results_dir / f"aad_loocv_s{subject_id}_results.png"
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
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE - AAD Classification Summary")
        print(f"{'='*60}")
        
        if all_results:
            accuracies = [res['accuracy'] for res in all_results]
            aucs = [res['auc'] for res in all_results]
            
            print(f"Processed {len(all_results)} subject(s)")
            for res in all_results:
                print(f"  S{res['subject_id']}: Accuracy={res['accuracy']:.4f}, AUC={res['auc']:.4f}")
            
            if len(all_results) > 1:
                print(f"\nOverall Statistics:")
                print(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
                print(f"  Mean AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
                print(f"  Best Accuracy: {np.max(accuracies):.4f}")
                print(f"  Best AUC:      {np.max(aucs):.4f}")


def main():
    """
    Main entry point for experiment.
    """
    # Run in quick mode (S1 only)
    experiment = AADExperiment(quick_mode=True)
    experiment.run()


if __name__ == "__main__":
    main()
