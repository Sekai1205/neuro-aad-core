"""
Auditory Attention Detection (AAD) Classifier using EEG Features
Performs left/right attention classification using EEG time-domain statistics
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from typing import Tuple, Optional


class AADClassifier:
    """
    AAD Classifier using EEG time-domain features.
    
    Strategy:
    1. Extract time-domain statistics from EEG (mean, std, RMS per channel)
    2. Train logistic regression to map features -> attended_track (Track1/Track2)
    
    This simplified approach avoids audio file dependencies and focuses on EEG-based features.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize AAD classifier.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        
        # Logistic regression classifier
        self.logistic_clf = LogisticRegression(
            random_state=random_seed,
            max_iter=1000,
            solver='lbfgs'
        )
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        self.n_features = None
    
    def _extract_eeg_features(self, eeg: np.ndarray) -> np.ndarray:
        """
        Extract time-domain statistics from EEG.
        
        Features per channel: mean, std, RMS, min, max, peak-to-peak
        
        Args:
            eeg: EEG signal (n_samples, n_channels)
        
        Returns:
            features: Extracted features (n_channels * n_features_per_channel,)
        """
        features = []
        
        for ch in range(eeg.shape[1]):
            signal = eeg[:, ch]
            
            # Time-domain statistics
            features.extend([
                np.mean(signal),           # Mean
                np.std(signal),            # Standard deviation
                np.sqrt(np.mean(signal**2)),  # RMS
                np.min(signal),            # Minimum
                np.max(signal),            # Maximum
                np.max(signal) - np.min(signal),  # Peak-to-peak
            ])
        
        return np.array(features)
    
    def fit(self, eeg_list: list, labels: np.ndarray) -> 'AADClassifier':
        """
        Train AAD classifier on multiple trials.
        
        Args:
            eeg_list: List of EEG arrays, each (n_samples, n_channels)
            labels: Binary labels for each trial (n_trials,), 0=Track1, 1=Track2
        
        Returns:
            self
        """
        print(f"\n{'='*60}")
        print("AAD Classifier Training")
        print(f"{'='*60}")
        print(f"Training on {len(eeg_list)} trials")
        
        # Extract features from each trial
        features_list = []
        for idx, eeg in enumerate(eeg_list):
            features = self._extract_eeg_features(eeg)
            features_list.append(features)
        
        # Stack features
        X = np.array(features_list)  # (n_trials, n_features)
        y = labels
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution: Track1={np.sum(y==0)}, Track2={np.sum(y==1)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train logistic regression
        self.logistic_clf.fit(X_scaled, y)
        
        self.is_fitted = True
        self.n_features = X.shape[1]
        
        print(f"Classifier training complete")
        print(f"Number of features: {self.n_features}")
        
        return self
    
    def predict(self, eeg: np.ndarray) -> Tuple[int, float]:
        """
        Predict attention direction for a single trial.
        
        Args:
            eeg: EEG signal (n_samples, n_channels)
        
        Returns:
            prediction: Predicted track (0 or 1)
            confidence: Prediction confidence (probability)
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        # Extract features
        features = self._extract_eeg_features(eeg)
        X = features.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.logistic_clf.predict(X_scaled)[0]
        confidence = np.max(self.logistic_clf.predict_proba(X_scaled)[0])
        
        return prediction, confidence
    
    def predict_batch(self, eeg_list: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict attention direction for multiple trials.
        
        Args:
            eeg_list: List of EEG arrays
        
        Returns:
            predictions: Predicted tracks (n_trials,)
            confidences: Prediction confidences (n_trials,)
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        features_list = []
        for eeg in eeg_list:
            features = self._extract_eeg_features(eeg)
            features_list.append(features)
        
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.logistic_clf.predict(X_scaled)
        probabilities = self.logistic_clf.predict_proba(X_scaled)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
    
    def predict_proba(self, eeg: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for both classes.
        
        Args:
            eeg: EEG signal (n_samples, n_channels)
        
        Returns:
            proba: Probabilities for [Track1, Track2]
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        features = self._extract_eeg_features(eeg)
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        return self.logistic_clf.predict_proba(X_scaled)[0]
    
    def score(self, eeg_list: list, labels: np.ndarray) -> float:
        """
        Evaluate classifier accuracy on test set.
        
        Args:
            eeg_list: List of EEG arrays
            labels: Ground truth labels
        
        Returns:
            accuracy: Classification accuracy
        """
        predictions, _ = self.predict_batch(eeg_list)
        accuracy = accuracy_score(labels, predictions)
        return accuracy
    
    def evaluate(self, eeg_list: list, labels: np.ndarray) -> dict:
        """
        Comprehensive evaluation including AUC and confusion matrix.
        
        Args:
            eeg_list: List of EEG arrays
            labels: Ground truth labels
        
        Returns:
            metrics: Dictionary with accuracy, auc, confusion_matrix
        """
        # Get predictions and probabilities
        features_list = []
        for eeg in eeg_list:
            features = self._extract_eeg_features(eeg)
            features_list.append(features)
        
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.logistic_clf.predict(X_scaled)
        probabilities = self.logistic_clf.predict_proba(X_scaled)[:, 1]  # Probability for class 1
        
        # Metrics
        accuracy = accuracy_score(labels, predictions)
        auc = roc_auc_score(labels, probabilities)
        cm = confusion_matrix(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
