import numpy as np

class EuclideanAlignment:
    """
    Layer 1: Online Euclidean Alignment (EA) for spatial whitening.
    Cancels individual gains and impedance variation by mapping average 
    spatial covariance matrices to the identity matrix.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.R_inv_sqrt = None  # (n_channels, n_channels) spatial whitening matrix
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'EuclideanAlignment':
        """
        Compute the spatial whitening matrix R^{-1/2} from calibration EEG data.
        X: (n_samples, n_channels) or (n_channels, n_samples)
        We standardise the input to represent (n_channels, n_samples) internally.
        """
        # Ensure shape is (n_channels, n_samples)
        if X.shape[0] > X.shape[1]:
            X = X.T
            
        n_channels, n_samples = X.shape
        
        # Calculate spatial covariance matrix R
        # Add epsilon to diagonal for numerical stability
        R = (X @ X.T) / n_samples + self.epsilon * np.eye(n_channels)
        
        # Eigendecomposition (R is symmetric, so we use eigh)
        eigvals, eigvecs = np.linalg.eigh(R)
        
        # Prevent division by zero / negative eigenvals
        eigvals = np.maximum(eigvals, self.epsilon)
        
        # Compute R^{-1/2} = U * Sigma^{-1/2} * U^T
        diag_inv_sqrt = 1.0 / np.sqrt(eigvals)
        self.R_inv_sqrt = eigvecs @ np.diag(diag_inv_sqrt) @ eigvecs.T
        self.is_fitted = True
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the calculated spatial whitening matrix to new EEG data frames.
        X: (n_samples, n_channels)
        Returns: Aligned EEG of shape (n_samples, n_channels)
        """
        if not self.is_fitted:
            raise ValueError("EuclideanAlignment is not fitted. Call fit() first.")
            
        # Transform each time point: X_aligned = R_inv_sqrt @ X_t
        # If X is (n_samples, n_channels), we compute: X_aligned = X @ R_inv_sqrt.T
        # Proof: (R_inv_sqrt @ X.T).T = X @ R_inv_sqrt.T = X @ R_inv_sqrt (since symmetric)
        return X @ self.R_inv_sqrt
