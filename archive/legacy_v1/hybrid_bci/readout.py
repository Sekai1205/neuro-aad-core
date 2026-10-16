import numpy as np

class StiefelMAMLReadout:
    """
    Layer 4: Stiefel-MAML Orthogonal Readout Solver (1-output Unit-Norm version).
    Learns a linear readout weight vector W_RO (n_reservoir,) mapping ESN states 
    to reconstruct the ATTENDED envelope. 
    Strict normalization is enforced using a projection onto the Stiefel Manifold for vectors (||W_RO||_2 = 1).
    """
    def __init__(self, n_reservoir: int = 200, lambda_: float = 0.001):
        self.n_reservoir = n_reservoir
        self.lambda_ = lambda_
        self.W_RO = None  # (n_reservoir,) Stiefel unit-norm readout weight vector
        self.is_fitted = False

    def project_to_stiefel(self, W: np.ndarray) -> np.ndarray:
        """
        Project a general weight vector W onto the Stiefel Manifold (normalize to unit length).
        """
        norm = np.linalg.norm(W)
        if norm > 1e-12:
            return W / norm
        else:
            return W

    def set_prior_from_others(self, train_readouts: list):
        """
        Set empirical prior weights based on training subjects' readouts, 
        projected onto the Stiefel manifold to enforce unit norm.
        """
        W_mean = np.mean(train_readouts, axis=0)
        self.W_RO = self.project_to_stiefel(W_mean)
        self.is_fitted = True

    def fit_online(self, H: np.ndarray, y: np.ndarray) -> 'StiefelMAMLReadout':
        """
        Online adaptation on calibration data using a Bayesian meta-prior.
        H: (n_samples, n_reservoir) reservoir states
        y: (n_samples,) target attended envelope
        """
        n_samples, n_reservoir = H.shape
        assert len(y) == n_samples, "Length mismatch between reservoir state and targets."
        
        # Z-score target envelope to resolve zero-mean feature vs positive-mean target bias mismatch
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std < 1e-12:
            y_std = 1.0
        y_z = (y - y_mean) / y_std
        
        # 1. Continuous normalized closed-form Bayesian regularized linear solve:
        # Solving: (H.T @ H / N + (lambda + gamma)*I) * W_sol = H.T @ y_z / N + gamma * mu_0
        XtX = (H.T @ H) / n_samples
        Xty = (H.T @ y_z) / n_samples
        
        # Load empirical prior mean mu_0 (from other 15 subjects)
        if self.W_RO is not None:
            mu_0 = self.W_RO
        else:
            mu_0 = np.zeros(n_reservoir)
            
        # Prior anchoring strength (gamma). Since calib is extremely short (10s), 
        # we anchor heavily to the robust empirical BCI prior (gamma = 10.0) to prevent overfitting.
        gamma = 10.0
        
        A = XtX + (self.lambda_ + gamma) * np.eye(n_reservoir)
        b = Xty + gamma * mu_0
        
        W_sol = np.linalg.solve(A, b)
        
        # 2. Project onto Stiefel Manifold (normalize to unit norm)
        self.W_RO = self.project_to_stiefel(W_sol)
        self.is_fitted = True
        
        return self

    def predict(self, H: np.ndarray) -> np.ndarray:
        """
        Reconstruct the attended speech envelope from reservoir states.
        H: (n_samples, n_reservoir)
        Returns: reconstructed envelope of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Readout is not fitted yet.")
            
        return H @ self.W_RO

