import numpy as np
from scipy import stats

class HMMStateSmoothing:
    """
    Layer 5: Stateful Hidden Markov Model (HMM) Attention State Filtering.
    Tracks attention states dynamically by modeling attention transition probability 
    and likelihood functions to filter out high-frequency decision jitter.
    """
    def __init__(self, transition_prob: float = 0.02, default_mu: float = 0.05, 
                 default_sigma: float = 0.08):
        self.transition_prob = transition_prob
        self.mu = default_mu
        self.sigma = default_sigma
        
        # Transition probability matrix:
        # Row 0: state 1 (Track 1), Row 1: state 2 (Track 2)
        # self.A[i, j] = P(z_t = j | z_{t-1} = i)
        p = transition_prob
        self.A = np.array([
            [1.0 - p, p],
            [p, 1.0 - p]
        ])
        
        # Stateful belief state (posterior) P(z_t | d_{1:t})
        # Initialized to uniform prior [0.5, 0.5]
        self.belief = np.array([0.5, 0.5])
        self.is_calibrated = False

    def calibrate_emission(self, calib_evidences: np.ndarray):
        """
        Dynamically calibrate the emission distribution parameters (mu, sigma) 
        from calibration evidence signals to adapt to target subject's specific SNR.
        """
        if len(calib_evidences) < 2:
            return
            
        # We assume the calibration evidences represent correct tracking
        # So the absolute value reflects the standard attention distance
        abs_ev = np.abs(calib_evidences)
        self.mu = float(np.mean(abs_ev))
        self.sigma = float(np.std(abs_ev))
        
        # Prevent division by zero / negative parameters
        if self.sigma < 1e-4:
            self.sigma = 0.08
        if self.mu < 1e-4:
            self.mu = 0.05
            
        self.is_calibrated = True

    def reset(self):
        """Reset belief state for a new session."""
        self.belief = np.array([0.5, 0.5])

    def _gaussian_likelihood(self, val: float, mean: float, std: float) -> float:
        """Helper to calculate Gaussian likelihood."""
        # Equivalent to scipy.stats.norm.pdf(val, mean, std)
        return (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((val - mean) / std) ** 2)

    def update(self, evidence: float) -> float:
        """
        Update the HMM belief state recursively using the forward algorithm.
        evidence: raw differential correlation (r1 - r2)
        Returns: P(z_t = 1 | d_{1:t}), i.e., posterior probability of attending Track 1.
        """
        # 1. Compute emission likelihoods for state 1 (attending Track 1) and state 2 (Track 2)
        # If attending Track 1, r1 > r2 -> evidence > 0 (mean = +mu)
        # If attending Track 2, r1 < r2 -> evidence < 0 (mean = -mu)
        p_emit_1 = self._gaussian_likelihood(evidence, self.mu, self.sigma)
        p_emit_2 = self._gaussian_likelihood(evidence, -self.mu, self.sigma)
        
        # Prevent absolute zero likelihoods
        p_emit_1 = max(p_emit_1, 1e-12)
        p_emit_2 = max(p_emit_2, 1e-12)
        
        # 2. Forward step: P(z_t = j, d_{1:t}) = P(d_t | z_t = j) * sum_i P(z_t = j | z_{t-1} = i) * P(z_{t-1} = i | d_{1:t-1})
        alpha_1 = p_emit_1 * (self.belief[0] * self.A[0, 0] + self.belief[1] * self.A[1, 0])
        alpha_2 = p_emit_2 * (self.belief[0] * self.A[0, 1] + self.belief[1] * self.A[1, 1])
        
        # 3. Normalise belief state to obtain posterior P(z_t | d_{1:t})
        total = alpha_1 + alpha_2
        self.belief = np.array([alpha_1 / total, alpha_2 / total])
        
        # Return posterior probability of attending Track 1
        return self.belief[0]
