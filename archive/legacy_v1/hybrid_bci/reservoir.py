import numpy as np

class EchoStateNetwork:
    """
    Layer 3: Echo State Network (ESN / Reservoir Computing).
    Maps low-dimensional time-frequency feature inputs into a high-dimensional, 
    non-linear dynamical state space to capture spatiotemporal dynamics.
    """
    def __init__(self, n_reservoir: int = 200, spectral_radius: float = 0.95, 
                 sparsity: float = 0.1, leaking_rate: float = 0.15, seed: int = 42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.seed = seed
        
        self.W_in = None   # Input weights (n_reservoir, n_features)
        self.W_RC = None   # Internal reservoir weights (n_reservoir, n_reservoir)
        self.states = None # Stateful history (n_samples, n_reservoir)
        self.is_initialized = False

    def initialize(self, n_features: int):
        """Initialize fixed random input and reservoir weights."""
        rng = np.random.default_rng(self.seed)
        
        # 1. Initialize input weight matrix W_in ~ Uniform(-0.1, 0.1)
        self.W_in = rng.uniform(-0.1, 0.1, (self.n_reservoir, n_features))
        
        # 2. Initialize internal sparse weight matrix W_RC ~ Normal(0, 1)
        W_raw = rng.normal(0, 1.0, (self.n_reservoir, self.n_reservoir))
        
        # Enforce sparsity
        mask = rng.uniform(0.0, 1.0, W_raw.shape) > self.sparsity
        W_raw[mask] = 0.0
        
        # Scale to match target spectral radius
        eigvals = np.linalg.eigvals(W_raw)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > 0:
            self.W_RC = W_raw * (self.spectral_radius / max_eig)
        else:
            self.W_RC = W_raw
            
        self.is_initialized = True

    def reset_state(self):
        """Reset state history for a new continuous session."""
        self.states = None
        self.h_prev = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Stream input features through the reservoir to obtain state transitions.
        X: (n_samples, n_features)
        Returns: Reservoir state history of shape (n_samples, n_reservoir)
        """
        n_samples, n_features = X.shape
        if not self.is_initialized:
            self.initialize(n_features)
            
        # Z-score standardisation to prevent ESN saturation from positive envelopes
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std < 1e-12] = 1.0
        X_z = (X - X_mean) / X_std
        
        # Scale inputs into highly sensitive linear activation range of tanh [-0.1, 0.1]
        X_scaled = X_z * 0.1
        
        states_history = np.zeros((n_samples, self.n_reservoir))
        
        # Start state is persistent to maintain temporal continuity in streaming
        if not hasattr(self, 'h_prev') or self.h_prev is None:
            self.h_prev = np.zeros(self.n_reservoir)
            
        h_prev = self.h_prev
        
        # Process step-by-step to maintain stateful recurrent updates
        for t in range(n_samples):
            x_t = X_scaled[t]
            # State equation: h_t = (1 - b) * h_{t-1} + b * tanh(W_in @ x_t + W_RC @ h_{t-1})
            net_input = self.W_in @ x_t + self.W_RC @ h_prev
            h_t = (1.0 - self.leaking_rate) * h_prev + self.leaking_rate * np.tanh(net_input)
            
            states_history[t] = h_t
            h_prev = h_t
            
        # Store last state for the next streaming window
        self.h_prev = h_prev
        return states_history
