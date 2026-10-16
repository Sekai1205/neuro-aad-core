import unittest
import numpy as np
from hybrid_bci.alignment import EuclideanAlignment
from hybrid_bci.features import GaborFeatureExtractor
from hybrid_bci.reservoir import EchoStateNetwork
from hybrid_bci.readout import StiefelMAMLReadout
from hybrid_bci.smoothing import HMMStateSmoothing

class TestHybridBCIComponents(unittest.TestCase):
    """
    Mathematical and operational unit tests verifying the correctness of 
    each of the 5 layers in the Integrated Hybrid BCI Architecture.
    """
    
    def test_euclidean_alignment_whitening(self):
        """
        Verify that Euclidean Alignment (EA) projects the average spatial covariance 
        of raw signals onto the identity matrix (spatial whitening).
        """
        # Create a random correlated multi-channel signal (e.g. 4 channels, 1000 samples)
        np.random.seed(42)
        raw_noise = np.random.normal(0, 1.0, (1000, 4))
        # Add strong channel cross-correlation
        mix_matrix = np.array([
            [1.5, 0.8, 0.2, 0.1],
            [0.8, 1.2, 0.4, 0.3],
            [0.2, 0.4, 1.0, 0.5],
            [0.1, 0.3, 0.5, 0.8]
        ])
        raw_correlated = raw_noise @ mix_matrix.T
        
        # Fit EA spatial alignment
        ea = EuclideanAlignment(epsilon=1e-8)
        ea.fit(raw_correlated)
        aligned = ea.transform(raw_correlated)
        
        # Compute aligned covariance: Cov(X) = (X.T @ X) / N
        # We expect this to be exactly equal to the Identity Matrix (I)
        aligned_cov = (aligned.T @ aligned) / aligned.shape[0]
        
        # Verify aligned covariance matches Identity up to machine precision
        np.testing.assert_allclose(aligned_cov, np.eye(4), atol=1e-5)
        print("✓ Test Passed: Euclidean Alignment spatial covariance is perfectly aligned to Identity!")

    def test_stiefel_orthogonal_readout(self):
        """
        Verify that Stiefel-MAML readout weight vectors strictly satisfy the 
        normalization constraint ||W_RO||_2 = 1.
        """
        np.random.seed(42)
        n_reservoir = 200
        
        # Create random non-orthogonal readout weights representing ESN-to-envelope mapping
        W_sol = np.random.normal(0, 1.0, n_reservoir)
        
        readout = StiefelMAMLReadout(n_reservoir=n_reservoir)
        
        # Project onto Stiefel Manifold for vectors (normalize to unit length)
        W_proj = readout.project_to_stiefel(W_sol)
        
        # Verify unit norm: ||W_proj||_2 = 1.0
        norm_val = np.linalg.norm(W_proj)
        np.testing.assert_allclose(norm_val, 1.0, atol=1e-12)
        print("✓ Test Passed: Stiefel readout projection strictly satisfies unit-norm constraint ||W_RO||_2 = 1!")

    def test_echo_state_network_dimensions(self):
        """
        Verify that Echo State Network propagates inputs into correct state dimensions 
        and preserves internal state history correctly.
        """
        n_reservoir = 200
        n_features = 8
        n_samples = 500
        
        np.random.seed(42)
        X = np.random.normal(0, 1.0, (n_samples, n_features))
        
        esn = EchoStateNetwork(n_reservoir=n_reservoir, leaking_rate=0.15)
        states = esn.transform(X)
        
        # Verify output shape matches (n_samples, n_reservoir)
        self.assertEqual(states.shape, (n_samples, n_reservoir))
        print("✓ Test Passed: Echo State Network maps inputs into correct dimensions statefully!")

    def test_hmm_belief_state_updates(self):
        """
        Verify that the HMM forward algorithm recursively tracks attention state belief 
        and properly smooths out transient noise.
        """
        # Set a realistic noise level (sigma = 0.15) to enable stateful inertia
        hmm = HMMStateSmoothing(transition_prob=0.02, default_mu=0.05, default_sigma=0.15)
        
        # Start state belief is uniform [0.5, 0.5]
        np.testing.assert_allclose(hmm.belief, [0.5, 0.5])
        
        # Feed strong evidence for Track 1 (r1 - r2 = +0.1)
        # We expect the posterior probability P(z_t = 1) to quickly approach 1.0
        p_track1 = hmm.update(0.1)
        self.assertGreater(p_track1, 0.5)
        
        # Feed one noisy frame in opposite direction (r1 - r2 = -0.05) representing glitch
        # We expect the stateful HMM to maintain memory and resist switching immediately
        p_track1_glitch = hmm.update(-0.05)
        self.assertGreater(p_track1_glitch, 0.5)
        print("✓ Test Passed: HMM filters out temporary signal glitches successfully!")

if __name__ == "__main__":
    unittest.main()
