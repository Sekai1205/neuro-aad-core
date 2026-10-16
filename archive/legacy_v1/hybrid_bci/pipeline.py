import numpy as np
from scipy import stats, signal
from hybrid_bci.alignment import EuclideanAlignment
from hybrid_bci.features import GaborFeatureExtractor
from hybrid_bci.reservoir import EchoStateNetwork
from hybrid_bci.readout import StiefelMAMLReadout
from hybrid_bci.smoothing import HMMStateSmoothing
from models.mtrf_torch import build_lag_matrix
from pipelines.run_bayesian import bandpass_filter

class HybridBCIPipeline:
    """
    Orchestrates the 5-layer Integrated Hybrid BCI Architecture:
    EEG -> EA -> ST (1-8Hz bandpass) -> 104ch Lag Matrix -> ESN -> Stiefel-MAML Readout -> HMM Smoothing.
    """
    def __init__(self, fs: float = 100, fs_target: float = 64, n_reservoir: int = 200, 
                 lambda_: float = 0.001, transition_prob: float = 0.02, 
                 tmin: float = 0.0, tmax: float = 0.25, epsilon: float = 1e-6):
        self.fs = fs
        self.fs_target = fs_target
        self.tmin = tmin
        self.tmax = tmax
        
        # Instantiate Layers
        self.ea = EuclideanAlignment(epsilon=epsilon)
        self.st = GaborFeatureExtractor(fs=fs, fs_target=fs_target)
        self.esn = EchoStateNetwork(n_reservoir=n_reservoir, leaking_rate=0.15)
        self.readout = StiefelMAMLReadout(n_reservoir=n_reservoir, lambda_=lambda_)
        self.hmm = HMMStateSmoothing(transition_prob=transition_prob)

    def calibrate(self, raw_eeg: np.ndarray, raw_envelopes: np.ndarray, 
                  train_readouts: list, attended_track: int = 1) -> 'HybridBCIPipeline':
        """
        Performs 10-second ultra-fast rocket start calibration.
        raw_eeg: (n_samples, n_channels)
        raw_envelopes: (n_samples, 2) - Track 1 and Track 2 actual envelopes
        train_readouts: list of (n_reservoir,) prior weights from other subjects.
        attended_track: 1 if user is attending to Track 1, 2 if Track 2.
        """
        # 1. Downsample EEG and envelopes to fs_target (e.g. 64Hz)
        eeg_ds = self.st.downsample_signal(raw_eeg)
        env_ds = self.st.downsample_signal(raw_envelopes)
        
        # Extract and downsample the attended envelope
        raw_attended = raw_envelopes[:, 0] if attended_track == 1 else raw_envelopes[:, 1]
        env_attended_ds = self.st.downsample_signal(raw_attended.reshape(-1, 1)).squeeze(1)
        
        # 2. Euclidean Alignment spatial calibration
        self.ea.fit(eeg_ds)
        eeg_aligned = self.ea.transform(eeg_ds)
        
        # 3. Time-frequency bandpass Gabor scattering (1-8 Hz Theta/Alpha Auditory Cortical Band)
        eeg_filt = bandpass_filter(eeg_aligned, self.fs_target, 1.0, 8.0)
        
        # 4. Construct 104-dimensional Lag Feature Matrix (4ch * 26 lags) to resolve acausality
        X_lag, vs, ve = build_lag_matrix(eeg_filt, self.tmin, self.tmax, self.fs_target)
        env_ds_valid = env_ds[vs:ve]
        env_attended_ds_valid = env_attended_ds[vs:ve]
        
        # 5. Echo State Network random mapping
        self.esn.initialize(X_lag.shape[1])
        states = self.esn.transform(X_lag)
        
        # 6. Load empirical readout prior from training subjects
        self.readout.set_prior_from_others(train_readouts)
        
        # 7. Adapt readout weights online via closed-form Stiefel unit-norm projection
        self.readout.fit_online(states, env_attended_ds_valid)
        
        # 8. Collect evidences and calibrate HMM emission parameters dynamically
        calib_len = states.shape[0]
        win_size_ds = int(10.0 * self.fs_target)
        if calib_len <= win_size_ds:
            win_size_ds = calib_len
            
        steps = max(1, (calib_len - win_size_ds) // int(2.0 * self.fs_target) + 1)
        Y_pred = self.readout.predict(states)
        
        calib_evidences = []
        for s in range(steps):
            start = s * int(2.0 * self.fs_target)
            end = start + win_size_ds
            if end > calib_len:
                break
                
            e1 = env_ds_valid[start:end, 0]
            e2 = env_ds_valid[start:end, 1]
            p = Y_pred[start:end]
            
            if np.std(e1) < 1e-12 or np.std(p) < 1e-12 or np.std(e2) < 1e-12:
                continue
                
            r1, _ = stats.pearsonr(e1, p)
            r2, _ = stats.pearsonr(e2, p)
            evidence = (r1 - r2) if attended_track == 1 else (r2 - r1)
            calib_evidences.append(evidence)
            
        if calib_evidences:
            self.hmm.calibrate_emission(np.array(calib_evidences))
            
        return self

    def decode_attention(self, raw_eeg: np.ndarray, raw_env1: np.ndarray, 
                         raw_env2: np.ndarray) -> tuple:
        """
        Decodes attention on a streaming 10-second window.
        raw_eeg: (n_samples, n_channels)
        raw_env1: (n_samples,) Track 1 actual envelope
        raw_env2: (n_samples,) Track 2 actual envelope
        Returns:
            prob_track1: P(z_t = 1 | d_{1:t}) - HMM posterior belief of attending Track 1
            raw_evidence: raw differential correlation r1 - r2
        """
        # 1. Downsample window
        eeg_ds = self.st.downsample_signal(raw_eeg)
        env1_ds = self.st.downsample_signal(raw_env1.reshape(-1, 1)).squeeze(1)
        env2_ds = self.st.downsample_signal(raw_env2.reshape(-1, 1)).squeeze(1)
        
        # 2. Euclidean Alignment
        eeg_aligned = self.ea.transform(eeg_ds)
        
        # 3. Auditory theta bandpass filtering
        eeg_filt = bandpass_filter(eeg_aligned, self.fs_target, 1.0, 8.0)
        
        # 4. Build 104-dimensional Lag Feature Matrix
        X_lag, vs, ve = build_lag_matrix(eeg_filt, self.tmin, self.tmax, self.fs_target)
        
        # 5. Echo State Network propagation
        states = self.esn.transform(X_lag)
        
        # 6. Stiefel Reconstruction
        Y_pred = self.readout.predict(states)
        
        # 7. Pearson correlation scoring
        env1_valid = env1_ds[vs:ve]
        env2_valid = env2_ds[vs:ve]
        
        if len(Y_pred) < 2 or np.std(Y_pred) < 1e-12 or np.std(env1_valid) < 1e-12 or np.std(env2_valid) < 1e-12:
            raw_evidence = 0.0
        else:
            r1, _ = stats.pearsonr(env1_valid, Y_pred)
            r2, _ = stats.pearsonr(env2_valid, Y_pred)
            raw_evidence = r1 - r2
            
        # 8. Stateful HMM update
        prob_track1 = self.hmm.update(raw_evidence)
        
        return prob_track1, raw_evidence
