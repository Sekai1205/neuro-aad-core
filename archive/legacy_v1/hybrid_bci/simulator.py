import numpy as np
import pandas as pd
from scipy import stats
from hybrid_bci.pipeline import HybridBCIPipeline
from hybrid_bci.readout import StiefelMAMLReadout
from hybrid_bci.features import GaborFeatureExtractor
from hybrid_bci.reservoir import EchoStateNetwork
from hybrid_bci.alignment import EuclideanAlignment

def train_subject_readout(raw_eeg: np.ndarray, raw_attended_env: np.ndarray, 
                          fs: float, fs_target: float, n_reservoir: int, lambda_: float) -> np.ndarray:
    """
    Fits a unit-norm Stiefel readout W_RO (n_reservoir,) for a training subject.
    """
    # 1. Downsample EEG and envelope
    st = GaborFeatureExtractor(fs=fs, fs_target=fs_target)
    eeg_ds = st.downsample_signal(raw_eeg)
    env_ds = st.downsample_signal(raw_attended_env.reshape(-1, 1)).squeeze(1)
    
    # 2. Euclidean Alignment
    ea = EuclideanAlignment()
    ea.fit(eeg_ds)
    eeg_aligned = ea.transform(eeg_ds)
    
    # 3. Time-frequency bandpass Gabor scattering (1-8 Hz Theta/Alpha Auditory Cortical Band)
    from pipelines.run_bayesian import bandpass_filter
    from models.mtrf_torch import build_lag_matrix
    eeg_filt = bandpass_filter(eeg_aligned, fs_target, 1.0, 8.0)
    
    # 4. Construct 104-dimensional Lag Feature Matrix (4ch * 26 lags) to resolve acausality
    X_lag, vs, ve = build_lag_matrix(eeg_filt, 0.0, 0.25, fs_target)
    env_ds_valid = env_ds[vs:ve]
    
    # 5. Echo State Network
    esn = EchoStateNetwork(n_reservoir=n_reservoir)
    esn.initialize(X_lag.shape[1])
    states = esn.transform(X_lag)
    
    # 6. Fit Readout without prior (starts from scratch)
    readout = StiefelMAMLReadout(n_reservoir=n_reservoir, lambda_=lambda_)
    readout.fit_online(states, env_ds_valid)
    
    return readout.W_RO


class HybridBCISimulator:
    """
    PC-based Virtual Real-Time BCI Attention Tracking Simulator.
    Simulates a continuous AAD hearing aid session using the 5-layer hybrid BCI pipeline.
    Uses a mathematically pristine rolling-buffer continuous stream simulation.
    """
    def __init__(self, fs: float = 100, fs_target: float = 64, n_reservoir: int = 200, 
                 lambda_: float = 0.001, transition_prob: float = 0.02, 
                 calib_s: float = 10.0, step_s: float = 2.0):
        self.fs = fs
        self.fs_target = fs_target
        self.n_reservoir = n_reservoir
        self.lambda_ = lambda_
        self.transition_prob = transition_prob
        self.calib_s = calib_s
        self.step_s = step_s
        
        self.pipeline = None
        self.precomputed_readouts = {}

    def run_simulation(self, subjects_data: dict, test_sub_id: int) -> pd.DataFrame:
        """
        Runs the continuous 77-minute real-time simulation on a test subject.
        """
        # 1. Train prior orthogonal readouts from other subjects
        print(f"  [Simulator] Constructing Stiefel priors from other subjects...")
        train_ids = [s for s in range(1, 17) if s != test_sub_id]
        train_readouts = []
        for train_id in train_ids:
            if train_id not in self.precomputed_readouts:
                sub_trials = subjects_data[train_id]
                eeg_c = np.concatenate([t['eeg'] for t in sub_trials], axis=0)
                env_att_c = np.concatenate([
                    t['envelope_track1'] if t['ground_truth_track'] == 1 else t['envelope_track2']
                    for t in sub_trials
                ], axis=0)
                
                W_ro = train_subject_readout(eeg_c, env_att_c, self.fs, self.fs_target, 
                                             self.n_reservoir, self.lambda_)
                self.precomputed_readouts[train_id] = W_ro
            train_readouts.append(self.precomputed_readouts[train_id])
            
        # 2. Concatenate test subject trials chronologically
        print(f"  [Simulator] Loading test subject S{test_sub_id} continuous session...")
        test_trials = subjects_data[test_sub_id]
        eeg_concat = np.concatenate([t['eeg'] for t in test_trials], axis=0)
        env1_concat = np.concatenate([t['envelope_track1'] for t in test_trials], axis=0)
        env2_concat = np.concatenate([t['envelope_track2'] for t in test_trials], axis=0)
        
        # Build continuous ground truth sequence and attended envelopes
        attended_env_concat = []
        ground_truth_seq = []
        for t in test_trials:
            n_samples = t['eeg'].shape[0]
            gt = t['ground_truth_track']
            ground_truth_seq.extend([gt] * n_samples)
            attended_env = t['envelope_track1'] if gt == 1 else t['envelope_track2']
            attended_env_concat.extend(attended_env)
            
        ground_truth_seq = np.array(ground_truth_seq)
        attended_env_concat = np.array(attended_env_concat)
        
        # 3. Downsample entire session EEG and envelopes to fs_target continuously
        print(f"  [Simulator] Downsampling continuous session to target rate {self.fs_target}Hz...")
        self.pipeline = HybridBCIPipeline(
            fs=self.fs, fs_target=self.fs_target, n_reservoir=self.n_reservoir,
            lambda_=self.lambda_, transition_prob=self.transition_prob
        )
        
        eeg_ds = self.pipeline.st.downsample_signal(eeg_concat)
        env1_ds = self.pipeline.st.downsample_signal(env1_concat.reshape(-1, 1)).squeeze(1)
        env2_ds = self.pipeline.st.downsample_signal(env2_concat.reshape(-1, 1)).squeeze(1)
        env_ds = np.column_stack([env1_ds, env2_ds])
        
        env_att_ds = self.pipeline.st.downsample_signal(attended_env_concat.reshape(-1, 1)).squeeze(1)
        
        # Map ground truth indices matching the downsampled rate
        n_samples_ds = eeg_ds.shape[0]
        gt_indices = np.round(np.arange(n_samples_ds) * eeg_concat.shape[0] / n_samples_ds).astype(int)
        gt_ds = ground_truth_seq[np.minimum(gt_indices, ground_truth_seq.shape[0] - 1)]
        
        # 4. Euclidean Alignment and continuous ESN mapping on the entire session
        print("  [Simulator] Spatial alignment and continuous ESN state expansion...")
        # Fit EA on first 10 seconds of downsampled EEG
        calib_samples_ds = int(self.calib_s * self.fs_target)
        self.pipeline.ea.fit(eeg_ds[:calib_samples_ds])
        eeg_aligned = self.pipeline.ea.transform(eeg_ds)
        
        # Extract features and construct lag feature matrix continuously
        from pipelines.run_bayesian import bandpass_filter
        from models.mtrf_torch import build_lag_matrix
        eeg_filt = bandpass_filter(eeg_aligned, self.fs_target, 1.0, 8.0)
        
        X_lag, vs, ve = build_lag_matrix(eeg_filt, self.pipeline.tmin, self.pipeline.tmax, self.fs_target)
        env_ds_valid = env_ds[vs:ve]
        env_att_ds_valid = env_att_ds[vs:ve]
        
        self.pipeline.esn.initialize(X_lag.shape[1])
        states_all = self.pipeline.esn.transform(X_lag)
        
        # 5. Calibration Phase
        print(f"  [Simulator] Calibrating orthogonal readout with empirical priors...")
        self.pipeline.readout.set_prior_from_others(train_readouts)
        self.pipeline.readout.fit_online(states_all[:calib_samples_ds], env_att_ds_valid[:calib_samples_ds])
        
        # Calibrate HMM emission probability dynamically
        calib_len = calib_samples_ds
        win_size_ds = int(10.0 * self.fs_target)
        if calib_len <= win_size_ds:
            win_size_ds = calib_len
            
        steps = max(1, (calib_len - win_size_ds) // int(2.0 * self.fs_target) + 1)
        Y_pred_calib = self.pipeline.readout.predict(states_all[:calib_samples_ds])
        
        calib_evidences = []
        for s in range(steps):
            start = s * int(2.0 * self.fs_target)
            end = start + win_size_ds
            if end > calib_len:
                break
            e1 = env_ds_valid[start:end, 0]
            e2 = env_ds_valid[start:end, 1]
            p = Y_pred_calib[start:end]
            if np.std(e1) < 1e-12 or np.std(p) < 1e-12 or np.std(e2) < 1e-12:
                continue
            r1, _ = stats.pearsonr(e1, p)
            r2, _ = stats.pearsonr(e2, p)
            evidence = (r1 - r2) if gt_ds[np.minimum(start + win_size_ds // 2 + vs, gt_ds.shape[0] - 1)] == 1 else (r2 - r1)
            calib_evidences.append(evidence)
            
        if calib_evidences:
            self.pipeline.hmm.calibrate_emission(np.array(calib_evidences))
            
        # 6. Sliding Window Online Real-Time Decoding
        print(f"  [Simulator] Simulating online tracking with {self.step_s}s step-size...")
        time_history = []
        raw_evidences = []
        hmm_posteriors = []
        gt_history = []
        predictions = []
        
        win_size_ds = int(10.0 * self.fs_target)
        step_size_ds = int(self.step_s * self.fs_target)
        
        # Reset HMM state
        self.pipeline.hmm.reset()
        
        start_ds = calib_samples_ds
        while start_ds + win_size_ds <= states_all.shape[0]:
            end_ds = start_ds + win_size_ds
            current_time = (start_ds + vs) / self.fs_target
            
            win_states = states_all[start_ds:end_ds]
            win_env1 = env_ds_valid[start_ds:end_ds, 0]
            win_env2 = env_ds_valid[start_ds:end_ds, 1]
            win_gt = gt_ds[np.minimum(start_ds + win_size_ds // 2 + vs, gt_ds.shape[0] - 1)]
            
            # Predict reconstruction
            Y_pred = self.pipeline.readout.predict(win_states)
            
            if len(Y_pred) < 2 or np.std(Y_pred) < 1e-12 or np.std(win_env1) < 1e-12 or np.std(win_env2) < 1e-12:
                raw_evidence = 0.0
            else:
                r1, _ = stats.pearsonr(win_env1, Y_pred)
                r2, _ = stats.pearsonr(win_env2, Y_pred)
                raw_evidence = r1 - r2
                
            # HMM forward recursive update
            prob_track1 = self.pipeline.hmm.update(raw_evidence)
            pred = 1 if prob_track1 > 0.5 else 2
            
            time_history.append(current_time)
            raw_evidences.append(raw_evidence)
            hmm_posteriors.append(prob_track1)
            gt_history.append(win_gt)
            predictions.append(pred)
            
            start_ds += step_size_ds
            
        print(f"  [Simulator] Simulation Finished! Logged {len(time_history)} decision steps.")
        
        sim_df = pd.DataFrame({
            'time_s': time_history,
            'raw_evidence': raw_evidences,
            'hmm_posterior_track1': hmm_posteriors,
            'ground_truth': gt_history,
            'prediction': predictions
        })
        
        return sim_df
