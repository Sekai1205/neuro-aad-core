import sys
sys.path.insert(0, ".")
from mtrf.experiment_aad_mtrf import AADmTRFExperiment
exp = AADmTRFExperiment(quick_mode=True)
data1 = exp.load_trial_data_with_envelopes(1, 0, target_fs=100)
print(list(exp.envelopes_cache.keys()))
