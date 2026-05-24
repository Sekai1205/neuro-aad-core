import sys
sys.path.insert(0, ".")
import scipy.io
from pathlib import Path
from config import DATA_DIR

all_stimuli = set()
for s in range(1, 17):
    mat_path = Path(DATA_DIR) / f"S{s}.mat"
    mat = scipy.io.loadmat(str(mat_path))
    trials = mat['trials']
    for t in range(20):
        trial = trials[0][t]
        for track in [1, 2]:
            st = trial[f'stimuli_track{track}'][0][0]
            if len(st) > 0 and isinstance(st[0], str):
                all_stimuli.add(st[0])
            else:
                all_stimuli.add(str(st))

print(f"Total distinct stimuli: {len(all_stimuli)}")
print(all_stimuli)
