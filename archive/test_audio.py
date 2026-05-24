import sys
sys.path.insert(0, ".")
import scipy.io
from pathlib import Path
from config import DATA_DIR
import numpy as np

mat = scipy.io.loadmat(str(Path(DATA_DIR) / "S1.mat"))
for t in range(20):
    tr = mat['trials'][0][t]
    print(tr['stimuli_track1'][0][0], tr['stimuli_track2'][0][0])
