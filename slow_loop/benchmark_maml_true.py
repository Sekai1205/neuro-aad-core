import learn2learn as l2l
import torch, time, numpy as np
from models.ear_eeg_cnn import EarEEGCNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = EarEEGCNN(num_channels=4, virtual_channels=8, num_classes=2).to(device)
maml = l2l.algorithms.MAML(model, lr=0.01, first_order=False)
loss_fn = torch.nn.CrossEntropyLoss()

dummy_input  = torch.randn(8, 4, 256).to(device)  # support set
dummy_target = torch.randint(0, 2, (8,)).to(device)

# ウォームアップ
for _ in range(5):
    learner = maml.clone()
    for _ in range(3):  # inner-loop 3 steps
        loss = loss_fn(learner(dummy_input), dummy_target)
        learner.adapt(loss)

# 安定計測
times = []
for _ in range(20):
    t0 = time.perf_counter()
    learner = maml.clone()
    for _ in range(3):
        loss = loss_fn(learner(dummy_input), dummy_target)
        learner.adapt(loss)
    # メタ勾配まで計算する場合はさらにouter-lossのbackwardも含める
    times.append(time.perf_counter() - t0)

print(f"True MAML inner-loop (3 steps): "
      f"{np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms")