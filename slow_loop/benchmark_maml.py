import torch
import time
import numpy as np
from thop import profile
from models.ear_eeg_cnn import EarEEGCNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Target Device: {device}")

model = EarEEGCNN(num_channels=4, virtual_channels=8, num_classes=2).to(device)
dummy_input = torch.randn(1, 4, 256).to(device)
dummy_target = torch.tensor([0]).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# FLOPs計算
macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
print(f"Model FLOPs (Forward): {macs * 2 / 1e6:.2f} Mega FLOPs")

print("\n--- ウォームアップフェーズ（初期化オーバーヘッドの消化） ---")
for i in range(10):
    optimizer.zero_grad()
    loss = loss_fn(model(dummy_input), dummy_target)
    loss.backward()
    optimizer.step()
print("ウォームアップ完了。")

print("\n--- 安定レイテンシ計測フェーズ ---")
times = []
for i in range(10):
    t0 = time.perf_counter()
    optimizer.zero_grad()
    loss = loss_fn(model(dummy_input), dummy_target)
    loss.backward()
    optimizer.step()
    times.append(time.perf_counter() - t0)

latency_mean = np.mean(times) * 1000
latency_std = np.std(times) * 1000

print(f"🎯 安定したMAML Inner-loop (3 steps) レイテンシ: {latency_mean:.1f} ± {latency_std:.1f} ms")

if latency_mean < 100:
    print("✅ 結論: リアルタイム処理要件（100ms以内）をクリア。時分割/並列実行の余地あり。")
else:
    print(f"⚠️ 結論: 依然として {latency_mean:.1f} ms。Slow Loopの更新周期設計、またはInner-loop step数の削減（3→1）等の再検討が必要。")