import torch
import time
import numpy as np
from models.ear_eeg_cnn import EarEEGCNN
import scipy.signal as signal

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 実環境のパラメータ設定
num_channels = 4
window_samples = 256  # 2.56s @ 100Hz
batch_size = 8        # 4-shot * 2 classes
inner_steps = 3
inner_lr = 0.01

model = EarEEGCNN(num_channels=num_channels, virtual_channels=8, num_classes=2).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)

# ダミーの生EEGデータ（バッファから取得された直後を想定）
raw_buffer_data = np.random.randn(batch_size, num_channels, window_samples)
dummy_target = torch.randint(0, 2, (batch_size,)).to(device)

def simulate_end_to_end_slow_loop():
    t_start = time.perf_counter()
    
    # 1. 前処理フェーズ (Bandpass filter & Z-score)
    t_pre_start = time.perf_counter()
    processed_data = np.zeros_like(raw_buffer_data)
    b, a = signal.butter(4, [1.0, 40.0], btype='bandpass', fs=100)
    
    for i in range(batch_size):
        for c in range(num_channels):
            # フィルタリング
            filtered = signal.filtfilt(b, a, raw_buffer_data[i, c])
            # Z-score正規化
            processed_data[i, c] = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
            
    input_tensor = torch.tensor(processed_data, dtype=torch.float32).to(device)
    t_pre_end = time.perf_counter()

    # 2. FOMAML 適応フェーズ (First-order SGD)
    t_adapt_start = time.perf_counter()
    for _ in range(inner_steps):
        optimizer.zero_grad()
        loss = loss_fn(model(input_tensor), dummy_target)
        loss.backward()
        optimizer.step()
    t_adapt_end = time.perf_counter()

    # 3. Fast Loopへの重みスワップ (シミュレーション: dict copy)
    t_swap_start = time.perf_counter()
    _ = {k: v.clone() for k, v in model.state_dict().items()}
    t_swap_end = time.perf_counter()
    
    t_total_end = time.perf_counter()
    
    return {
        'total': (t_total_end - t_start) * 1000,
        'preprocess': (t_pre_end - t_pre_start) * 1000,
        'adaptation': (t_adapt_end - t_adapt_start) * 1000,
        'swap': (t_swap_end - t_swap_start) * 1000
    }

# ウォームアップ
for _ in range(5):
    _ = simulate_end_to_end_slow_loop()

# 安定計測
results = {'total': [], 'preprocess': [], 'adaptation': [], 'swap': []}
for _ in range(20):
    res = simulate_end_to_end_slow_loop()
    for k in results.keys():
        results[k].append(res[k])

print(f"--- Slow Loop End-to-End Latency (Batch={batch_size}, Shape={num_channels}x{window_samples}) ---")
print(f"Total E2E Latency: {np.mean(results['total']):.1f} ± {np.std(results['total']):.1f} ms")
print(f"  ├─ Preprocessing:  {np.mean(results['preprocess']):.1f} ms")
print(f"  ├─ Adaptation:     {np.mean(results['adaptation']):.1f} ms")
print(f"  └─ Weights Swap:   {np.mean(results['swap']):.1f} ms")