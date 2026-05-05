import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy.io as sio
import numpy as np
import os
import json
import higher
from models.eeg_1d_cnn import EEG1DCNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ============================================================
# 【改修③】 DataLoaderによるショット数最適化
# 旧実装: インデックスで直接スライス (shot=32, 全データの1/10のみ使用)
# 新実装: TensorDataset + DataLoader
#   - shuffle=True でエポックごとにランダム化
#   - drop_last=True でBatchNorm崩壊防止
#   - num_workers=2 で並列プリフェッチ
#   → shot=256まで引き上げ (旧比8倍のコンテキスト長)
# ============================================================
def load_ku_leuven_data(file_path, max_windows_per_trial=None):
    """
    Returns:
        TensorDataset: X shape (N,64,256), y shape (N,)
    """
    print(f"  📂 データ読み込み中: {file_path}")
    mat = sio.loadmat(file_path, simplify_cells=True)
    trials = mat['trials']
    X_list, y_list = [], []
    window_size = 256

    for i in range(len(trials)):
        trial = trials[i]
        eeg = trial['RawData']['EegData']
        label = 0 if trial['attended_ear'] == 'L' else 1
        num_windows = len(eeg) // window_size
        if max_windows_per_trial is not None:
            num_windows = min(num_windows, max_windows_per_trial)
        for w in range(num_windows):
            s = w * window_size
            X_list.append(eeg[s:s+window_size, :].T)
            y_list.append(label)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.long)
    print(f"  ✅ ロード完了: {len(X)}サンプル  shape={X.shape}")
    return TensorDataset(X, y)


def main():
    print(f"=== MAML Evaluation v2: SpatialFilter + DataLoader (device={device}) ===\n")

    # ------------------------------------------------------------------
    # ハイパーパラメータ
    # TASK_BATCH_SIZE: support+query 合計ショット数  旧=32 → 新=256
    # ------------------------------------------------------------------
    TASK_BATCH_SIZE = 256
    INNER_LR   = 0.01
    INNER_STEPS = 3
    EPOCHS     = 20
    META_LR    = 0.001

    # データロード (max_windows_per_trial=None で全データ解禁)
    dataset = load_ku_leuven_data(
        "data/KULeuven data set/S1.mat",
        max_windows_per_trial=None
    )

    n_train = int(len(dataset) * 0.8)
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"  📊 Train: {n_train} / Val: {n_val}")

    loader_kw = dict(drop_last=True, num_workers=2, persistent_workers=True, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=TASK_BATCH_SIZE, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=TASK_BATCH_SIZE, shuffle=False, **loader_kw)

    # 【改修①】 空間フィルタ付きモデル
    model = EEG1DCNN(num_channels=64, num_spatial_filters=8, num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  🧠 EEG1DCNN with SpatialFilter | params={n_params:,}\n")

    meta_optimizer = optim.Adam(model.parameters(), lr=META_LR)
    loss_fn = nn.CrossEntropyLoss()
    inference_log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, acc_sum, n_tasks = 0.0, 0.0, 0

        for X_batch, y_batch in train_loader:
            half = TASK_BATCH_SIZE // 2
            x_spt = X_batch[:half].to(device); y_spt = y_batch[:half].to(device)
            x_qry = X_batch[half:].to(device); y_qry = y_batch[half:].to(device)

            meta_optimizer.zero_grad()
            inner_opt = optim.SGD(model.parameters(), lr=INNER_LR)

            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                for _ in range(INNER_STEPS):
                    diffopt.step(loss_fn(fmodel(x_spt), y_spt))

                qry_preds = fmodel(x_qry)
                qry_loss  = loss_fn(qry_preds, y_qry)
                qry_loss.backward()

                _, pred = torch.max(qry_preds, 1)
                acc = (pred == y_qry).sum().item() / len(y_qry) * 100
                inference_log.append({"epoch": epoch, "task": n_tasks, "raw_accuracy": round(acc, 2)})

            meta_optimizer.step()
            loss_sum += qry_loss.item(); acc_sum += acc; n_tasks += 1

        # 検証
        model.eval()
        val_acc_sum, val_n = 0.0, 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                p = model(Xv.to(device))
                _, pv = torch.max(p, 1)
                val_acc_sum += (pv == yv.to(device)).sum().item() / len(yv) * 100
                val_n += 1
        val_acc = val_acc_sum / val_n if val_n else 0.0

        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {loss_sum/n_tasks:.4f} | "
              f"Train: {acc_sum/n_tasks:.1f}% | Val: {val_acc:.1f}%")

    # 推論ログ保存 (C++シミュレーションが読み込む)
    os.makedirs("data", exist_ok=True)
    with open("data/inference_log.json", "w") as f:
        json.dump(inference_log, f, indent=2)
    print(f"\n  💾 inference_log.json 保存完了 ({len(inference_log)} records)")

    # 空間フィルタ重み保存
    np.save("data/spatial_filter_weights.npy", model.get_spatial_weights().numpy())
    print("  💾 spatial_filter_weights.npy 保存完了")
    print("\n🎉 全プロセス完了！")


if __name__ == '__main__':
    main()