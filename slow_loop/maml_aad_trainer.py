import os
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import higher

# 🚨 [新兵器] 新しく作ったEar-EEG専用モデルをインポート！
from models.ear_eeg_cnn import EarEEGCNN

# MacのGPU（MPS）が使える場合は使用、なければCPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 🚨 【Ear-EEG シミュレーション】
# 64チャンネルの中から、耳の周り（側頭部・頭頂側頭部）付近の4チャンネルだけを抽出する
# （※国際10-20法に基づく想定インデックス：左耳付近と右耳付近）
EAR_CHANNELS = [22, 26, 54, 58] 

def load_single_subject(file_path, window_size=256):
    """1人の被験者のデータを読み込み、4chだけ抽出して正規化する"""
    mat = sio.loadmat(file_path, simplify_cells=True)
    trials = mat['trials']
    X_list, y_list = [], []

    for trial in trials:
        eeg = trial['RawData']['EegData']
        label = 0 if trial['attended_ear'] == 'L' else 1
        
        for w in range(len(eeg) // window_size):
            start = w * window_size
            window_data = eeg[start:start + window_size, :].T # shape: (64, 256)
            
            # 🚨 ここで60チャンネルを捨て、耳周りの4チャンネルだけを抽出！
            ear_window_data = window_data[EAR_CHANNELS, :] # shape: (4, 256)
            
            # Z-score正規化
            mean = np.mean(ear_window_data, axis=1, keepdims=True)
            std = np.std(ear_window_data, axis=1, keepdims=True) + 1e-6
            normalized_window = (ear_window_data - mean) / std
            
            X_list.append(normalized_window)
            y_list.append(label)

    return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.long)


def prepare_loso_dataset(data_dir, test_subject='S16'):
    train_tasks = {}
    test_task = None
    print(f"📂 データをロード中... (メタ・テスト被験者として {test_subject} を隔離します)")

    for i in range(1, 17):
        subject_id = f"S{i}"
        file_path = os.path.join(data_dir, f"{subject_id}.mat")

        if not os.path.exists(file_path):
            continue

        X, y = load_single_subject(file_path)

        if subject_id == test_subject:
            test_task = (X, y)
            print(f"  🎯 {subject_id} をテスト用に完全隔離: {X.shape} 👈 4チャンネル化成功！")
        else:
            train_tasks[subject_id] = (X, y)
            print(f"  🧠 {subject_id} を学習用にタスク登録: {X.shape}")

    return train_tasks, test_task


def main():
    print(f"=== Starting Ear-EEG MAML (LOSO) Simulation on {device} ===")
    
    data_dir = "data/KULeuven data set"
    train_tasks, test_task = prepare_loso_dataset(data_dir, test_subject='S16')
    train_subject_ids = list(train_tasks.keys())

    # 🚨 モデルを「EarEEGCNN」に変更（入力4ch、仮想空間フィルター8ch）
    model = EarEEGCNN(num_channels=4, virtual_channels=8, num_classes=2).to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    inner_lr = 0.01
    inner_steps = 3
    meta_epochs = 20
    task_batch_size = 4 
    shot = 32           

    print("\n🚀 メタ・トレーニング（Outer Loop）開始...")

    for epoch in range(1, meta_epochs + 1):
        meta_loss = 0.0
        meta_acc = 0.0
        
        sampled_subjects = np.random.choice(train_subject_ids, task_batch_size, replace=False)
        meta_optimizer.zero_grad()

        for subject in sampled_subjects:
            X_data, y_data = train_tasks[subject]
            
            indices = torch.randperm(len(X_data))
            X_data, y_data = X_data[indices], y_data[indices]
            
            x_spt, y_spt = X_data[:shot].to(device), y_data[:shot].to(device)
            x_qry, y_qry = X_data[shot:shot*2].to(device), y_data[shot:shot*2].to(device)

            with higher.innerloop_ctx(model, optim.SGD(model.parameters(), lr=inner_lr), copy_initial_weights=False) as (fmodel, diffopt):
                for _ in range(inner_steps):
                    spt_preds = fmodel(x_spt)
                    diffopt.step(loss_fn(spt_preds, y_spt))
                
                qry_preds = fmodel(x_qry)
                task_loss = loss_fn(qry_preds, y_qry)
                
                meta_loss += task_loss
                
                _, predicted = torch.max(qry_preds, 1)
                correct = (predicted == y_qry).sum().item()
                meta_acc += (correct / len(y_qry)) * 100
        
        meta_loss.backward()
        meta_optimizer.step()
        
        avg_loss = meta_loss.item() / task_batch_size
        avg_acc = meta_acc / task_batch_size
        print(f"Epoch {epoch:2d}/{meta_epochs} | Meta-Loss: {avg_loss:.4f} | Meta-Train 適応後正解率: {avg_acc:.1f}%")

    # ==========================================
    # 🎯 最終決戦：未知の被験者 (S16) でテスト
    # ==========================================
    print("\n🏁 メタ・トレーニング完了！隔離していた未知の被験者 (S16) でテストします！")
    
    X_test, y_test = test_task
    
    indices_test = torch.randperm(len(X_test))
    X_test, y_test = X_test[indices_test], y_test[indices_test]

    test_shot = 64 
    x_spt_test, y_spt_test = X_test[:test_shot].to(device), y_test[:test_shot].to(device)
    x_qry_test, y_qry_test = X_test[test_shot:test_shot*2].to(device), y_test[test_shot:test_shot*2].to(device)

    test_optimizer = optim.SGD(model.parameters(), lr=inner_lr)
    for _ in range(inner_steps):
        test_optimizer.zero_grad()
        spt_preds = model(x_spt_test)
        loss = loss_fn(spt_preds, y_spt_test)
        loss.backward()
        test_optimizer.step()

    model.eval()
    with torch.no_grad():
        final_preds = model(x_qry_test)
        _, predicted = torch.max(final_preds, 1)
        correct = (predicted == y_qry_test).sum().item()
        final_acc = (correct / len(y_qry_test)) * 100

    print(f"🏆 【Ear-EEG シミュレーション】4チャンネルでの未知被験者(S16) 適応後テスト正解率: {final_acc:.1f}%")

if __name__ == '__main__':
    main()