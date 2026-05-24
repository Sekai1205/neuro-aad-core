import os
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import higher
import matplotlib.pyplot as plt

from models.ear_eeg_cnn import EarEEGCNN

# デバイス設定
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 🚨 【実験A】チャンネル削減の検証条件
CHANNEL_CONFIGS = {
    "64ch\n(Full)": list(range(64)),
    "16ch\n(Moderate)": [7, 8, 9, 10, 12, 13, 22, 26, 45, 46, 54, 58, 20, 21, 52, 53], 
    "8ch\n(Wearable)": [22, 26, 45, 46, 54, 58, 12, 13], 
    "4ch\n(Ear-EEG)": [22, 26, 54, 58] 
}

def load_single_subject(file_path, channels, window_size=256):
    mat = sio.loadmat(file_path, simplify_cells=True)
    trials = mat['trials']
    X_list, y_list = [], []

    for trial in trials:
        eeg = trial['RawData']['EegData']
        label = 0 if trial['attended_ear'] == 'L' else 1
        
        for w in range(len(eeg) // window_size):
            start = w * window_size
            window_data = eeg[start:start + window_size, :].T 
            
            selected_data = window_data[channels, :]
            
            # Z-score正規化
            mean = np.mean(selected_data, axis=1, keepdims=True)
            std = np.std(selected_data, axis=1, keepdims=True) + 1e-6
            normalized_window = (selected_data - mean) / std
            
            X_list.append(normalized_window)
            y_list.append(label)

    return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.long)

def prepare_loso_dataset(data_dir, channels, test_subject='S16'):
    train_tasks = {}
    test_task = None

    for i in range(1, 17):
        subject_id = f"S{i}"
        file_path = os.path.join(data_dir, f"{subject_id}.mat")
        if not os.path.exists(file_path): continue

        X, y = load_single_subject(file_path, channels)
        if subject_id == test_subject:
            test_task = (X, y)
        else:
            train_tasks[subject_id] = (X, y)

    return train_tasks, test_task

def run_maml_evaluation(data_dir, config_name, channels, test_subject='S16'):
    config_name_flat = config_name.replace('\n', ' ')
    print(f"\n========================================")
    print(f"🚀 条件開始: {config_name_flat} (電極数: {len(channels)})")
    print(f"========================================")
    
    train_tasks, test_task = prepare_loso_dataset(data_dir, channels, test_subject)
    train_subject_ids = list(train_tasks.keys())

    virtual_ch = max(8, len(channels)) if len(channels) < 64 else 64
    model = EarEEGCNN(num_channels=len(channels), virtual_channels=virtual_ch, num_classes=2).to(device)
    
    meta_optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    inner_lr, inner_steps, meta_epochs, task_batch_size, shot = 0.01, 3, 10, 4, 32 

    # --- メタ・トレーニング（Outer Loop） ---
    for epoch in range(1, meta_epochs + 1):
        meta_loss, meta_acc = 0.0, 0.0
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
                meta_loss += loss_fn(qry_preds, y_qry)
                meta_acc += ((qry_preds.argmax(1) == y_qry).float().mean().item()) * 100
        
        meta_loss.backward()
        meta_optimizer.step()
        print(f"  Epoch {epoch:2d}/{meta_epochs} | Meta-Train Acc: {meta_acc/task_batch_size:.1f}%")

    # --- 未知被験者テスト（バグ修正済） ---
    X_test, y_test = test_task
    
    # 🚨【致命的なバグの修正】テストデータをシャッフルし、ラベルの偏りを防ぐ
    test_indices = torch.randperm(len(X_test))
    X_test, y_test = X_test[test_indices], y_test[test_indices]

    test_shot = 64 
    x_spt_t, y_spt_t = X_test[:test_shot].to(device), y_test[:test_shot].to(device)
    
    eval_size = min(512, len(X_test) - test_shot)
    x_qry_t, y_qry_t = X_test[test_shot:test_shot+eval_size].to(device), y_test[test_shot:test_shot+eval_size].to(device)

    test_opt = optim.SGD(model.parameters(), lr=inner_lr)
    for _ in range(inner_steps):
        test_opt.zero_grad()
        loss_fn(model(x_spt_t), y_spt_t).backward()
        test_opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(x_qry_t).argmax(1)
        final_acc = (preds == y_qry_t).float().mean().item() * 100
        
        # デバッグ: クラスの偏りがないか確認（50%前後なら正常）
        zero_ratio = (y_qry_t == 0).float().mean().item() * 100
        print(f"  [Debug] テスト評価データのラベル 'L(0)' の割合: {zero_ratio:.1f}%")
        
    print(f"🎯 【結果】{config_name_flat} -> テスト正解率: {final_acc:.1f}%")
    return final_acc

def main():
    print(f"=== Experiment A: Systematic Channel Reduction Evaluation ===")
    data_dir = "data/KULeuven data set"
    
    results = {}
    for config_name, channels in CHANNEL_CONFIGS.items():
        acc = run_maml_evaluation(data_dir, config_name, channels)
        results[config_name] = acc

    # 📊 棒グラフの描画と保存（最適化済）
    labels = list(results.keys())
    accuracies = list(results.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 減っていく様子を色で表現

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 50%（ランダム・チャンスレベル）の赤線を引く
    plt.axhline(50.0, color='red', linestyle='--', linewidth=2, label='Chance Level (50%)')
    
    # バーの上に数値を表示
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title("Effect of EEG Channel Reduction on MAML Adaptation Accuracy", fontsize=15, fontweight='bold')
    plt.ylabel("Test Accuracy on Unseen Subject (%)", fontsize=12)
    plt.ylim(40, max(accuracies) + 10)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    save_path = "experiment_a_results_bar.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n🎉 実験A完了！結果の棒グラフを '{save_path}' に保存しました。")

if __name__ == '__main__':
    main()