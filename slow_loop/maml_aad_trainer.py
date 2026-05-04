import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import os
import higher  # 🔮 これがMAMLの魔法のライブラリ
from models.eeg_1d_cnn import EEG1DCNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def load_ku_leuven_data(file_path):
    mat = sio.loadmat(file_path, simplify_cells=True)
    trials = mat['trials']
    
    X_list, y_list = [], []
    for i in range(5):
        trial = trials[i]
        eeg = trial['RawData']['EegData']
        ear = trial['attended_ear']
        label = 0 if ear == 'L' else 1
        
        window_size = 256
        num_windows = len(eeg) // window_size
        
        for w in range(num_windows):
            start = w * window_size
            end = start + window_size
            windowed_eeg = eeg[start:end, :].T 
            X_list.append(windowed_eeg)
            y_list.append(label)
            
            if w >= 50: break
                
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.long)
    return X_tensor, y_tensor

def main():
    print(f"=== Starting MAML (Meta-Learning) on {device} ===")
    
    file_path = "data/KULeuven data set/S1.mat"
    X_data, y_data = load_ku_leuven_data(file_path)
    
    # データをランダムにシャッフル
    dataset_size = len(X_data)
    indices = torch.randperm(dataset_size)
    X_data = X_data[indices]
    y_data = y_data[indices]

    # モデルとメタ最適化ツール（「適応力」を鍛える大元の先生）
    model = EEG1DCNN(num_channels=64, num_classes=2).to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # --- MAMLの心臓部（ハイパーパラメータ） ---
    inner_lr = 0.01  # 個人適応（キャリブレーション）時の学習率
    inner_steps = 3  # たった3回のステップで適応させる（3分間の初期設定を模擬）
    epochs = 15
    task_batch_size = 32 # 1回の「タスク」に使うデータの数

    print("🚀 MAMLループを起動します...「適応のしかた」を学習中...")

    for epoch in range(1, epochs + 1):
        meta_loss = 0.0
        
        for i in range(0, dataset_size - task_batch_size, task_batch_size):
            # MAMLでは、データを「Support(練習用)」と「Query(本番テスト用)」に分けます
            half = task_batch_size // 2
            x_spt, y_spt = X_data[i : i+half].to(device), y_data[i : i+half].to(device)
            x_qry, y_qry = X_data[i+half : i+task_batch_size].to(device), y_data[i+half : i+task_batch_size].to(device)

            meta_optimizer.zero_grad()

            # --- 🔮 higherライブラリによるMAMLの魔法 ---
            # innerloop_ctx が、AIの脳の「一時的なクローン(fmodel)」を作ってシミュレーションさせます
            with higher.innerloop_ctx(model, optim.SGD(model.parameters(), lr=inner_lr), copy_initial_weights=False) as (fmodel, diffopt):
                
                # 【内側のループ（Inner Loop）】
                # Supportデータ（数分間のキャリブレーション）を使って、クローンを「今の被験者」に素早く適応させる
                for step in range(inner_steps):
                    spt_preds = fmodel(x_spt)
                    spt_loss = loss_fn(spt_preds, y_spt)
                    diffopt.step(spt_loss)
                
                # 【外側のループ（Outer Loop）】
                # 適応したクローンが、未知のQueryデータ（本番）でどれくらい通用するかをテストする
                qry_preds = fmodel(x_qry)
                qry_loss = loss_fn(qry_preds, y_qry)
                
                # そのテストの誤差から「どうすればもっと上手く適応できるベースモデルになれるか」を計算！
                qry_loss.backward()
            
            # 大元のメタモデルを更新（適応能力がアップする）
            meta_optimizer.step()
            meta_loss += qry_loss.item()
            
        print(f"Epoch {epoch:2d}/{epochs} | Meta-Loss (適応能力の誤差): {meta_loss:.4f}")

    print("\n🎉 MAMLのコアシステムが完成しました！")

if __name__ == '__main__':
    main()