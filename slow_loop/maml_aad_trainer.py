import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import os
import higher
from models.eeg_1d_cnn import EEG1DCNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def load_ku_leuven_data(file_path):
    mat = sio.loadmat(file_path, simplify_cells=True)
    trials = mat['trials']
    
    X_list, y_list = [], []
    for i in range(5):
        trial = trials[i]
        eeg = trial['RawData']['EegData']
        label = 0 if trial['attended_ear'] == 'L' else 1
        
        window_size = 256
        num_windows = len(eeg) // window_size
        
        for w in range(num_windows):
            start = w * window_size
            end = start + window_size
            X_list.append(eeg[start:end, :].T)
            y_list.append(label)
            if w >= 50: break
                
    return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.long)

def main():
    print(f"=== Starting MAML Evaluation on {device} ===")
    X_data, y_data = load_ku_leuven_data("data/KULeuven data set/S1.mat")
    
    dataset_size = len(X_data)
    indices = torch.randperm(dataset_size)
    X_data, y_data = X_data[indices], y_data[indices]

    model = EEG1DCNN(num_channels=64, num_classes=2).to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    inner_lr, inner_steps, epochs, task_batch_size = 0.01, 3, 15, 32

    for epoch in range(1, epochs + 1):
        meta_loss = 0.0
        meta_acc = 0.0  # 🎯 新規追加：正解率を記録
        num_tasks = 0
        
        for i in range(0, dataset_size - task_batch_size, task_batch_size):
            half = task_batch_size // 2
            x_spt, y_spt = X_data[i : i+half].to(device), y_data[i : i+half].to(device)
            x_qry, y_qry = X_data[i+half : i+task_batch_size].to(device), y_data[i+half : i+task_batch_size].to(device)

            meta_optimizer.zero_grad()

            with higher.innerloop_ctx(model, optim.SGD(model.parameters(), lr=inner_lr), copy_initial_weights=False) as (fmodel, diffopt):
                # 3回のステップで「適応」させる
                for _ in range(inner_steps):
                    spt_preds = fmodel(x_spt)
                    diffopt.step(loss_fn(spt_preds, y_spt))
                
                # 適応済みの脳で、初見のテスト（Query）に挑む
                qry_preds = fmodel(x_qry)
                qry_loss = loss_fn(qry_preds, y_qry)
                qry_loss.backward()
                
                # 🎯 新規追加：AIの予測が合っていたか答え合わせ
                _, predicted = torch.max(qry_preds, 1)
                correct = (predicted == y_qry).sum().item()
                meta_acc += (correct / len(y_qry)) * 100
            
            meta_optimizer.step()
            meta_loss += qry_loss.item()
            num_tasks += 1
            
        avg_loss = meta_loss / num_tasks
        avg_acc = meta_acc / num_tasks
        print(f"Epoch {epoch:2d}/{epochs} | Meta-Loss: {avg_loss:.4f} | 適応後テスト正解率: {avg_acc:.1f}%")

    print("\n🎉 MAMLの評価ループ完了！")

if __name__ == '__main__':
    main()