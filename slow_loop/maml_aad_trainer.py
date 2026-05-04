import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import os
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
    print(f"=== Starting Training with Validation on {device} ===")
    
    file_path = "data/KULeuven data set/S1.mat"
    X_data, y_data = load_ku_leuven_data(file_path)
    
    dataset_size = len(X_data)
    print(f"✅ 全データ数: {dataset_size} 個")

    # ---------------------------------------------------------
    # 🆕 データを「学習用(80%)」と「テスト用(20%)」にシャッフルして分割
    # ---------------------------------------------------------
    train_size = int(0.8 * dataset_size)
    indices = torch.randperm(dataset_size) # データをランダムにシャッフル
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, y_train = X_data[train_indices], y_data[train_indices]
    X_val, y_val = X_data[val_indices], y_data[val_indices]
    
    print(f"📚 学習用(過去問): {len(X_train)} 個")
    print(f"📝 テスト用(初見): {len(X_val)} 個\n")

    model = EEG1DCNN(num_channels=64, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 32
    epochs = 15

    for epoch in range(1, epochs + 1):
        # --- 📚 学習フェーズ（過去問を解いて賢くなる） ---
        model.train() # 学習モードON
        train_loss = 0.0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / (len(X_train) / batch_size)

        # --- 📝 テストフェーズ（初見のデータで実力を測る） ---
        model.eval() # 学習モードOFF（ズル禁止）
        correct = 0
        total = len(X_val)
        
        # テスト中は脳のネットワークを更新しない
        with torch.no_grad():
            val_X = X_val.to(device)
            val_y = y_val.to(device)
            val_preds = model(val_X)
            
            # AIが「左(0)」「右(1)」どちらの確率が高いと判断したかを取得
            _, predicted_labels = torch.max(val_preds, 1)
            correct = (predicted_labels == val_y).sum().item()
            
        val_accuracy = (correct / total) * 100 # 正解率(%)

        # 結果の表示
        print(f"Epoch {epoch:2d}/{epochs} | 学習Loss: {avg_train_loss:.4f} | テスト正解率: {val_accuracy:.1f}%")

    print("\n🎉 検証ループ完了！")

if __name__ == '__main__':
    main()