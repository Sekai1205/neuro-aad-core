import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import os
from models.eeg_1d_cnn import EEG1DCNN

# M5 MacのApple Siliconパワーを解放
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def load_ku_leuven_data(file_path):
    print(f"本物のEEGデータを {file_path} から読み込んでいます...")
    mat = sio.loadmat(file_path, simplify_cells=True)
    trials = mat['trials']
    
    X_list = []
    y_list = []
    
    # 最初の5つの試行（実験）のデータを抽出する
    for i in range(5):
        trial = trials[i]
        eeg = trial['RawData']['EegData'] # 脳波データ (サンプル数, 64ch)
        ear = trial['attended_ear']       # 'L'(左) または 'R'(右)
        
        # AIが理解できるように、Lなら「0」、Rなら「1」の正解ラベルにする
        label = 0 if ear == 'L' else 1
        
        # 巨大な脳波を「2秒間（128Hz × 2秒 = 256サンプル）」ごとに切り刻む
        window_size = 256
        num_windows = len(eeg) // window_size
        
        for w in range(num_windows):
            start = w * window_size
            end = start + window_size
            
            # AI (PyTorch) は「チャンネル数 × 時間」の形が好きなので、縦横を入れ替える（.T）
            windowed_eeg = eeg[start:end, :].T 
            X_list.append(windowed_eeg)
            y_list.append(label)
            
            # テスト用に、1つの試行から50個（100秒分）だけ取り出したらストップ
            if w >= 50:
                break
                
    # データを PyTorch のテンソル（AI専用の形式）に変換
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.long)
    return X_tensor, y_tensor

def main():
    print(f"=== Starting REAL DATA Training on {device} ===")
    
    # データの場所（※さっきエラーを直したパスと同じです）
    file_path = "data/KULeuven data set/S1.mat"
    if not os.path.exists(file_path):
        print(f"❌ エラー: {file_path} が見つかりません。")
        return
        
    # 1. データの準備
    X_data, y_data = load_ku_leuven_data(file_path)
    print(f"✅ データ準備完了！ 切り出した脳波ブロックの総数: {len(X_data)} 個")

    # 2. AIモデルとオプティマイザの準備
    model = EEG1DCNN(num_channels=64, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 32
    epochs = 15
    dataset_size = len(X_data)

    print("\n🚀 本物の脳波データでの学習ループを開始します...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        # バッチサイズ（32個のブロック）ごとにまとめてAIに学習させる
        for i in range(0, dataset_size, batch_size):
            batch_X = X_data[i:i+batch_size].to(device)
            batch_y = y_data[i:i+batch_size].to(device)

            optimizer.zero_grad()             # 記憶をリセット
            predictions = model(batch_X)      # 予測する
            loss = loss_fn(predictions, batch_y) # 正解との誤差を計算
            loss.backward()                   # 修正ポイントを計算
            optimizer.step()                  # 脳のネットワークを更新
            
            epoch_loss += loss.item()
            
        # 1エポック（全データを1周）ごとの平均誤差を表示
        avg_loss = epoch_loss / (dataset_size / batch_size)
        print(f"Epoch {epoch:2d}/{epochs} | Average Loss (誤差): {avg_loss:.4f}")

    print("\n🎉 本物の脳波を用いた学習テストが正常に完了しました！")

if __name__ == '__main__':
    main()