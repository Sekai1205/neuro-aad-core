import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import socket
import time
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
        for w in range(len(eeg) // window_size):
            start = w * window_size
            X_list.append(eeg[start:start + window_size, :].T)
            y_list.append(label)
            if w >= 50: break
    return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.long)

def main():
    print(f"=== TRUE SLOW LOOP: MAML Brain -> C++ Ear ===")
    
    # 1. データの準備
    X_data, y_data = load_ku_leuven_data("data/KULeuven data set/S1.mat")
    
    # 2. UDPソケット（トンネルの入り口）の準備
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 5005)

    model = EEG1DCNN(num_channels=64, num_classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    task_batch_size = 32
    half = task_batch_size // 2
    
    # 最初の半分のデータでキャリブレーション、残りの半分でリアルタイムテスト
    x_spt, y_spt = X_data[0:half].to(device), y_data[0:half].to(device)
    x_qry = X_data[half:task_batch_size].to(device)

    print("⏳ 最初の数ブロックを使って、AIのキャリブレーション（適応）を実行中...")
    
    with higher.innerloop_ctx(model, optim.SGD(model.parameters(), lr=0.01), copy_initial_weights=False) as (fmodel, diffopt):
        # MAMLの魔法：3回のステップで瞬時に適応
        for _ in range(3): 
            spt_preds = fmodel(x_spt)
            diffopt.step(loss_fn(spt_preds, y_spt))
        
        print("✅ 適応完了！リアルタイム推論とC++への送信を開始します。\n")
        
        # --- リアルタイム推論ストリーミングのシミュレーション ---
        for i in range(len(x_qry)):
            single_eeg = x_qry[i:i+1] # 2秒分の脳波ブロックを1つだけ取り出す
            
            # AIが推論する
            prediction = fmodel(single_eeg)
            _, predicted_label = torch.max(prediction, 1)
            
            # 0ならL（左）、1ならR（右）
            command = "L" if predicted_label.item() == 0 else "R"
            
            print(f"🧠 [推論 {i+1}/{len(x_qry)}] AIの解析結果: {command} -> C++へ送信！")
            sock.sendto(command.encode(), server_address)
            
            # 実際のシステムに合わせて、2秒間隔で処理を回す
            time.sleep(2)

    sock.close()
    print("\n🎉 全推論タスク完了！システムを正常終了します。")

if __name__ == '__main__':
    main()