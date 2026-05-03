import torch
import torch.nn as nn
import torch.optim as optim
from models.eeg_1d_cnn import EEG1DCNN # さっき作ったモデルを呼び出す

# デバイス設定 (M5 Macのパワーを解放)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def main():
    print(f"--- Starting AI Training Test on {device} ---")
    
    # 1. モデルと最適化ツール（AIの先生役）の準備
    model = EEG1DCNN(num_channels=64, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("モデルの初期化完了。ダミー脳波データで学習テストを開始します...\n")

    # 2. 学習ループ（100回まわしてみる）
    for epoch in range(1, 101):
        # [ダミーデータの生成] 
        # 64チャンネル、256サンプル(約1秒分)のランダムな波形を4つ作る
        dummy_eeg = torch.randn(4, 64, 256).to(device) 
        # 答え(0:左の声に集中, 1:右の声に集中)をランダムに設定
        dummy_labels = torch.randint(0, 2, (4,)).to(device)

        # --- AIの学習ステップ ---
        optimizer.zero_grad()             # 記憶をリセット
        predictions = model(dummy_eeg)    # AIに予測させる
        loss = loss_fn(predictions, dummy_labels) # 答え合わせ（誤差を計算）
        loss.backward()                   # どこを直せばいいか計算
        optimizer.step()                  # 脳のネットワークを更新

        # 10回ごとに進捗（Loss: 誤差）を表示
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/100 | Loss (誤差): {loss.item():.4f}")

    print("\n🎉 ダミーデータでの学習テストが正常に完了しました！")
    print("M5 Macの MPS (Apple Silicon) コアは完璧に動作しています。")

if __name__ == '__main__':
    main()