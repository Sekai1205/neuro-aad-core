import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    print("=== KULeuven EEG Data Plotter ===")
    
    # さっき成功したファイルパス
    file_path = "data/KULeuven data set/S1.mat"
    
    if not os.path.exists(file_path):
        print(f"❌ エラー: {file_path} が見つかりません。")
        return

    print("S1.mat を読み込んでいます...")
    # simplify_cells=True をつけると、MATLAB特有の複雑な入れ子を綺麗なPython辞書にしてくれます
    mat = sio.loadmat(file_path, simplify_cells=True)
    
    try:
        # 1. 最初の実験データ（Trial 1）を取得
        trial_1 = mat['trials'][0] 
        
        # 2. EEGデータを取り出す (サンプル数 × チャンネル数)
        eeg_data = trial_1['RawData']['EegData']
        
        # 3. サンプリング周波数 (1秒間に何回データを取ったか。今回は128Hz)
        fs = int(trial_1['FileHeader']['SampleRate'])
        
        print(f"✅ データ抽出成功！")
        print(f"EEGデータの全体サイズ: {eeg_data.shape} (サンプル数 × チャンネル数)")

        # ------------------------------------------------
        # グラフの描画設定
        # ------------------------------------------------
        print("\nグラフを作成しています...")
        
        # 全部描画すると真っ黒になるので、最初の「3秒間」だけを切り取る
        seconds_to_plot = 3
        samples_to_plot = fs * seconds_to_plot
        
        # 時間の軸（X軸）を作る [0.0秒, 0.0078秒, 0.0156秒, ..., 3.0秒]
        time_axis = np.linspace(0, seconds_to_plot, samples_to_plot)
        
        # 1番目のチャンネル（Ch 1）の波形データを取り出す
        # ※ eeg_data は [時間, チャンネル] の順番で並んでいます
        ch1_data = eeg_data[:samples_to_plot, 0]

        # グラフを描画する
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, ch1_data, color='blue', linewidth=1.2)
        
        plt.title("Subject 1 - EEG Waveform (Channel 1, First 3 seconds)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        print("🚀 グラフを表示します！画面にウィンドウが出るのを確認してください。")
        plt.show()

    except Exception as e:
        print(f"❌ データの抽出中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()