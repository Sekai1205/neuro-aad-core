import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=== Python(AI) × C++(Audio) 統合シミュレーション ===")
    
    total_steps = 100   # 約100回のAI推論（例：200msごとに推論なら約20秒分）
    accuracy = 0.641    # 先ほど叩き出した「64.1%」の汎化性能
    
    # 状況設定：ユーザーはずっと「左耳（L: 0）」に注意を向けているとする
    true_label = 0
    
    # 1. AI推論のシミュレーション（Python側）
    np.random.seed(42) # 再現性のため
    # 64.1%の確率で正解(0)、残りの確率で誤判定(1)を出力するAI
    ai_predictions = np.where(np.random.rand(total_steps) < accuracy, true_label, 1)
    
    # 2. オーディオ制御のシミュレーション（C++側のEMAロジック）
    alpha = 0.1  # ipc_receiver.cpp に実装されているスムージング係数
    
    current_gain_L = 0.5  # 初期音量 L
    current_gain_R = 0.5  # 初期音量 R
    
    history_gain_L = []
    history_gain_R = []
    
    for pred in ai_predictions:
        # AIの判定から「目標の音量（ターゲットゲイン）」を決定
        target_gain_L = 1.0 if pred == 0 else 0.0
        target_gain_R = 0.0 if pred == 0 else 1.0
        
        # 🚨 ここが C++ 側の最強ロジック（指数移動平均: EMA）
        # 目標に向かって「α(10%)」だけ近づく。残りの90%は前の状態を維持する。
        current_gain_L = current_gain_L * (1.0 - alpha) + target_gain_L * alpha
        current_gain_R = current_gain_R * (1.0 - alpha) + target_gain_R * alpha
        
        history_gain_L.append(current_gain_L)
        history_gain_R.append(current_gain_R)

    # ==========================================
    # 3. グラフ描画（面談のキラー資料）
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    # AIの「生」の判定（64%なのでバタバタ暴れる）
    plt.subplot(2, 1, 1)
    plt.title(f"Python AI Predictions (Accuracy: {accuracy*100:.1f}%)", fontsize=14)
    plt.step(range(total_steps), [1 if p==0 else 0 for p in ai_predictions], where='post', color='red', alpha=0.6, label='AI Raw Output (1=L, 0=R)')
    plt.yticks([0, 1], ['R (Wrong)', 'L (Correct)'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    
    # C++のエッジ処理を通した「最終的な音量」（安定する！）
    plt.subplot(2, 1, 2)
    plt.title("C++ Audio Output with EMA Smoothing (alpha=0.1)", fontsize=14)
    plt.plot(history_gain_L, label='L Ear Volume (Target)', color='blue', linewidth=2.5)
    plt.plot(history_gain_R, label='R Ear Volume (Noise)', color='orange', linewidth=2.5, linestyle='--')
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.8)
    plt.ylabel('Audio Gain')
    plt.xlabel('Time Steps')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center right')
    
    plt.tight_layout()
    
    # 画像として保存！
    save_path = "system_simulation_result.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n🎉 シミュレーション完了！結果を画像として保存しました: {save_path}")
    
if __name__ == '__main__':
    main()