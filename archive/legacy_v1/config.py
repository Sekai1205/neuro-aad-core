# config.py - プロジェクトルート設定ファイル
from pathlib import Path

# === パス ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "KULeuven data set"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# === KULeuvenデータセット仕様（実測確認済み） ===
FS = 128                    # サンプリングレート（Hz）
N_CHANNELS_FULL = 64        # フルチャンネル数
N_SUBJECTS = 16             # 被験者数
N_TRIALS_PER_SUBJECT = 20   # 1被験者あたりのトライアル数

# === ウィンドウ設定 ===
# 注：KULeuven先行研究（Geirnaert 2021）は10秒ウィンドウを使用
# Go/No-Go基準（75% baseline → 80% Go条件）は10秒ウィンドウで定義されている
# 本実装でも10秒ウィンドウに統一する
WINDOW_SEC = 10.0
WINDOW_SAMPLES = int(FS * WINDOW_SEC)  # = 1280サンプル
STRIDE_SAMPLES = int(FS * 1.0)         # = 128サンプル（1秒スライド）

# === 実験A：チャンネル削減ステップ ===
CHANNEL_STEPS = [64, 16, 8, 4]

# === Go/No-Go基準（Geirnaert 2021との比較） ===
BASELINE_ACCURACY = 0.75   # Geirnaert 2021の10秒ウィンドウ精度
GO_THRESHOLD = 0.80        # 本研究のGo条件

# === FOMAML ===
INNER_STEPS = 3
INNER_LR = 0.01
SUPPORT_SET_SIZE = 8       # 4-shot × 2クラス

# === モデル ===
N_CLASSES = 2

# === トレーニング ===
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_SEED = 42
