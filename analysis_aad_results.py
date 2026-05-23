"""
mTRF-AAD実装レビュー：結果分析と改善提案
2026-05-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 結果ファイル読み込み
csv_path = Path("/Users/sekai/.gemini/antigravity/scratch/neuro-aad-core/results/aad_mtrf_envelope/aad_mtrf_s1.csv")
df = pd.read_csv(csv_path)

print("=" * 70)
print("AAD mTRF実装レビュー：S1結果分析")
print("=" * 70)

# 1. 基本統計
print("\n【基本統計】")
print(f"総試行: {len(df)}")
print(f"Track1試行: {(df['ground_truth']==1).sum()}")
print(f"Track2試行: {(df['ground_truth']==2).sum()}")
print(f"正解数: {(df['prediction']==df['ground_truth']).sum()}")
print(f"精度: {(df['prediction']==df['ground_truth']).sum()}/{len(df)} = {(df['prediction']==df['ground_truth']).mean():.2%}")

# 2. トラック別の感度・特異度
track1_mask = df['ground_truth'] == 1
track2_mask = df['ground_truth'] == 2

track1_correct = ((df['prediction'] == 1) & track1_mask).sum()
track2_correct = ((df['prediction'] == 2) & track2_mask).sum()

track1_sensitivity = track1_correct / track1_mask.sum()
track2_specificity = track2_correct / track2_mask.sum()

print(f"\n【トラック別性能】")
print(f"Track1感度 (recall): {track1_correct}/{track1_mask.sum()} = {track1_sensitivity:.2%}")
print(f"Track2特異度 (recall): {track2_correct}/{track2_mask.sum()} = {track2_specificity:.2%}")

# 3. 相関値の分析
print(f"\n【相関値（r）の分布】")
print(f"r_track1: mean={df['r_track1'].mean():.4f}, std={df['r_track1'].std():.4f}")
print(f"         min={df['r_track1'].min():.4f}, max={df['r_track1'].max():.4f}")
print(f"r_track2: mean={df['r_track2'].mean():.4f}, std={df['r_track2'].std():.4f}")
print(f"         min={df['r_track2'].min():.4f}, max={df['r_track2'].max():.4f}")
print(f"delta_r:  mean={df['delta_r'].mean():.4f}, std={df['delta_r'].std():.4f}")

# 4. 誤分類パターン分析
print(f"\n【誤分類パターン】")
errors = df[df['prediction'] != df['ground_truth']].copy()
print(f"誤分類数: {len(errors)}/{len(df)} = {len(errors)/len(df):.2%}")

if len(errors) > 0:
    print("\nTrack1→Track2誤分類（Track1だがTrack2と予測）:")
    t1_to_t2 = errors[(errors['ground_truth']==1) & (errors['prediction']==2)]
    if len(t1_to_t2) > 0:
        print(f"  数: {len(t1_to_t2)}")
        print(f"  delta_r: mean={t1_to_t2['delta_r'].mean():.4f} (負の値が多いはず)")
    
    print("\nTrack2→Track1誤分類（Track2だがTrack1と予測）:")
    t2_to_t1 = errors[(errors['ground_truth']==2) & (errors['prediction']==1)]
    if len(t2_to_t1) > 0:
        print(f"  数: {len(t2_to_t1)}")
        print(f"  delta_r: mean={t2_to_t1['delta_r'].mean():.4f} (正の値が多いはず)")

# 5. 正解・誤分類の相関値比較
print(f"\n【正解と誤分類の相関値比較】")
correct = df[df['prediction'] == df['ground_truth']]
incorrect = df[df['prediction'] != df['ground_truth']]

print(f"正解時のdelta_r: mean={correct['delta_r'].mean():.4f}, std={correct['delta_r'].std():.4f}")
print(f"誤分時のdelta_r: mean={incorrect['delta_r'].mean():.4f}, std={incorrect['delta_r'].std():.4f}")

# 6. 問題点と原因分析
print("\n" + "=" * 70)
print("【問題点と原因分析】")
print("=" * 70)

print(f"""
1. 精度55%（チャンスレベル50%から+5%）
   - 予測精度が大きく改善されていない
   - クラス不均衡（Track1:16, Track2:4）の影響
   
2. Track1感度{track1_sensitivity:.0%}と低い
   - Track1試行でTrack2と誤分類されている（{len(t1_to_t2)}回）
   - 相関値が負になる場合もある
   
3. 相関値が全体的に低い（最大0.134）
   - 文献報告値（0.15-0.40）より大幅に低い
   - エンベロープ抽出の品質に問題の可能性
   
4. ランダム分類に近い性能
   - 正解・誤分のdelta_rの平均差が小さい
   - モデルがTrack1/Track2を区別できていない

【推定される原因】
a) エンベロープ品質が低い
   - Gammatone filterbank実装の問題
   - 周波数範囲やフィルタ次数の不最適
   
b) クラス不均衡への対応が不十分
   - LOOCV時に訓練セットが極端に不均衡
   - Track2学習に過適合の可能性
   
c) 信号整列・同期の問題
   - EEGとエンベロープの時間同期
   - ダウンサンプリング時の位相ずれ
""")

# 7. 改善提案
print("【改善提案】")
print("""
1. エンベロープ抽出パイプラインの検証
   □ Gammatone filterのパラメータ見直し（Q値、フィルタ次数）
   □ Power lawの指数確認
   □ ローパスフィルタ（10Hz）の効果検証
   □ 他の周波数帯域でのテスト
   
2. クラス不均衡への対処
   □ LOOCV時にbalanced learningを検討
   □ 訓練時の重み調整（Track2に低い重み）
   □ Thresholdの調整
   
3. mTRF実装の検証
   □ Ridge regression regularization (lambda=100)の最適化
   □ 時間ラグ範囲（-50～400ms）の検証
   □ 標準化・正規化の確認
   
4. 信号処理の検証
   □ EEGのデータ品質確認
   □ エンベロープとEEGの整列確認
   □ 異なるダウンサンプリング方法のテスト
""")

print("\n" + "=" * 70)
print("詳細は /results/aad_mtrf_envelope/aad_mtrf_s1.png を参照")
print("=" * 70)
