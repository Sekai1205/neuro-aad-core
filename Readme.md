# Neuro-AAD: Personalized Auditory Attention Decoding

## 概要 (Overview)
脳波（EEG）を用いた聴覚注意デコーディング（AAD）システム。
「MAML（メタ学習）」による超高速個人適応と、「C++ / Python 非同期デュアルループ」による遅延制約（< 10ms）の突破を目指すプロトタイプ実装。

## アーキテクチャ (Architecture)
* **Fast Loop (C++ / Metal):** * 役割: 音響レンダリング・フィルタ処理
  * 制約: `< 10ms`
  * 特徴: Lock-free SPSC Queueによる完全なノンブロッキング処理
* **Slow Loop (Python / PyTorch):**
  * 役割: EEGデコーディング・MAMLファインチューニング
  * 制約: `< 2.0s` (Switch Detection Latency)
  * 特徴: `learn2learn` を用いた3分間フューショット適応