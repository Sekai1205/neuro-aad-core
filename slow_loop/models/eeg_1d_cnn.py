import torch
import torch.nn as nn

# ============================================================
# 【改修①】 Spatial Filter CNN
# ネットワークの先頭に「空間フィルタ層」を追加。
# AI自身が「64個の電極のどれを重視すべきか」を自動学習する。
#
# アーキテクチャ解説:
#   [入力: (Batch, 64ch, 256time)]
#       ↓
#   SpatialFilter: Conv1d(64→8, kernel=1)
#     → 電極64本を重み付き線形結合で8本の「仮想チャンネル」に圧縮。
#       kernel_size=1 なので時間方向は触らず、空間のみを操作。
#       ※CSP (Common Spatial Pattern) のニューラルネット版。
#   BatchNorm + Dropout で過学習を抑制
#       ↓
#   Temporal Feature Extractor: 時間方向の特徴抽出
#     Conv1d(8→16, k=5) → ReLU
#     Conv1d(16→32, k=3) → ReLU
#     AdaptiveAvgPool1d(1) → 時系列をスカラーに圧縮
#       ↓
#   Classifier: Linear(32→2) → Left / Right
# ============================================================

class EEG1DCNN(nn.Module):
    def __init__(self, num_channels=64, num_spatial_filters=8, num_classes=2):
        super(EEG1DCNN, self).__init__()

        # -------------------------------------------------------------------
        # 【空間フィルタ層】AI版CSP
        # kernel_size=1 → 時間方向には一切触れず、64電極の「重み付き合成」のみ。
        # bias=False → BatchNormと一緒に使うときは不要（慣例）。
        # -------------------------------------------------------------------
        self.spatial_filter = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,       # 入力: 64電極
                out_channels=num_spatial_filters, # 出力: 8仮想チャンネル
                kernel_size=1,                   # 空間のみ操作（時間方向=1）
                bias=False
            ),
            nn.BatchNorm1d(num_spatial_filters),  # 学習安定化
            nn.Dropout(p=0.25),                   # 過学習抑制（電極ランダムマスク）
        )

        # -------------------------------------------------------------------
        # 【時間特徴抽出層】元のCNNをそのまま継承
        # 入力チャンネル数が num_spatial_filters(8) に変わったことに注意
        # -------------------------------------------------------------------
        self.temporal_features = nn.Sequential(
            nn.Conv1d(num_spatial_filters, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # データをギュッと圧縮
        )

        # 分類（左の音か、右の音か）
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (Batch, 64, 256)
        x = self.spatial_filter(x)     # → (Batch, 8, 256)
        x = self.temporal_features(x)  # → (Batch, 32, 1)
        x = x.view(x.size(0), -1)     # → (Batch, 32)
        x = self.classifier(x)         # → (Batch, 2)
        return x

    def get_spatial_weights(self):
        """
        【可視化用ヘルパー】
        学習済みの空間フィルタ重みを返す。
        shape: (num_spatial_filters, num_channels) = (8, 64)
        各行が「1つの仮想チャンネルを作るための64電極の重み」を表す。
        plt.imshow() で ヒートマップとして可視化できる。
        """
        weight = self.spatial_filter[0].weight.data.cpu()  # (8, 64, 1)
        return weight.squeeze(-1)  # (8, 64)