import torch
import torch.nn as nn

# M5 Macのエッジ環境（低遅延）で動かすための、超軽量な1次元CNN
class EEG1DCNN(nn.Module):
    def __init__(self, num_channels=64, num_classes=2):
        super(EEG1DCNN, self).__init__()
        # 特徴抽出（フィルタリング）
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # データをギュッと圧縮
        )
        # 分類（左の音か、右の音か）
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # 一列に並べ直す
        x = self.classifier(x)
        return x