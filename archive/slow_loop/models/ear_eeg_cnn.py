import torch
import torch.nn as nn
import torch.nn.functional as F

class EarEEGCNN(nn.Module):
    """
    Phase 2: スマートイヤホン/小型電極 (Ear-EEG) 向け特化型モデル
    数チャンネルの入力から、Neural Spatial Filter で特徴を増幅し、STFT + 2D-CNN で判定する。
    """
    def __init__(self, num_channels=4, virtual_channels=8, num_classes=2, n_fft=64, hop_length=16):
        super(EarEEGCNN, self).__init__()
        
        # 🚨 [新兵器] Neural Spatial Filter (空間フィルター)
        # kernel_size=1 の畳み込みは「時間方向には動かず、チャンネル間でのみ重み付き和を計算する」魔法の層です。
        # 例：4つの実際の電極から、8つの「理想的な仮想電極」をAIが合成します。
        self.spatial_filter = nn.Conv1d(
            in_channels=num_channels, 
            out_channels=virtual_channels, 
            kernel_size=1, 
            bias=False # 純粋なフィルターとして機能させるためバイアスは無し
        )
        self.bn_spatial = nn.BatchNorm1d(virtual_channels)

        # 以下のSTFTと2D-CNNは、仮想チャンネル(virtual_channels)を処理するように進化
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.conv1 = nn.Conv2d(in_channels=virtual_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, num_channels(例:4), seq_len)
        
        # 1. 空間フィルタリング（実際の電極 -> 仮想の電極へ変換）
        x = self.spatial_filter(x)
        x = self.bn_spatial(x)
        # 変換後 shape: (batch, virtual_channels(例:8), seq_len)
        
        # 2. STFT（周波数画像への変換）
        batch_size, v_ch, seq_len = x.shape
        x_reshaped = x.view(batch_size * v_ch, seq_len)
        
        window = torch.hann_window(self.n_fft).to(x.device)
        stft_out = torch.stft(x_reshaped, n_fft=self.n_fft, hop_length=self.hop_length, 
                              window=window, return_complex=True)
        spectrogram = torch.abs(stft_out) 
        
        _, freq_bins, time_frames = spectrogram.shape
        spectrogram = spectrogram.view(batch_size, v_ch, freq_bins, time_frames)
        
        # 3. 2D-CNNによる特徴抽出
        x = self.conv1(spectrogram)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.adaptive_pool(x).view(batch_size, -1)
        out = self.fc(x)
        return out