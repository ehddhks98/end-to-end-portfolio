import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.leaky_relu1 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.leaky_relu2 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.leaky_relu1, self.drop1,
            self.conv2, self.chomp2, self.bn2, self.leaky_relu2, self.drop2
        )

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None
        
        self.final_activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_activation(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            pad = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, ch, kernel_size, stride=1, 
                dilation=dilation, padding=pad, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(num_channels[-1], hidden_size),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        # Conv1d를 위한 전치: (batch, seq, features) → (batch, features, seq)
        x = x.transpose(1, 2)
        
        # TCN 레이어들을 통과
        out = self.network(x)
        
        # 마지막 타임스텝 선택: (batch, channels, seq) → (batch, channels)
        out = out[:, :, -1]
        
        # 최종 은닉 크기로 투영
        out = self.output_projection(out)
        return out
