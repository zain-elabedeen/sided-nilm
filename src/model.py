import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, : -self.conv.padding[0]]
        return self.norm(self.relu(out))

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(1, 32, 3, 1),
            TCNBlock(32, 32, 3, 2),
            TCNBlock(32, 32, 3, 4),
        )
        self.fc = nn.Conv1d(32, 5, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.fc(x)
        return x.transpose(1, 2)