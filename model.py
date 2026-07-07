# =============================================================
# model.py  —  1D Convolutional Spiking Neural Network
# =============================================================
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class ConvSNN1D(nn.Module):
    """
    1D Conv SNN for PCG classification.

    Input tensor per forward call:  (batch, n_mels, T_frames)
    The network is called once per simulation time step inside
    the training loop, accumulating membrane potentials across
    T_sim steps.

    Architecture:
        Conv1D(40→32, k=7) → BN → LIF
        MaxPool1D(2)
        Conv1D(32→64, k=5) → BN → LIF
        AdaptiveAvgPool1D(1)          ← collapses time
        Linear(64 → 2)               ← readout
    """
    def __init__(self, n_mels: int = 40, beta: float = 0.9):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Block 1
        self.conv1 = nn.Conv1d(n_mels, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.pool1 = nn.MaxPool1d(2)

        # Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.pool2 = nn.AdaptiveAvgPool1d(1)   # (B, 64, 1)

        # Readout
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        """
        x: (batch, n_mels, T_frames)  — one time step of the simulation
        Returns: spike tensor (batch, 2)
        """
        # Block 1
        out = self.conv1(x)          # (B, 32, T)
        out = self.bn1(out)
        out = self.lif1(out)         # LIF: returns spikes (B, 32, T)

        out = self.pool1(out)        # (B, 32, T//2)

        # Block 2
        out = self.conv2(out)        # (B, 64, T//2)
        out = self.bn2(out)
        out = self.lif2(out)         # LIF: returns spikes (B, 64, T//2)

        out = self.pool2(out)        # (B, 64, 1)
        out = out.squeeze(-1)        # (B, 64)

        out = self.fc(out)           # (B, 2)
        return out