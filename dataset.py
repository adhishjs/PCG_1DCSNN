# =============================================================
# dataset.py  —  PCG Spike Dataset  (encoder-agnostic)
# =============================================================
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from encoding import BaseEncoder, MelEncoder


class PCGSpikeDataset(Dataset):
    """
    Loads .npy PCG segments, delegates all feature extraction to an
    encoder, and returns rate-encoded spike tensors.

    File naming convention:  a{id:04d}_{label}_seg{n}.npy
    label is 'normal' or 'abnormal'

    Args:
        folder  : path to FILTERED_DATA or FINAL_VAL
        encoder : any BaseEncoder subclass instance
                  (default: MelEncoder with sr=2000, n_mels=24)
        T_sim   : number of SNN simulation time steps
    """
    def __init__(self, folder: str,
                 encoder: BaseEncoder = None,
                 T_sim: int = 50):
        self.folder  = folder
        self.encoder = encoder or MelEncoder()
        self.T_sim   = T_sim

        self.files  = []
        self.labels = []

        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.npy'):
                continue
            if '_normal_' in fname:
                label = 0
            elif '_abnormal_' in fname:
                label = 1
            else:
                continue
            self.files.append(os.path.join(folder, fname))
            self.labels.append(label)

        print(f"Loaded {len(self.files)} segments from {folder}")
        print(f"  Normal: {self.labels.count(0)}  |  Abnormal: {self.labels.count(1)}")
        print(f"  Encoder: {self.encoder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = np.load(self.files[idx]).astype(np.float32)

        # All feature extraction lives in the encoder
        spikes = self.encoder.encode_to_spikes(audio, T_sim=self.T_sim)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spikes, label
