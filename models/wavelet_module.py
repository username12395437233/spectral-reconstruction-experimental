import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

class WaveletDecomposition(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
    
    def forward(self, x):
        # x: (B, C, H, W)
        yl, yh = self.dwt(x)
        # yl: (B, C, H/2, W/2)
        # yh: list of length J, each element (B, C, 3, H/2, W/2) for J=1
        lh, hl, hh = yh[0][:, :, 0, :, :], yh[0][:, :, 1, :, :], yh[0][:, :, 2, :, :]
        return yl, lh, hl, hh

class WaveletReconstruction(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        self.idwt = DWTInverse(wave=wave, mode=mode)
    
    def forward(self, yl, lh, hl, hh):
        # Объединяем обратно в формат (B, C, 3, H, W)
        yh = torch.stack([lh, hl, hh], dim=2)
        return self.idwt((yl, [yh]))