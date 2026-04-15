import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

class WaveletDecomposition(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
    
    def forward(self, x):
        yl, yh = self.dwt(x)
        lh, hl, hh = yh[0][:, :, 0, :, :], yh[0][:, :, 1, :, :], yh[0][:, :, 2, :, :]
        return yl, lh, hl, hh

class WaveletReconstruction(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        self.idwt = DWTInverse(wave=wave, mode=mode)
    
    def forward(self, yl, lh, hl, hh):
        yh = torch.stack([lh, hl, hh], dim=2)
        return self.idwt((yl, [yh]))