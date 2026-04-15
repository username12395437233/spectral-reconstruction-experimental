import torch
import torch.nn as nn
from mamba_ssm import Mamba3 # Импортируем Mamba3

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=64):
        super().__init__()
        self.mamba = Mamba3(
            d_model=d_model, 
            d_state=d_state,
            d_conv=d_conv,     
            expand=expand,     
            headdim=headdim,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        out = self.mamba(self.norm(x_flat))
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out + x