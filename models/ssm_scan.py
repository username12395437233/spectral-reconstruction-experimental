import torch
import torch.nn as nn
from einops import rearrange, repeat

class SS2D_Enhanced(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Используем Mamba2 как базовый сканирующий блок
        from mamba_ssm.modules.mamba2 import Mamba2
        self.mamba = Mamba2(
            d_model=d_model,
            d_ssm=d_model,
            headdim=d_model // 4,  # 4 головы
            expand=expand
        )
        # Пространственно-спектральное внимание
        self.spectral_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model//4, 1),
            nn.ReLU(),
            nn.Conv2d(d_model//4, d_model, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Mamba ожидает (B, L, C) — переставляем, делаем сканирование
        x_perm = rearrange(x, 'b c h w -> b (h w) c')
        out = self.mamba(x_perm)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        # применить спектральное внимание
        attn = self.spectral_attn(out)
        return out * attn