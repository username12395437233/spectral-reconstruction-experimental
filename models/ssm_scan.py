import math

import torch
import torch.nn as nn
from einops import rearrange


class SS2D_Enhanced(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=None, ssm_version="mamba3"):
        super().__init__()

        if headdim is None:
            headdim = max(8, d_model // 4)
        if d_model % headdim != 0:
            headdim = math.gcd(d_model, headdim)
        if headdim <= 0:
            raise ValueError(f"Failed to derive a valid headdim for d_model={d_model}")

        self.norm = nn.LayerNorm(d_model)
        ssm_version = ssm_version.lower()

        if ssm_version == "mamba3":
            # Mamba3's Triton rotary-QK kernel needs the effective QK dimension to be >= 16.
            # With the default rope_fraction=0.5 in upstream Mamba3, this implies d_state >= 64.
            if d_state < 64:
                raise ValueError(
                    f"Mamba3 requires d_state >= 64 for the current kernel path, got d_state={d_state}. "
                    "Set model.d_state to 64 or 128, or switch model.ssm_version to 'mamba2'."
                )
            from mamba_ssm import Mamba3

            self.mamba = Mamba3(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            )
        elif ssm_version == "mamba2":
            from mamba_ssm.modules.mamba2 import Mamba2

            self.mamba = Mamba2(
                d_model=d_model,
                d_ssm=d_model,
                headdim=headdim,
                expand=expand,
            )
        else:
            raise ValueError(f"Unsupported ssm_version: {ssm_version}")

        self.spectral_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        x_perm = rearrange(x, "b c h w -> b (h w) c")
        out = self.mamba(self.norm(x_perm))
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        attn = self.spectral_attn(out)
        return x + out * attn
