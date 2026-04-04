import torch
import torch.nn as nn

from models.gradient_attention import GradientAttention
from models.mif_module import MIFModule
from models.wavelet_module import WaveletDecomposition


class ResidualRefinementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x, skip):
        fused = x + skip
        return fused + self.block(fused)


class UltraHSINet(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=16,
        ssm_version="mamba3",
        use_wavelet=True,
        use_gradient_attn=True,
        num_spectral=31,
    ):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_gradient_attn = use_gradient_attn

        if use_wavelet:
            self.wavelet = WaveletDecomposition()
            self.ll_conv = nn.Sequential(
                nn.Conv2d(3, d_model, 3, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            )
            self.hf_cnn = nn.Sequential(
                nn.Conv2d(9, d_model, 3, padding=1),
                nn.BatchNorm2d(d_model),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            )
        else:
            self.input_conv = nn.Conv2d(3, d_model, 3, padding=1)

        self.down1 = nn.Conv2d(d_model, d_model * 2, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(d_model * 2, d_model * 4, 4, stride=2, padding=1)

        self.mid_encoder = MIFModule(
            d_model * 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ssm_version=ssm_version,
        )
        self.bottleneck = MIFModule(
            d_model * 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ssm_version=ssm_version,
        )

        self.up2 = nn.ConvTranspose2d(d_model * 4, d_model * 2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(d_model * 2, d_model, 4, stride=2, padding=1)
        self.fuse2 = FeatureFusionBlock(d_model * 2)
        self.fuse1 = FeatureFusionBlock(d_model)

        if use_gradient_attn:
            self.grad_attn2 = GradientAttention(d_model * 2)
            self.grad_attn1 = GradientAttention(d_model)

        self.pre_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            ResidualRefinementBlock(d_model),
            ResidualRefinementBlock(d_model),
        )
        self.spectral_mixer = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 1),
        )
        self.rgb_embed = nn.Sequential(
            nn.Conv2d(3, d_model // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model // 2, 3, padding=1),
            nn.GELU(),
        )
        self.rgb_skip = nn.Sequential(
            nn.Conv2d(3, d_model // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, num_spectral, 1),
        )
        self.coarse_head = nn.Sequential(
            nn.Conv2d(d_model, num_spectral, 3, padding=1),
            nn.Sigmoid(),
        )
        self.refinement_head = nn.Sequential(
            nn.Conv2d(d_model + d_model // 2 + num_spectral, d_model, 3, padding=1),
            nn.GELU(),
            ResidualRefinementBlock(d_model),
            nn.Conv2d(d_model, num_spectral, 3, padding=1),
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, rgb, return_aux=False):
        if self.use_wavelet:
            ll, lh, hl, hh = self.wavelet(rgb)
            lf = self.ll_conv(ll)
            hf = torch.cat([lh, hl, hh], dim=1)
            hf = self.hf_cnn(hf)
            x = lf + hf
        else:
            x = self.input_conv(rgb)

        e1 = self.down1(x)
        e1 = self.mid_encoder(e1)
        e2 = self.down2(e1)
        b = self.bottleneck(e2)

        d2 = self.up2(b)
        if self.use_gradient_attn:
            d2 = self.grad_attn2(d2)
        d2 = self.fuse2(d2, e1)

        d1 = self.up1(d2)
        if self.use_gradient_attn:
            d1 = self.grad_attn1(d1)
        d1 = self.fuse1(d1, x)

        head_feat = self.pre_head(d1)
        head_feat = head_feat + self.spectral_mixer(head_feat)

        coarse_logits = self.coarse_head[0](head_feat) + self.rgb_skip(rgb)
        coarse = self.coarse_head[1](coarse_logits)

        refine_input = torch.cat([head_feat, self.rgb_embed(rgb), coarse], dim=1)
        residual_logits = self.refinement_head(refine_input)
        out = self.final_activation(coarse_logits + residual_logits)

        if return_aux:
            return out, {"coarse": coarse}
        return out
