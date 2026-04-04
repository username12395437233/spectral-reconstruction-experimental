import torch
import torch.nn as nn
from models.wavelet_module import WaveletDecomposition
from models.gradient_attention import GradientAttention
from models.mif_module import MIFModule


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
            # LL (низкие частоты) -> проекция в d_model и апскейл до исходного размера
            self.ll_conv = nn.Sequential(
                nn.Conv2d(3, d_model, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
            # Высокочастотные компоненты (LH, HL, HH) объединяются в 9 каналов
            self.hf_cnn = nn.Sequential(
                nn.Conv2d(9, d_model, 3, padding=1),
                nn.BatchNorm2d(d_model),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        else:
            self.input_conv = nn.Conv2d(3, d_model, 3, padding=1)
        
        # Encoder
        self.down1 = nn.Conv2d(d_model, d_model*2, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(d_model*2, d_model*4, 4, stride=2, padding=1)

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
        
        self.up2 = nn.ConvTranspose2d(d_model*4, d_model*2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(d_model*2, d_model, 4, stride=2, padding=1)
        self.fuse2 = FeatureFusionBlock(d_model * 2)
        self.fuse1 = FeatureFusionBlock(d_model)
        
        if use_gradient_attn:
            self.grad_attn2 = GradientAttention(d_model*2)
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
        self.rgb_skip = nn.Sequential(
            nn.Conv2d(3, d_model // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, num_spectral, 1),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(d_model, num_spectral, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, rgb):
        if self.use_wavelet:
            LL, LH, HL, HH = self.wavelet(rgb)      # каждый: (B,3,H/2,W/2)
            lf = self.ll_conv(LL)                   # (B,d_model,H,W)
            # Объединяем LH, HL, HH в 9 каналов
            hf = torch.cat([LH, HL, HH], dim=1)     # (B,9,H/2,W/2)
            hf = self.hf_cnn(hf)                    # (B,d_model,H,W)
            x = lf + hf
        else:
            x = self.input_conv(rgb)                # (B,d_model,H,W)
        
        e1 = self.down1(x)      # (B,d_model*2,H/2,W/2)
        e1 = self.mid_encoder(e1)
        e2 = self.down2(e1)     # (B,d_model*4,H/4,W/4)
        b = self.bottleneck(e2) # (B,d_model*4,H/4,W/4)
        d2 = self.up2(b)        # (B,d_model*2,H/2,W/2)
        if self.use_gradient_attn:
            d2 = self.grad_attn2(d2)
        d2 = self.fuse2(d2, e1)
        d1 = self.up1(d2)       # (B,d_model,H,W)
        if self.use_gradient_attn:
            d1 = self.grad_attn1(d1)
        d1 = self.fuse1(d1, x)
        head_feat = self.pre_head(d1)
        head_feat = head_feat + self.spectral_mixer(head_feat)
        spectral_logits = self.output_conv[0](head_feat) + self.rgb_skip(rgb)
        out = self.output_conv[1](spectral_logits)
        return out
