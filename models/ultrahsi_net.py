import torch
import torch.nn as nn
from models.gradient_attention import GradientAttention
from models.mif_module import MIFModule

class UltraHSINet(nn.Module):
    def __init__(self, d_model=64, use_wavelet=False, use_gradient_attn=True, num_spectral=31):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_gradient_attn = use_gradient_attn
        
        self.input_conv = nn.Conv2d(3, d_model, 3, padding=1)
        
        if use_wavelet:
            from models.wavelet_module import WaveletDecomposition
            self.wavelet = WaveletDecomposition()
            self.hf_cnn = nn.Sequential(
                nn.Conv2d(3, d_model, 3, padding=1),
                nn.BatchNorm2d(d_model),
                nn.ReLU()
            )
            # Для LL нужна отдельная свёртка с учётом уменьшенного размера
            self.ll_conv = nn.Sequential(
                nn.Conv2d(3, d_model, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        
        # Encoder
        self.down1 = nn.Conv2d(d_model, d_model*2, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(d_model*2, d_model*4, 4, stride=2, padding=1)
        
        self.bottleneck = MIFModule(d_model*4)
        
        self.up2 = nn.ConvTranspose2d(d_model*4, d_model*2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(d_model*2, d_model, 4, stride=2, padding=1)
        
        if use_gradient_attn:
            self.grad_attn2 = GradientAttention(d_model*2)
            self.grad_attn1 = GradientAttention(d_model)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(d_model, num_spectral, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb):
        if self.use_wavelet:
            LL, LH, HL, HH = self.wavelet(rgb)
            lf = self.ll_conv(LL)                     # (B, d_model, H, W)
            hf = torch.stack([LH, HL, HH], dim=2)     # (B, 3, 3, H/2, W/2) — упростим
            hf = hf.view(hf.size(0), 3, hf.size(3), hf.size(4))  # (B, 3, H/2, W/2)
            hf = nn.functional.interpolate(hf, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 3, H, W)
            hf = self.hf_cnn(hf)                      # (B, d_model, H, W)
            x = lf + hf
        else:
            x = self.input_conv(rgb)
        
        e1 = self.down1(x)
        e2 = self.down2(e1)
        b = self.bottleneck(e2)
        d2 = self.up2(b)
        if self.use_gradient_attn:
            d2 = self.grad_attn2(d2)
        d2 = d2 + e1
        d1 = self.up1(d2)
        if self.use_gradient_attn:
            d1 = self.grad_attn1(d1)
        d1 = d1 + x
        out = self.output_conv(d1)
        return out