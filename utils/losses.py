import torch
import torch.nn as nn
import torch.fft

class SpectralAngleMapperLoss(nn.Module):
    def forward(self, pred, target):
        dot = (pred * target).sum(dim=1)
        norm_pred = torch.norm(pred, dim=1)
        norm_target = torch.norm(target, dim=1)
        cos = dot / (norm_pred * norm_target + 1e-8)
        return torch.acos(torch.clamp(cos, -0.999, 0.999)).mean()

class FastFourierLoss(nn.Module):
    def forward(self, pred, target):
        fft_pred = torch.fft.fftn(pred, dim=(-2,-1))
        fft_target = torch.fft.fftn(target, dim=(-2,-1))
        return nn.functional.l1_loss(fft_pred.real, fft_target.real) + \
               nn.functional.l1_loss(fft_pred.imag, fft_target.imag)

class CombinedLoss(nn.Module):
    def __init__(self, weight_l1=1.0, weight_sam=0.1, weight_fft=0.01):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sam = SpectralAngleMapperLoss()
        self.fft = FastFourierLoss()
        self.w_l1 = weight_l1
        self.w_sam = weight_sam
        self.w_fft = weight_fft
    
    def forward(self, pred, target):
        return self.w_l1 * self.l1(pred, target) + \
               self.w_sam * self.sam(pred, target) + \
               self.w_fft * self.fft(pred, target)