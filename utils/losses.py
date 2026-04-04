import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class SpectralAngleMapperLoss(nn.Module):
    def forward(self, pred, target):
        dot = (pred * target).sum(dim=1)
        norm_pred = torch.norm(pred, dim=1)
        norm_target = torch.norm(target, dim=1)
        cos = dot / (norm_pred * norm_target + 1e-8)
        return torch.acos(torch.clamp(cos, -0.999, 0.999)).mean()


class FastFourierLoss(nn.Module):
    def forward(self, pred, target):
        fft_pred = torch.fft.fftn(pred, dim=(-2, -1))
        fft_target = torch.fft.fftn(target, dim=(-2, -1))
        return F.l1_loss(fft_pred.real, fft_target.real) + F.l1_loss(fft_pred.imag, fft_target.imag)


class SpectralDifferenceLoss(nn.Module):
    def forward(self, pred, target):
        pred_diff = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_diff = target[:, 1:, :, :] - target[:, :-1, :, :]
        return F.l1_loss(pred_diff, target_diff)


class SpectralCurvatureLoss(nn.Module):
    def forward(self, pred, target):
        pred_diff2 = pred[:, 2:, :, :] - 2 * pred[:, 1:-1, :, :] + pred[:, :-2, :, :]
        target_diff2 = target[:, 2:, :, :] - 2 * target[:, 1:-1, :, :] + target[:, :-2, :, :]
        return F.l1_loss(pred_diff2, target_diff2)


class SpectralCorrelationLoss(nn.Module):
    def forward(self, pred, target):
        b, c, h, w = pred.shape
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, c)
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, c)
        pred_flat = F.normalize(pred_flat, dim=1)
        target_flat = F.normalize(target_flat, dim=1)
        cosine = (pred_flat * target_flat).sum(dim=1)
        return 1.0 - cosine.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        weight_l1=1.0,
        weight_sam=0.2,
        weight_fft=0.01,
        weight_spectral_diff=0.25,
        weight_spectral_curvature=0.15,
        weight_spectral_corr=0.15,
    ):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sam = SpectralAngleMapperLoss()
        self.fft = FastFourierLoss()
        self.spectral_diff = SpectralDifferenceLoss()
        self.spectral_curvature = SpectralCurvatureLoss()
        self.spectral_corr = SpectralCorrelationLoss()
        self.w_l1 = weight_l1
        self.w_sam = weight_sam
        self.w_fft = weight_fft
        self.w_spectral_diff = weight_spectral_diff
        self.w_spectral_curvature = weight_spectral_curvature
        self.w_spectral_corr = weight_spectral_corr

    def forward(self, pred, target):
        return (
            self.w_l1 * self.l1(pred, target)
            + self.w_sam * self.sam(pred, target)
            + self.w_fft * self.fft(pred, target)
            + self.w_spectral_diff * self.spectral_diff(pred, target)
            + self.w_spectral_curvature * self.spectral_curvature(pred, target)
            + self.w_spectral_corr * self.spectral_corr(pred, target)
        )
