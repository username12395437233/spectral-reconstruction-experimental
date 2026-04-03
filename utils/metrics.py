import torch
import numpy as np

def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target)**2)
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))

def sam(pred, target):
    dot = (pred * target).sum(dim=1)
    norm_pred = torch.norm(pred, dim=1)
    norm_target = torch.norm(target, dim=1)
    cos = dot / (norm_pred * norm_target + 1e-8)
    return torch.rad2deg(torch.acos(torch.clamp(cos, -0.999, 0.999))).mean()

def ergas(pred, target, ratio=4):
    # ratio = spatial resolution ratio (обычно 4 для CAVE 512->128? уточнить)
    err = pred - target
    mse_bands = torch.mean(err**2, dim=(2,3))
    mean_target = torch.mean(target, dim=(2,3))
    ergas_val = torch.sqrt(torch.mean(mse_bands / (mean_target**2 + 1e-8))) * 100 / ratio
    return ergas_val.mean()