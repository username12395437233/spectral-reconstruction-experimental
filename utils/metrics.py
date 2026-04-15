import torch
import numpy as np
from skimage.metrics import structural_similarity

def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target)**2)
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2) + 1e-8)

def sam(pred, target):
    dot = (pred * target).sum(dim=1)
    norm_pred = torch.norm(pred, dim=1)
    norm_target = torch.norm(target, dim=1)
    cos = dot / (norm_pred * norm_target + 1e-8)
    return torch.rad2deg(torch.acos(torch.clamp(cos, -0.999, 0.999))).mean()

def mssim(pred, target, data_range=1.0):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    scores = []
    for batch_idx in range(pred_np.shape[0]):
        for band_idx in range(pred_np.shape[1]):
            pred_band = pred_np[batch_idx, band_idx]
            target_band = target_np[batch_idx, band_idx]
            scores.append(
                structural_similarity(
                    pred_band,
                    target_band,
                    data_range=data_range,
                )
            )

    return torch.tensor(float(np.mean(scores)), dtype=pred.dtype, device=pred.device)

def ergas(pred, target, ratio=4):
    err = pred - target
    mse_bands = torch.mean(err**2, dim=(2,3))
    mean_target = torch.mean(target, dim=(2,3))
    ergas_val = torch.sqrt(torch.mean(mse_bands / (mean_target**2 + 1e-8))) * 100 / ratio
    return ergas_val.mean()
