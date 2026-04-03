import torch
import torch.nn as nn
from models.ssm_scan import SS2D_Enhanced

class MIFModule(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=None, ssm_version="mamba3"):
        super().__init__()
        self.global_branch = SS2D_Enhanced(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ssm_version=ssm_version,
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 1)
        )
        self.adaptive_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        global_feat = self.global_branch(x)
        local_feat = self.local_branch(x)
        weights = self.adaptive_router(x)  # (B,2)
        out = (weights[:,0:1].view(-1,1,1,1) * global_feat + 
               weights[:,1:2].view(-1,1,1,1) * local_feat)
        return out
