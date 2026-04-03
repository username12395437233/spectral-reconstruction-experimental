import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        self.sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.conv_grad = nn.Conv2d(in_channels*2, in_channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        grad_x = F.conv2d(x, sobel_x.expand(C,1,3,3), groups=C, padding=1)
        grad_y = F.conv2d(x, sobel_y.expand(C,1,3,3), groups=C, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        combined = torch.cat([x, grad_mag], dim=1)
        attn = torch.sigmoid(self.conv_grad(combined))
        return x * attn