import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from data.cave_dataset import CAVEDataset
from models.ultrahsi_net import UltraHSINet
from utils.losses import CombinedLoss
from utils.metrics import psnr, sam
import yaml
from tqdm import tqdm

def train():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['device'])
    
    train_dataset = CAVEDataset(
        Path(config['data']['path']),
        train=True,
        train_ratio=config['data']['train_ratio'],
        spatial_size=tuple(config['data']['spatial_size'])
    )
    val_dataset = CAVEDataset(
        Path(config['data']['path']),
        train=False,
        train_ratio=config['data']['train_ratio'],
        spatial_size=tuple(config['data']['spatial_size'])
    )
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = UltraHSINet(
        d_model=config['model']['d_model'],
        use_wavelet=config['model']['use_wavelet'],
        use_gradient_attn=config['model']['use_gradient_attn']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    criterion = CombinedLoss()
    
    best_psnr = 0
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        for rgb, hsi in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            rgb, hsi = rgb.to(device), hsi.to(device)
            optimizer.zero_grad()
            pred = model(rgb)
            loss = criterion(pred, hsi)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_psnr = 0
        val_sam = 0
        with torch.no_grad():
            for rgb, hsi in val_loader:
                rgb, hsi = rgb.to(device), hsi.to(device)
                pred = model(rgb)
                val_psnr += psnr(pred, hsi).item()
                val_sam += sam(pred, hsi).item()
        val_psnr /= len(val_loader)
        val_sam /= len(val_loader)
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, PSNR={val_psnr:.2f} dB, SAM={val_sam:.2f}°')
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Training finished. Best PSNR: {best_psnr:.2f} dB')

if __name__ == '__main__':
    train()