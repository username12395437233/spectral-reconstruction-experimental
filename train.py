import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.cave_dataset import CAVEDataset
from models.ultrahsi_net import UltraHSINet
from utils.losses import CombinedLoss
from utils.metrics import psnr, rmse, sam, mssim


def train():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"])

    train_dataset = CAVEDataset(
        Path(config["data"]["path"]),
        split="train",
        train_ratio=config["data"]["train_ratio"],
        spatial_size=tuple(config["data"]["spatial_size"]),
        patch_size=tuple(config["data"].get("train_patch_size", [])) or None,
        patches_per_scene=config["data"].get("patches_per_scene", 1),
        augment=config["data"].get("augment", True),
        normalization=config["data"].get("normalization", "fixed_255"),
        split_scenes=config["data"].get("splits"),
    )
    val_dataset = CAVEDataset(
        Path(config["data"]["path"]),
        split="val",
        train_ratio=config["data"]["train_ratio"],
        spatial_size=tuple(config["data"]["spatial_size"]),
        patch_size=tuple(config["data"].get("val_patch_size", [])) or None,
        patches_per_scene=1,
        augment=False,
        normalization=config["data"].get("normalization", "fixed_255"),
        split_scenes=config["data"].get("splits"),
    )
    test_dataset = CAVEDataset(
        Path(config["data"]["path"]),
        split="test",
        train_ratio=config["data"]["train_ratio"],
        spatial_size=tuple(config["data"]["spatial_size"]),
        patch_size=tuple(config["data"].get("test_patch_size", [])) or None,
        patches_per_scene=1,
        augment=False,
        normalization=config["data"].get("normalization", "fixed_255"),
        split_scenes=config["data"].get("splits"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(
        f"Split sizes: train={len(train_dataset.scenes)}, "
        f"val={len(val_dataset.scenes)}, test={len(test_dataset.scenes)}"
    )

    model = UltraHSINet(
        d_model=config["model"]["d_model"],
        d_state=config["model"].get("d_state", 16),
        d_conv=config["model"].get("d_conv", 4),
        expand=config["model"].get("expand", 2),
        headdim=config["model"].get("headdim", 16),
        ssm_version=config["model"].get("ssm_version", "mamba3"),
        use_wavelet=config["model"]["use_wavelet"],
        use_gradient_attn=config["model"]["use_gradient_attn"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = CombinedLoss()

    best_psnr = 0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0
        for rgb, hsi in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            rgb, hsi = rgb.to(device), hsi.to(device)
            optimizer.zero_grad()
            pred = model(rgb)
            loss = criterion(pred, hsi)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        model.eval()
        val_psnr = 0
        val_rmse = 0
        val_sam = 0
        val_mssim = 0
        with torch.no_grad():
            for rgb, hsi in val_loader:
                rgb, hsi = rgb.to(device), hsi.to(device)
                pred = model(rgb)
                val_psnr += psnr(pred, hsi).item()
                val_rmse += rmse(pred, hsi).item()
                val_sam += sam(pred, hsi).item()
                val_mssim += mssim(pred, hsi).item()

        val_psnr /= len(val_loader)
        val_rmse /= len(val_loader)
        val_sam /= len(val_loader)
        val_mssim /= len(val_loader)

        print(
            f"Epoch {epoch + 1}: Loss={epoch_loss / len(train_loader):.4f}, "
            f"PSNR={val_psnr:.2f} dB, RMSE={val_rmse:.4f}, "
            f"SAM={val_sam:.2f} deg, MSSIM={val_mssim:.4f}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Training finished. Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    train()
