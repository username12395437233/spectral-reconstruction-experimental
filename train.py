import copy
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cave_dataset import CAVEDataset
from models.ultrahsi_net import UltraHSINet
from utils.losses import CombinedLoss
from utils.metrics import mssim, psnr, rmse, sam


def build_datasets(config):
    common_kwargs = {
        "root_dir": Path(config["data"]["path"]),
        "train_ratio": config["data"]["train_ratio"],
        "spatial_size": tuple(config["data"]["spatial_size"]),
        "normalization": config["data"].get("normalization", "fixed_255"),
        "split_scenes": config["data"].get("splits"),
    }

    train_dataset = CAVEDataset(
        split="train",
        patch_size=tuple(config["data"].get("train_patch_size", [])) or None,
        patches_per_scene=config["data"].get("patches_per_scene", 1),
        augment=config["data"].get("augment", True),
        **common_kwargs,
    )
    val_dataset = CAVEDataset(
        split="val",
        patch_size=tuple(config["data"].get("val_patch_size", [])) or None,
        patches_per_scene=1,
        augment=False,
        **common_kwargs,
    )
    test_dataset = CAVEDataset(
        split="test",
        patch_size=tuple(config["data"].get("test_patch_size", [])) or None,
        patches_per_scene=1,
        augment=False,
        **common_kwargs,
    )
    return train_dataset, val_dataset, test_dataset


def build_model(config, device):
    return UltraHSINet(
        d_model=config["model"]["d_model"],
        d_state=config["model"].get("d_state", 16),
        d_conv=config["model"].get("d_conv", 4),
        expand=config["model"].get("expand", 2),
        headdim=config["model"].get("headdim", 16),
        ssm_version=config["model"].get("ssm_version", "mamba3"),
        use_wavelet=config["model"]["use_wavelet"],
        use_gradient_attn=config["model"]["use_gradient_attn"],
    ).to(device)


def create_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        model_state = model.state_dict()
        for name, ema_param in ema_model.state_dict().items():
            ema_param.copy_(ema_param * decay + model_state[name] * (1.0 - decay))


def evaluate(model, data_loader, device):
    model.eval()
    totals = {"psnr": 0.0, "rmse": 0.0, "sam": 0.0, "mssim": 0.0}

    with torch.no_grad():
        for rgb, hsi in data_loader:
            rgb, hsi = rgb.to(device), hsi.to(device)
            pred = model(rgb)
            if isinstance(pred, tuple):
                pred = pred[0]
            totals["psnr"] += psnr(pred, hsi).item()
            totals["rmse"] += rmse(pred, hsi).item()
            totals["sam"] += sam(pred, hsi).item()
            totals["mssim"] += mssim(pred, hsi).item()

    num_batches = len(data_loader)
    return {name: value / num_batches for name, value in totals.items()}


def format_metrics(prefix, metrics):
    return (
        f"{prefix}: "
        f"PSNR={metrics['psnr']:.2f} dB, "
        f"RMSE={metrics['rmse']:.4f}, "
        f"SAM={metrics['sam']:.2f} deg, "
        f"MSSIM={metrics['mssim']:.4f}"
    )


def train():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"])
    train_dataset, val_dataset, test_dataset = build_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(
        f"Split sizes: train={len(train_dataset.scenes)}, "
        f"val={len(val_dataset.scenes)}, test={len(test_dataset.scenes)}"
    )

    model = build_model(config, device)
    ema_model = create_ema_model(model)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = CombinedLoss()
    coarse_loss_weight = config["training"].get("coarse_loss_weight", 0.3)
    ema_decay = config["training"].get("ema_decay", 0.999)

    best_psnr = float("-inf")
    best_sam = float("inf")
    best_psnr_model_path = Path("best_psnr_model.pth")
    best_sam_model_path = Path("best_sam_model.pth")

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0

        for rgb, hsi in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            rgb, hsi = rgb.to(device), hsi.to(device)
            optimizer.zero_grad()
            pred, aux_outputs = model(rgb, return_aux=True)
            loss = criterion(pred, hsi)
            coarse = aux_outputs.get("coarse")
            if coarse is not None:
                loss = loss + coarse_loss_weight * criterion(coarse, hsi)
            loss.backward()
            optimizer.step()
            update_ema(model, ema_model, ema_decay)
            epoch_loss += loss.item()

        scheduler.step()
        val_metrics = evaluate(ema_model, val_loader, device)

        print(
            f"Epoch {epoch + 1}: Loss={epoch_loss / len(train_loader):.4f}, "
            f"PSNR={val_metrics['psnr']:.2f} dB, "
            f"RMSE={val_metrics['rmse']:.4f}, "
            f"SAM={val_metrics['sam']:.2f} deg, "
            f"MSSIM={val_metrics['mssim']:.4f}"
        )

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save(ema_model.state_dict(), best_psnr_model_path)
        if val_metrics["sam"] < best_sam:
            best_sam = val_metrics["sam"]
            torch.save(ema_model.state_dict(), best_sam_model_path)

    print(f"Training finished. Best val PSNR: {best_psnr:.2f} dB, Best val SAM: {best_sam:.2f} deg")

    best_psnr_model = build_model(config, device)
    best_psnr_model.load_state_dict(torch.load(best_psnr_model_path, map_location=device))
    best_psnr_test_metrics = evaluate(best_psnr_model, test_loader, device)
    print(format_metrics("Final test (best PSNR)", best_psnr_test_metrics))

    best_sam_model = build_model(config, device)
    best_sam_model.load_state_dict(torch.load(best_sam_model_path, map_location=device))
    best_sam_test_metrics = evaluate(best_sam_model, test_loader, device)
    print(format_metrics("Final test (best SAM)", best_sam_test_metrics))


if __name__ == "__main__":
    train()
