import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import yaml

from data.cave_dataset import CAVEDataset
from models.ultrahsi_net import UltraHSINet
from utils.metrics import psnr, rmse, sam, mssim


def build_test_dataset(config):
    return CAVEDataset(
        root_dir=Path(config["data"]["path"]),
        split="test",
        train_ratio=config["data"]["train_ratio"],
        spatial_size=tuple(config["data"]["spatial_size"]),
        patch_size=tuple(config["data"].get("test_patch_size", [])) or None,
        patches_per_scene=1,
        augment=False,
        normalization=config["data"].get("normalization", "fixed_255"),
        split_scenes=config["data"].get("splits"),
    )


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


def evaluate(model, data_loader, device):
    model.eval()
    totals = {
        "psnr": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "mssim": 0.0,
    }

    with torch.no_grad():
        for rgb, hsi in data_loader:
            rgb, hsi = rgb.to(device), hsi.to(device)
            pred = model(rgb)
            totals["psnr"] += psnr(pred, hsi).item()
            totals["rmse"] += rmse(pred, hsi).item()
            totals["sam"] += sam(pred, hsi).item()
            totals["mssim"] += mssim(pred, hsi).item()

    num_batches = len(data_loader)
    return {name: value / num_batches for name, value in totals.items()}


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"])
    # checkpoint_name = sys.argv[1] if len(sys.argv) > 1 else "best_psnr_model_final.pth"
    checkpoint_name = sys.argv[1] if len(sys.argv) > 1 else "best_sam_model.pth"
    checkpoint_path = Path(checkpoint_name)
    print(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_name} not found. Train the model first.")

    test_dataset = build_test_dataset(config)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(test_dataset.scenes)} test scenes")

    model = build_model(config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    metrics = evaluate(model, test_loader, device)
    print(
        f"{checkpoint_name} test: "
        f"PSNR={metrics['psnr']:.2f} dB, "
        f"RMSE={metrics['rmse']:.4f}, "
        f"SAM={metrics['sam']:.2f} deg, "
        f"MSSIM={metrics['mssim']:.4f}"
    )


if __name__ == "__main__":
    main()
