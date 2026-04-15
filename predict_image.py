import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from models.ultrahsi_net import UltraHSINet

CONFIG_PATH = "config.yaml"
CHECKPOINT_PATH = "best_psnr_model_final.pth"
IMAGE_PATH = "/mnt/c/Users/User/repo/datasets/test1.png"

OUTPUT_NPY_PATH = "/mnt/c/Users/User/repo/saved/test_pred_hsi.npy"
OUTPUT_PREVIEW_PATH = "/mnt/c/Users/User/repo/saved/test_preview.png"

# Если True, вход будет приведен к spatial_size из config.yaml.
RESIZE_TO_CONFIG = False
PAD_MULTIPLE = 4


def build_model(config, device):
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
    return model


def load_rgb_image(image_path, spatial_size=None):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    if spatial_size is not None:
        image = image.resize((spatial_size[1], spatial_size[0]), Image.BILINEAR)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return image_tensor, original_size


def pad_to_multiple(tensor, multiple=4):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)

    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (pad_h, pad_w)


def crop_back(tensor, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return tensor

    _, _, h, w = tensor.shape
    return tensor[:, :, :h - pad_h, :w - pad_w]


def save_hsi_cube(hsi_tensor, output_path):
    hsi_np = hsi_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    np.save(output_path, hsi_np)
    return hsi_np.shape


def save_preview(hsi_tensor, output_path):
    hsi = hsi_tensor.squeeze(0).cpu().numpy()  # (C, H, W)
    rgb = np.stack([
        hsi[25],  # R
        hsi[15],  # G
        hsi[5],   # B
    ], axis=-1)

    rgb = np.clip(rgb, 0.0, 1.0)

    rgb = rgb - rgb.min()
    max_val = rgb.max()
    if max_val > 0:
        rgb = rgb / max_val

    preview_img = Image.fromarray((rgb * 255).astype(np.uint8))
    preview_img.save(output_path)


def main():
    config_path = Path(CONFIG_PATH)
    checkpoint_path = Path(CHECKPOINT_PATH)
    image_path = Path(IMAGE_PATH)
    output_npy_path = Path(OUTPUT_NPY_PATH)
    output_preview_path = Path(OUTPUT_PREVIEW_PATH)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"])
    spatial_size = tuple(config["data"]["spatial_size"]) if RESIZE_TO_CONFIG else None

    model = build_model(config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    rgb_tensor, original_size = load_rgb_image(image_path, spatial_size=spatial_size)

    input_h = rgb_tensor.shape[2]
    input_w = rgb_tensor.shape[3]

    rgb_tensor, pad_hw = pad_to_multiple(rgb_tensor, multiple=PAD_MULTIPLE)
    padded_h = rgb_tensor.shape[2]
    padded_w = rgb_tensor.shape[3]

    rgb_tensor = rgb_tensor.to(device)

    with torch.no_grad():
        pred = model(rgb_tensor)

    pred = crop_back(pred, pad_hw)

    shape = save_hsi_cube(pred, output_npy_path)
    save_preview(pred, output_preview_path)

    print(f"Input image: {image_path}")
    print(f"Original size (W, H): {original_size}")
    print(f"Model input size before pad: ({input_h}, {input_w})")
    print(f"Model input size after pad: ({padded_h}, {padded_w})")
    print(f"Output HSI shape: {shape}")  # (H, W, 31)
    print(f"Saved cube: {output_npy_path}")
    print(f"Saved preview: {output_preview_path}")


if __name__ == "__main__":
    main()