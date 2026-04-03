import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset


class CAVEDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        split="train",
        train_ratio=0.8,
        spatial_size=(512, 512),
        patch_size=None,
        patches_per_scene=1,
        augment=True,
        normalization="fixed_255",
        split_scenes=None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.spatial_size = spatial_size
        self.patch_size = patch_size
        self.patches_per_scene = max(1, int(patches_per_scene))
        self.augment = augment and split == "train"
        self.normalization = normalization
        self.split_scenes = split_scenes or {}

        self.scenes = []
        for scene_dir in self.root_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            inner_dir = scene_dir / scene_dir.name
            if not inner_dir.exists():
                inner_dir = scene_dir

            rgb_candidates = list(inner_dir.glob("*RGB.bmp")) + list(inner_dir.glob("*RGB.BMP"))
            if not rgb_candidates:
                continue
            rgb_path = rgb_candidates[0]

            all_png = [f for f in inner_dir.glob("*.png") if "RGB" not in f.name and "Thumbs" not in f.name]

            def extract_number(path):
                nums = re.findall(r"\d+", path.stem)
                return int(nums[-1]) if nums else 0

            png_files = sorted(all_png, key=extract_number)

            if len(png_files) == 31:
                self.scenes.append((scene_dir.name.lower(), rgb_path, png_files))
            else:
                print(f"Warning: {inner_dir} has {len(png_files)} PNG files, expected 31")

        if not self.scenes:
            raise FileNotFoundError(f"No CAVE scenes found in {self.root_dir}")

        explicit_splits = any(self.split_scenes.get(name) for name in ("train", "val", "test"))
        if explicit_splits:
            split_names = {name.lower() for name in self.split_scenes.get(self.split, [])}
            if self.split == "train":
                excluded = {
                    name.lower()
                    for subset in ("val", "test")
                    for name in self.split_scenes.get(subset, [])
                }
                if split_names:
                    self.scenes = [scene for scene in self.scenes if scene[0] in split_names]
                else:
                    self.scenes = [scene for scene in self.scenes if scene[0] not in excluded]
            else:
                self.scenes = [scene for scene in self.scenes if scene[0] in split_names]
        else:
            total = len(self.scenes)
            split_idx = int(train_ratio * total)
            if self.split == "train":
                self.scenes = self.scenes[:split_idx]
            else:
                self.scenes = self.scenes[split_idx:]

        print(f"Loaded {len(self.scenes)} scenes for split '{self.split}'")

    def __len__(self):
        if self.patch_size is not None:
            return len(self.scenes) * self.patches_per_scene
        return len(self.scenes)

    def _normalize(self, rgb, hsi):
        if self.normalization == "fixed_255":
            return rgb / 255.0, hsi / 255.0
        if self.normalization == "by_dtype":
            rgb_scale = self.rgb_scale if self.rgb_scale is not None else 255.0
            hsi_scale = self.hsi_scale if self.hsi_scale is not None else 65535.0
            return rgb / rgb_scale, hsi / hsi_scale
        if self.normalization == "per_image_max":
            return rgb / (rgb.max() + 1e-8), hsi / (hsi.max() + 1e-8)
        raise ValueError(f"Unknown normalization mode: {self.normalization}")

    def _crop_patch(self, rgb, hsi):
        patch_h, patch_w = self.patch_size
        h, w = hsi.shape[:2]

        if h < patch_h or w < patch_w:
            target_h = max(h, patch_h)
            target_w = max(w, patch_w)
            hsi = resize(hsi, (target_h, target_w, hsi.shape[-1]), preserve_range=True)
            rgb = resize(rgb, (target_h, target_w, rgb.shape[-1]), preserve_range=True)
            h, w = target_h, target_w

        if self.augment:
            top = random.randint(0, h - patch_h)
            left = random.randint(0, w - patch_w)
        else:
            top = max(0, (h - patch_h) // 2)
            left = max(0, (w - patch_w) // 2)

        rgb = rgb[top:top + patch_h, left:left + patch_w]
        hsi = hsi[top:top + patch_h, left:left + patch_w]

        if self.augment:
            if random.random() < 0.5:
                rgb = np.flip(rgb, axis=0).copy()
                hsi = np.flip(hsi, axis=0).copy()
            if random.random() < 0.5:
                rgb = np.flip(rgb, axis=1).copy()
                hsi = np.flip(hsi, axis=1).copy()
            rotations = random.randint(0, 3)
            if rotations:
                rgb = np.rot90(rgb, rotations, axes=(0, 1)).copy()
                hsi = np.rot90(hsi, rotations, axes=(0, 1)).copy()

        return rgb, hsi

    def __getitem__(self, idx):
        if self.patch_size is not None:
            idx = idx % len(self.scenes)

        _, rgb_path, png_paths = self.scenes[idx]

        rgb_raw = np.array(Image.open(rgb_path))
        self.rgb_scale = float(np.iinfo(rgb_raw.dtype).max) if np.issubdtype(rgb_raw.dtype, np.integer) else 255.0
        rgb = rgb_raw.astype(np.float32)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        elif rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]

        hsi_channels = []
        hsi_scale = None
        for path in png_paths:
            img_raw = np.array(Image.open(path))
            if hsi_scale is None:
                hsi_scale = float(np.iinfo(img_raw.dtype).max) if np.issubdtype(img_raw.dtype, np.integer) else 255.0
            img = img_raw.astype(np.float32)
            if img.ndim == 3:
                img = img[:, :, 0]
            hsi_channels.append(img)
        self.hsi_scale = hsi_scale
        hsi = np.stack(hsi_channels, axis=-1)

        h, w, c = hsi.shape
        if (h, w) != self.spatial_size:
            hsi = resize(hsi, (*self.spatial_size, c), preserve_range=True)
            rgb = resize(rgb, (*self.spatial_size, 3), preserve_range=True)

        if self.patch_size is not None:
            rgb, hsi = self._crop_patch(rgb, hsi)

        rgb, hsi = self._normalize(rgb, hsi)

        hsi = torch.from_numpy(hsi).permute(2, 0, 1).float()
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()

        return rgb, hsi
