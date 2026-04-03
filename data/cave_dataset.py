import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.transform import resize
import re

class CAVEDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, train_ratio=0.8, spatial_size=(512,512)):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.spatial_size = spatial_size
        
        self.scenes = []
        for scene_dir in self.root_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            inner_dir = scene_dir / scene_dir.name
            if not inner_dir.exists():
                inner_dir = scene_dir  # если нет двойной вложенности
            
            rgb_candidates = list(inner_dir.glob("*RGB.bmp")) + list(inner_dir.glob("*RGB.BMP"))
            if not rgb_candidates:
                continue
            rgb_path = rgb_candidates[0]
            
            # Все PNG, исключая RGB и Thumbs
            all_png = [f for f in inner_dir.glob("*.png") if "RGB" not in f.name and "Thumbs" not in f.name]
            # Сортировка по номеру канала (извлекаем число из имени)
            def extract_number(p):
                nums = re.findall(r'\d+', p.stem)
                return int(nums[-1]) if nums else 0
            png_files = sorted(all_png, key=extract_number)
            
            if len(png_files) == 31:
                self.scenes.append((rgb_path, png_files))
            else:
                print(f"Warning: {inner_dir} имеет {len(png_files)} PNG, ожидается 31")
        
        if not self.scenes:
            raise FileNotFoundError(f"Не найдено сцен в {self.root_dir}")
        
        total = len(self.scenes)
        split = int(train_ratio * total)
        self.scenes = self.scenes[:split] if train else self.scenes[split:]
        print(f"Загружено {len(self.scenes)} сцен")
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        rgb_path, png_paths = self.scenes[idx]
        
        rgb = np.array(Image.open(rgb_path)).astype(np.float32)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        elif rgb.shape[-1] == 4:
            rgb = rgb[:,:,:3]
        
        hsi_channels = []
        for p in png_paths:
            img = np.array(Image.open(p)).astype(np.float32)
            if img.ndim == 3:
                img = img[:,:,0]
            hsi_channels.append(img)
        hsi = np.stack(hsi_channels, axis=-1)
        
        h, w, c = hsi.shape
        if (h, w) != self.spatial_size:
            hsi = resize(hsi, (*self.spatial_size, c), preserve_range=True)
            rgb = resize(rgb, (*self.spatial_size, 3), preserve_range=True)
        
        hsi = hsi / (hsi.max() + 1e-8)
        rgb = rgb / (rgb.max() + 1e-8)
        
        hsi = torch.from_numpy(hsi).permute(2,0,1).float()
        rgb = torch.from_numpy(rgb).permute(2,0,1).float()
        
        return rgb, hsi