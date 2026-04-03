from pathlib import Path
from data.cave_dataset import CAVEDataset

dataset_path = Path("/mnt/c/Users/User/repo/datasets/CAVE")
dataset = CAVEDataset(dataset_path, train=True, spatial_size=(256,256))
rgb, hsi = dataset[0]
print(f"RGB shape: {rgb.shape}, HSI shape: {hsi.shape}")