import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ForensicFingerprintDataset(Dataset):
    def __init__(self, csv_path: str, stage: str = "stage1", image_size: int = 256):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.stage = stage.lower()
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def _load_grayscale(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        target_img = self._load_grayscale(row["exemplar_path"])
        mask = (self._load_grayscale(row["mask_path"]) > 0.5).astype(np.float32)
        partial_img = target_img * mask
        return {
            "partial_img": torch.from_numpy(partial_img).unsqueeze(0),
            "target_img": torch.from_numpy(target_img).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "prompt": str(row["prompt"]),
        }

def create_dataloaders(csv_path: str, stage: str = "stage1", image_size: int = 256, batch_size: int = 8, val_split: float = 0.2, num_workers: int = 2):
    full_ds = ForensicFingerprintDataset(csv_path=csv_path, stage=stage, image_size=image_size)
    val_size = max(1, int(len(full_ds) * val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
