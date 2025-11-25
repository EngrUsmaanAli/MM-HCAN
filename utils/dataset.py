import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class TemporalSpectralDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 transform=None,
                 target_transform=None,
                 sequence_length: int = 1024): 
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.sequence_length = sequence_length
        
        self.classes = sorted(os.listdir(os.path.join(root_dir, "spectral_features")))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Store raw temporal signals
        self.temporal_data = [] 
        
        for cls in self.classes:
            cls_csv = os.path.join(root_dir, "temporal_features", cls, "output1.csv")
            df = pd.read_csv(cls_csv, header=None if cls == "Healthy" else 0, dtype=np.float32)
            arr = df.values 
            self.temporal_data.append(arr)

        self.items = []      
        for cls in self.classes:
            class_idx = self.class_to_idx[cls]
            cls_img_dir = os.path.join(root_dir, "spectral_features", cls)
            img_files = sorted(f for f in os.listdir(cls_img_dir) if f.endswith(".png"))

            num_signals = self.temporal_data[class_idx].shape[1] 
            
            for col_idx, fname in enumerate(img_files):
                if col_idx >= num_signals: break 
                img_path = os.path.join(cls_img_dir, fname)
                self.items.append((img_path, class_idx, col_idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, class_idx, col_idx = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        raw_signal = self.temporal_data[class_idx][:, col_idx]
        
        if len(raw_signal) > self.sequence_length:
            raw_signal = raw_signal[:self.sequence_length]
        elif len(raw_signal) < self.sequence_length:
            padding = np.zeros(self.sequence_length - len(raw_signal), dtype=np.float32)
            raw_signal = np.concatenate([raw_signal, padding])
        mu = np.mean(raw_signal)
        sigma = np.std(raw_signal) + 1e-8
        norm_signal = (raw_signal - mu) / sigma
        
        signal_tensor = torch.from_numpy(norm_signal).float() 

        label = class_idx
        if self.target_transform:
            label = self.target_transform(label)

        return signal_tensor, img, label
