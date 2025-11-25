import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import TemporalSpectralDataset

def get_dataloaders(root_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),        
    ])

    full_dataset = TemporalSpectralDataset(root_dir, transform=transform)

    indices = torch.randperm(len(full_dataset)).tolist()
    full_dataset = torch.utils.data.Subset(full_dataset, indices)

    total = len(full_dataset)
    train_size = int(0.8 * total)
    test_size = total - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    ft_size = int(0.75 * train_size) 
    val_size = train_size - ft_size
    ft_ds, val_ds = random_split(train_ds, [ft_size, val_size])

    return (
        DataLoader(ft_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )
