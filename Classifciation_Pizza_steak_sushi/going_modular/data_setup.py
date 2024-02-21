"""
contains funcitonality for creating pytorch data loaders for image
classification data.
"""

import os
import torch
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader


print(torch.__version__)

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int, 
        num_workers: int = NUM_WORKERS
):
    train_data = datasets.ImageFolder(train_dir,transform=transform)
    test_data = datasets.ImageFolder(test_dir,transform=transform)

    class_names = train_data.classes
    
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle = True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle = False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names