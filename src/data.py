# src/data.py
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

def get_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int, pin_memory: bool):
    """
    Returns (train_loader, val_loader).
    Transforms:
      train: RandomHorizontalFlip, ToTensor, Normalize mean=(0.5,0.5,0.5) std=(0.5,0.5,0.5)
      val:   ToTensor, Normalize same
    Images normalized to [-1, 1] range.
    """
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    val_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_val
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
