import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

def get_chestxray_dataloaders(data_root, batch_size=32, val_split=0.1):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    full_train_dataset = ImageFolder(train_dir, transform=train_transform)
    
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_test_transform
    
    test_dataset = ImageFolder(test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(full_train_dataset.classes)
    
    return train_loader, val_loader, test_loader, num_classes
