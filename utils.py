import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO


def get_dataloaders(batch_size=128):
    data_flag = "pneumoniamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = DataClass(split="train", transform=transform, download=False, root="./medmnist_data")
    val_dataset = DataClass(split="val", transform=transform, download=False, root="./medmnist_data")
    test_dataset = DataClass(split="test", transform=transform, download=False, root="./medmnist_data")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader