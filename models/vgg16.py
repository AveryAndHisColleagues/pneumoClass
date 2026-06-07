import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder # 为kaggle的chest x-ray数据集准备的，直接使用ImageFolder加载数据
# import medmnist
# from medmnist import INFO

# data_flag = "pneumoniamnist"
# info = INFO[data_flag]
# DataClass = getattr(medmnist, info["python_class"])
# num_classes = len(info["label"])

# data_root = r"../medmnist_data"

# train_transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),
#     transforms.RandomRotation(15),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])
train_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# train_dataset = DataClass(split="train", transform=train_transform, download=False, root=data_root)
# val_dataset = DataClass(split="val", transform=val_test_transform, download=False, root=data_root)
# test_dataset = DataClass(split="test", transform=val_test_transform, download=False, root=data_root)
train_root = "./chest_xray_split/train"
val_root = "./chest_xray_split/val"

# 官方测试集
test_root = "./chest_xray/chest_xray/test"
train_dataset = ImageFolder(train_root, transform=train_transform)
val_dataset = ImageFolder(val_root, transform=val_test_transform)
test_dataset = ImageFolder(test_root, transform=val_test_transform)

num_classes = len(train_dataset.classes)
print("类别：", train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 50
patience = 7
bad_epochs = 0
best_val_acc = 0.0

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = correct / total

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    print(
        f"Epoch {epoch+1}: "
        f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
        f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        bad_epochs = 0
        torch.save(model.state_dict(), "checkpoints/best_vgg16.pth")
        print(f"Saved best VGG16! Best Val Acc={best_val_acc:.4f}")
    else:
        bad_epochs += 1
        print(f"No improvement. bad_epochs={bad_epochs}/{patience}")

    if bad_epochs >= patience:
        print("Early stopping triggered!")
        break

model.load_state_dict(torch.load("checkpoints/best_vgg16.pth", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Final VGG16 Test Acc = {test_acc:.4f}")