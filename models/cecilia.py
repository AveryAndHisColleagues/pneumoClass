import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
import os

# ----------------------------
# 数据增强和数据集
# ----------------------------
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_flag = "pneumoniamnist"
info = INFO[data_flag]
DataClass = getattr(medmnist, info["python_class"])
num_classes = len(info["label"])
in_channels = info["n_channels"]

data_root = r"../medmnist_data"

train_dataset = DataClass(split="train", transform=train_transform, download=False, root=data_root)
val_dataset   = DataClass(split="val", transform=val_test_transform, download=False, root=data_root)
test_dataset  = DataClass(split="test", transform=val_test_transform, download=False, root=data_root)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 模型定义
# ----------------------------
class Cecilia(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

model = Cecilia().to(device)

# ----------------------------
# 训练配置
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
epochs = 50
patience = 7
bad_epochs = 0
best_val_acc = 0.0

os.makedirs("checkpoints", exist_ok=True)

# ----------------------------
# 训练循环
# ----------------------------
for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_train_loss = train_loss / len(train_loader)

    # 验证集
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        bad_epochs = 0
        torch.save(model.state_dict(), "checkpoints/best_cecilia.pth")
        print(f"Saved best model! Best Val Acc={best_val_acc:.4f}")
    else:
        bad_epochs += 1
        print(f"No improvement. bad_epochs={bad_epochs}/{patience}")
        if bad_epochs >= patience:
            print("Early stopping triggered!")
            break

# ----------------------------
# 测试集
# ----------------------------
model.load_state_dict(torch.load("checkpoints/best_cecilia.pth"))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Final Test Acc = {test_acc:.4f}")