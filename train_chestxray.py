import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datasets.chestxray import get_chestxray_dataloaders
from utils.roc_utils import plot_roc_curves, evaluate_with_roc

class MiniVGG(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Cecilia(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device, model_name):
    best_val_acc = 0.0
    bad_epochs = 0
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
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
        
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), f"checkpoints/best_{model_name}.pth")
            print(f"Saved best {model_name}! Best Val Acc={best_val_acc:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered!")
                break
    
    return model

def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    test_acc = correct / total
    return test_acc

def main():
    data_root = "../chest_xray"
    batch_size = 32
    epochs = 50
    patience = 7
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader, num_classes = get_chestxray_dataloaders(data_root, batch_size)
    print(f"Number of classes: {num_classes}")
    
    models_dict = {
        'cecilia': Cecilia(num_classes=num_classes),
        'minivgg': MiniVGG(num_classes=num_classes),
        'vgg16': models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        'resnet18': models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    }
    
    models_dict['vgg16'].classifier[6] = nn.Linear(4096, num_classes)
    models_dict['resnet18'].fc = nn.Linear(models_dict['resnet18'].fc.in_features, num_classes)
    
    trained_models = []
    model_names = []
    
    for name, model in models_dict.items():
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        print(f"\nTraining {name}...")
        model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device, name)
        model.load_state_dict(torch.load(f"checkpoints/best_{name}.pth", map_location=device))
        
        test_acc = test_model(model, test_loader, device)
        print(f"Final {name} Test Acc = {test_acc:.4f}")
        
        trained_models.append(model)
        model_names.append(name)
    
    print("\nGenerating ROC curve comparison...")
    plot_roc_curves(trained_models, model_names, test_loader, device)
    
    print("\nEvaluation results:")
    for model, name in zip(trained_models, model_names):
        result = evaluate_with_roc(model, test_loader, device, name)
        print(f"{name}: Accuracy={result['accuracy']:.4f}, AUC={result['auc']:.4f}")

if __name__ == "__main__":
    main()
