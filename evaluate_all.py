import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


def evaluate_model(model, test_loader, device, model_name):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    # pneumonia 作为阳性类
    sensitivity = recall_score(all_labels, all_preds, pos_label=1)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    precision = precision_score(all_labels, all_preds, pos_label=1)

    result = {
        "Model": model_name,
        "Accuracy": acc,
        "AUC": auc,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    return result