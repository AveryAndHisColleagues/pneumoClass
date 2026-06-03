import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

def evaluate_model(model, test_loader, device, model_name):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)

    sensitivity = recall_score(all_labels, all_preds, pos_label=1)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    precision = precision_score(all_labels, all_preds, pos_label=1)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    result = {
        "Model": model_name,
        "Accuracy": acc,
        "AUC": auc_score,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "fpr": fpr,
        "tpr": tpr
    }

    return result

def plot_roc_curves(results, save_path='roc_comparison.png'):
    plt.figure(figsize=(10, 8))

    for result in results:
        plt.plot(
            result['fpr'], 
            result['tpr'], 
            label=f"{result['Model']} (AUC = {result['AUC']:.4f})"
        )

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()

def print_results(results):
    df = pd.DataFrame(results)
    df = df.drop(columns=['fpr', 'tpr'])
    print("\nModel Comparison Results:")
    print(df.to_string(index=False))
    return df

def save_results(results, filename='results.csv'):
    df = pd.DataFrame(results)
    df = df.drop(columns=['fpr', 'tpr'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
