import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    f1_score, recall_score, precision_score, roc_curve, auc
)


def calculate_metrics(y_true, y_pred, y_proba, average='macro'):
    cm = confusion_matrix(y_true, y_pred)

    TN = np.diag(cm)
    FP = cm.sum(axis=0) - TN
    FN = cm.sum(axis=1) - TN
    TP = TN

    specificity = np.mean(TN / (TN + FP + 1e-8))
    sensitivity = recall_score(y_true, y_pred, average=average)
    precision   = precision_score(y_true, y_pred, average=average, zero_division=0)
    f1          = f1_score(y_true, y_pred, average=average)
    acc         = (y_true == y_pred).sum() / len(y_true)
    auc_score   = None
    
    try:
        auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
    except ValueError:
        auc_score = 0.0  # veya float('nan')

    return {
        'accuracy': acc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'precision': precision,
        'f1_score': f1,
        'auc': auc_score,
        'confusion_matrix': cm
    }


def evaluate_full(model, dataloader, device):
    import torch
    model.eval()
    y_true, y_pred, y_proba = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
            y_proba.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_proba)


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc(y_true, y_proba, class_names):
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = len(class_names)
    y_onehot = np.zeros((len(y_true), n_classes))
    y_onehot[np.arange(len(y_true)), y_true] = 1

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_metric_curves(metrics_log, keys=None):
    if keys is None:
        keys = ['loss', 'accuracy', 'f1_score', 'specificity', 'sensitivity']

    for key in keys:
        plt.figure()
        plt.plot(metrics_log[f'train_{key}'], label='Train')
        plt.plot(metrics_log[f'val_{key}'], label='Val')
        plt.title(key.upper())
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
def plot_learning_rate_curve(metrics_log):
    lr_values = metrics_log.get('learning_rate', [])
    if not lr_values:
        print("❗ Learning rate verisi yok.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(lr_values, marker='o')
    plt.title("Learning Rate Değişimi")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
