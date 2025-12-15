import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from TraditionalClassifier import TraditionalClassifier
def evaluate_classifier(classifier, test_root):
    y_true = []
    y_pred = []

    for class_name in os.listdir(test_root):
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(class_dir, filename)
            pred = classifier.classify(img_path)

            y_true.append(class_name)
            y_pred.append(pred)

    # Metrikleri hesapla
    print("\n🔍 Classification Report:\n")
    print(classification_report(y_true, y_pred))

    labels = sorted(set(y_true + y_pred))  # Hem doğru hem tahmin edilen tüm sınıflar

    print("\n📊 Confusion Matrix:\n")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                    index=labels,
                    columns=labels))

    print("\n🎯 Accuracy:", accuracy_score(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred)

import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Sınıf')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


classifier = TraditionalClassifier()
evaluate_classifier(classifier, test_root="Datasets/")
