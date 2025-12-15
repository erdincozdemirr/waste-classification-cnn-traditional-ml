# CNN/main_4_xgboost.py

import sys
sys.path.insert(0, r"C:\Users\eozdemir.ext\Desktop\msc\waste-classification")

import os
import time
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

from lib.dataset_precreator import get_images_by_class
from lib.metrics import calculate_metrics


# -------------------------------------------------------
# ANA KLASÖRLER
# -------------------------------------------------------

RESULTS_ROOT = "Traditional/Results"
MODELS_DIR = "Traditional/Model"

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# -------------------------------------------------------
# HELPER FONKSİYONLAR
# -------------------------------------------------------

def save_confusion_matrix(cm, class_names, title, save_path):
    """Confusion matrix'i PNG olarak kaydeder."""
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_roc(y_true, y_proba, class_names, title, save_path):
    """ROC eğrilerini PNG olarak kaydeder (multi-class)."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure(figsize=(8, 6))

    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics_txt(save_dir, train_m, val_m, test_m):
    """Train/Val/Test metriklerini txt dosyasına kaydeder."""
    path = os.path.join(save_dir, "metrics.txt")

    def fmt_block(title, m):
        return (
            f"===== {title} METRİKLERİ =====\n"
            f"Accuracy   : {m['accuracy']:.4f}\n"
            f"F1 Score   : {m['f1_score']:.4f}\n"
            f"Sensitivity: {m['sensitivity']:.4f}\n"
            f"Specificity: {m['specificity']:.4f}\n"
            f"AUC        : {m['auc']:.4f}\n"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(fmt_block("TRAIN", train_m))
        f.write("\n")
        f.write(fmt_block("VAL", val_m))
        f.write("\n")
        f.write(fmt_block("TEST", test_m))

    print(f">> Metrikler txt olarak kaydedildi: {path}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print("=== XGBOOST PIPELINE BAŞLIYOR ===")

    # Sınıf isimleri
    print(">> [DATA] Sınıf isimleri okunuyor...")
    _, class_names = get_images_by_class("Datasets", max_per_class=1)
    print(f">> [DATA] Sınıflar: {class_names}")

    # Feature'lar
    print(">> [DATA] Feature .npy dosyaları yükleniyor...")
    X_train = np.load("Src/Features/X_train.npy")
    y_train = np.load("Src/Features/y_train.npy")
    X_val   = np.load("Src/Features/X_val.npy")
    y_val   = np.load("Src/Features/y_val.npy")
    X_test  = np.load("Src/Features/X_test.npy")
    y_test  = np.load("Src/Features/y_test.npy")

    print(f">> [DATA] X_train shape: {X_train.shape}")
    print(f">> [DATA] X_val   shape: {X_val.shape}")
    print(f">> [DATA] X_test  shape: {X_test.shape}")

    # ---------------------------------------------------
    # MODEL KONFİGÜRASYONLARI
    # ---------------------------------------------------
    configs = [

        #4 tane net senaryo var:
        #Model 1: “çok ağaç & yüksek lr”
        #Model 2: “çok ağaç & düşük lr”
        #Model 3: “az ağaç & yüksek lr”
        #Model 4: “az ağaç & düşük lr”

         # Model 1: Referans model.
        # Orta sayıda ağaç ve daha yüksek öğrenme oranı ile hızlı eğitim.
        dict(
            name="xgboost_model_1",
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.3,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        ),

        # Model 2: Aynı kapasite, daha küçük öğrenme oranı.
        # Adımlar küçülür, daha stabil ama biraz daha yavaş bir yapı elde edilir.
        dict(
            name="xgboost_model_2",
            n_estimators=40,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.3,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        ),

        # Model 3: Daha az ağaç ile daha hafif bir model.
        # Eğitim süresini azaltırken performanstaki değişimi görmek için kullanılabilir.
        dict(
            name="xgboost_model_3",
            n_estimators=20,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.3,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        ),

        # Model 4: Hem ağaç sayısı hem öğrenme oranı azaltılmış yapı.
        # En hafif, en temkinli yapı; hız ve genelleme dengesi açısından referans noktası olabilir.
        dict(
            name="xgboost_model_4",
            n_estimators=20,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.3,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        ),
    ]

    # ---------------------------------------------------
    # TRAIN LOOP
    # ---------------------------------------------------
    for cfg in configs:

        model_name = cfg["name"]
        model_result_dir = os.path.join(RESULTS_ROOT, model_name)
        os.makedirs(model_result_dir, exist_ok=True)

        print("\n====================================")
        print(f"EĞİTİM BAŞLIYOR: {model_name}")
        print("====================================\n")

        params = cfg.copy()
        params.pop("name")

        xgb_clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(class_names),
            eval_metric="mlogloss",
            **params
        )

        # Eğitim
        start = time.time()
        xgb_clf.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        duration = time.time() - start
        print(f">> Eğitim süresi: {duration:.2f} sn")

        # Modeli kaydet
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        joblib.dump(xgb_clf, model_path)
        print(f">> Model kaydedildi: {model_path}")

        # Tahminler
        print(">> [XGB] Train/Val/Test tahminleri hesaplanıyor...")
        y_train_pred  = xgb_clf.predict(X_train)
        y_val_pred    = xgb_clf.predict(X_val)
        y_test_pred   = xgb_clf.predict(X_test)

        y_train_proba = xgb_clf.predict_proba(X_train)
        y_val_proba   = xgb_clf.predict_proba(X_val)
        y_test_proba  = xgb_clf.predict_proba(X_test)

        # METRİKLER
        print(">> [METRICS] Metrikler hesaplanıyor...")
        train_m = calculate_metrics(y_train, y_train_pred, y_train_proba)
        val_m   = calculate_metrics(y_val,   y_val_pred,   y_val_proba)
        test_m  = calculate_metrics(y_test,  y_test_pred,  y_test_proba)

        print(f"\n>>> {model_name} - Validation Accuracy: {val_m['accuracy']:.4f}")
        print(f">>> {model_name} - Test Accuracy      : {test_m['accuracy']:.4f}")

        # TXT kaydet
        save_metrics_txt(model_result_dir, train_m, val_m, test_m)

        # Confusion Matrix & ROC PNG
        print(">> [PLOT] Confusion Matrix & ROC PNG kaydediliyor...")

        save_confusion_matrix(
            val_m["confusion_matrix"],
            class_names,
            f"{model_name} - Validation CM",
            os.path.join(model_result_dir, "confusion_val.png")
        )

        save_confusion_matrix(
            test_m["confusion_matrix"],
            class_names,
            f"{model_name} - Test CM",
            os.path.join(model_result_dir, "confusion_test.png")
        )

        save_roc(
            y_val, y_val_proba, class_names,
            f"{model_name} - Validation ROC",
            os.path.join(model_result_dir, "roc_val.png")
        )

        save_roc(
            y_test, y_test_proba, class_names,
            f"{model_name} - Test ROC",
            os.path.join(model_result_dir, "roc_test.png")
        )

        print(f">> {model_name} için tüm sonuçlar kaydedildi: {model_result_dir}")

    print("\n=== TÜM XGBOOST MODELLERİ TAMAMLANDI ===")


if __name__ == "__main__":
    main()
