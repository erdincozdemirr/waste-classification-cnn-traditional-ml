# CNN/main_5_svm.py

import sys
sys.path.insert(0, r"C:\Users\eozdemir.ext\Desktop\msc\waste-classification")

import os
import time
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.svm import SVC
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


def decision_scores_to_proba(scores: np.ndarray) -> np.ndarray:
    """
    SVC(probability=False) kullandığımız için decision_function çıktısını
    softmax ile pseudo-olasılığa çeviriyoruz.
    """
    scores = np.asarray(scores)

    # Binary case: (n_samples,) → (n_samples, 2)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T

    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return proba


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print("=== SVM (svm_main) PIPELINE BAŞLIYOR ===")

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
    # SVM MODEL KONFİGÜRASYONLARI
    # Her biri farklı C / max_iter kombinasyonları ile
    # hız – karar sınırı esnekliği – aşırı öğrenme dengesini test eder.
    #
    # Notlar:
    #  - C küçük → karar sınırı daha geniş → daha fazla düzenleme → daha hızlı ama biraz düşük performans.
    #  - C büyük → karar sınırı daha dar → daha esnek → potansiyel olarak daha iyi accuracy ama yavaş.
    #  - max_iter arttıkça optimizasyon daha uzun sürer ama daha doğru çözüme yaklaşabilir.
    #
    # Amaç: linear kernel + farklı kapasite ayarlarıyla 4 hafif SVM varyasyonunu karşılaştırmak.
    # ---------------------------------------------------
    configs = [

        # -------------------------------------------------------
        # Model 1 — "Hafif düzenlenmiş, en hızlı model"
        # C = 0.1 → büyük bir margin, daha fazla regularization.
        # Hızlı eğitilir, genellemesi stabil olur ama accuracy düşük olabilir.
        # -------------------------------------------------------
        dict(
            name="svm_model_1",
            C=0.1,
            kernel="linear",
            tol=1e-2,
            max_iter=800,
            decision_function_shape="ovr",
            probability=False,
        ),

        # -------------------------------------------------------
        # Model 2 — "Dengeli model (default SVM hissiyatı)"
        # C = 1.0 → regularization ve esneklik dengede.
        # Genelde baseline olarak beklenen performansı verir.
        # -------------------------------------------------------
        dict(
            name="svm_model_2",
            C=1.0,
            kernel="linear",
            tol=1e-2,
            max_iter=1000,
            decision_function_shape="ovr",
            probability=False,
        ),

        # -------------------------------------------------------
        # Model 3 — "Daha agresif model"
        # C = 10 → daha dar margin, daha az regularization.
        # Hard-margin varianta yaklaşır, karmaşık sınırlarda daha iyi accuracy verebilir.
        # Ancak overfit riski ve eğitim süresi artar.
        # -------------------------------------------------------
        dict(
            name="svm_model_3",
            C=10.0,
            kernel="linear",
            tol=1e-2,
            max_iter=1000,
            decision_function_shape="ovr",
            probability=False,
        ),

        # -------------------------------------------------------
        # Model 4 — "İleri iterasyon, ince ayarlı"
        # C = 1.0 (Model 2 gibi), fakat:
        # tol = 5e-3 → daha sıkı çözüm
        # max_iter = 1500 → daha uzun optimizasyon
        # Yavaş ama daha temiz karar sınırı kurabilir.
        # -------------------------------------------------------
        dict(
            name="svm_model_4",
            C=1.0,
            kernel="linear",
            tol=5e-3,
            max_iter=1500,
            decision_function_shape="ovr",
            probability=False,
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

        # SVC parametrelerini hazırla
        params = cfg.copy()
        params.pop("name")

        svm_clf = SVC(**params)

        # Eğitim
        print(">> [SVM] Eğitim başlıyor...")
        start = time.time()
        svm_clf.fit(X_train, y_train)
        duration = time.time() - start
        print(f">> [SVM] Eğitim süresi: {duration:.2f} sn")

        # Modeli kaydet
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        joblib.dump(svm_clf, model_path)
        print(f">> Model kaydedildi: {model_path}")

        # Tahminler
        print(">> [SVM] Train/Val/Test tahminleri hesaplanıyor...")
        y_train_pred = svm_clf.predict(X_train)
        y_val_pred   = svm_clf.predict(X_val)
        y_test_pred  = svm_clf.predict(X_test)

        # decision_function → pseudo-proba
        train_scores = svm_clf.decision_function(X_train)
        val_scores   = svm_clf.decision_function(X_val)
        test_scores  = svm_clf.decision_function(X_test)

        y_train_proba = decision_scores_to_proba(train_scores)
        y_val_proba   = decision_scores_to_proba(val_scores)
        y_test_proba  = decision_scores_to_proba(test_scores)

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
            os.path.join(model_result_dir, "confusion_val.png"),
        )

        save_confusion_matrix(
            test_m["confusion_matrix"],
            class_names,
            f"{model_name} - Test CM",
            os.path.join(model_result_dir, "confusion_test.png"),
        )

        save_roc(
            y_val,
            y_val_proba,
            class_names,
            f"{model_name} - Validation ROC",
            os.path.join(model_result_dir, "roc_val.png"),
        )

        save_roc(
            y_test,
            y_test_proba,
            class_names,
            f"{model_name} - Test ROC",
            os.path.join(model_result_dir, "roc_test.png"),
        )

        print(f">> {model_name} için tüm sonuçlar kaydedildi: {model_result_dir}")

    print("\n=== TÜM SVM MODELLERİ TAMAMLANDI ===")


if __name__ == "__main__":
    main()
