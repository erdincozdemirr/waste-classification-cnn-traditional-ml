# ./CNN/main_1_extract_features.py

import sys
sys.path.insert(0, r"C:\Users\eozdemir.ext\Desktop\msc\waste-classification")

import os
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from lib.dataset_precreator import get_images_by_class, flatten_image_dict
from lib.feature_extractor import FeatureDataset, ImageFeatureExtractor


RESULTS_ROOT = "Src/Features"
os.makedirs(RESULTS_ROOT, exist_ok=True)


def extract_X_y(loader):
    X_list, y_list = [], []
    for batch_x, batch_y in tqdm(loader, desc="Extracting features"):
        # batch_x: (batch_size, feat_dim) — zaten tensor
        X_list.append(batch_x.numpy())
        y_list.append(batch_y.numpy())
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y


def main():
    print("=== FEATURE EXTRACTION BAŞLIYOR ===")

    # 1) Görselleri ve label'ları al
    print(">> [DATA] Görseller yükleniyor...")
    image_dict, class_names = get_images_by_class("Datasets", max_per_class=837)
    image_paths, labels = flatten_image_dict(image_dict, class_names)
    print(f">> Toplam görsel: {len(image_paths)} | Sınıflar: {class_names}")

    # 2) Train / Val / Test split
    print(">> [SPLIT] Train/Val/Test bölünüyor...")
    train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        valtest_paths,
        valtest_labels,
        test_size=0.5,
        stratify=valtest_labels,
        random_state=42
    )

    print(f">> Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    # 3) Extractor & Dataset & DataLoader
    print(">> [FEATURE] ImageFeatureExtractor oluşturuluyor...")
    extractor = ImageFeatureExtractor()

    train_dataset = FeatureDataset(train_paths, train_labels, extractor)
    val_dataset   = FeatureDataset(val_paths,   val_labels,   extractor)
    test_dataset  = FeatureDataset(test_paths,  test_labels,  extractor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # 4) Feature çıkarma
    print(">> [FEATURE] Train set feature çıkarılıyor...")
    X_train, y_train = extract_X_y(train_loader)

    print(">> [FEATURE] Val set feature çıkarılıyor...")
    X_val, y_val = extract_X_y(val_loader)

    print(">> [FEATURE] Test set feature çıkarılıyor...")
    X_test, y_test = extract_X_y(test_loader)

    # 5) Kaydet
    np.save(os.path.join(RESULTS_ROOT, "X_train.npy"), X_train)
    np.save(os.path.join(RESULTS_ROOT, "y_train.npy"), y_train)
    np.save(os.path.join(RESULTS_ROOT, "X_val.npy"),   X_val)
    np.save(os.path.join(RESULTS_ROOT, "y_val.npy"),   y_val)
    np.save(os.path.join(RESULTS_ROOT, "X_test.npy"),  X_test)
    np.save(os.path.join(RESULTS_ROOT, "y_test.npy"),  y_test)

    print("=== FEATURE EXTRACTION TAMAMLANDI ===")
    print(f">> Kaydedilen dizin: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
