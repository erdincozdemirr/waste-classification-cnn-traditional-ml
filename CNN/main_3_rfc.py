import sys

# Proje kökünü (waste-classification) sys.path'e ELLE ekliyoruz
sys.path.insert(0, r"C:\Users\eozdemir.ext\Desktop\msc\waste-classification")

from lib.dataset_precreator import get_images_by_class, flatten_image_dict
image_dict, class_names = get_images_by_class("Datasets", max_per_class=837)
image_paths, labels = flatten_image_dict(image_dict, class_names)
from sklearn.model_selection import train_test_split
from lib.feature_extractor import FeatureDataset, ImageFeatureExtractor
from torch.utils.data import DataLoader
import joblib
import sys

# extractor = ImageFeatureExtractor()
# train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
#     image_paths, labels,
#     test_size=0.2, stratify=labels, random_state=42
# )
# val_paths, test_paths, val_labels, test_labels = train_test_split(
#     valtest_paths, valtest_labels,
#     test_size=0.5, stratify=valtest_labels, random_state=42
# )
# train_dataset = FeatureDataset(train_paths, train_labels, extractor)
# val_dataset   = FeatureDataset(val_paths, val_labels, extractor)
# test_dataset  = FeatureDataset(test_paths, test_labels, extractor)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=32)
# test_loader  = DataLoader(test_dataset, batch_size=32)
# import numpy as np
# from tqdm import tqdm
# def extract_X_y(loader):
#     X, y = [], []
#     for batch_x, batch_y in tqdm(loader, desc="Extracting features"):
#         X.append(batch_x.numpy())
#         y.append(batch_y.numpy())
#     return np.vstack(X), np.hstack(y)
# X_train, y_train = extract_X_y(train_loader)
# X_val, y_val     = extract_X_y(val_loader)
# X_test, y_test   = extract_X_y(test_loader)
# np.save("Src/Features/X_train.npy", X_train)
# np.save("Src/Features/y_train.npy", y_train)
# np.save("Src/Features/X_val.npy", X_val)
# np.save("Src/Features/y_val.npy", y_val)
# np.save("Src/Features/X_test.npy", X_test)
# np.save("Src/Features/y_test.npy", y_test)
# 
# sys.exit()


from sklearn.ensemble import RandomForestClassifier
from lib.metrics import calculate_metrics, plot_confusion_matrix, plot_roc
import numpy as np

# 0. class names
class_names = ['battery', 'cardboard', 'clothes', 'metal', 'paper', 'plastic']

# 1. load your saved .npy files
X_train = np.load("Src/Features/X_train.npy")
y_train = np.load("Src/Features/y_train.npy")
X_val = np.load("Src/Features/X_val.npy")
y_val = np.load("Src/Features/y_val.npy")
X_test = np.load("Src/Features/X_test.npy")
y_test = np.load("Src/Features/y_test.npy")
print("X_train shape:", X_train.shape)
sys.exit()
# 2. model eğit
clf = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, "Traditional/Model/random_forest_model.pkl")

# 3. predict
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

y_train_proba = clf.predict_proba(X_train)
y_val_proba = clf.predict_proba(X_val)
y_test_proba = clf.predict_proba(X_test)

# 4. metrik hesapla
train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

# 5. yazdır
def print_metrics(name, m):
    print(f"\n📊 [{name.upper()}]")
    print(f"ACC : {m['accuracy']:.4f}")
    print(f"LOSS: {1 - m['accuracy']:.4f}")
    print(f"F1  : {m['f1_score']:.4f}")
    print(f"SENS: {m['sensitivity']:.4f}")
    print(f"SPEC: {m['specificity']:.4f}")
    print(f"AUC : {m['auc']:.4f}")

print_metrics("Train", train_metrics)
print_metrics("Val", val_metrics)
print_metrics("Test", test_metrics)

# 6. confusion matrix ve roc çiz
plot_confusion_matrix(val_metrics['confusion_matrix'], class_names, title="Validation Confusion Matrix")
plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, title="Test Confusion Matrix")

plot_roc(y_val, y_val_proba, class_names)
plot_roc(y_test, y_test_proba, class_names)

