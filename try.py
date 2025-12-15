import torch
import torchvision.transforms as transforms
from CNN.Models.alexnet_classifier import AlexNetClassifier
from lib.feature_extractor import ImageFeatureExtractor  # RFC için kullandığın class
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import numpy as np

# 🎯 Sınıf isimleri
classes = {'battery': 0, 'cardboard': 1, 'kıyafet': 2, 'metal': 3, 'paper': 4, 'plastic': 5}
idx_to_class = {v: k for k, v in classes.items()}

# 📁 Görsel yolu
# image_path = "Image/2.png"
# image_path = "Image/4.png"
# image_path = "Image/6.png"
# image_path = "Image/7.png"
# image_path = "Image/8.png"
# image_path = "Image/9.png"
# image_path = "Image/10.png"
# image_path = "Image/11.png"
# image_path = "Image/12.png"
image_path = "Image/13.png"
# image_path = "Image/14.png"
# image_path = "Image/15.png"
#image_path = "Image/16.png"


# 🧠 RFC, XGBoost, SVM modeli ve feature extractor
rfc = joblib.load("Traditional/Model/random_forest_model.pkl")
xgb = joblib.load("Traditional/Model/xgboost_model.pkl")   
svm = joblib.load("Traditional/Model/svm_model.pkl")

extractor = ImageFeatureExtractor(device="cpu")
features = extractor.extract(image_path).reshape(1, -1)

# 🔮 RFC tahmini
pred_rfc_idx = rfc.predict(features)[0]
pred_rfc_label = idx_to_class[pred_rfc_idx]

# ⚡ XGBoost tahmini
pred_xgb_idx = xgb.predict(features)[0]
pred_xgb_label = idx_to_class[pred_xgb_idx]

# 🧠 SVM tahmini
pred_svm_idx = svm.predict(features)[0]
pred_svm_label = idx_to_class[pred_svm_idx]

# 🧠 CNN modeli
device = "cpu"
cnn_model = AlexNetClassifier(num_classes=6, pretrained=True).to(device)
checkpoint = torch.load("CNN/Outputs/alexnet_model10.pth", map_location=device)
cnn_model.load_state_dict(checkpoint['model_state_dict'])
cnn_model.eval()

# 🔁 Görsel transform (CNN için)
image = Image.open(image_path).convert("RGB")
_image_for_gui = image.copy()  # matplotlib için
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image_tensor = cnn_transform(image).unsqueeze(0).to(device)

# 🎯 CNN tahmini
with torch.no_grad():
    output = cnn_model(image_tensor)
    pred_cnn_idx = torch.argmax(output, dim=1).item()
    pred_cnn_label = idx_to_class[pred_cnn_idx]

# 🖨️ Terminalde yazdır
print(f"📸 Görsel: {image_path}")
print(f"🔹 CNN Tahmini : {pred_cnn_label}")
print(f"🌲 RFC Tahmini : {pred_rfc_label}")
print(f"⚡ XGB Tahmini : {pred_xgb_label}")
print(f"🧠 SVM Tahmini : {pred_svm_label}")

# 🖼️ Beyaz pencere + tahminler
plt.figure(figsize=(4, 4))
plt.imshow(_image_for_gui)
plt.title(f"CNN: {pred_cnn_label} | RFC: {pred_rfc_label} | XGB: {pred_xgb_label} | SVM: {pred_svm_label}", fontsize=12)
plt.axis("off")
plt.tight_layout()
plt.show()
