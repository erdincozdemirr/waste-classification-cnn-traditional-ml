import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
pd.set_option("display.max_rows", None)      # Tüm satırlar
pd.set_option("display.max_columns", None)   # Tüm sütunlar
pd.set_option("display.width", None)         # Satır kesilmesin
pd.set_option("display.max_colwidth", None)  # Sütun içeriği tam görünsün
class TraditionalClassifier:

    def __init__(self, brightness_threshold=200):
        self.brightness_threshold = brightness_threshold

    def extract_features(self, img_path):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = {}

        # 1️⃣ RGB Histogram (16-bin)
        for i, color in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([img_rgb], [i], None, [16], [0, 256]).flatten()
            hist_norm = hist / hist.sum()
            for j in range(16):
                features[f'{color}_hist_{j}'] = hist_norm[j]

        # 2️⃣ LBP (Local Binary Pattern) Histogram
        lbp = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        for i in range(len(lbp_hist)):
            features[f'lbp_{i}'] = lbp_hist[i]

        # 3️⃣ Color Dominance
        R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        dominant_r = np.mean((R > G) & (R > B))
        dominant_g = np.mean((G > R) & (G > B))
        dominant_b = np.mean((B > R) & (B > G))
        features['dominant_r'] = dominant_r
        features['dominant_g'] = dominant_g
        features['dominant_b'] = dominant_b

        # 4️⃣ Shape Features (Contour + Circularity)
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularities = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                circularities.append(circularity)
        features['mean_circularity'] = np.mean(circularities) if circularities else 0
        features['num_contours'] = len(contours)

        # 5️⃣ Sharpness (Laplacian Variance)
        lap_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        features['laplacian_var'] = lap_var

        return features
    def classify(self, img_path):
        f = self.extract_features(img_path)

        # 🔋 BATTERY
        if (
            f["mean_circularity"] > 0.6 and
            f["laplacian_var"] > 2000 and
            f["dominant_b"] > 0.15 and
            f["num_contours"] < 50
        ):
            return "battery"

        # 📦 CARDBOARD
        elif (
            f["mean_circularity"] < 0.5 and
            f["laplacian_var"] < 1000 and
            f["num_contours"] < 90 and
            f["dominant_r"] > 0.05
        ):
            return "cardboard"

        # 👕 CLOTHES
        elif (
            f["num_contours"] > 180 and
            f["laplacian_var"] < 1500 and
            f["dominant_b"] > 0.2 and
            f["mean_circularity"] < 0.5
        ):
            return "clothes"

        # 🪙 METAL
        elif (
            f["laplacian_var"] > 1500 and
            f["dominant_b"] > 0.25 and
            f["num_contours"] < 100 and
            f["mean_circularity"] > 0.5
        ):
            return "metal"

        # 📄 PAPER
        elif (
            f["laplacian_var"] > 3000 and
            f["num_contours"] > 70 and
            f["mean_circularity"] < 0.6 and
            f["dominant_r"] < 0.1
        ):
            return "paper"

        # 🧴 PLASTIC
        elif (
            f["laplacian_var"] < 1000 and
            f["dominant_r"] > 0.2 and
            f["mean_circularity"] > 0.5
        ):
            return "plastic"

        # 🔁 fallback (isteğe bağlı)
        else:
            return "plastic"
def extract_features_in_memory(dataset_dir):
    classifier = TraditionalClassifier()
    all_data = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file in tqdm(os.listdir(class_dir), desc=class_name):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, file)
            try:
                features = classifier.extract_features(img_path)
                features["class"] = class_name
                all_data.append(features)
            except Exception as e:
                print(f"⚠️ Hata ({img_path}): {e}")

    df = pd.DataFrame(all_data)


    # Özet çıkar (sadece bellekte)
    summary = df.groupby("class").agg(["mean", "std"]).round(2)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    return df, summary
    
if __name__ == "__main__":
    df, summary = extract_features_in_memory("Datasets")
    summary.to_csv("features_summaryV3.csv")
    
        # classifier = TraditionalClassifier()
        
        
        # # classifier = TraditionalClassifier()
        # # print(classifier.classify("Datasets/battery/battery91.jpg"))
        
        
        
        
        
        
        # DATASET_DIR = "Datasets"
        # BRIGHTNESS_THRESHOLD = 200

        # data = []

        # for class_name in os.listdir(DATASET_DIR):
        #     class_dir = os.path.join(DATASET_DIR, class_name)
        #     if not os.path.isdir(class_dir):
        #         continue

        #     print(f"📂 Sınıf: {class_name}")
        
        #     for filename in tqdm(os.listdir(class_dir)):
        #         if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        #             continue
                
        #         img_path = os.path.join(class_dir, filename)
        #         img = cv2.imread(img_path)
        #         if img is None:
        #             continue
                
        #         img = cv2.resize(img, (224, 224))
        #         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #         h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        #         mean_hue = np.mean(h)
        #         mean_saturation = np.mean(s)
        #         mean_value = np.mean(v)

        #         # Parlaklık oranı (value > 200 olan pikseller)
        #         bright_ratio = np.sum(v > BRIGHTNESS_THRESHOLD) / v.size

        #         data.append({
        #             "class": class_name,
        #             "mean_hue": mean_hue,
        #             "mean_saturation": mean_saturation,
        #             "mean_value": mean_value,
        #             "bright_ratio": bright_ratio
        #         })

        # # DataFrame'e aktar
        # df = pd.DataFrame(data)
        # print("\n📊 Örnek veri:")
        # print(df.head())

        # # CSV olarak kaydet (isteğe bağlı)
        # df.to_csv("feature_summary.csv", index=False)
        # print("\n✅ Özellikler feature_summary.csv dosyasına kaydedildi.")

# CSV'den oku
    # df = pd.read_csv(os.getcwd() + "/Traditional/feature_summary.csv")

    # # Her sınıf için mean ve std hesapla
    # summary = df.groupby("class").agg(["mean", "std"]).round(2)

    # # Sütun isimlerini sadeleştir (çok katmanlı olmasın)
    # summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    # # Tabloyu gör
    # print(summary)

    # # İstersen bir Excel veya CSV olarak da dışa aktar
    # summary.to_csv("class_feature_summary.csv")