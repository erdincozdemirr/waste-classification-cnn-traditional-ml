# ♻️ Waste Classification with CNN & Traditional Machine Learning

This project focuses on **image-based waste classification** by comparing **deep learning** and **traditional machine learning** approaches.

The study includes:
- An **end-to-end CNN model (AlexNet)** trained directly on images
- **Traditional ML models (SVM, Random Forest, XGBoost)** trained on **MobileNet-extracted features**
- A detailed experimental pipeline with **confusion matrices, ROC curves, and multiple performance metrics**

Developed as part of an **MSc Machine Learning course**.

---

## 📌 Project Motivation

Waste classification is an important problem in recycling and environmental sustainability.  
This project aims to explore:

- How well a CNN (AlexNet) performs on raw images
- How traditional ML models perform when combined with deep feature extraction
- The trade-offs between **end-to-end deep learning** and **feature-based ML pipelines**

---

Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

---

## 🧠 Models Used

### Deep Learning
- **AlexNet (PyTorch)**
  - Trained from scratch (no pretrained weights)
  - Focal Loss + Label Smoothing
  - Early Stopping & LR Scheduler
  - Multiple hyperparameter configurations (ALEX1–ALEX4)

### Feature Extraction
- **MobileNet**
  - Used as a fixed feature extractor
  - Generates embeddings for traditional ML models

### Traditional Machine Learning
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier

---

## ⚙️ Installation

### 1️⃣ Create virtual environment
```bash
python -m venv .venv
Activate:

bash
Kodu kopyala
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
2️⃣ Install dependencies
bash
Kodu kopyala
pip install -r requirements.txt
🚀 Running the Project
🔹 Train AlexNet (4 configurations)
bash
Kodu kopyala
python CNN/main_2_alexNet.py
This will:

Train ALEX1–ALEX4

Save models to:

swift
Kodu kopyala
CNN/Outputs/ALEX1.pth
CNN/Outputs/ALEX2.pth
CNN/Outputs/ALEX3.pth
CNN/Outputs/ALEX4.pth
Save metrics, confusion matrices and ROC curves to:

swift
Kodu kopyala
Results/ALEX/ALEX1/
Results/ALEX/ALEX2/
Results/ALEX/ALEX3/
Results/ALEX/ALEX4/
Write the best model summary to:

swift
Kodu kopyala
Results/ALEX/BEST_MODEL.txt
🔹 Run inference & model comparison
bash
Kodu kopyala
python try.py
This script:

Loads a test image

Runs prediction using:

AlexNet (.pth)

SVM

Random Forest

XGBoost

Displays predictions together on the same image

📊 Evaluation Metrics
The following metrics are computed for train / validation / test sets:

Accuracy

F1 Score

Sensitivity (Recall)

Specificity

ROC-AUC

Confusion Matrix

ROC Curves (per class)

All results are saved as .txt and .png files for reproducibility.

🏆 Model Selection
Models are ranked based on validation accuracy

The best-performing AlexNet configuration is recorded in:

swift
Kodu kopyala
Results/ALEX/BEST_MODEL.txt
🧪 Key Takeaways
AlexNet provides strong end-to-end performance on raw images

MobileNet + traditional ML offers a competitive and efficient alternative

Feature-based ML pipelines are easier to train and faster to iterate

CNNs capture spatial patterns more effectively but require longer training

🧰 Technologies Used
Python

PyTorch & Torchvision

Scikit-learn

XGBoost

NumPy, SciPy

Matplotlib, Seaborn

OpenCV, Pillow

👨‍🎓 Academic Context
This project was developed as part of an MSc-level Machine Learning course, focusing on:

Model comparison

Experimental rigor

Reproducibility

Performance analysis
