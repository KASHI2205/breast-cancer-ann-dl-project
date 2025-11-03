# 🩺 Breast Cancer Prediction using Deep Learning (ANN)

A **Deep Learning–based Breast Cancer Classification System** built with **TensorFlow** and **Keras**.  
This project predicts whether a tumor is **Benign** or **Malignant** using diagnostic features such as radius, texture, area, and compactness.  
It also features a **Gradio-powered web interface** for interactive, real-time predictions.

---

## 🚀 Features

- 🧠 **Artificial Neural Network (ANN)** model built using TensorFlow / Keras  
- ⚙️ Data preprocessing and normalization using Scikit-learn  
- 📊 Evaluation metrics: Accuracy, Confusion Matrix, ROC-AUC  
- 💻 Interactive **Gradio UI** for user-friendly prediction  
- 🔍 Option for both **preset test cases** and **custom user inputs**

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Deep Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| UI / Visualization | Gradio |
| Dataset | [Kaggle – Breast Cancer Dataset](https://www.kaggle.com/datasets/adhamelkomy/breast-cancer) |

---

## 📦 Dataset

The dataset contains various diagnostic features such as:
- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, etc.  
Each record is labeled as either **Malignant (M)** or **Benign (B)**.

To load via `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("adhamelkomy/breast-cancer")
print("Path to dataset files:", path)
