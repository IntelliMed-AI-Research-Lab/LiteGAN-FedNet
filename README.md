# 🧠 Brain Tumor Classification using Mendeley & Figshare Datasets

## 📌 Overview

This project focuses on **brain tumor classification** using medical imaging datasets from **Mendeley** and **Figshare**.

The goal is to build a **robust deep learning pipeline** that can generalize across multiple datasets and improve classification performance.

---

## 🚀 Key Features

* ✅ Multi-dataset training (Mendeley + Figshare)
* ✅ Medical image classification pipeline
* ✅ Deep learning-based feature extraction
* ✅ Modular and reusable training code
* ✅ Scalable for research and real-world applications

---

## 🏗️ Project Structure

```

id="p2a91x"
Brain-Tumor-Classification/
│
├── data/
│   ├── mendeley/
│   ├── figshare/
│
├── notebooks/
│   ├── training_mendeley.ipynb
│   └── training_figshare.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── results/
│   ├── plots/
│   └── metrics.txt
│
├── models/
│   └── saved_model.pth
│
├── requirements.txt
└── README.md

```

---

## 🗂️ Datasets

### 📍 Mendeley Dataset

* Brain tumor MRI images
* Labeled into multiple tumor categories

---

### 📍 Figshare Dataset

* Publicly available brain MRI dataset
* Used to improve model generalization

---

## ⚙️ Methodology

### 1️⃣ Data Loading

* Load MRI images from both datasets
* Organize into class folders

---

### 2️⃣ Preprocessing

* Resize images (e.g., 224×224)
* Normalize pixel values
* Apply augmentations:

  * Rotation
  * Flipping
  * Scaling

---

### 3️⃣ Model Architecture

Typical pipeline:

* CNN / Transformer backbone
* Feature extraction
* Fully connected classification head

---

### 4️⃣ Training Strategy

* Train separately on:

  * Mendeley dataset
  * Figshare dataset
* Compare performance across datasets
* Optional: Combine datasets for better generalization

---

### 5️⃣ Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

## 🧪 How to Run

### 🔹 1. Clone Repository

---
bash id="mnd1"
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification
```

```

### 🔹 2. Install Dependencies

---
bash id="mnd2"
pip install -r requirements.txt
```

```

### 🔹 3. Run Training

#### ▶ Train on Mendeley Dataset

---
bash id="mnd3"
jupyter notebook notebooks/training_mendeley.ipynb
```
```
#### ▶ Train on Figshare Dataset

---
bash id="mnd4"
jupyter notebook notebooks/training_figshare.ipynb
```

```

## 📦 Requirements

---
id="mnd5"
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
opencv-python
```

```

## 📊 Results

| Dataset  | Performance |
| -------- | ----------- |
| Mendeley | High        |
| Figshare | Moderate    |
| Combined | ✅ Best      |

> Training on multiple datasets improves model robustness and generalization.

---

## 💾 Model Saving

---
python id="mnd6"
torch.save(model.state_dict(), "saved_model.pth")
```

```

## 📈 Future Improvements

* 🔥 Swin Transformer integration
* 🔥 Hybrid CNN + TCN architecture
* 🔥 Grad-CAM for explainability
* 🔥 Cross-dataset validation
* 🔥 Deployment (Streamlit / Flask)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and improve the project.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Md Irfanul Kabir**
AI Researcher | Deep Learning Enthusiast 🚀

---
