# 🧠 LiteGAN-FedNet: Privacy-Preserving and Explainable Federated Learning for Brain Tumor MRI Classification

## 📌 Overview

This repository contains the implementation used to produce the results reported in *"Privacy-Preserving and Explainable Federated Learning for Brain Tumor MRI Classification Using LiteGAN-FedNet."* The pipeline extracts 512-D feature embeddings from brain tumor MRI slices using a fine-tuned ResNet-18, augments minority classes with a per-client conditional GAN (cGAN), compresses embeddings to 100-D via a single shared PCA basis, and trains a lightweight MLP classifier under Federated Averaging (FedAvg) across simulated clients. Model behavior is interpreted post-hoc using Grad-CAM, SHAP, and LIME.

Evaluated on two public benchmarks:
- **Figshare Brain MRI Dataset** (6,536 images, 4 classes)
- **Mendeley Brain Tumor MRI Dataset** (11,148 images, 4 classes)

Both datasets share the same four-class label set: glioma, meningioma, pituitary tumor, no-tumor.

---

## ⚠️ Important note on results reproducibility

This repository contains **two related but distinct sets of experiments**, and it's important not to conflate them:

1. **`training_mendeley.ipynb` / `training_figshare.ipynb`** — the main pipeline (ResNet-18 fine-tuning → cGAN augmentation → shared PCA → **centralized** MLP training on pooled client embeddings). This produces the headline accuracy numbers reported in the paper's main results table.
2. **`ablation_mendeley.ipynb` / `ablation_figshare.ipynb`** — a separate, true per-client FedAvg implementation (local training + weighted weight-averaging over `K=3` clients, `T=50` rounds), run across 5 seeds, used for the paper's ablation study. This is the more methodologically faithful representation of the federated protocol described in the paper, and produces different (lower) accuracy than the centralized pipeline above — this discrepancy is disclosed explicitly in the paper's Statistical Significance and Robustness Analysis section.

Reconciling these two into a single unified federated implementation is an open item — see **Future Work** below.

---

## 🏗️ Repository Structure

```
LiteGAN-FedNet/
│
├── training_mendeley.ipynb     # Main pipeline: ResNet-18 → cGAN → PCA → centralized MLP (Mendeley)
├── training_figshare.ipynb     # Main pipeline: ResNet-18 → cGAN → PCA → centralized MLP (Figshare)
├── ablation_mendeley.ipynb     # True per-client FedAvg ablation study (Mendeley), 5 seeds
├── ablation_figshare.ipynb     # True per-client FedAvg ablation study (Figshare), 5 seeds
├── Datasest Splition.png       # Dataset train/test split diagram
├── requirements.txt            # Pinned dependency versions (see below)
└── README.md
```

---

## 🗂️ Datasets

| Dataset | Total Images | Train / Test | Classes | Source |
|---|---|---|---|---|
| Figshare | 6,536 | 5,228 / 1,308 | glioma, meningioma, pituitary, no-tumor | [DOI: 10.6084/m9.figshare.14778750](https://doi.org/10.6084/m9.figshare.14778750) |
| Mendeley | 11,148 | 9,291 / 1,857 | glioma, meningioma, pituitary, no-tumor | [DOI: 10.17632/zwr4ntf94j.6](https://data.mendeley.com/datasets/zwr4ntf94j/6) |

Neither source dataset includes patient identifiers, so splits are at the image/slice level rather than the patient level — see the paper's Dataset Description section for the full discussion of this limitation.

---

## ⚙️ Methodology

1. **Feature Extraction** — ResNet-18 (ImageNet-pretrained) fine-tuned end-to-end for 10 epochs, then used purely as a 512-D feature extractor.
2. **cGAN Feature Augmentation** — a per-client conditional GAN, trained with the WGAN-GP objective (300 epochs, gradient-penalty λ=10, 5:1 discriminator:generator update ratio), generates 200 synthetic embeddings per minority class.
3. **PCA Compression** — a single shared basis, fit once on the pooled training embeddings, projects 512-D → 100-D (an 80.5% reduction in the dominant transmitted-weight dimension; see the Communication Cost Accounting section of the paper for the full byte-level breakdown).
4. **Classification** — a lightweight MLP (hidden size 128) trained either centrally on pooled embeddings (`training_*.ipynb`) or federated via true FedAvg across `K=3` clients (`ablation_*.ipynb`).
5. **Explainability** — Grad-CAM (spatial), SHAP (global, on PCA components), and LIME (instance-level, on raw images), applied post-hoc.

Full architecture details, hyperparameters, and the exact library versions used are listed in `requirements.txt` and in the paper's Reproducibility Appendix.

---

## 🧪 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/irfanulkabirhira/LiteGAN-FedNet.git
cd LiteGAN-FedNet
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the main pipeline
```bash
jupyter notebook training_mendeley.ipynb   # or training_figshare.ipynb
```
This produces the ResNet-18 fine-tuning, cGAN training, PCA-fitting, and centralized MLP classification results.

### 4. Run the ablation study (optional, requires Step 3 completed in the same runtime)
Run `training_mendeley.ipynb` (or `training_figshare.ipynb`) through the cGAN training step only, then, **in the same runtime/session**, run:
```bash
jupyter notebook ablation_mendeley.ipynb   # or ablation_figshare.ipynb
```
This reuses the already-trained cGAN and extracted features to run the true per-client FedAvg ablation across 5 seeds.

> **Note:** these notebooks were developed and run on Google Colab with an NVIDIA T4 GPU. Running locally requires a CUDA-capable GPU with at least the VRAM listed in `requirements.txt`'s comments, or expect substantially longer CPU runtimes.

---

## 📦 Requirements

Exact pinned versions (also listed in the paper's Reproducibility Appendix):

```
torch==2.9.0+cu126
torchvision==0.24.0+cu126
torchaudio==2.9.0+cu126
scikit-learn==1.6.1
flwr==1.24.0
shap==0.50.0
lime
albumentations==2.0.8
xgboost==3.1.2
imbalanced-learn
opencv-python==4.12.0
pandas==2.2.2
matplotlib==3.10.0
Pillow==11.3.0
tqdm==4.67.1
```

---

## 📊 Results

Results are reported separately per dataset and per experiment type (centralized pipeline vs. true-FedAvg ablation), matching the paper. See the paper's Results Analysis and Ablation Study sections for full tables, including per-configuration mean ± standard deviation over 5 seeds for the ablation study. We do not summarize results as a single "High/Moderate/Best" table here, since headline single-run numbers and multi-seed ablation numbers are not directly comparable — see the paper's Statistical Significance and Robustness Analysis section for the full disclosure of this distinction.

---

## 📈 Future Work

- Unify the centralized pipeline (`training_*.ipynb`) and the true-FedAvg ablation implementation (`ablation_*.ipynb`) into a single consistent federated training script.
- Implement and evaluate differential privacy (currently architecturally supported but not enabled or evaluated in reported results).
- Add feature-space diversity/collapse metrics for cGAN-generated embeddings.
- Multi-center clinical validation beyond the two public benchmarks used here.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👥 Authors

MD Irfanul Kabir Hira, Mst Moriom Akter Bithee, Md Sohag Hossain, Md. Kowsar Ahmed, Md. Mamun Ur Rashid, Umme Sara, Mohammad Shorif Uddin
