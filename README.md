# 🏥 Medical Diagnosis — 3D Imaging + Clinical Text

> Multimodal AI system combining 3D CT scan analysis (CNN3D) and clinical report understanding (BERT) for automated diagnostic assistance.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![BERT](https://img.shields.io/badge/HuggingFace-BERT-FFD21E?logo=huggingface)
![MLOps](https://img.shields.io/badge/MLOps-Ready-4CAF50)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🇬🇧 English

### Overview

This project implements a **multimodal diagnostic assistance system** that fuses:
- **3D CT scans** (315 GB processed) via a 3D CNN (reconstruction, segmentation, normalization)
- **1,000 clinical reports** via BERT for descriptor extraction

Both modalities are fused into a unified prediction pipeline with an explainability layer and an interactive result interface.

### Key Features

- 🧠 **3D CNN** — volumetric reconstruction, segmentation, and normalization of XCT scans
- 📄 **BERT NLP** — clinical descriptor extraction from free-text reports
- 🔀 **Late fusion** — learned combination of image + text embeddings
- 🔍 **Explainability** — Grad-CAM for 3D volumes + SHAP for text features
- 📊 **Visual interface** — slice viewer, attention heatmaps, prediction confidence

### Results

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|----|-----------|--------|
| CNN3D only | 0.84 | 0.79 | 0.81 | 0.77 |
| BERT only | 0.80 | 0.75 | 0.78 | 0.72 |
| **Multimodal fusion** | **0.91** | **0.87** | **0.89** | **0.85** |

---

## 🇫🇷 Français

### Vue d'ensemble

Ce projet implémente un **système d'aide au diagnostic multimodal** combinant :
- **Imagerie CT 3D** (315 Go traités) via un CNN 3D (reconstruction, segmentation, normalisation)
- **1 000 rapports cliniques** via BERT pour l'extraction de descripteurs

Les deux modalités sont fusionnées dans un pipeline de prédiction unifié avec une couche d'explicabilité et une interface visuelle interactive.

### Fonctionnalités clés

- 🧠 **CNN 3D** — reconstruction volumétrique, segmentation et normalisation de scans XCT
- 📄 **BERT NLP** — extraction de descripteurs cliniques à partir de rapports textuels
- 🔀 **Fusion tardive** — combinaison apprise des embeddings image + texte
- 🔍 **Explicabilité** — Grad-CAM 3D + SHAP pour les features textuelles
- 📊 **Interface visuelle** — visionneuse de coupes, heatmaps d'attention, confiance de prédiction

---

## 🗂️ Repository Structure

```
medical-diagnosis-multimodal/
│
├── data/
│   ├── raw/                        # Raw DICOM / NIfTI scans + clinical reports
│   └── processed/                  # Normalized volumes + tokenized texts
│
├── notebooks/
│   ├── 01_volume_eda.ipynb         # 3D scan exploration, slice visualization
│   ├── 02_text_eda.ipynb           # Clinical report EDA, vocabulary analysis
│   ├── 03_cnn3d_training.ipynb     # CNN3D training + Grad-CAM
│   ├── 04_bert_training.ipynb      # BERT fine-tuning + SHAP
│   └── 05_fusion_eval.ipynb        # Multimodal fusion + final evaluation
│
├── src/
│   ├── preprocess/
│   │   ├── volume_preprocess.py    # DICOM loading, resizing, normalization
│   │   └── text_preprocess.py      # Clinical report cleaning + tokenization
│   ├── models/
│   │   ├── cnn3d.py                # 3D CNN encoder + segmentation head
│   │   ├── bert_encoder.py         # BERT fine-tuning for clinical NLP
│   │   └── fusion.py               # Late fusion model (image + text)
│   ├── train.py                    # Unified training loop
│   ├── evaluate.py                 # AUC, F1, confusion matrix, calibration
│   └── explainability.py           # Grad-CAM 3D + SHAP text attribution
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Methodology

### 1. 3D Volume Preprocessing
- Load DICOM series → NIfTI conversion
- Resample to isotropic 1mm³ spacing
- Clip HU values [-1000, 400], MinMax normalization
- Resize to fixed shape (128×128×64)
- Data augmentation: random flips, rotations, elastic deformation

### 2. Clinical Text Preprocessing
- De-identification (regex + NER-based)
- Segmentation into anatomical sections
- Tokenization with `bert-base-uncased` (max 512 tokens)
- Descriptor extraction: symptoms, findings, anatomical locations

### 3. 3D CNN
- Architecture: 4× Conv3D blocks with BatchNorm + MaxPool3D
- Segmentation head: U-Net-like skip connections
- Classification head: Global Average Pooling → FC → Sigmoid
- Loss: BCE + Dice (joint segmentation + classification)

### 4. BERT Fine-tuning
- Base: `bert-base-uncased` (110M params)
- Fine-tuned on 1,000 clinical reports
- [CLS] token embedding as report representation
- Task: multi-label classification (pathology presence)

### 5. Multimodal Fusion
- Late fusion: concatenate CNN3D embedding + BERT [CLS] embedding
- Learnable attention weights per modality
- Final MLP classifier with dropout

---

## 🚀 Getting Started

```bash
git clone https://github.com/HAMZAZAROUALI/medical-diagnosis-multimodal.git
cd medical-diagnosis-multimodal
pip install -r requirements.txt

# Preprocess
python src/preprocess/volume_preprocess.py
python src/preprocess/text_preprocess.py

# Train
python src/train.py --mode fusion --epochs 30

# Evaluate
python src/evaluate.py

# Explainability
python src/explainability.py
```

---

## 📦 Requirements

```
torch>=2.0
torchvision>=0.15
transformers>=4.35
nibabel>=5.0
pydicom>=2.4
SimpleITK>=2.3
scikit-learn>=1.3
shap>=0.43
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
pandas>=2.0
numpy>=1.24
jupyter
```

---

## ⚠️ Data Privacy

All patient data used in this project is **anonymized and de-identified** in compliance with GDPR and HIPAA standards. No raw medical data is versioned in this repository.

---

## 👤 Author

**Hamza Zarouali** — AI & Data Science Engineer
[LinkedIn](https://linkedin.com/in/HAMZAZAROUALI) · [Email](mailto:hamzazarouali100@gmail.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
