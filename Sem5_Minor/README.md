# 🩺 PneumoScan: Explainable, Uncertainty-Aware Pneumonia Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Explainable, uncertainty-aware pneumonia detection from chest X-rays using U-Net, DenseNet121, Grad-CAM++, and Monte Carlo Dropout.**  
> *Developed at Maulana Azad National Institute of Technology (MANIT), Bhopal.*

---

## 📌 Overview

Traditional deep learning models for medical imaging function as "black boxes" that yield high-confidence predictions without explanatory justification. Crucially, standard models lack the ability to say **"I am not sure"**, risking silent misclassifications in safety-critical clinical environments.

**PneumoScan** addresses this fundamental challenge by combining:
1. **U-Net Lung Segmentation**: Accurately delineates the paired lung boundaries, cropping out text labels, bones, and image artifacts to prevent AI shortcut learning.
2. **DenseNet121 Transfer Learning**: Classifies pneumonia with fine-grained feature reuse.
3. **Grad-CAM++ Visual Interpretability**: Overlays weighted heatmaps onto the segmented lung parenchyma, highlighting alveolar consolidations and opacities.
4. **Monte Carlo Dropout (30 stochastic passes)**: Quantifies model uncertainty using predictive entropy and spread, automatically flagging equivocal cases for human radiologist review.

---

## 🚀 Key Features

* **🩺 Single X-ray Analysis**:
  * Upload any chest X-ray (`.png`, `.jpg`, `.jpeg`) or load pre-evaluated test samples with one click.
  * **Dynamic Status Banners**:
    * 🟢 **"Prediction is stable"** when entropy is below the review threshold.
    * 🟠 **"Uncertain, refer for expert review"** when entropy exceeds the threshold.
    * 🔴 **Safety Net Warning** if the lungs could not be located (lung coverage < 5%).
  * **Four Result Cards**: Predicted Class (`NORMAL` / `PNEUMONIA`), Confidence %, MC Spread (±$\sigma$), and Predictive Entropy ($H$ in bits).
  * **Interactive Uncertainty Gauge**: Visual scale comparing entropy against the review threshold.
  * **Three Side-by-Side Visualizations**:
    1. Original radiograph with segmented lung contour.
    2. Segmented 320×320 lung crop with CLAHE enhancement.
    3. Grad-CAM++ heatmap overlay.
  * **Monte Carlo Histogram**: Frequency distribution across 30 passes with decision boundary at $P = 0.50$.
  * **Details & Export Panel**: Diagnostic metrics table and button to download the Grad-CAM++ heatmap.
  * **"How to Read This" Guide**: Plain-language clinical walkthrough.

* **📁 Batch Analysis**:
  * Multi-file uploader to analyze cohorts of radiographs simultaneously.
  * Live progress bar and summary metric cards (Total, Normal, Pneumonia, Flagged %, Fallbacks).
  * Interactive results table with color-coded triage indicators.
  * **CSV Export**: Download full batch results for clinical records or audit trails.
  * Detailed single-case drill-down right from the batch view.

* **📊 Architecture & Performance Dashboard**:
  * Complete end-to-end processing pipeline diagram.
  * Benchmark results on 586 holdout test images (**96.59% Accuracy**, **98.13% Recall**, **97.22% Precision**).
  * Selective prediction analysis showing how uncertainty flags catch **19 out of 20 misclassifications**.
  * Confusion matrix and per-class metrics.

---

## 📈 Benchmark Performance (Holdout Test Set)

Evaluated on 586 holdout chest X-rays from the pooled 5,856 dataset:

| Metric | Score | Clinical Relevance |
| :--- | :---: | :--- |
| **Accuracy** | **96.59%** | 566 of 586 test scans correctly classified |
| **Pneumonia Recall** | **98.13%** | Only 8 missed pneumonia cases out of 428 |
| **Pneumonia Precision** | **97.22%** | Only 12 false positives out of 432 positive calls |
| **Pneumonia F1-Score** | **0.9767** | Robust harmonic mean on imbalanced data |
| **Error Rate on Confident Cases** | **0.25%** | Near-zero errors when model is confident |
| **Error Rate on Flagged Cases** | **10.22%** | High-risk cases successfully referred to experts |

---

## 🛠️ Easy Deployment to Streamlit Cloud via GitHub

Deploying PneumoScan to [Streamlit Community Cloud](https://share.streamlit.io/) is fast and free:

### Step 1: Push to GitHub
1. Create a new repository on your GitHub account (e.g., `PneumoScan` or `Sem5_Minor`).
2. In this folder (`c:\Projects\Sem5_Minor`), initialize git and push:
```bash
git init
git add .
git commit -m "Initial commit: PneumoScan Streamlit app"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Navigate to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository: `<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>`.
4. Set branch: `main`.
5. Set Main file path: `app.py`.
6. Click **"Deploy!"**.
   *(Build takes ~30 seconds because all dependencies are lightweight and headless-optimized).*

---

## 💻 Running Locally

### 1. Clone repository
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
Sem5_Minor/
├── app.py                   # Main Streamlit web application
├── pipeline.py              # Segmentation, MC Dropout & Grad-CAM++ engine
├── sample_data.py           # Pre-packaged base64 demo chest radiographs
├── samples/                 # Sample JPG images (Normal, Pneumonia, Uncertain, Fallback)
├── requirements.txt         # Minimal, fast Streamlit Cloud dependencies
├── .streamlit/
│   └── config.toml          # Custom dark medical theme
├── Minor_PPT.pptx           # Academic presentation slides
├── pneumonia.ipynb          # Source training & evaluation notebook
└── README.md                # Project documentation & deployment guide
```

---

## ⚠️ Medical Disclaimer

**Educational prototype for academic research and demonstration only.**  
This application is **not a certified medical device** and is **not intended for autonomous clinical diagnosis or medical treatment planning**. Any clinical decision must be made by a qualified healthcare professional.

---

## 👥 Authors & Academic Attribution

**Maulana Azad National Institute of Technology (MANIT), Bhopal**  
Department of Computer Science & Engineering  

* **Balaka Laluth Vardhan** (Scholar No. 24112011116)  
* **Padagala Sathvik** (Scholar No. 24112011159)  
* **Shashank Agarwal** (Scholar No. 24112011114)
