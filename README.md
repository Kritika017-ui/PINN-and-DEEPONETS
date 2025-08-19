
# Cardiovascular Disease Detection using Physics-Informed Neural Networks (PINNs) and DeepONets

This repository implements an **end-to-end AI pipeline** for detecting cardiovascular diseases from medical images. The pipeline combines **Physics-Informed Neural Networks (PINNs)**, **DeepONet surrogates**, and a **classifier** to identify:

- **Aneurysms** (bulging of vessels)  
- **Stenosis** (narrowing of vessels)  
- **Atherosclerosis** (plaque buildup)  
- **Cardiac Valve Defects**

A Gradio-based dashboard allows users to upload medical images and obtain **velocity, pressure, wall shear stress (WSS) maps**, along with an automated **disease prediction**.

---

## 🚀 Project Overview

**Pipeline Steps:**

1. **Image Preprocessing** → Image + Mask → PINN-ready geometry & collocation points.  
2. **PINN Training** → Solve Navier–Stokes equations (u, v, p, WSS).  
3. **Dataset Creation** → Collect PINN outputs into a structured dataset.  
4. **DeepONet Surrogate** → Train surrogate for fast inference of flow fields.  
5. **Feature Extraction + Classifier** → Extract hemodynamic & geometric features → disease classification.  
6. **Dashboard** → Upload image, run prediction, visualize results.

---

## 📂 File Structure

```bash
project/
├── data/
│   ├── raw/                # Original images and masks
│   ├── preproc/            # Preprocessed masks, SDF, collocation points
│   │   └── case_0001/
│   │       ├── image.png
│   │       ├── mask.png
│   │       ├── sdf.npy
│   │       ├── pts_interior.npy
│   │       ├── pts_boundary.npy
│   │       └── inlet_mask.npy
│   └── pinn_outputs/       # PINN predictions (u,v,p,wss)
│       └── case_0001/
│           ├── u_128.npy
│           ├── v_128.npy
│           ├── p_128.npy
│           └── wss_128.npy
│
├── deepo_dataset/          # Downsampled dataset for DeepONet
│   └── case_0001/
│       ├── mask_64.npy
│       ├── u_64.npy
│       ├── v_64.npy
│       ├── p_64.npy
│       └── bc.json
│
├── models/                 # Saved models
│   ├── pinn_case_0001.pth
│   ├── deeponet.pth
│   └── classifier.pkl
│
├── notebooks/              # Jupyter notebooks for each stage
│   ├── 01_preprocess.ipynb
│   ├── 02_train_pinn.ipynb
│   ├── 03_train_deeponet.ipynb
│   ├── 04_train_classifier.ipynb
│   └── 05_dashboard.ipynb
│
└── README.md               # This file
````

---

## ⚙️ Installation & Requirements

### Environment

* Python 3.9+
* GPU recommended (CUDA 11+)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Key Libraries

* `numpy`, `scipy`, `scikit-image`, `opencv-python`
* `torch`, `deepxde`, `matplotlib`
* `pandas`, `tqdm`
* `gradio` (for dashboard)
* `scikit-learn` / `xgboost` (for classifier)

---

## 🧩 Pipeline Details

### Step A — Preprocessing

* **Input:** Medical image (PNG/JPG/DICOM) + mask.
* **Output:**

  * `mask.npy`, `sdf.npy`, `pts_interior.npy`, `pts_boundary.npy`, `inlet_mask.npy`
  * Visualization of mask, boundary, sampled points

Run:

```bash
jupyter notebook notebooks/01_preprocess.ipynb
```

---

### Step B — PINN Training

* Solves **Navier–Stokes equations** to predict `u, v, p` fields and `WSS`.
* Each geometry trained separately (per-geometry PINN).
* Outputs saved as `.npy` arrays.

Run:

```bash
jupyter notebook notebooks/02_train_pinn.ipynb
```

---

### Step C — Dataset Creation

* Collect all PINN outputs → downsample → normalize → save into `deepo_dataset/`.
* Generates `index.csv` for DeepONet training.

Run:

```bash
jupyter notebook notebooks/03_train_deeponet.ipynb
```

---

### Step D — DeepONet Surrogate

* Learns fast operator mapping: **(mask + BC) → (u, v, p)**.
* Uses CNN (branch net) + MLP (trunk net).
* Outputs fast predictions on unseen cases.

---

### Step E — Feature Extraction & Classifier

* Extracts hemodynamic & geometric features:

  * Mean / max / variance of WSS
  * Pressure drop (ΔP)
  * Velocity/vorticity statistics
  * Geometry descriptors
* Trains classifier (RandomForest/XGBoost).

Run:

```bash
jupyter notebook notebooks/04_train_classifier.ipynb
```

---

### Step F — Dashboard (Gradio)

* Upload medical image → segmentation → DeepONet inference → disease prediction.
* Outputs velocity, pressure, WSS maps, and classification result.

Run:

```bash
jupyter notebook notebooks/05_dashboard.ipynb
```

Example Gradio output:

* **Input:** Uploaded 2D slice
* **Output:**

  * Segmented mask
  * Velocity / pressure / WSS visualization
  * Disease label + probability

---

## 📊 Evaluation

* **PINN:**

  * Mean divergence < `1e-3`
  * Visual inspection: streamlines, velocity magnitude

* **DeepONet:**

  * RMSE(u, v, p) on held-out cases
  * Inference speed comparison vs PINN

* **Classifier:**

  * Accuracy, F1-score, AUC
  * Confusion matrix

---

## 🎯 Suggested Milestones

1. Preprocess 10 cases → validate masks, SDF, inlet detection.
2. Train PINN for 1–2 cases at low resolution.
3. Build dataset of \~50 cases from PINN outputs.
4. Train DeepONet → test on unseen geometry.
5. Extract features → train classifier → evaluate metrics.
6. Deploy Gradio dashboard.

---

## 🔗 Colab Notebook

You can try the pipeline directly in Colab:
[Colab Link](https://colab.research.google.com/drive/1HV5iS1bKiGQNtGp9vyGH9RZxl55uEhxq#scrollTo=DOL0_JHsguDa)

---

## 👩‍💻 Authors & Contributions

* **Kritika Rana** (Final Year B.Tech CSE-AIML, Chandigarh University)

  * Research & Development of PINN + DeepONet framework
  * Dataset preparation & feature engineering
  * Dashboard implementation

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

```

---

Would you like me to also generate a **`requirements.txt`** file to go along with this README (so you don’t have to manually list dependencies on GitHub)?
```
