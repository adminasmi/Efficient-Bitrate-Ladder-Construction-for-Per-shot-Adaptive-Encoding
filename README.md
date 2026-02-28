This repository contains the code and scripts for the paper:

*Efficient Bitrate Ladder Construction for Per-shot Adaptive Encoding* 

🎆 **VCIP 2024 Best Student Paper Runner-Up**

https://drive.google.com/file/d/1FdXBqToebhHYvNRflmflIs44_cGlz7AA/view

---

## Overview
We build **per-shot content-adaptive bitrate ladders** efficiently using a two-stage pipeline:

1. **Curve Fitting:** encode only a small set of operating points and fit RD curves.
2. **Cross-Curve Prediction:** predict the remaining curve parameters across **resolution** and/or **preset** with lightweight ML.

This reduces total encoding cost while approximating per-shot CAE quality.

---

## Dataset Preparation (Encoding + Processing)

You need to encode videos with a **specified encoder** under:
- multiple **resolutions**
- multiple **bitrates / QPs**
- optional multiple **presets**

Main directories:
- `enc-dec/`  
  Encoding scripts and notebooks (currently includes **AV1**, **vvenc**, and **x265**).  
  The corresponding `.ipynb` files contain step-by-step usage and notes.

- `dataset/`  
  Dataset processing utilities:
  - multi-resolution generation
  - shot / scene splitting
  - metrics computation

- `dataset/analyse/`  
  Evaluation helpers (PSNR / SSIM / VMAF). Core logic is in `metrics.py`.

**Tip:** enable CPU *performance mode* during transcoding to speed up encoding.

---

## Fast CAE Ladder Construction

We follow two steps: **curve fitting** → **cross-curve prediction**.

> Some `.py` files share names with notebooks (`.ipynb`). They mainly extract the time-consuming parts into scripts for easier batch running.

### 1) Curve Fitting
- Notebook / script:
  - `curvefit.ipynb`
  - `curvefit.py`
- Contains multiple fitting functions and comparisons.
- Fit evaluation results are saved under:
  - `evalFit/`

### 2) Cross-Curve Prediction
1. Build the ML dataset:
   - `prepareData.py`

2. Predict curve parameters:
   - `predCurve.py`  
     Cross-**preset** *or* cross-**resolution** prediction
   - `predCurveDual.py`  
     Joint prediction across both **preset** and **resolution**

- Prediction evaluation results are saved under:
  - `evalPred/`

---

## Plotting
- Intermediate plots are typically inside the corresponding notebooks.
- Final plots (including paper-style figures) are generated in:
  - `draw.ipynb`
