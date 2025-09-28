# CLS Hidden Geometry

Research tools for exploring Vision Transformer (ViT) `[CLS]`–state geometry on ImageNet-1k.  
This repo provides:

- A drop-in subclass of `transformers.ViTForImageClassification` with:
  - Per-layer `[CLS]` extraction and caching for any ImageNet class
  - Cosine MMD and **MK-MMD** (mixture-kernel MMD) across layers / labels
  - Per-layer kernel mean embedding scores $ \hat{\mu}_{{t,y}}(c_t(z)) $ for query images
  - Optional PCA **energy-threshold** projection utility
  - Lightweight, per-key caching of features and Gram matrices
- Reproducible experiment scripts (plots, heatmaps, pairwise tables, tie-like candidates)

> **TL;DR**: Point the class at a ViT classifier, it ensures an on-disk ImageNet-256 split, pulls per-layer `[CLS]` states for classes you care about, and lets you compute/plot MMD & μ̂ trajectories—optionally with PCA + multi-kernel tricks.

---

## Table of Contents

- [Installation](#installation)
- [Dataset & Local Cache](#dataset--local-cache)
- [Quickstart](#quickstart)
- [Repo Structure](#repo-structure)
- [Key Features](#key-features)
- [API Cheatsheet](#api-cheatsheet)
- [Tips & Troubleshooting](#tips--troubleshooting)

---

## Installation

```bash
# Python 3.9+ recommended
pip install torch torchvision torchaudio   # choose the wheel that matches your CUDA/MPS/CPU setup
pip install transformers datasets pillow matplotlib numpy scikit-learn

# For entropy/eccentricity summaries (optional but supported in this repo)
pip install giotto-tda
```

If you’re on Apple Silicon, `device="mps"` works out of the box with recent PyTorch.

---

## Dataset & Local Cache

This project uses a **Hugging Face `datasets`** mirror of ImageNet-1k resized to 256 px:

- HF repo id: `evanarlian/imagenet_1k_resized_256`
- On first use, the class will **download and save to disk** under:

```
./ImageNet/imagenet_1k_resized_256/train/
./ImageNet/imagenet_1k_resized_256/val/
./ImageNet/imagenet_1k_resized_256/test/   # only if you access it
```

> **Note**: Make sure you have the right to access/use ImageNet content in your environment. This repo only automates pulling a resized mirror via HF `datasets`.

---

## Quickstart

```python
from src.ClsHiddenGeometry import ClsHiddenGeometry
from PIL import Image

# 1) Load a ViT classifier (e.g., ViT-Base fine-tuned on ImageNet-1k).
m = ClsHiddenGeometry("google/vit-base-patch16-224", device="cuda:0")  # or "mps"/"cpu"

# 2) Cache per-layer [CLS] states for a class (string or int id).
m.compute_hidden_cls_states(class_label="tabby, tabby cat", num_samples=700, batch_size=100)

# 3) Persist (and later reload) the cached entry.
m.write_hidden_cls_states("tabby, tabby cat", 700)
m.read_hidden_cls_states("data/google__vit-base-patch16-224/tabby_tabby_cat_700.pt")

# 4) Compare two labels via cosine MMD across layers.
vals = m.cosine_mmd2_between_labels_along_layers("tabby, tabby cat", 700, "Siamese cat, Siamese", 700)

# 5) Compute μ̂ trajectories for a query image (cosine or MK-RBF).
img = Image.open("path/to/cat.jpg").convert("RGB")
curve, mean_val = m.mu_hat_along_layers("tabby, tabby cat", 700, image_z=img, true_label="cat", kernel="cosine")
```

---

## Repo Structure

```
.
├── src/
│   └── ClsHiddenGeometry.py   # the main class + EnergyWhitener utility
├── experiment_scripts/
│   ├── mu_hat_experiment.py   # per-image μ̂ curves vs many labels (cos/MK)
│   ├── mmd_experiment.py      # cosine + MK pairwise MMD across labels
│   ├── ties.py                # μ̂ curves for “tie-like” ambiguous image
│   └── make_data.py           # build label groups and persist caches
├── data/                      # saved .pt caches and experiment payloads (.pkl)
├── plots/                     # figures written by the experiment scripts
└── ImageNet/                  # HF dataset saved to disk on first run
```

---

## Key Features

### Per-layer `[CLS]` capture (ImageNet-1k train split)

Sample `n` images from class `y` and record the `[CLS]` state **after each transformer block**:

- Stored as `list[Tensor]` with length `L = num_hidden_layers`
- Each tensor has shape `(n, d_model)`
- Cached under key `"{label}_{n}"`

### Cosine MMD and MK-MMD

- **Cosine** on unit-sphere normalized features (fast; no bandwidth)
- **MK-MMD (RBF/Laplace)** per layer:
  - Build a geometric **sigma grid** centered by the median distance
  - Optionally apply **PCA energy projection** (per layer/key/union)
  - Learn mixture weights to **maximize standardized MMD²** (effect / std)

### μ̂ Trajectories

For a query image $ z $, compute $ \hat{\mu}_{t,y}(c_t(z)) $ along layers against $P_{t,y}$:

- `kernel="cosine" | "mk_rbf" | "mk_laplace"`
- Optionally scan PCA thresholds and pick the best via bootstrap std-score
- Returns the trajectory and its mean—handy for ranking or plots

### “Tie-like” Candidate Mining

Scan a dataset split and rank images by **tie-likeness** (uniform top-m mass + small margins):

- Saves top examples as JPEGs with metrics in filenames
- Great to stress-test μ̂ curves on ambiguous cases (e.g., water jug vs pitcher)

### PCA Energy Whitener

A lightweight wrapper that picks `n_components` to hit a desired **explained variance** threshold and (optionally) whitens. 

---


## API Cheatsheet

> See in-code docstrings for details

### Construction

```python
m = ClsHiddenGeometry(
    model_id="google/vit-base-patch16-224",
    device="cuda:0",                  # or "mps", "cpu"
    trust_remote_code=False,
    # **from_pretrained_kwargs forwarded to HF
)
```

### Convenience

- `m.labels` → label names in index order (from `config.id2label`)
- `m.classify(images, top_k=5)` → quick top-k predictions for one or many images

### Dataset utilities

- Ensures local HF dataset exists:
  - `_ensure_local_imagenet256()`
- Finds feature columns:
  - `_find_cols(ds) -> (image_col, label_col)`
- Class resolution:
  - `_label_to_index(ds, label_col, class_label)`

### Hidden state caching

- `compute_hidden_cls_states(class_label, num_samples, batch_size)`
- `write_hidden_cls_states(class_label, n_samples)` → save `.pt`
- `read_hidden_cls_states(path)` → load `.pt` into memory
- `merge_hidden_cls_states(keys_to_merge, new_label, delete_old=True)`
- `clear_hidden_cls_states(class_label, n_samples)`

### Statistics

- `compute_shannon_entropy(class_label, n_samples, plot=True)` → `(n, L)`
- `compute_eccentricity(class_label, n_samples, plot=True)` → `(n, L)`

### MMD & μ̂

- `cosine_mmd2_between_labels_along_layers(A, nA, B, nB) -> (L,)`
- `plot_cosine_mmd_between_labels(A, nA, B, nB)`
- `mk_mmd2_between_labels_along_layers(A, nA, B, nB, **kw) -> (vals, sigmas_per_layer, std_scores)`
- `plot_mk_mmd_between_labels(A, nA, B, nB, **kw)`
- `mu_hat_along_layers(class_label, n_samples, image_z, true_label, kernel=...) -> (vals, mean)`

### Tie-like mining & labels

- `find_tie_like_candidates(split="val", max_candidates=10, m=3, ...) -> list[dict]`
- `print_true_labels_for_val_indices(idxs, split="val")`

---

## Tips & Troubleshooting

- **Device**
  - CUDA: `device="cuda:0"`; Apple Silicon: `device="mps"`; otherwise `cpu`.
- **Dataset errors**
  - If you see `ImportError: datasets`, run `pip install datasets`.
  - If `_find_cols` fails, check the dataset features; this class expects an HF `Image` column and a `ClassLabel` column.
- **Keys & naming**
  - Cache keys are `"{label}_{n}"`. When `label` is a string, commas/spaces are sanitized.
  - Use `merge_hidden_cls_states` to **rename** a single entry to a nice alias (e.g., `"cellular telephone, ..."` → `"cellphone"`).
- **Plots**
  - All scripts save plots under `./plots/{model_id_sanitized}/...` or subfolders noted in each script.

---

