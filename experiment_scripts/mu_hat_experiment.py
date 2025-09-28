# experiment_scripts/mu_hat_experiment.py
# ==============================================================================
# Goal
# ----
# For a set of ImageNet class labels and a single test image per label, compute
# and visualize the per-layer kernel mean embedding scores
#   μ̂_{p_{t,y}}(z_t)
# comparing the query image (z) against each class distribution p_{t,y}.
#
# We produce two plots per image:
#   1) Cosine kernel          (unit-sphere features)
#   2) MK-RBF (multi-kernel)  (RBF mixture with sigma grid + optional PCA)
#
# Prereqs
# -------
# - Precomputed hidden [CLS] states saved at:
#     data/google__vit-base-patch16-224/{label}_{n_samples}.pt
# - A test image per class at:
#     data/test_images/{label}.jpg
#
# Outputs
# -------
# - Per-image plots saved to:
#     plots/single_mu_hat_trajectories/{label}__vs_all__cos.png
#     plots/single_mu_hat_trajectories/{label}__vs_all__mk.png
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.ClsHiddenGeometry import ClsHiddenGeometry

# ------------------------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------------------------
# Classes to compare. Use names that match your dataset's ClassLabel metadata.
labels = ["robin", "jay", "chickadee", "beagle", "lab", "boxer",
          "tabby", "siamese", "persian", "lizard", "snake"]

# Number of cached samples per class used to build p_{t,y}.
# (These should match the saved .pt files you’ve prepared.)
n_samples = 700

# MK-RBF settings:
# - n_scales: number of bandwidths around the median distance per layer
# - thr_grid: PCA energy thresholds to try; best is chosen by bootstrap std-score
n_scales = 3
thr_grid = [0.95, 0.97, 0.99]
# ------------------------------------------------------------------------------

# Instantiate the ViT-based analyzer. Choose "cuda:0", "mps", or "cpu" as available.
m = ClsHiddenGeometry("google/vit-base-patch16-224", device="mps")

# Load precomputed hidden [CLS] states for each label.
# (If missing, you could instead call `compute_hidden_cls_states` to generate them.)
for label in labels:
    pt_path = f"data/google__vit-base-patch16-224/{label}_{n_samples}.pt"
    if not os.path.exists(pt_path):
        print(f"[warn] missing cached states: {pt_path} — this label will not plot correctly.")
        continue
    m.read_hidden_cls_states(pt_path)

# ----------------------------------------
# μ̂ trajectories for test images (cos & mk)
# ----------------------------------------
out_dir = "../plots/single_mu_hat_trajectories"
os.makedirs(out_dir, exist_ok=True)

# Layer axis (1..L) for plotting
L = int(m.config.num_hidden_layers)
x = np.arange(1, L + 1)

# Use a stable, distinct color per label across both figures
from matplotlib.colors import ListedColormap
idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1]  # pick distinct hues from tab20
cmap = ListedColormap([plt.cm.tab20.colors[i] for i in idx])
colors = {lab: cmap(i % 11) for i, lab in enumerate(labels)}

# Loop over test images; for each, compare against all class distributions.
for img_label in labels:
    img_path = f"data/test_images/{img_label}.jpg"
    if not os.path.exists(img_path):
        print(f"[warn] missing image: {img_path} — skipping")
        continue

    # Load the test image (ensure RGB for processor compatibility).
    img = Image.open(img_path).convert("RGB")

    # -------- COSINE --------
    # Cosine kernel works on unit-sphere features; no PCA is applied here.
    plt.figure(figsize=(10, 6))
    for y_label in labels:
        # μ̂_{p_{t,y}}(z_t) across layers, using cosine kernel.
        vals, _ = m.mu_hat_along_layers(
            class_label=y_label,
            n_samples=n_samples,
            image_z=img,
            true_label=img_label,   # only used to name internal plot paths in the method
            kernel="cosine",
            plot_res=False          # we do our own plotting below
        )
        plt.plot(x, vals, marker="o", linewidth=1.8, label=y_label, color=colors[y_label])
        print(f"[info] COS μ̂ for test img '{img_label}' vs class '{y_label}'")

    plt.xlabel("Layer t")
    plt.ylabel(r"$\hat{\mu}_{p_{t,y}}(z_t)$ [cosine]")
    plt.title(f"{img_label} vs all (cos)")
    plt.legend(loc="best", fontsize=9, ncols=1)
    plt.tight_layout()
    safe = img_label.replace("/", "__").replace(" ", "_")
    plt.savefig(os.path.join(out_dir, f"{safe}__vs_all__cos.png"), dpi=300)
    plt.close()

    # -------- MK (RBF) --------
    # MK-RBF blends multiple RBF bandwidths; we optionally apply per-layer PCA
    # (chosen by bootstrap std-score) before building the sigma grid.
    plt.figure(figsize=(10, 6))
    for y_label in labels:
        vals, _ = m.mu_hat_along_layers(
            class_label=y_label,
            n_samples=n_samples,
            image_z=img,
            true_label=img_label,
            kernel="mk_rbf",          # MK with RBF base kernels
            mk_n_scales=n_scales,     # number of sigma values around the center
            mk_ratio=0.5,             # geometric step between adjacent sigmas
            mk_sigma_center=None,     # use median distance heuristic per layer
            normalize_each_kernel=True,  # currently a no-op (kept for future use)
            weights=None,             # uniform mixture weights for μ̂ (by design)
            energy_threshold=thr_grid, # try multiple PCA thresholds; pick by std-score
            plot_res=False
        )
        plt.plot(x, vals, marker="o", linewidth=1.8, label=y_label, color=colors[y_label])
        print(f"[info] MK-RBF μ̂ for test img '{img_label}' vs class '{y_label}'")

    plt.xlabel("Layer t")
    plt.ylabel(r"$\hat{\mu}_{p_{t,y}}(z_t)$ [MK-RBF]")
    plt.title(f"{img_label} vs all (mk)")
    plt.legend(loc="best", fontsize=9, ncols=1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{safe}__vs_all__mk.png"), dpi=300)
    plt.close()

