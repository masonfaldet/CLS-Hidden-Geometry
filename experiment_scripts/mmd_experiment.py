# experiment_scripts/mmd_experiment.py
# ==============================================================================
# Goal
# ----
# Compute layer-wise distances between multiple ImageNet classes using:
#   • Cosine MMD^2 on unit-sphere features
#   • MK-MMD^2 (multi-kernel RBF with per-layer sigma grids and optional PCA)
#
# We build pairwise tables (N × N × L) where:
#   - N = number of class labels
#   - L = number of transformer blocks (layers)
# Each entry [i, j, t] stores the distance between class_i and class_j at layer t.
#
# Then we:
#   1) Plot, for each anchor class i, the row curves {i vs j} across layers (cosine & MK).
#   2) Render heatmaps per layer (12 subplots for ViT-Base) showing all pairs.
#
# Prereqs
# -------
# - Precomputed hidden [CLS] states saved at:
#     data/google__vit-base-patch16-224/{label}_{n_samples}.pt
# - Run make_data.py
#
# Outputs
# -------
# - Per-class vs-all line plots (PDFs) under:
#     plots/single_pairs/{class}__vs_all__cos.pdf
#     plots/single_pairs/{class}__vs_all__mk.pdf
# - Heatmap grids (one figure per kernel) under:
#     plots/single_pairs/cos_heatmaps_layers.pdf
#     plots/single_pairs/mk_heatmaps_layers.pdf
# - Serialized pairwise tables saved to:
#     data/google__vit-base-patch16-224/pairwise_mmd_tables__{timestamp}.pkl
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from src.ClsHiddenGeometry import ClsHiddenGeometry
import pickle  # for saving the final tables

# ------------------------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------------------------
# Choose a set of classes. Names should match those used in your cached states.
labels = [
    "robin",
    "jay",
    "chikadee",
    "beagle",
    "lab",
    "boxer",
    "tabby",
    "siamese",
    "persian",
    "lizard",
    "snake"
]

# Number of cached samples per class used to build p_{t,y}.
# (These must match the .pt filenames you’ve prepared.)
n_samples = 700

# MK-RBF configuration:
# - n_scales: number of bandwidths around the per-layer median distance
# - thr_grid: PCA energy thresholds to try (best picked by bootstrap std-score)
n_scales = 3
thr_grid = [0.95, 0.97, 0.99]
MODEL_ID = "facebook/deit-base-patch16-224"
# ------------------------------------------------------------------------------

# Instantiate the analyzer. Use "cuda:0", "mps", or "cpu" as appropriate.
m = ClsHiddenGeometry(MODEL_ID, device="mps")

# Load precomputed hidden [CLS] states (.pt) for each label.
# If a file is missing, you can generate it via:
#   m.compute_hidden_cls_states(label, num_samples=n_samples, batch_size=64)
#   m.write_hidden_cls_states(label, n_samples)
for label in labels:
    base_dir = os.path.join("../", "data", MODEL_ID.replace("/", "__"))
    pt_path = base_dir + f"/{label}_{n_samples}.pt"
    if not os.path.exists(pt_path):
        print(f"[warn] missing cached states: {pt_path} — this label will not compute correctly.")
        continue
    m.read_hidden_cls_states(pt_path)

# ----------------------------------------
# Build pairwise tables (N × N × L)
# ----------------------------------------
N = len(labels)
L = int(m.config.num_hidden_layers)  # ViT-Base = 12; adjust if your model differs

# Preallocate result tensors: [i, j, layer]
cos_tbl = np.zeros((N, N, L), dtype=float)
mk_tbl = np.zeros((N, N, L), dtype=float)

# Fill only the upper triangle (i < j) and mirror to (j, i) to save work.
for i in range(N):
    for j in range(i + 1, N):
        # --- Cosine MMD^2 across layers (returns a 1D array of length L) ---
        cos_vals = m.cosine_mmd2_between_labels_along_layers(
            class_label_A=labels[i], nA=n_samples,
            class_label_B=labels[j], nB=n_samples,
        )
        cos_tbl[i, j, :] = cos_vals
        cos_tbl[j, i, :] = cos_vals  # symmetric

        # --- MK-MMD^2 across layers (returns vals, sigma_lists, std_scores) ---
        # Uses an RBF sigma grid per layer; if `thr_grid` is supplied, a PCA energy
        # threshold is chosen by maximizing standardized MMD^2.
        mk_vals, _, _ = m.mk_mmd2_between_labels_along_layers(
            class_label_A=labels[i], nA=n_samples,
            class_label_B=labels[j], nB=n_samples,
            n_scales=n_scales,
            energy_threshold=thr_grid,  # try multiple PCA thresholds; pick best per layer
        )
        mk_tbl[i, j, :] = mk_vals
        mk_tbl[j, i, :] = mk_vals  # symmetric
    print(f"[info] completed: {labels[i]} vs all")

# If you want to load previously saved tables instead, uncomment:
# pkl_path = "data/google__vit-base-patch16-224/pairwise_mmd_tables__20250924-214854.pkl"
# with open(pkl_path, "rb") as f:
#     d = pickle.load(f)
#     cos_tbl = d["cos_tbl"]
#     mk_tbl = d["mk_tbl"]

# ----------------------------------------
# Per-class “vs all others” line plots
# ----------------------------------------
out_dir = f"../plots/{m.model_id.replace('/', '__')}/single_pairs"
os.makedirs(out_dir, exist_ok=True)

# Use a categorical colormap with enough distinct colors for the N classes.
from matplotlib.colors import ListedColormap
idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1]  # pick distinct hues from tab20
cmap = ListedColormap([plt.cm.tab20.colors[i] for i in idx])

colors = {j: cmap(j % 11) for j in range(N)}
x = np.arange(1, L + 1)  # layer indices 1..L

for i in range(N):
    safe_name = labels[i].replace("/", "__").replace(" ", "_")

    # ----- COSINE FIG -----
    # Plot row i (i vs j for all j) across layers.
    d_cos = cos_tbl[i, :, :]  # shape (N, L)
    plt.figure(figsize=(10, 6))
    for j in range(N):
        if j == i:
            continue  # skip self-distance
        plt.plot(x, d_cos[j, :], marker="o", linewidth=2.0, label=labels[j], color=colors[j])
    plt.xlabel("Layer t", fontsize=16)
    plt.ylabel("Cosine MMD$^2$", fontsize=16)
    plt.legend(loc="best", fontsize=16, ncols=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{safe_name}__vs_all__cos.pdf"))
    plt.close()

    # ----- MK FIG -----
    # Same plot for MK-MMD^2.
    d_mk = mk_tbl[i, :, :]
    plt.figure(figsize=(10, 6))
    for j in range(N):
        if j == i:
            continue
        plt.plot(x, d_mk[j, :], marker="o", linewidth=2.0, label=labels[j], color=colors[j])
    plt.xlabel("Layer t", fontsize=16)
    plt.ylabel("MK-MMD$^2$", fontsize=16)
    plt.legend(loc="best", fontsize=16, ncols=2)
    plt.legend(loc="best", fontsize=16, ncols=2)  # (duplicate legend is harmless; keeps behavior identical)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{safe_name}__vs_all__mk.pdf"))
    plt.close()

# -------------------------------------------------
# Heatmaps: one figure per kernel, 12 subplots (1 per layer for ViT-Base)
# -------------------------------------------------
L = cos_tbl.shape[2]             # number of layers
rows, cols = 3, 4                # 3x4 grid = 12 (adjust if your model has !=12 layers)
assert rows * cols >= L, "Grid too small for number of layers."

# ---------- COSINE heatmaps ----------
vmin_c = np.nanmin(cos_tbl)      # shared color limits across layers (comparability)
vmax_c = np.nanmax(cos_tbl)

fig_c, axes_c = plt.subplots(rows, cols, figsize=(18, 12), constrained_layout=True)
for t in range(L):
    r, c = divmod(t, cols)
    ax = axes_c[r, c]
    im_c = ax.imshow(cos_tbl[:, :, t], vmin=vmin_c, vmax=vmax_c, cmap="viridis", aspect="auto")
    ax.set_title(f"Layer {t + 1}", fontsize=16)

    # Tidy tick labels: only on bottom row / leftmost column
    if r == rows - 1:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
    else:
        ax.set_xticks([])
    if c == 0:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=16)
    else:
        ax.set_yticks([])

# Hide any unused subplots (if rows*cols > L)
for k in range(L, rows * cols):
    axes_c.flat[k].axis("off")

cbar_c = fig_c.colorbar(im_c, ax=axes_c.ravel().tolist(), shrink=0.9)
cbar_c.set_label("Cosine MMD$^2$")
fig_c.suptitle("Cosine MMD$^2$ by layer", y=1.02, fontsize=18)
fig_c.savefig(os.path.join(out_dir, "cos_heatmaps_layers.pdf"))
plt.close(fig_c)

# ---------- MK heatmaps ----------
vmin_m = np.nanmin(mk_tbl)
vmax_m = np.nanmax(mk_tbl)

fig_m, axes_m = plt.subplots(rows, cols, figsize=(18, 12), constrained_layout=True)
for t in range(L):
    r, c = divmod(t, cols)
    ax = axes_m[r, c]
    im_m = ax.imshow(mk_tbl[:, :, t], vmin=vmin_m, vmax=vmax_m, cmap="viridis", aspect="auto")
    ax.set_title(f"Layer {t + 1}", fontsize=16)

    if r == rows - 1:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
    else:
        ax.set_xticks([])
    if c == 0:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=16)
    else:
        ax.set_yticks([])

for k in range(L, rows * cols):
    axes_m.flat[k].axis("off")

cbar_m = fig_m.colorbar(im_m, ax=axes_m.ravel().tolist(), shrink=0.9)
cbar_m.set_label("MK-MMD$^2$")
fig_m.suptitle("MK-MMD$^2$ by layer", y=1.02, fontsize=18)
fig_m.savefig(os.path.join(out_dir, "mk_heatmaps_layers.pdf"))
plt.close(fig_m)

# =========================
# Save pairwise MMD tables
# =========================
import datetime as dt

save_dir = f"../data/{m.model_id.replace('/', '__')}"
os.makedirs(save_dir, exist_ok=True)

timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
base = f"pairwise_mmd_tables__{timestamp}"

# Package payload (metadata + numeric arrays). Numpy is fine for .pkl.
payload = {
    "model_id": m.model_id,
    "labels": labels,
    "n_samples": n_samples,
    "num_layers": int(m.config.num_hidden_layers),
    "mk_params": {"n_scales": n_scales, "thr_grid": thr_grid},
    "cos_tbl": cos_tbl,  # shape (N, N, L)
    "mk_tbl": mk_tbl,    # shape (N, N, L)
}

pkl_path = os.path.join(save_dir, base + ".pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"[done] saved PKL: {pkl_path}")
