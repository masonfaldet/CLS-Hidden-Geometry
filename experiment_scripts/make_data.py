# experiment_scripts/make_data.py
# ==============================================================================
# Goal
# ----
# Build class *groups* by:
#   1) Sampling ImageNet-1k training images for specified label IDs,
#   2) Extracting and caching per-layer hidden [CLS] states for each label,
#   3) Merging one or more labels into a single “group” entry, and
#   4) Persisting the merged caches to disk for reuse in later experiments.
#
# Why group?
# ----------
# Some downstream analyses may want to treat a *concept* (e.g., “clock”) as
# coming from multiple ImageNet classes, or just give a friendly alias for a
# single numeric label. Merging packs the rows together per layer and stores
# them under a clean key (e.g., "clock_700").
#
# Outputs
# -------
# - Torch files under ./data/{model_id_sanitized}/{group}_{n}.pt
#   (e.g., data/google__vit-base-patch16-224/clock_700.pt)
#
# Notes
# -----
# - Label IDs here are ImageNet-1k class *indices* that match the model’s
#   `config.id2label` mapping. If you prefer to use class *names*, you can
#   pass strings directly to `compute_hidden_cls_states`.
# - Adjust `num_samples` and `batch_size` depending on GPU/VRAM.
# ==============================================================================

from src.ClsHiddenGeometry import ClsHiddenGeometry

if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # 1) Load a pretrained ViT classifier with the geometry utilities attached.
    #    Choose a device that fits your setup: "cuda:0", "mps" (Apple), or "cpu".
    # --------------------------------------------------------------------------
    m = ClsHiddenGeometry("google/vit-base-patch16-224", device="mps")

    # Optional: print a couple label names to sanity-check your IDs.
    # id2label = getattr(m.config, "id2label", {})
    # for k in (409, 426, 635):
    #     print(k, id2label.get(k, "<unknown>"))

    # --------------------------------------------------------------------------
    # 2) Define groups.
    #    - Keys are friendly group names.
    #    - Values are lists of ImageNet class IDs to include in the group.
    #    - Many of these examples have only one ID: that’s fine—merging a
    #      single key effectively *renames* it to the friendly group label.
    # --------------------------------------------------------------------------
    groups = {
        "clock" : [409],
        "barometer" : [426],
        "compass" : [635],
        "banjo": [420],
        "cello": [486],
        "violin": [889],
        "orange": [950],
        "lemon": [951],
        "pomegranate": [957],
    }

    # --------------------------------------------------------------------------
    # 3) For each raw label ID, sample images and cache per-layer [CLS] states.
    #    - num_samples=700: number of training images per label to include.
    #    - batch_size=100 : forward-pass batch size (tune for your memory).
    #    This writes nothing to disk yet—results live in memory under keys like
    #    "{label_id}_700" inside `m.hidden_cls_states`.
    # --------------------------------------------------------------------------
    for labels in groups.values():
        for label in labels:
            m.compute_hidden_cls_states(class_label=label, num_samples=700, batch_size=100)
        print("group")

    # --------------------------------------------------------------------------
    # 4) Merge labels into group keys.
    #    - keys_to_merge is a list of (label, n_samples) pairs referencing the
    #      in-memory entries produced above.
    #    - new_label is the friendly group name.
    #    - delete_old=True removes the original label entries after merging.
    #    After this, you will have entries like "clock_700", "violin_700", etc.
    # --------------------------------------------------------------------------
    for g, labels in groups.items():
        m.merge_hidden_cls_states(
            keys_to_merge=[(label, 700) for label in labels],
            new_label=f"{g}",
            delete_old=True
        )
        print(f"merged {g}")

    # --------------------------------------------------------------------------
    # 5) Persist each merged group to disk.
    #    - Files are saved under ./data/{model_id_sanitized}/{group}_{n}.pt
    #    - `write_hidden_cls_states` will compute on demand if missing, but here
    #      we’ve already built everything in steps 3–4.
    # --------------------------------------------------------------------------
    for g in groups.keys():
        m.write_hidden_cls_states(f"{g}", 700)
