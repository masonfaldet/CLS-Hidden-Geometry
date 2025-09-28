# ClsHiddenGeometry.py
# ------------------------------------------------------------
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from gtda.mapper import Eccentricity
from gtda.mapper import Entropy
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, ViTForImageClassification

try:
    from datasets import (
        load_dataset,
        load_from_disk,
        ClassLabel,
        Image as HFImage,
        Dataset,
    )
except Exception as e:
    load_dataset = None  # type: ignore[assignment]
    load_from_disk = None  # type: ignore[assignment]
    ClassLabel = None  # type: ignore[assignment]
    HFImage = None  # type: ignore[assignment]
    Dataset = None  # type: ignore[assignment]

ImageLike = Union[Image.Image]

_IMNET256_REPO = "evanarlian/imagenet_1k_resized_256"  # 256 (shorter side), has train/val/test
_IMNET_LOCAL_ROOT = os.path.join(".", "ImageNet")
_IMNET_LOCAL_TRAIN = os.path.join(_IMNET_LOCAL_ROOT, "imagenet_1k_resized_256", "train")


# ---------------- PCA projection utility (optionally whitening) ----------------
class EnergyWhitener:
    """
    PCA projector that chooses n_components so cumulative explained variance >= energy_threshold.
    (Naming note: this is a PCA *projection*; set whiten=True to actually whiten.)

    Use wherever you need dimension reduction (per key, per layer, or per union) — no global state assumed.
    """
    def __init__(
        self,
        energy_threshold: float = 0.90,
        whiten: bool = False,
        max_components: int | None = None,
        svd_solver: str = "full",
        random_state: int = 42,
    ):
        if not (0.0 < energy_threshold <= 1.0):
            raise ValueError("energy_threshold must be in (0, 1].")
        self.energy_threshold = float(energy_threshold)
        self.max_components = max_components
        self.svd_solver = svd_solver
        self.random_state = random_state
        self.whiten = whiten

        self.pca: PCA | None = None
        self.fitted: bool = False
        self.n_components_: int | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cum_explained_variance_: np.ndarray | None = None

    def fit(self, Xref: np.ndarray):
        """Fit PCA, pick r == min components meeting the energy threshold, then fit PCA(n_components=r)."""
        full_pca = PCA(n_components=None, svd_solver=self.svd_solver, whiten=False, random_state=self.random_state)
        full_pca.fit(Xref)

        evr = np.asarray(full_pca.explained_variance_ratio_)
        cum = np.cumsum(evr)
        r = int(np.searchsorted(cum, self.energy_threshold, side="left") + 1)

        rank_cap = min(Xref.shape[0], Xref.shape[1])
        if self.max_components is not None:
            r = min(r, int(self.max_components))
        r = max(1, min(r, rank_cap))

        self.pca = PCA(
            n_components=r,
            svd_solver=self.svd_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        self.pca.fit(Xref)

        self.n_components_ = r
        self.explained_variance_ratio_ = evr
        self.cum_explained_variance_ = cum
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or self.pca is None:
            raise RuntimeError("EnergyWhitener not fitted. Call fit(Xref) first.")
        return self.pca.transform(X)


class ClsHiddenGeometry(ViTForImageClassification):
    """
    Vision Transformer classifier with tooling for [CLS]–state geometry, ImageNet-1k.

    This subclass wraps a pretrained ImageNet-1k ViT checkpoint and provides a
    research-oriented API for:
      • Capturing per-layer hidden `[CLS]` states for a given class.
      • Computing cosine MMD and multi-kernel MMD (MK-MMD) across layers/labels.
      • Evaluating per-layer kernel mean embeddings μ̂_{p_{t,y}}(z_t) for query images.
      • Optional PCA energy projection (via `EnergyWhitener`) at key steps.
      • Lightweight caching to avoid recomputation (sphere features, Gram blocks, PCAs).

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier (e.g., "google/vit-base-patch16-224") for a
        fine-tuned ImageNet-1k classifier.
    device : str, optional
        Torch device to move the model to, e.g. "cuda:0", "mps", or "cpu".
        If omitted, uses the default device of the loaded model.
    trust_remote_code : bool, default=False
        Passed to `from_pretrained` for both model and processor.
    **from_pretrained_kwargs
        Forwarded verbatim to `ViTForImageClassification.from_pretrained`.

    Attributes
    ----------
    model_id : str
        The identifier used to load the model/processor.
    image_processor : transformers.ImageProcessor
        Paired image processor associated with `model_id`.
    train_ds : datasets.Dataset
        Local, on-disk ImageNet-1k (resized to 256) **train** split.
    image_col : str
        Column name in `train_ds` containing images.
    label_col : str
        Column name in `train_ds` containing labels (`ClassLabel`).
    hidden_cls_states : dict[str, list[torch.Tensor]]
        Cached per-layer `[CLS]` matrices for keys of the form "{label}_{n}".
        Each value is a list (length L) of tensors with shape `(n, d_model)`.
    kernel_cache : dict[str, KernelCache]
        Per-key cache for derived artifacts (unit-sphere arrays, Gram matrices, PCAs).

    Notes
    -----
    • Dataset layout: this class expects an HF `datasets` copy of ImageNet-1k resized to
      256px, saved locally under `./ImageNet/imagenet_1k_resized_256/{split}/`.
      It will be created automatically if missing.

    • PCA policy: for MK-MMD and μ̂ computations, PCA “energy-threshold” projections can be
      applied per key, per layer, or on pooled unions depending on the routine
      (see method docstrings for details).

    • Construction uses `__new__` rather than `__init__` to return a patched instance of the
      pretrained base class while attaching extra attributes and caches.
    """

    # ---------------- Construction ----------------
    def __new__(
        cls,
        model_id: str,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        **from_pretrained_kwargs,
    ):
        """
        Allocate and initialize a pretrained ViT classifier, then upgrade it to this subclass.

        This method:
          1) Loads `ViTForImageClassification` from `model_id`.
          2) Rebinds its `__class__` to `ClsHiddenGeometry'.
          3) Attaches the paired `AutoImageProcessor`.
          4) Initializes in-memory caches/stores.
          5) Ensures a local ImageNet-1k (resized-256) **train** split exists and loads it.
          6) Detects the image/label columns.
          7) Moves to `device` (if provided) and sets eval mode.

        Returns
        -------
        ClsHiddenGeometry
            A fully initialized instance backed by the pretrained weights.

        Notes
        -----
        • Using `__new__` allows us to return the already-constructed HF model while
          augmenting it with additional attributes and methods from this subclass.
        """
        # 1) Load the pretrained classification backbone.
        base = ViTForImageClassification.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            **from_pretrained_kwargs,
        )

        # 2) Rebind the class so the returned object exposes subclass methods/APIs.
        base.__class__ = cls  # type: ignore[attr-defined]

        # 3) Attach bookkeeping + paired processor.
        base.model_id = model_id  # type: ignore[attr-defined]
        base.image_processor = AutoImageProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )  # type: ignore[attr-defined]

        # 4) Initialize caches/stores used by higher-level methods.
        base.hidden_cls_states = {}         # type: ignore[attr-defined]
        base.kernel_cache = {}              # type: ignore[attr-defined]

        # 5) Ensure local ImageNet-1k (256) TRAIN split exists on disk, then load it.
        base._require_datasets()            # type: ignore[attr-defined]
        base._ensure_local_imagenet256()    # type: ignore[attr-defined]
        base.train_ds = load_from_disk(_IMNET_LOCAL_TRAIN)  # type: ignore[attr-defined]

        # 6) Detect column names for images/labels in the dataset.
        base.image_col, base.label_col = base._find_cols(base.train_ds)  # type: ignore[attr-defined]

        # 7) Optional device placement and eval mode for inference-friendly defaults.
        if device is not None:
            base.to(device)
        base.eval()
        return base

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        **from_pretrained_kwargs,
    ):
        """
        No-op initializer.

        All heavy lifting is performed in `__new__`; `__init__` is intentionally empty to
        avoid re-initialization side effects after the subclass rebind.
        """
        # All work done in __new__
        pass

    # ---------------- Convenience helpers ----------------
    @property
    def labels(self) -> List[str]:
        """
        Class label names in index order.

        Returns
        -------
        list of str
            Label names derived from `self.config.id2label` ordered by class index.
            If `id2label` is missing or empty, returns an empty list.
        """
        id2label = getattr(self.config, "id2label", None) or {}
        return [id2label[i] for i in sorted(id2label.keys())] if id2label else []

    @property
    def model(self):
        """
        Identity accessor for compatibility with code that expects a `.model` attribute.

        Returns
        -------
        Any
            Returns `self`. Useful when mirroring libraries that expose `.model`.
        """
        return self

    def preprocess(
            self,
            images: Union[ImageLike, Sequence[ImageLike]],
            return_tensors: str = "pt",
    ):
        """
        Preprocess one or more images using the paired image processor.

        Parameters
        ----------
        images : ImageLike or sequence of ImageLike
            A single image or a sequence. If a single `PIL.Image.Image` is provided,
            it is wrapped into a list automatically.
        return_tensors : {"pt", "np"}, default="pt"
            Tensor format requested from the processor.

        Returns
        -------
        dict[str, Any]
            Batch dict as returned by `self.image_processor`, typically including
            keys like `pixel_values`.

        Notes
        -----
        This is a thin wrapper around `self.image_processor(...)`.
        """
        # Normalize to a list input expected by HF processors.
        if isinstance(images, Image.Image):
            images = [images]
        return self.image_processor(images=list(images), return_tensors=return_tensors)  # type: ignore[attr-defined]

    @torch.no_grad()
    def classify(
            self,
            images: Union[ImageLike, Sequence[ImageLike]],
            top_k: int = 5,
            device: Optional[str] = None,
    ) -> List[List[dict]]:
        """
        Run a forward pass and return top-k class predictions per image.

        Parameters
        ----------
        images : ImageLike or sequence of ImageLike
            A single image or a sequence of images to classify.
        top_k : int, default=5
            The number of highest-probability classes to return per image.
            Clamped to the number of classes.
        device : str, optional
            Torch device override (e.g., "cuda:0", "mps", "cpu"). If omitted,
            uses the device of the first model parameter.

        Returns
        -------
        list of list of dict
            For each input image, a list of length `top_k` (or fewer) with entries:
            `{"label": str, "score": float, "index": int}`.
            - `label` is resolved via `config.id2label` when available, else the index as string.
            - `score` is the softmax probability.
            - `index` is the integer class id.

        Notes
        -----
        The method temporarily switches the module to eval mode and restores
        the previous training state upon exit.
        """
        was_training = self.training
        self.eval()

        batch = self.preprocess(images)
        device_ = device or next(self.parameters()).device
        batch = {k: v.to(device_) for k, v in batch.items()}

        outputs = self(**batch)
        logits = outputs.logits
        probs = logits.softmax(dim=-1)

        topk = min(top_k, probs.shape[-1])
        scores, indices = probs.topk(topk, dim=-1)

        id2label = getattr(self.config, "id2label", {}) or {}

        def idx_to_label(i: int) -> str:
            return id2label.get(int(i), str(int(i)))

        results: List[List[dict]] = []
        for row_scores, row_indices in zip(scores, indices):
            entries = []
            for s, i in zip(row_scores.tolist(), row_indices.tolist()):
                entries.append(
                    {"label": idx_to_label(i), "score": float(s), "index": int(i)}
                )
            results.append(entries)

        if was_training:
            self.train()
        return results

    # ----------------- Dataset utilities ------------------
    def _require_datasets(self):
        """
        Ensure the optional 'datasets' dependency is available.

        Raises
        ------
        ImportError
            If either `load_dataset` or `load_from_disk` is not importable.
        """
        if load_dataset is None or load_from_disk is None:
            raise ImportError(
                "The 'datasets' package is required. Install via: pip install datasets"
            )

    def _ensure_local_imagenet256(self) -> None:
        """
        Ensure a local on-disk copy of resized ImageNet-1k (256 px) training split exists.

        The expected path is:
        `./ImageNet/imagenet_1k_resized_256/train/`

        Behavior
        --------
        - If the directory already exists and contains `dataset_info.json`, do nothing.
        - Otherwise, loads the dataset from `_IMNET256_REPO` and saves it to disk.

        Notes
        -----
        This uses Hugging Face `datasets` under the hood and assumes `_IMNET_LOCAL_TRAIN`
        and `_IMNET256_REPO` are configured module-level constants.
        """
        train_dir = _IMNET_LOCAL_TRAIN
        # Fast path: already present on disk.
        if os.path.isdir(train_dir) and os.path.exists(os.path.join(train_dir, "dataset_info.json")):
            return
        os.makedirs(train_dir, exist_ok=True)
        ds_train = load_dataset(_IMNET256_REPO, split="train")  # type: ignore[misc]
        ds_train.save_to_disk(train_dir)

    def _find_cols(self, ds: "Dataset") -> Tuple[str, str]:
        """
        Infer the image and label column names for a `datasets.Dataset`.

        Parameters
        ----------
        ds : datasets.Dataset
            Dataset whose features will be inspected.

        Returns
        -------
        (str, str)
            A tuple `(image_col, label_col)`.

        Raises
        ------
        ValueError
            If no image-like column or no `ClassLabel` label column can be found.

        Notes
        -----
        Heuristics:
        - Prefer common image column names: {"image", "img", "image_file", "image_path"}.
        - Otherwise, pick the first feature whose type is `datasets.Image`.
        - For labels, prefer a `"label"` column of type `datasets.ClassLabel`; else use
          the first `ClassLabel`-typed feature found.
        """
        feats = ds.features

        # --- Find image column ---
        image_col = None
        for cand in ("image", "img", "image_file", "image_path"):
            if cand in feats and (HFImage is None or isinstance(feats[cand], HFImage)):
                image_col = cand
                break
        if image_col is None:
            for k, v in feats.items():
                if HFImage is not None and isinstance(v, HFImage):
                    image_col = k
                    break
        if image_col is None:
            raise ValueError("Could not locate an image column in the dataset.")

        # --- Find label column ---
        label_col = None
        if "label" in feats and (ClassLabel is None or isinstance(feats["label"], ClassLabel)):
            label_col = "label"
        else:
            for k, v in feats.items():
                if ClassLabel is not None and isinstance(v, ClassLabel):
                    label_col = k
                    break
        if label_col is None:
            raise ValueError("Could not locate a ClassLabel label column in the dataset.")
        return image_col, label_col

    def _label_to_index(
            self,
            ds: "Dataset",
            label_col: str,
            class_label: Union[str, int],
    ) -> int:
        """
        Map a user-provided class label (string name or int id) to its class index.

        Parameters
        ----------
        ds : datasets.Dataset
            Dataset providing the label space.
        label_col : str
            Column name containing `ClassLabel`-typed labels.
        class_label : str or int
            Either the exact class name (case-insensitive fallback) or an integer id.

        Returns
        -------
        int
            The resolved class index in `[0, num_classes)`.

        Raises
        ------
        ValueError
            If the integer id is out of range, class names are unavailable, or the
            provided string cannot be matched to any class.

        Notes
        -----
        String matching tries an exact match first, then a lowercase match.
        """
        feat = ds.features[label_col]
        if isinstance(class_label, int):
            if 0 <= class_label < feat.num_classes:
                return int(class_label)
            raise ValueError(f"class_label index {class_label} is out of range [0, {feat.num_classes}).")

        names = list(getattr(feat, "names", []))
        if not names:
            raise ValueError("Class names not available in dataset metadata.")
        if class_label in names:
            return names.index(class_label)

        # Case-insensitive fallback.
        lower_map = {n.lower(): i for i, n in enumerate(names)}
        key = str(class_label).lower()
        if key in lower_map:
            return lower_map[key]
        raise ValueError(
            f"Class label '{class_label}' not found. First few labels: {names[:10]} ... (total {len(names)})"
        )

    def _make_key(self, class_label: Union[str, int], n_samples: Union[str, int]) -> str:
        """
        Build a simple cache key for (class_label, n_samples).

        Parameters
        ----------
        class_label : str or int
            Class identifier; commas and spaces are sanitized for the key.
        n_samples : str or int
            Number of samples or a tag describing the sampling strategy.

        Returns
        -------
        str
            A key of the form `"{class_label}_{n_samples}"` with commas removed and
            spaces replaced by underscores in `class_label`.
        """
        return f"{str(class_label).replace(',', '').replace(' ', '_')}_{n_samples}"

    # ------------ CLS functions -------------------------
    @torch.no_grad()
    def compute_hidden_cls_states(
            self,
            class_label: Union[str, int],
            num_samples: int,
            batch_size: int,
            *,
            seed: int = 0,
    ) -> None:
        """
        Sample images of a class from the local ImageNet train split and cache per-layer [CLS] states.

        For each sampled image, this records the hidden state of the `[CLS]` token
        after every transformer block (i.e., excluding the patch embedding layer).

        Parameters
        ----------
        class_label : str or int
            Class identifier. Strings are matched to dataset class names (case-insensitive fallback);
            integers are treated as class indices.
        num_samples : int
            Number of images to sample from the class.
        batch_size : int
            Batch size for forward passes.
        seed : int, default=0
            RNG seed used for sampling images.

        Returns
        -------
        None
            Results are stored in `self.hidden_cls_states` under the key
            `f"{class_label}_{num_samples}"`.

        Side Effects
        ------------
        self.hidden_cls_states[key] : list[torch.Tensor]
            A list of length `num_hidden_layers`, where each tensor has shape
            `(num_samples, d_model)` containing the per-image [CLS] hidden states
            at that layer.

        Raises
        ------
        RuntimeError
            If `config.num_hidden_layers` cannot be determined or hidden states
            returned by the backbone are inconsistent.
        ValueError
            If there are fewer than `num_samples` images available for the class.

        Notes
        -----
        - Assumes `self.train_ds`, `self.image_col`, and `self.label_col` are initialized
          and refer to an on-disk HF `datasets.Dataset` for ImageNet-1k resized to 256.
        - Uses `self.vit(..., output_hidden_states=True)` and extracts `[CLS]` as `[:, 0, :]`.
        """
        self._require_datasets()
        random.seed(seed)
        device = next(self.parameters()).device

        num_layers = int(getattr(self.config, "num_hidden_layers", 0))
        if num_layers <= 0:
            raise RuntimeError("Could not infer number of transformer blocks from config.num_hidden_layers.")

        ds = self.train_ds
        image_col, label_col = self.image_col, self.label_col

        # Resolve class index from user input.
        target_idx = self._label_to_index(ds, label_col, class_label)

        # Collect candidate row indices for this class.
        labels = ds[label_col]
        idxs = [i for i, y in enumerate(labels) if int(y) == int(target_idx)]
        if len(idxs) < num_samples:
            raise ValueError(
                f"Requested {num_samples} samples but found only {len(idxs)} for class '{class_label}'."
            )
        chosen = random.sample(idxs, k=num_samples)

        # Some datasets need Python format for PIL images; fall back gracefully.
        try:
            ds_py = ds.with_format(type="python")
        except Exception:
            ds_py = ds

        # Accumulate per-layer batches to avoid huge tensors in memory.
        per_layer_batches: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            rows = [ds_py[int(i)] for i in chosen[start:end]]
            imgs: List[Image.Image] = [row[image_col] for row in rows]

            # Encode and move to the model device.
            enc = self.image_processor(images=imgs, return_tensors="pt")  # type: ignore[attr-defined]
            enc = {k: v.to(device) for k, v in enc.items()}

            vit_outputs = self.vit(**enc, output_hidden_states=True)  # type: ignore[attr-defined]
            hidden_states = vit_outputs.hidden_states
            if hidden_states is None or len(hidden_states) != (num_layers + 1):
                raise RuntimeError("Unexpected hidden_states; expected embeddings + one per block.")

            # Extract [CLS] token after each block (skip embeddings at index 0).
            for k in range(num_layers):
                hs_k = hidden_states[k + 1][:, 0, :].detach().cpu()
                per_layer_batches[k].append(hs_k)

        # Concatenate minibatches per layer and sanity-check row count.
        per_layer_tensors: List[torch.Tensor] = []
        for k in range(num_layers):
            layer_tensor = torch.cat(per_layer_batches[k], dim=0)
            if layer_tensor.shape[0] != num_samples:
                raise RuntimeError(f"Layer {k}: got {layer_tensor.shape[0]} rows, expected {num_samples}.")
            per_layer_tensors.append(layer_tensor)

        key = self._make_key(class_label, num_samples)
        self.hidden_cls_states[key] = per_layer_tensors  # type: ignore[attr-defined]

    def write_hidden_cls_states(self, class_label, n_samples):
        """
        Persist cached [CLS] hidden states for a class to disk.

        Ensures the corresponding entry exists in `self.hidden_cls_states`, computing
        it if necessary, then saves to:
        `./data/{self.model_id.replace('/', '__')}/{class_label}_{n_samples}.pt`

        Parameters
        ----------
        class_label : str or int
            Class name or index used to build the cache key.
        n_samples : int
            Number of samples in the cached entry.

        Returns
        -------
        None
        """
        key = self._make_key(class_label, n_samples)
        if key not in self.hidden_cls_states:
            bs = min(64, int(n_samples))
            self.compute_hidden_cls_states(
                class_label=class_label, num_samples=int(n_samples), batch_size=bs
            )
        out_dir = os.path.join("./", "data", self.model_id.replace("/", "__"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{key}.pt")
        torch.save(self.hidden_cls_states[key], out_path)

    def read_hidden_cls_states(self, path):
        """
        Load per-layer [CLS] hidden states from disk and store them in the in-memory cache.

        Expects filenames of the form:
        `.../<class_label>_<n_samples>.pt`

        Parameters
        ----------
        path : str
            Path to a `.pt` file containing a list of `torch.Tensor` objects
            (one per transformer block), each of shape `(n_samples, d_model)`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the filename does not end with `_<n_samples>.pt`, or the loaded
            object is not a list/tuple of tensors.
        """
        base = os.path.basename(path)
        if base.endswith(".pt"):
            base = base[:-3]
        idx = base.rfind("_")
        if idx == -1:
            raise ValueError("Filename must end with '_<n_samples>.pt' to recover the key.")
        class_label = base[:idx]
        try:
            n_samples = int(base[idx + 1:])
        except ValueError:
            raise ValueError("Could not parse <n_samples> as an integer from the filename.")

        data = torch.load(path, map_location="cpu")
        if not isinstance(data, (list, tuple)) or not all(isinstance(t, torch.Tensor) for t in data):
            raise ValueError("Loaded object must be a list of torch.Tensors (one per transformer block).")

        num_layers = int(getattr(self.config, "num_hidden_layers", 0))
        if num_layers and len(data) != num_layers:
            print(f"Warning: loaded {len(data)} layer tensors, model has {num_layers} blocks.")

        key = f"{class_label}_{n_samples}"
        self.hidden_cls_states[key] = list(data)
        return None

    def clear_hidden_cls_states(self, class_label, n_samples):
        """
        Remove a cached [CLS] entry from memory.

        Parameters
        ----------
        class_label : str or int
            Class name or index used to build the cache key.
        n_samples : int
            Number of samples in the cached entry.

        Returns
        -------
        None
        """
        key = self._make_key(class_label, n_samples)
        if key in self.hidden_cls_states:
            self.hidden_cls_states.pop(key)
        return None

    def merge_hidden_cls_states(
            self,
            keys_to_merge: Sequence[Tuple[Union[str, int], Union[str, int]]],
            new_label: str,
            delete_old: bool = True,
    ) -> None:
        """
        Concatenate multiple cached [CLS] entries (per layer) into a single merged entry.

        Parameters
        ----------
        keys_to_merge : sequence of (label, n_samples)
            Pairs referencing existing cache keys to merge. All entries must have
            the same number of layers and `d_model`.
        new_label : str
            Label prefix used for the merged key. The merged `n_samples` equals
            the sum of rows across inputs.
        delete_old : bool, default=True
            If True, remove original entries after merging.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any referenced key is missing.
        RuntimeError
            If layer counts or tensor shapes are inconsistent across entries.
        """
        keys = [self._make_key(lbl, n) for (lbl, n) in keys_to_merge]
        for k in keys:
            if k not in self.hidden_cls_states:
                raise ValueError(f"No CLS hidden state for key {k}; run compute_hidden_cls_states first.")

        num_layers = int(getattr(self.config, "num_hidden_layers", 0))
        if num_layers <= 0:
            raise RuntimeError("Could not infer number of transformer blocks from config.num_hidden_layers.")

        first = self.hidden_cls_states[keys[0]]
        if len(first) != num_layers:
            raise RuntimeError(f"Entry {keys[0]} has {len(first)} layers; model reports {num_layers}.")
        d_model = first[0].shape[1]

        # Validate consistency across all entries.
        for k in keys:
            layers = self.hidden_cls_states[k]
            if len(layers) != num_layers:
                raise RuntimeError(f"Entry {k} has {len(layers)} layers; expected {num_layers}.")
            for i, t in enumerate(layers):
                if t.ndim != 2 or t.shape[1] != d_model:
                    raise RuntimeError(f"Entry {k}, layer {i} has shape {tuple(t.shape)}; expected (*, {d_model}).")

        # Concatenate per-layer tensors vertically.
        merged: List[torch.Tensor] = []
        for i in range(num_layers):
            to_cat = [self.hidden_cls_states[k][i] for k in keys]
            merged.append(torch.cat(to_cat, dim=0))

        total_rows = sum(self.hidden_cls_states[k][0].shape[0] for k in keys)
        new_key = self._make_key(new_label, total_rows)
        self.hidden_cls_states[new_key] = merged

        if delete_old:
            for k in keys:
                self.hidden_cls_states.pop(k, None)
        return None

    # --------------- Statistical Functions ----------------------
    def compute_shannon_entropy(self, class_label, n_samples, plot=True):
        """
        Compute per-sample Shannon entropy of [CLS] states at each layer.

        For each layer `ℓ`, applies `gtda.mapper.Entropy().fit_transform` to the matrix
        `X_ℓ ∈ R^{n_samples × d_model}` and horizontally stacks the resulting column
        vectors into an array of shape `(n_samples, n_layers)`.

        Parameters
        ----------
        class_label : str or int
            Class identifier used to locate cached states; computed on-demand if missing.
        n_samples : int
            Number of samples in the cached entry.
        plot : bool, default=True
            If True, saves a boxplot of entropies per layer to
            `./plots/{model_id}/{key}_entropies`.

        Returns
        -------
        ndarray of shape (n_samples, n_layers)
            Shannon entropies per sample (rows) and per layer (columns).

        Notes
        -----
        This function materializes tensors to CPU and converts them to NumPy arrays
        before calling `gtda` transforms.
        """
        key = self._make_key(class_label, n_samples)
        if key not in self.hidden_cls_states:
            bs = min(64, int(n_samples))
            self.compute_hidden_cls_states(class_label=class_label, num_samples=int(n_samples), batch_size=bs)

        states_t = self.hidden_cls_states[key]
        states_np = [state.detach().cpu().numpy() for state in states_t]

        ent = Entropy()
        entropy_ls = [ent.fit_transform(state) for state in states_np]
        entropies_arr = np.hstack(entropy_ls)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.boxplot(entropies_arr, positions=np.arange(1, len(states_np) + 1))
            plt.xlabel('Layer')
            plt.ylabel('Shannon Entropy')
            plt.title('Shannon entropies of hidden [CLS] states')
            plt.tight_layout()
            out_path = f"./plots/{self.model_id.replace('/', '__')}/{key}_entropies"
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            plt.savefig(out_path, dpi=300)
            plt.close()
        return entropies_arr

    def compute_eccentricity(self, class_label, n_samples, plot=True):
        """
        Compute per-sample eccentricity of [CLS] states at each layer.

        For each layer `ℓ`, applies `gtda.mapper.Eccentricity().fit_transform` to the
        matrix `X_ℓ ∈ R^{n_samples × d_model}` and horizontally stacks the outputs into
        an array of shape `(n_samples, n_layers)`.

        Parameters
        ----------
        class_label : str or int
            Class identifier used to locate cached states; computed on-demand if missing.
        n_samples : int
            Number of samples in the cached entry.
        plot : bool, default=True
            If True, saves a boxplot of eccentricities per layer to
            `./plots/{model_id}/{key}_eccentricities`.

        Returns
        -------
        ndarray of shape (n_samples, n_layers)
            Eccentricities per sample (rows) and per layer (columns).
        """
        key = self._make_key(class_label, n_samples)
        if key not in self.hidden_cls_states:
            bs = min(64, int(n_samples))
            self.compute_hidden_cls_states(class_label=class_label, num_samples=int(n_samples), batch_size=bs)

        states_t = self.hidden_cls_states[key]
        states_np = [state.detach().cpu().numpy() for state in states_t]

        ecc = Eccentricity()
        eccentricity_ls = [ecc.fit_transform(state) for state in states_np]
        eccentricity_arr = np.hstack(eccentricity_ls)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.boxplot(eccentricity_arr, positions=np.arange(1, len(states_np) + 1))
            plt.xlabel('Layer')
            plt.ylabel('Eccentricity')
            plt.title('Eccentricities of hidden [CLS] states')
            plt.tight_layout()
            out_path = f"./plots/{self.model_id.replace('/', '__')}/{key}_eccentricities"
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            plt.savefig(out_path, dpi=300)
            plt.close()
        return eccentricity_arr

    # -------------------------------------------------------------------------
    # ----------------------- Kernel / MMD integration ------------------------
    # -------------------------------------------------------------------------

    # ======== Low-level helpers ========
    @staticmethod
    def _safe_row_norms(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Row-wise ℓ2 norms with a numerical floor.

        Parameters
        ----------
        X : (N, D) ndarray
            Input matrix whose rows are vectors to be normalized.
        eps : float, default=1e-12
            Minimum allowed norm to avoid division by zero.

        Returns
        -------
        (N, 1) ndarray
            Max(nrm, eps) per row.
        """
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        return np.maximum(nrm, eps)

    @staticmethod
    def _normalize_to_sphere(X: np.ndarray) -> np.ndarray:
        """
        Normalize rows of `X` to the unit sphere.

        Parameters
        ----------
        X : (N, D) ndarray
            Input matrix.

        Returns
        -------
        (N, D) ndarray
            Row-normalized data with unit ℓ2 norm (up to numerical floor).
        """
        return X / ClsHiddenGeometry._safe_row_norms(X)

    @staticmethod
    def _cosine_gram(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Cosine Gram matrix for already unit-normalized rows.

        Parameters
        ----------
        A : (N, D) ndarray
        B : (M, D) ndarray

        Returns
        -------
        (N, M) ndarray
            Cosine similarities `A @ B.T`.
        """
        # assume both already unit-normalized
        return A @ B.T

    @staticmethod
    def _rbf_gram(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
        """
        Gaussian RBF Gram matrix.

        k(x, y) = exp(-||x - y||^2 / (2 sigma^2))

        Parameters
        ----------
        A : (N, D) ndarray
        B : (M, D) ndarray
        sigma : float
            Bandwidth (> 0).

        Returns
        -------
        (N, M) ndarray
            RBF kernel matrix.
        """
        A2 = np.sum(A * A, axis=1, keepdims=True)
        B2 = np.sum(B * B, axis=1, keepdims=True).T
        sq = A2 + B2 - 2.0 * (A @ B.T)
        gamma = 1.0 / (2.0 * (sigma ** 2))
        return np.exp(-gamma * sq)

    @staticmethod
    def _laplace_gram(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
        """
        Laplace (ℓ2) kernel Gram matrix.

        k(x, y) = exp(-||x - y||_2 / sigma)

        Parameters
        ----------
        A : (N, D) ndarray
        B : (M, D) ndarray
        sigma : float
            Scale (> 0).

        Returns
        -------
        (N, M) ndarray
            Laplace kernel matrix.
        """
        A2 = np.sum(A * A, axis=1, keepdims=True)
        B2 = np.sum(B * B, axis=1, keepdims=True).T
        sq = A2 + B2 - 2.0 * (A @ B.T)
        np.maximum(sq, 0.0, out=sq)  # numerical safety
        D = np.sqrt(sq, dtype=A.dtype)
        s = max(float(sigma), 1e-12)
        return np.exp(- D / s)

    @staticmethod
    def _mmd2_unbiased_from_gram(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
        """
        Unbiased MMD^2 from Gram blocks.

        Parameters
        ----------
        Kxx : (n, n) ndarray
            Kernel matrix within X (diagonal excluded in estimator).
        Kyy : (m, m) ndarray
            Kernel matrix within Y (diagonal excluded in estimator).
        Kxy : (n, m) ndarray
            Cross-kernel matrix between X and Y.

        Returns
        -------
        float
            Unbiased MMD^2 estimate. NaN if n < 2 or m < 2.
        """
        n = Kxx.shape[0]
        m = Kyy.shape[0]
        if n < 2 or m < 2:
            return float("nan")
        sum_Kxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        sum_Kyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        sum_Kxy = Kxy.sum() / (n * m)
        return sum_Kxx + sum_Kyy - 2.0 * sum_Kxy

    @staticmethod
    def _median_pairwise_distance(X: np.ndarray, max_samples: int = 5000) -> float:
        """
        Median pairwise Euclidean distance (sqrt of median squared distance).

        Parameters
        ----------
        X : (N, D) ndarray
            Input data.
        max_samples : int, default=5000
            Random subsample size for efficiency.

        Returns
        -------
        float
            Median distance. Returns 1.0 if insufficient pairs.
        """
        N = X.shape[0]
        idx = np.random.RandomState(42).choice(N, size=min(N, max_samples), replace=False)
        Z = X[idx]
        Z2 = np.sum(Z * Z, axis=1, keepdims=True)
        sq = Z2 + Z2.T - 2.0 * (Z @ Z.T)
        tri = sq[np.triu_indices_from(sq, k=1)]
        tri = tri[tri > 0]
        return float(np.sqrt(np.median(tri))) if tri.size else 1.0

    # ======== Cache struct per (label, n) key ========
    @dataclass
    class KernelCache:
        """
        Per-key cache of intermediate artifacts to avoid recomputation.

        Attributes
        ----------
        sphere_np : list of (N, D) ndarray or None
            Per-layer unit-sphere versions of hidden states.
        cosine_Kxx : list of (N, N) ndarray or None
            Per-layer cosine Gram matrices for X vs X.
        key_whiten_np_by_thr : dict[float, list[ndarray]] or None
            (Optional) Per-threshold cached whitened arrays by layer.
        key_pca_by_thr : dict[float, EnergyWhitener] or None
            (Optional) Per-threshold global projector cache.
        layer_pca_by_thr : dict[(int, float), EnergyWhitener] or None
            (Optional) Per-(layer, threshold) projector cache.
        """
        sphere_np: Optional[List[np.ndarray]] = None
        cosine_Kxx: Optional[List[np.ndarray]] = None
        key_whiten_np_by_thr: Optional[Dict[float, List[np.ndarray]]] = None
        key_pca_by_thr: Optional[Dict[float, "EnergyWhitener"]] = None
        layer_pca_by_thr: Optional[Dict[Tuple[int, float], "EnergyWhitener"]] = None

    def _key(self, class_label, n_samples) -> str:
        """Alias for `_make_key`; keeps naming consistent in the kernel module."""
        return self._make_key(class_label, n_samples)

    def _get_cache(self, class_label, n_samples) -> "ClsHiddenGeometry.KernelCache":
        """
        Get or initialize the per-key kernel cache; compute states if missing.

        Parameters
        ----------
        class_label : str or int
        n_samples : int

        Returns
        -------
        KernelCache
            Mutable cache object associated with the key.
        """
        key = self._key(class_label, n_samples)
        if key not in self.hidden_cls_states:
            bs = min(64, int(n_samples))
            self.compute_hidden_cls_states(class_label=class_label, num_samples=int(n_samples), batch_size=bs)
        if key not in self.kernel_cache:
            self.kernel_cache[key] = ClsHiddenGeometry.KernelCache()
        return self.kernel_cache[key]

    # ======== Cosine (unit-sphere) pipeline ========
    def _ensure_sphere_cached(self, class_label, n_samples):
        """
        Ensure per-layer unit-sphere arrays are cached for the given key.

        Populates `KernelCache.sphere_np` from `self.hidden_cls_states`.
        """
        cache = self._get_cache(class_label, n_samples)
        if cache.sphere_np is not None:
            return
        key = self._key(class_label, n_samples)
        layers = self.hidden_cls_states[key]
        sphere = [self._normalize_to_sphere(t.detach().cpu().numpy()) for t in layers]
        cache.sphere_np = sphere

    def _ensure_cosine_gram_cached(self, class_label, n_samples):
        """
        Ensure per-layer cosine Gram matrices (X vs X) are cached.

        Uses cached unit-sphere data; computes `A @ A.T` per layer.
        """
        cache = self._get_cache(class_label, n_samples)
        if cache.cosine_Kxx is not None:
            return
        self._ensure_sphere_cached(class_label, n_samples)
        Ks = []
        for X in cache.sphere_np:
            Ks.append(self._cosine_gram(X, X))
        cache.cosine_Kxx = Ks

    def cosine_mmd2_between_labels_along_layers(self, class_label_A, nA, class_label_B, nB) -> np.ndarray:
        """
        Per-layer unbiased MMD^2 under the cosine kernel between two cached keys.

        Parameters
        ----------
        class_label_A, class_label_B : str or int
            Class identifiers for the two distributions.
        nA, nB : int
            Sample counts used when caching the hidden states.

        Returns
        -------
        (L,) ndarray
            MMD^2(X_{t,yA}, X_{t,yB}) for t = 1..L.
        """
        self._ensure_sphere_cached(class_label_A, nA)
        self._ensure_sphere_cached(class_label_B, nB)
        A = self._get_cache(class_label_A, nA).sphere_np
        B = self._get_cache(class_label_B, nB).sphere_np
        if len(A) != len(B):
            raise RuntimeError("Layer counts do not match.")
        L = len(A)
        vals = np.zeros(L, dtype=float)
        for t in range(L):
            Xa, Xb = A[t], B[t]
            Kaa = self._cosine_gram(Xa, Xa)
            Kbb = self._cosine_gram(Xb, Xb)
            Kab = self._cosine_gram(Xa, Xb)
            vals[t] = self._mmd2_unbiased_from_gram(Kaa, Kbb, Kab)
        return vals

    def plot_cosine_mmd_between_labels(self, class_label_A, nA, class_label_B, nB):
        """
        Plot per-layer cosine MMD^2 between two keys and save the figure.

        Figure is saved to:
        `./plots/{model_id}/{keyA}__vs__{keyB}_cos_mmd_by_layer`
        """
        vals = self.cosine_mmd2_between_labels_along_layers(class_label_A, nA, class_label_B, nB)
        plt.figure()
        plt.plot(np.arange(1, len(vals) + 1), vals, marker="o")
        plt.xlabel("Layer t")
        plt.ylabel("Cosine MMD$^2$")
        plt.title(f"Cosine MMD$^2$ vs layer: {class_label_A} (n={nA}) vs {class_label_B} (n={nB})")
        out_path = f"./plots/{self.model_id.replace('/', '__')}/{self._key(class_label_A, nA)}__vs__{self._key(class_label_B, nB)}_cos_mmd_by_layer"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    # ---------------- MK-MMD with learned weights (standardized MMD^2) ----------------
    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
        """
        Euclidean projection onto the probability simplex { w >= 0, sum(w) = 1 }.

        Parameters
        ----------
        v : (S,) ndarray

        Returns
        -------
        (S,) ndarray
            Projected vector.
        """
        if v.ndim != 1:
            v = v.ravel()
        n = v.size
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        w = np.maximum(v - theta, 0.0)
        s = w.sum()
        return w if s > 0 else np.ones_like(w) / len(w)

    def _bootstrap_mmd2_cov(
            self,
            *,
            mode: str,  # "H1" or "H0"
            Kaa_list: Optional[List[np.ndarray]] = None,
            Kbb_list: Optional[List[np.ndarray]] = None,
            Kab_list: Optional[List[np.ndarray]] = None,
            Kpp_list: Optional[List[np.ndarray]] = None,  # pooled gram for H0
            n: Optional[int] = None,
            m: Optional[int] = None,
            B: int = 200,
            subsample: Optional[int] = None,
            seed: int = 123,
            rescale_if_subsample: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap the covariance of the per-sigma unbiased MMD^2 vector.

        Parameters
        ----------
        mode : {"H1", "H0"}
            "H1": resample within X and within Y using Kaa/Kbb/Kab.
            "H0": random permutations on pooled indices using Kpp (size n+m).
        Kaa_list, Kbb_list, Kab_list : list of ndarray, optional
            Gram blocks per sigma (required for "H1").
        Kpp_list : list of ndarray, optional
            Pooled Gram matrices per sigma (required for "H0"). X rows first, then Y.
        n, m : int, optional
            Original group sizes (required for "H0").
        B : int, default=200
            Number of bootstrap replicas.
        subsample : int, optional
            If provided and smaller than group sizes, bootstrap on subsamples for efficiency.
        seed : int, default=123
            RNG seed for reproducibility.
        rescale_if_subsample : bool, default=True
            Rescale covariance when subsampling to roughly match full-sample variance.

        Returns
        -------
        a : (S,) ndarray
            Per-sigma unbiased MMD^2 computed on the original partition (full data).
        Sigma : (S, S) ndarray
            Estimated covariance matrix across sigmas (with a small ridge added).

        Raises
        ------
        ValueError
            When required inputs are missing or inconsistent.
        """
        rng = np.random.default_rng(seed)

        if mode not in ("H1", "H0"):
            raise ValueError("mode must be 'H1' or 'H0'")

        if mode == "H1":
            if Kaa_list is None or Kbb_list is None or Kab_list is None:
                raise ValueError("H1 mode requires Kaa_list, Kbb_list, Kab_list.")

            # Set L <- |sigma_grid|
            L = len(Kaa_list)
            if L == 0:
                return np.array([]), np.zeros((0, 0))

            # n_full <- |X_{t,yA}|, m_full <- |X_{t,yB}|
            n_full = Kaa_list[0].shape[0]
            m_full = Kbb_list[0].shape[0]

            # a[i] <- MMD^2_{K_{sigma[i]}}(X, Y) on full data
            a = np.zeros(L, dtype=np.float64)
            for l in range(L):
                a[l] = self._mmd2_unbiased_from_gram(Kaa_list[l], Kbb_list[l], Kab_list[l])

            # choose resample sizes
            n_b = n_full if (subsample is None or subsample >= n_full) else subsample
            m_b = m_full if (subsample is None or subsample >= m_full) else subsample

            boot = np.zeros((B, L), dtype=np.float64)
            for b in range(B):
                ia = rng.integers(0, n_full, n_b)
                ib = rng.integers(0, m_full, m_b)
                for l in range(L):
                    Kxx_b = Kaa_list[l][np.ix_(ia, ia)]
                    Kyy_b = Kbb_list[l][np.ix_(ib, ib)]
                    Kxy_b = Kab_list[l][np.ix_(ia, ib)]
                    boot[b, l] = self._mmd2_unbiased_from_gram(Kxx_b, Kyy_b, Kxy_b)

            Sigma = np.cov(boot, rowvar=False, ddof=1)
            Sigma = np.atleast_2d(np.asarray(Sigma))

            # crude but effective rescale using variance ~ (1/n + 1/m)
            if rescale_if_subsample and (n_b < n_full or m_b < m_full):
                scale = (1.0 / n_b + 1.0 / m_b) / (1.0 / n_full + 1.0 / m_full)
                Sigma = Sigma * max(scale, 1e-8)

        else:  # mode == "H0"
            if Kpp_list is None or n is None or m is None:
                raise ValueError("H0 mode requires Kpp_list and sizes n, m (original partition).")
            L = len(Kpp_list)
            if L == 0:
                return np.array([]), np.zeros((0, 0))
            N_full = Kpp_list[0].shape[0]
            if N_full != n + m:
                raise ValueError("Inconsistent sizes for H0: Kpp size must equal n+m.")

            # Original A=range(n), B=range(n, n+m) (caller must have stacked like this)
            a = np.zeros(L, dtype=np.float64)
            idxA_full = np.arange(n)
            idxB_full = np.arange(n, n + m)
            for l in range(L):
                Kpp = Kpp_list[l]
                Kaa = Kpp[np.ix_(idxA_full, idxA_full)]
                Kbb = Kpp[np.ix_(idxB_full, idxB_full)]
                Kab = Kpp[np.ix_(idxA_full, idxB_full)]
                a[l] = self._mmd2_unbiased_from_gram(Kaa, Kbb, Kab)

            n_b = n if (subsample is None or subsample >= n) else subsample
            m_b = m if (subsample is None or subsample >= m) else subsample

            boot = np.zeros((B, L), dtype=np.float64)
            for b in range(B):
                perm = rng.permutation(N_full)
                ia = perm[:n_b]
                ib = perm[n_b:n_b + m_b]
                for l in range(L):
                    Kpp = Kpp_list[l]
                    Kxx_b = Kpp[np.ix_(ia, ia)]
                    Kyy_b = Kpp[np.ix_(ib, ib)]
                    Kxy_b = Kpp[np.ix_(ia, ib)]
                    boot[b, l] = self._mmd2_unbiased_from_gram(Kxx_b, Kyy_b, Kxy_b)
            Sigma = np.cov(boot, rowvar=False, ddof=1)
            Sigma = np.atleast_2d(np.asarray(Sigma))

            if rescale_if_subsample and (n_b < n or m_b < m):
                scale = (1.0 / n_b + 1.0 / m_b) / (1.0 / n + 1.0 / m)
                Sigma = Sigma * max(scale, 1e-8)

        # Add a tiny ridge for numerical stability.
        tr = float(np.trace(Sigma)) if Sigma.size else 0.0
        eps = 1e-8 * (tr / max(1, Sigma.shape[0]) + 1.0)
        Sigma = Sigma + eps * np.eye(Sigma.shape[0], dtype=np.float64)

        return a, Sigma

    def _learn_mk_weights_given_a_Sigma(self, a: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """
        Maximize standardized MMD^2 over the simplex via projected gradient ascent.

        Objective
        ---------
        maximize  (w^T a) / sqrt(w^T Σ w)
        subject to w >= 0, sum(w) = 1

        Parameters
        ----------
        a : (S,) ndarray
            Per-sigma unbiased MMD^2 on full data.
        Sigma : (S, S) ndarray
            Bootstrap covariance across sigmas.

        Returns
        -------
        (S,) ndarray
            Optimal mixture weights on the simplex (float32).
        """
        if a.size == 0:
            return np.array([], dtype=np.float32)
        try:
            w0 = np.linalg.solve(Sigma, a)
        except np.linalg.LinAlgError:
            w0 = a.copy()
        w = self._project_to_simplex(np.maximum(w0, 0.0))
        for _ in range(100):
            num = float(np.dot(w, a))
            den_sq = float(w @ Sigma @ w)
            if num <= 1e-12 or den_sq <= 1e-12:
                break
            grad = a / max(num, 1e-12) - (Sigma @ w) / max(den_sq, 1e-12)
            eta = 0.2  # fixed step works well in practice here
            w_new = self._project_to_simplex(w + eta * grad)
            if np.linalg.norm(w_new - w, 1) < 1e-6:
                w = w_new
                break
            w = w_new
        return w.astype(np.float32)

    def _learn_mk_weights_and_std_from_grams(
            self,
            *,
            mode: str,  # "H1" or "H0"
            Kaa_list: Optional[List[np.ndarray]] = None,
            Kbb_list: Optional[List[np.ndarray]] = None,
            Kab_list: Optional[List[np.ndarray]] = None,
            Kpp_list: Optional[List[np.ndarray]] = None,
            n: Optional[int] = None,
            m: Optional[int] = None,
            B: int = 200,
            subsample: Optional[int] = None,
            seed: int = 123,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Learn MK weights that maximize standardized MMD^2 and report the score.

        Parameters
        ----------
        mode : {"H1", "H0"}
            Variance estimation mode for Σ (bootstrap vs permutation).
        Kaa_list, Kbb_list, Kab_list : list of ndarray, optional
            Gram blocks per sigma (required for "H1").
        Kpp_list : list of ndarray, optional
            Pooled Gram matrices per sigma (required for "H0").
        n, m : int, optional
            Original group sizes (required for "H0").
        B : int, default=200
            Number of bootstrap/permute replicas.
        subsample : int, optional
            Bootstrap subsample size for efficiency.
        seed : int, default=123

        Returns
        -------
        weights : (S,) ndarray
            Learned simplex weights.
        std_mmd2 : float
            Standardized MMD^2 at the optimum.
        a_vector : (S,) ndarray
            Full-data per-sigma unbiased MMD^2.
        """
        a, Sigma = self._bootstrap_mmd2_cov(
            mode=mode, Kaa_list=Kaa_list, Kbb_list=Kbb_list, Kab_list=Kab_list,
            Kpp_list=Kpp_list, n=n, m=m, B=B, subsample=subsample, seed=seed
        )
        if a.size == 0:
            return np.array([], dtype=np.float32), 0.0, a
        w = self._learn_mk_weights_given_a_Sigma(a, Sigma)
        num = float(np.dot(w, a))
        den = float(np.sqrt(max(w @ Sigma @ w, 1e-12)))
        std_mmd2 = num / den if den > 0 else 0.0
        return w, std_mmd2, a

    def _bootstrap_mu_hat_stats(
            self,
            Kxz_list: List[np.ndarray],
            weights: np.ndarray,
            B: int = 80,
            subsample: Optional[int] = None,
            seed: int = 123,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap statistics for μ̂_p(z) under a kernel mixture.

        Parameters
        ----------
        Kxz_list : list of (N,) ndarray
            Per-sigma kernel vectors `k_sigma(x_i, z)` for i=1..N.
        weights : (S,) ndarray
            Mixture weights over sigmas; will be renormalized.
        B : int, default=80
            Number of bootstrap replicas (within-class resampling).
        subsample : int, optional
            If provided and < N, resample with replacement to this size.
        seed : int, default=123

        Returns
        -------
        mu_full : float
            Full-data mixture estimate of μ̂_p(z).
        std_boot : float
            Bootstrap standard deviation of the mixture estimate.
        std_score : float
            Signal-to-noise ratio `mu_full / std_boot`.
        """
        # weights normalized
        w = np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)

        # Full-data mean per sigma, then mix.
        means_full = np.array([float(Kxz.mean()) for Kxz in Kxz_list], dtype=np.float64)
        mu_full = float(np.dot(w, means_full))

        # Bootstrap (within-class resample)
        n = Kxz_list[0].shape[0]
        n_b = n if (subsample is None or subsample >= n) else subsample
        rng = np.random.default_rng(seed)
        boot_vals = np.zeros(B, dtype=np.float64)
        for b in range(B):
            ia = rng.integers(0, n, n_b)  # with replacement
            mix = 0.0
            for ws, Kxz in zip(w, Kxz_list):
                mix += float(ws) * float(Kxz[ia].mean())
            boot_vals[b] = mix

        std_boot = float(np.std(boot_vals, ddof=1)) if B > 1 else 0.0
        std_score = mu_full / (std_boot + 1e-12)
        return mu_full, std_boot, std_score

    def mk_mmd2_between_labels_along_layers(
            self,
            class_label_A,
            nA,
            class_label_B,
            nB,
            n_scales: int = 7,
            ratio: float = 0.5,
            normalize_each_kernel: bool = True,
            weights: Optional[np.ndarray] = None,
            energy_threshold: Optional[Union[float, Sequence[float]]] = None,
            base_kernel: str = "rbf",
            variance_mode: str = "H1",  # "H1" (within-class bootstrap) or "H0" (permutation on pooled)
            B: int = 100,
            subsample: Optional[int] = None,
            seed: int = 123,
    ) -> Tuple[np.ndarray, List[List[float]], np.ndarray]:
        """
        Layer-wise MK-MMD² between two classes with optional PCA and sigma selection.

        For each layer t:
          1) (Optionally) project X_a and X_b by EnergyWhitener with `energy_threshold`.
          2) Build a per-layer sigma grid centered at the median pairwise distance.
          3) Learn mixture weights to maximize standardized MMD^2 (or use provided `weights`).
          4) Record the mixed unbiased MMD^2 and its standardized score; keep the best threshold.

        Parameters
        ----------
        class_label_A, class_label_B : str or int
            Class identifiers for the two distributions.
        nA, nB : int
            Sample counts used when caching the hidden states.
        n_scales : int, default=7
            Number of bandwidths in the sigma grid per layer.
        ratio : float, default=0.5
            Geometric step between sigma scales (center * ratio^e).
        normalize_each_kernel : bool, default=True
            Placeholder for per-kernel normalization (currently a no-op).
        weights : (S,) ndarray, optional
            Fixed mixture weights; if None, learned per layer via `_learn_mk_weights_and_std_from_grams`.
        energy_threshold : float or sequence of float, optional
            PCA energy thresholds to try; `None` means no projection.
        base_kernel : {"rbf", "laplace"}, default="rbf"
            Base kernel for MK-MMD.
        variance_mode : {"H1", "H0"}, default="H1"
            Strategy to estimate Σ for the standardized score.
        B : int, default=100
            Number of bootstrap/permute replicas for Σ.
        subsample : int, optional
            Subsample size in bootstrap steps (for efficiency).
        seed : int, default=123

        Returns
        -------
        vals : (L,) ndarray
            MK-MMD^2 values per layer using the best threshold.
        sigma_lists : list[list[float]]
            Sigma grid used at each layer for the selected threshold.
        std_scores : (L,) ndarray
            Standardized scores (effect / std) associated with `vals`.

        Raises
        ------
        ValueError
            If `base_kernel` is not recognized.
        RuntimeError
            If the two keys have different layer counts.
        """
        keyA = self._key(class_label_A, nA)
        keyB = self._key(class_label_B, nB)
        if keyA not in self.hidden_cls_states:
            bs = min(64, int(nA))
            self.compute_hidden_cls_states(class_label_A, int(nA), bs)
        if keyB not in self.hidden_cls_states:
            bs = min(64, int(nB))
            self.compute_hidden_cls_states(class_label_B, int(nB), bs)

        A_layers = [t.detach().cpu().numpy() for t in self.hidden_cls_states[keyA]]
        B_layers = [t.detach().cpu().numpy() for t in self.hidden_cls_states[keyB]]
        if len(A_layers) != len(B_layers):
            raise RuntimeError("Layer counts do not match.")

        if base_kernel not in ("rbf", "laplace"):
            raise ValueError("base_kernel must be 'rbf' or 'laplace'")
        gram = self._rbf_gram if base_kernel == "rbf" else self._laplace_gram

        thr_grid = self._as_threshold_grid(energy_threshold)

        L = len(A_layers)
        vals = np.zeros(L, dtype=float)
        std_scores = np.zeros(L, dtype=float)
        sigma_lists: List[List[float]] = []

        for t in range(L):
            Xa_raw, Xb_raw = A_layers[t], B_layers[t]

            best_std = -np.inf
            best_val = 0.0
            best_sigmas: List[float] = []

            for thr in thr_grid:
                # Optional per-layer PCA on union (whiten=False)
                if thr is not None:
                    X_union = np.vstack([Xa_raw, Xb_raw])
                    union_proj = EnergyWhitener(energy_threshold=float(thr), whiten=False).fit(X_union)
                    Xa = union_proj.transform(Xa_raw)
                    Xb = union_proj.transform(Xb_raw)
                    pooled_for_sigma = np.vstack([Xa, Xb])
                else:
                    Xa, Xb = Xa_raw, Xb_raw
                    pooled_for_sigma = np.vstack([Xa, Xb])

                # Sigma grid for this layer & threshold
                sigma_center = self._median_pairwise_distance(pooled_for_sigma)
                half = n_scales // 2
                exps = np.arange(-half, half + 1)
                sigmas = [sigma_center * (ratio ** e) for e in exps]

                if variance_mode == "H1":
                    Kaa_list, Kbb_list, Kab_list = [], [], []
                    for s in sigmas:
                        Kaa = gram(Xa, Xa, s)
                        Kbb = gram(Xb, Xb, s)
                        Kab = gram(Xa, Xb, s)
                        if normalize_each_kernel:
                            pass  # placeholder for future normalization
                        Kaa_list.append(Kaa);
                        Kbb_list.append(Kbb);
                        Kab_list.append(Kab)

                    if weights is None:
                        w, std_mmd2, a_vec = self._learn_mk_weights_and_std_from_grams(
                            mode="H1", Kaa_list=Kaa_list, Kbb_list=Kbb_list, Kab_list=Kab_list,
                            B=B, subsample=subsample, seed=seed
                        )
                    else:
                        a_vec, Sigma = self._bootstrap_mmd2_cov(
                            mode="H1", Kaa_list=Kaa_list, Kbb_list=Kbb_list, Kab_list=Kab_list,
                            B=B, subsample=subsample, seed=seed
                        )
                        w = np.asarray(weights, dtype=np.float64)
                        w = w / (w.sum() + 1e-12)
                        num = float(np.dot(w, a_vec))
                        den = float(np.sqrt(max(w @ Sigma @ w, 1e-12)))
                        std_mmd2 = num / den if den > 0 else 0.0

                    val = float(np.dot(w, a_vec)) if a_vec.size else 0.0

                else:  # variance_mode == "H0"
                    # Build pooled Gram once per sigma with A stacked before B
                    Xp = np.vstack([Xa, Xb])  # original A first, then B
                    nA0, nB0 = Xa.shape[0], Xb.shape[0]
                    Kpp_list = [gram(Xp, Xp, s) for s in sigmas]
                    if weights is None:
                        w, std_mmd2, a_vec = self._learn_mk_weights_and_std_from_grams(
                            mode="H0", Kpp_list=Kpp_list, n=nA0, m=nB0, B=B, subsample=subsample, seed=seed
                        )
                    else:
                        a_vec, Sigma = self._bootstrap_mmd2_cov(
                            mode="H0", Kpp_list=Kpp_list, n=nA0, m=nB0, B=B, subsample=subsample, seed=seed
                        )
                        w = np.asarray(weights, dtype=np.float64)
                        w = w / (w.sum() + 1e-12)
                        num = float(np.dot(w, a_vec))
                        den = float(np.sqrt(max(w @ Sigma @ w, 1e-12)))
                        std_mmd2 = num / den if den > 0 else 0.0
                    val = float(np.dot(w, a_vec)) if a_vec.size else 0.0

                if std_mmd2 > best_std:
                    best_std = std_mmd2
                    best_val = val
                    best_sigmas = sigmas

            vals[t] = best_val
            std_scores[t] = best_std
            sigma_lists.append(best_sigmas)

        return vals, sigma_lists, std_scores

    def plot_mk_mmd_between_labels(self, class_label_A, nA, class_label_B, nB, **mk_kwargs):
        """
        Plot per-layer MK-MMD^2 between two keys and save the figure.

        Additional MK-MMD settings can be passed via `**mk_kwargs`.
        Figure is saved to:
        `./plots/{model_id}/{keyA}__vs__{keyB}_mk_mmd_by_layer`
        """
        vals, sigma_lists, std_scores = self.mk_mmd2_between_labels_along_layers(
            class_label_A, nA, class_label_B, nB, **mk_kwargs
        )
        plt.figure()
        plt.plot(np.arange(1, len(vals) + 1), vals, marker="o")
        plt.xlabel("Layer t")
        plt.ylabel("MK-MMD$^2$")
        sig_text = "layer-wise σ grids"
        plt.title(f"MK-MMD$^2$ vs layer: {class_label_A} (n={nA}) vs {class_label_B} (n={nB})\n({sig_text})")
        out_path = f"./plots/{self.model_id.replace('/', '__')}/{self._key(class_label_A, nA)}__vs__{self._key(class_label_B, nB)}_mk_mmd_by_layer"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    # ======== μ̂_{p_{t,y}}(z_t) along layers for a new image (optional layer-wise PCA) ========
    @torch.no_grad()
    def _cls_traj_for_image(self, image: Image.Image) -> List[np.ndarray]:
        """
        Extract per-layer [CLS] hidden vectors for a single image (as NumPy arrays).

        Matches the extraction used in `compute_hidden_cls_states` (i.e., token index 0
        after each transformer block).

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        list of (1, D) ndarray
            One row per layer (length = num_hidden_layers).
        """
        device = next(self.parameters()).device
        enc = self.image_processor(images=[image], return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        vit_outputs = self.vit(**enc, output_hidden_states=True)
        hidden_states = vit_outputs.hidden_states
        if hidden_states is None or len(hidden_states) != (int(self.config.num_hidden_layers) + 1):
            raise RuntimeError("Unexpected hidden_states length.")
        traj = []
        for k in range(int(self.config.num_hidden_layers)):
            hs_k = hidden_states[k + 1][:, 0, :].detach().cpu().numpy()  # (1 x d)
            traj.append(hs_k)
        return traj

    def mu_hat_along_layers(
            self,
            class_label,
            n_samples,
            image_z: Image.Image,
            true_label: str,
            kernel: str = "cosine",
            mk_n_scales: int = 7,
            mk_ratio: float = 0.5,
            mk_sigma_center: Optional[float] = None,
            normalize_each_kernel: bool = True,
            weights: Optional[np.ndarray] = None,
            energy_threshold: Optional[Union[float, Sequence[float]]] = None,
            plot_res=True
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluate μ̂_{p_{t,y}}(z_t) across layers for a query image and return the trajectory + mean.

        Parameters
        ----------
        class_label : str or int
            Class y whose empirical distribution p_{t,y} is used.
        n_samples : int
            Sample count used to build p_{t,y}.
        image_z : PIL.Image.Image
            Query image for which to compute {z_t}.
        true_label : str
            String used for plot naming; not used in computation.
        kernel : {"cosine", "mk_rbf", "mk_laplace"}, default="cosine"
            Kernel family for μ̂.
        mk_n_scales : int, default=7
            Number of scales in the MK grid (MK only).
        mk_ratio : float, default=0.5
            Geometric step between scales (MK only).
        mk_sigma_center : float, optional
            Override center scale (if None, use median heuristic per layer).
        normalize_each_kernel : bool, default=True
            Placeholder for future per-kernel normalization (no-op).
        weights : (S,) ndarray, optional
            Fixed mixture weights; if None, uniform weights are used in μ̂ (MK only).
        energy_threshold : float or sequence of float, optional
            Per-layer PCA energy thresholds to try (MK only); `None` means no PCA.
        plot_res : bool, default=True
            If True, save a trajectory plot.

        Returns
        -------
        vals : (L,) ndarray
            μ̂ values per layer.
        mean_val : float
            Mean of μ̂ across layers.

        Raises
        ------
        ValueError
            If `kernel` is not recognized.
        """
        traj = self._cls_traj_for_image(image_z)

        if kernel == "cosine":
            self._ensure_sphere_cached(class_label, n_samples)
            spheres = self._get_cache(class_label, n_samples).sphere_np
            vals = np.zeros(len(spheres), dtype=float)
            for t, X in enumerate(spheres):
                zt = self._normalize_to_sphere(traj[t])  # (1 x d)
                Kxz = self._cosine_gram(X, zt)  # (n x 1)
                vals[t] = float(Kxz.mean(axis=0))  # μ̂_p(z_t)

        elif kernel in ("mk_rbf", "mk_laplace"):
            gram = self._rbf_gram if kernel == "mk_rbf" else self._laplace_gram

            key = self._key(class_label, n_samples)
            if key not in self.hidden_cls_states:
                bs = min(64, int(n_samples))
                self.compute_hidden_cls_states(class_label, int(n_samples), bs)
            layers = self.hidden_cls_states[key]
            vals = np.zeros(len(layers), dtype=float)

            thr_grid = self._as_threshold_grid(energy_threshold)

            # mixture weights (fixed for μ̂)
            def _normalize_w(w):
                w = np.asarray(w, dtype=np.float64)
                return w / (w.sum() + 1e-12)

            for t in range(len(layers)):
                Xt_raw = layers[t].detach().cpu().numpy()
                zt_raw = traj[t]

                best_std = -np.inf
                best_mu = 0.0

                for thr in thr_grid:
                    # Optional PCA on pooled {Xt_raw, zt_raw}
                    if thr is not None:
                        X_pool = np.vstack([Xt_raw, zt_raw])
                        projector = EnergyWhitener(energy_threshold=float(thr), whiten=False).fit(X_pool)
                        Xt = projector.transform(Xt_raw)
                        zt = projector.transform(zt_raw)
                        pool_for_sigma = np.vstack([Xt, zt])
                    else:
                        Xt = Xt_raw
                        zt = zt_raw
                        pool_for_sigma = np.vstack([Xt, zt])

                    # Sigma grid (per layer/threshold)
                    if mk_sigma_center is None:
                        sigma_center = self._median_pairwise_distance(pool_for_sigma)
                    else:
                        sigma_center = float(mk_sigma_center)
                    half = mk_n_scales // 2
                    exps = np.arange(-half, half + 1)
                    sigmas = [sigma_center * (mk_ratio ** e) for e in exps]

                    # Per-sigma Kxz (store vectors for bootstrap)
                    Kxz_list = [gram(Xt, zt, s).reshape(-1) for s in sigmas]
                    ws = _normalize_w(np.ones(len(sigmas)) if weights is None else weights)

                    # Bootstrap std-score for μ̂ (within-class)
                    mu_full, std_boot, std_score = self._bootstrap_mu_hat_stats(
                        Kxz_list, ws, B=80, subsample=None, seed=123
                    )

                    if std_score > best_std:
                        best_std = std_score
                        best_mu = mu_full

                vals[t] = best_mu
        else:
            raise ValueError("kernel must be 'cosine', 'mk_rbf', or 'mk_laplace'.")

        if plot_res:
            plt.figure()
            plt.plot(np.arange(1, len(vals) + 1), vals, marker="o")
            plt.xlabel("Layer t")
            if kernel == "cosine":
                ylabel = r"$\hat{\mu}_{p_{t,y}}(z_t)$ [cosine]"
            elif kernel == "mk_rbf":
                ylabel = r"$\hat{\mu}_{p_{t,y}}(z_t)$ [MK-RBF]"
            else:
                ylabel = r"$\hat{\mu}_{p_{t,y}}(z_t)$ [MK-Laplace]"
            plt.ylabel(ylabel)
            plt.title(f"μ̂_p(z_t) across layers for {class_label} (n={n_samples}) - {kernel}")
            out_path = f"./plots/{self.model_id.replace('/', '__')}/{true_label}_vs_{self._key(class_label, n_samples)}_mu_hat_{kernel}"
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()

        return vals, float(vals.mean())

    # ======== Small utilities ========
    @staticmethod
    def _as_threshold_grid(energy_threshold: Optional[Union[float, Sequence[float]]]
                           ) -> List[Optional[float]]:
        """
        Normalize `energy_threshold` into a candidate list.

        Parameters
        ----------
        energy_threshold : float or sequence of float or None
            If None, return [None]. If scalar, return [scalar]. If sequence,
            cast each entry to float.

        Returns
        -------
        list of float or [None]
            Threshold candidates.
        """
        if energy_threshold is None:
            return [None]
        if isinstance(energy_threshold, (list, tuple, np.ndarray)):
            return [float(x) for x in energy_threshold]
        return [float(energy_threshold)]

    @staticmethod
    def _topm_metrics(probs_row: np.ndarray, logits_row: np.ndarray, m: int = 3):
        """
        Compute small-m mass/entropy and margins from a single prediction row.

        Parameters
        ----------
        probs_row : (C,) ndarray
            Class probabilities.
        logits_row : (C,) ndarray
            Raw logits (same ordering as `probs_row`).
        m : int, default=3
            Number of top probabilities to consider.

        Returns
        -------
        dict
            {
              "S_m": float,        # total mass in top-m (∈ (0,1])
              "uni_m": float,      # entropy of normalized top-m, scaled by log(m) ∈ [0,1]
              "p_margin": float,   # p1 - p2
              "z_margin": float,   # z1 - z2 (using global top-2 by logits)
              "top_idx": (m,) int ndarray,
              "top_p": (m,) float ndarray,
            }
        """
        p = np.asarray(probs_row, dtype=np.float64)
        p = np.maximum(p, 1e-12); p /= p.sum()
        z = np.asarray(logits_row, dtype=np.float64)

        idx = np.argsort(p)[::-1]
        pk = p[idx[:m]]
        zk = z[idx[:m]]

        S_m = float(pk.sum())
        q = pk / S_m                          # normalized top-m distribution
        Hq = float(-(q * np.log(q)).sum())
        uni = Hq / np.log(m)                  # ∈ [0, 1]

        # Margins (probability margin on top-m, logit margin on global top-2)
        p1, p2 = float(pk[0]), float(pk[1] if m >= 2 else pk[0])
        z_sorted = np.sort(z)[::-1]
        z1, z2 = float(z_sorted[0]), float(z_sorted[1] if z_sorted.size >= 2 else z_sorted[0])

        return {
            "S_m": S_m,
            "uni_m": uni,
            "p_margin": p1 - p2,
            "z_margin": z1 - z2,
            "top_idx": idx[:m],
            "top_p": pk,
        }

    @staticmethod
    def _tie_score_from_metrics(
        S_m,
        uni_m,
        p_margin,
        z_margin,
        tau_p: float = 0.05,
        tau_z: float = 0.5,
    ):
        """
        Heuristic tie-likeness score in [roughly] [0,1], higher = more tie-like.

        Combines:
        - Small-m mass S_m (higher is more tie-like),
        - Uniformity of mass over top-m (higher is more tie-like),
        - Small probability/logit margins (smaller gaps → higher score via exp decay).

        Parameters
        ----------
        S_m : float
            Sum of top-m probabilities.
        uni_m : float
            Scaled entropy of top-m (H/ln m), in [0, 1].
        p_margin : float
            p1 - p2 (probability gap).
        z_margin : float
            z1 - z2 (logit gap).
        tau_p : float, default=0.05
            Decay scale for probability gap.
        tau_z : float, default=0.5
            Decay scale for logit gap.

        Returns
        -------
        float
            Tie score (larger is better).
        """
        m_comp = float(np.exp(-max(p_margin, 0.0) / max(tau_p, 1e-6)))
        z_comp = float(np.exp(-max(z_margin, 0.0) / max(tau_z, 1e-6)))
        return 0.45 * uni_m + 0.35 * S_m + 0.20 * (0.5 * (m_comp + z_comp))

    @torch.no_grad()
    def find_tie_like_candidates(
        self,
        split: str = "val",
        batch_size: int = 128,
        max_candidates: int = 10,
        m: int = 3,                        # look for ~3-way ambiguity
        # soft thresholds (only used to PRINT flags; ranking is by tie_score):
        min_Sm: float = 0.60,
        min_uni: float = 0.90,
        max_p_margin: float = 0.08,        # ~8% top1-top2 gap
        max_z_margin: float = 0.5,         # logit units
        tau_p: float = 0.05,               # scales margin in tie score
        tau_z: float = 0.5,                # scales logit gap in tie score
        seed: int = 0,
        out_dir: str = "./candidates_ties",
        shuffle: bool = True,
        print_top_k: int = 10,
    ):
        """
        Rank images by “tie-likeness” and save/print the top N examples.

        The ranking score combines:
          • small-m uniformity: H(q)/log m (q = normalized top-m probs),
          • small-m coverage: S_m,
          • small top-2 margins (probability and logit).

        Parameters
        ----------
        split : {"train","val"}, default="val"
            Which ImageNet-1k resized-256 split to search.
        batch_size : int, default=128
            Inference batch size.
        max_candidates : int, default=10
            Number of highest-scoring examples to keep.
        m : int, default=3
            Size of the “small-m” subset considered for entropy/coverage.
        min_Sm : float, default=0.60
            Soft threshold used only for printing a ✓ flag (not for ranking).
        min_uni : float, default=0.90
            Soft threshold for uniformity flag (not for ranking).
        max_p_margin : float, default=0.08
            Soft threshold for probability-margin flag (not for ranking).
        max_z_margin : float, default=0.5
            Soft threshold for logit-margin flag (not for ranking).
        tau_p : float, default=0.05
            Scale for probability gap in tie score.
        tau_z : float, default=0.5
            Scale for logit gap in tie score.
        seed : int, default=0
            RNG seed for shuffling dataset indices.
        out_dir : str, default="./candidates_ties"
            Output directory for JPEG snapshots of candidates.
        shuffle : bool, default=True
            Shuffle dataset indices before scanning.
        print_top_k : int, default=10
            Number of top-k predictions printed for each saved candidate.

        Returns
        -------
        list of dict
            Each dict has keys:
            {
              "index": int,
              "path": str,              # saved JPEG path
              "tie_score": float,
              "S_m": float,
              "uni_m": float,
              "p_margin": float,
              "z_margin": float,
              "topk": list[(label, prob)],
            }

        Notes
        -----
        - The dataset split is downloaded/cached locally if needed.
        - Ranking is solely by the tie score; min/max thresholds only annotate output.
        """
        # Ensure split exists locally (download/save if missing).
        self._require_datasets()
        local_dir = os.path.join(_IMNET_LOCAL_ROOT, "imagenet_1k_resized_256", split)
        if not (os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "dataset_info.json"))):
            os.makedirs(local_dir, exist_ok=True)
            ds_split = load_dataset(_IMNET256_REPO, split=split)  # type: ignore[misc]
            ds_split.save_to_disk(local_dir)
        ds = load_from_disk(local_dir)  # type: ignore[misc]
        image_col, _ = self._find_cols(ds)

        # Use Python format to get PIL images.
        try:
            ds_py = ds.with_format(type="python")
        except Exception:
            ds_py = ds

        n_total = len(ds_py)
        idxs = list(range(n_total))
        if shuffle:
            rnd = random.Random(seed); rnd.shuffle(idxs)

        device = next(self.parameters()).device
        id2label = getattr(self.config, "id2label", {}) or {}

        def idx_to_label(i: int) -> str:
            return id2label.get(int(i), str(int(i)))

        import heapq
        heap = []  # min-heap storing best candidates by tie_score

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_indices = idxs[start:end]
            rows = [ds_py[i] for i in batch_indices]
            pil_imgs = [row[image_col] for row in rows]

            enc = self.image_processor(images=pil_imgs, return_tensors="pt")  # type: ignore[attr-defined]
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = self(**enc)
            logits = outputs.logits            # [B, C]
            probs  = logits.softmax(dim=-1)    # [B, C]

            probs_np  = probs.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()

            topk = min(print_top_k, probs.shape[-1])
            tk_scores, tk_idx = probs.topk(topk, dim=-1)
            tk_scores = tk_scores.detach().cpu().numpy()
            tk_idx    = tk_idx.detach().cpu().numpy()

            for bi, ds_idx in enumerate(batch_indices):
                mets = self._topm_metrics(probs_np[bi], logits_np[bi], m=m)
                score = self._tie_score_from_metrics(
                    mets["S_m"], mets["uni_m"], mets["p_margin"], mets["z_margin"],
                    tau_p=tau_p, tau_z=tau_z
                )

                # Maintain a bounded top-N via min-heap.
                item = (score, int(ds_idx), mets,
                        [(idx_to_label(int(tk_idx[bi, k])), float(tk_scores[bi, k])) for k in range(topk)])
                if len(heap) < max_candidates:
                    heapq.heappush(heap, item)
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, item)

        # Finalize: highest scores first and persist images with metric tags in filename.
        os.makedirs(out_dir, exist_ok=True)
        best = sorted(heap, key=lambda x: x[0], reverse=True)

        results = []
        for rank, (tie_score, ds_idx, mets, tk_list) in enumerate(best, start=1):
            im: Image.Image = ds_py[int(ds_idx)][image_col]
            fname = (f"{rank:02d}__idx{ds_idx:07d}"
                     f"__Sm{mets['S_m']:.2f}__uni{mets['uni_m']:.2f}"
                     f"__pm{mets['p_margin']:.3f}__zm{mets['z_margin']:.2f}.jpg")
            fpath = os.path.join(out_dir, fname)
            try:
                # Ensure 3-channel JPEG.
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                elif im.mode == "L":
                    im = im.convert("RGB")
                im.save(fpath, format="JPEG", quality=95, subsampling=0)
            except Exception as e:
                print(f"[WARN] save failed for idx={ds_idx}: {e}")

            # Compose printable flags for quick triage.
            top_flags = []
            if mets["S_m"] >= min_Sm: top_flags.append("Sm✓")
            if mets["uni_m"] >= min_uni: top_flags.append("uni✓")
            if mets["p_margin"] <= max_p_margin: top_flags.append("pmarg✓")
            if mets["z_margin"] <= max_z_margin: top_flags.append("zmarg✓")
            flags = ",".join(top_flags) if top_flags else "-"

            print("=" * 78)
            print(f"[{rank}/{len(best)}] idx={ds_idx} | tie_score={tie_score:.3f} | saved: {fpath}")
            print(f"  S_m={mets['S_m']:.3f}, uni_m={mets['uni_m']:.3f}, "
                  f"p_margin={mets['p_margin']:.3f}, z_margin={mets['z_margin']:.3f} | {flags}")
            for k, (lab, sc) in enumerate(tk_list, start=1):
                print(f"   {k:2d}. {lab:30s} p={sc:.4f}")
            print("=" * 78)

            results.append({
                "index": int(ds_idx),
                "path": fpath,
                "tie_score": float(tie_score),
                "S_m": float(mets["S_m"]),
                "uni_m": float(mets["uni_m"]),
                "p_margin": float(mets["p_margin"]),
                "z_margin": float(mets["z_margin"]),
                "topk": tk_list,
            })
        return results

    def print_true_labels_for_val_indices(
        self,
        idxs: Sequence[int],
        split: str = "val",
    ) -> List[Tuple[int, int, str]]:
        """
        Print and return ground-truth labels for given indices in the ImageNet-1k resized-256 split.

        Parameters
        ----------
        idxs : sequence of int
            Dataset indices (e.g., into the validation split).
        split : {"train","val"}, default="val"
            Which split to query.

        Returns
        -------
        list of (idx, label_idx, label_name)
            One tuple per valid index.

        Notes
        -----
        Downloads/caches the split locally if missing.
        """
        # Ensure the split is cached locally.
        self._require_datasets()
        local_dir = os.path.join(_IMNET_LOCAL_ROOT, "imagenet_1k_resized_256", split)
        if not (os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "dataset_info.json"))):
            os.makedirs(local_dir, exist_ok=True)
            ds_split = load_dataset(_IMNET256_REPO, split=split)  # type: ignore[misc]
            ds_split.save_to_disk(local_dir)

        ds = load_from_disk(local_dir)  # type: ignore[misc]
        _, label_col = self._find_cols(ds)
        feat = ds.features[label_col]

        # Python format (so we get native Python objects)
        try:
            ds_py = ds.with_format(type="python")
        except Exception:
            ds_py = ds

        n_total = len(ds_py)
        results: List[Tuple[int, int, str]] = []

        for idx in idxs:
            i = int(idx)
            if not (0 <= i < n_total):
                print(f"[WARN] idx {i} out of range [0, {n_total-1}] — skipped.")
                continue

            row = ds_py[i]
            y_true = int(row[label_col])

            # Map class index -> human-readable name
            if hasattr(feat, "int2str"):
                true_name = str(feat.int2str(y_true))
            else:
                names = getattr(feat, "names", None)
                true_name = str(names[y_true]) if names is not None and 0 <= y_true < len(names) else str(y_true)

            print(f"idx={i:7d}  |  y_true={y_true:4d}  |  label='{true_name}'")
            results.append((i, y_true, true_name))

        return results

