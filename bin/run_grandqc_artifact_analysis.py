#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import os
import re
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path


NOTICE_TEXT = """GrandQC integration notice

This pipeline step uses GrandQC model checkpoints and inference logic derived from:
Weng Z. et al. "GrandQC: a comprehensive solution to quality control problem in digital pathology"
Nature Communications (2024). DOI: 10.1038/s41467-024-54769-y

GrandQC repository: https://github.com/cpath-ukk/grandqc
GrandQC license: CC BY-NC-SA 4.0 (non-commercial)
Use of this step is subject to the original GrandQC license terms.
"""

TISSUE_RECORD_ID = "14507273"
ARTIFACT_RECORD_ID = "14041538"
TISSUE_MODEL_FILE = "Tissue_Detection_MPP10.pth"
ARTIFACT_MODEL_FILES = {
    1.0: "GrandQC_MPP1.pth",
    1.5: "GrandQC_MPP15.pth",
    2.0: "GrandQC_MPP2.pth",
}

QC_CLASS_MAPPING = {
    1: "Normal Tissue",
    2: "Fold",
    3: "Darkspot & Foreign Object",
    4: "PenMarking",
    5: "Edge & Air Bubble",
    6: "OOF",
    7: "Background",
}

QC_COLORS = [
    [128, 128, 128],
    [255, 99, 71],
    [0, 255, 0],
    [255, 0, 0],
    [255, 0, 255],
    [75, 0, 130],
    [255, 255, 255],
]

GRANDQC_BASE_DEP_SPECS = {
    "numpy": "numpy==1.26.4",
    "PIL": "pillow==10.4.0",
    "cv2": "opencv-python-headless==4.7.0.72",
    "tifffile": "tifffile==2023.4.12",
    "zarr": "zarr==2.16.1",
    "numcodecs": "numcodecs==0.11.0",
    "skimage": "scikit-image==0.21.0",
    "imagecodecs": "imagecodecs==2024.12.30",
    "tqdm": "tqdm==4.65.0",
}

GRANDQC_INFERENCE_DEP_SPECS = {
    "segmentation_models_pytorch": "segmentation-models-pytorch==0.3.1",
    "timm": "timm==0.4.12",
    "pretrainedmodels": "pretrainedmodels==0.7.4",
    "efficientnet_pytorch": "efficientnet-pytorch==0.7.1",
    "munch": "munch==4.0.0",
}


def discover_bundled_grandqc_home() -> Path | None:
    candidates = []
    env_home = os.environ.get("GRANDQC_HOME", "").strip()
    if env_home:
        candidates.append(Path(env_home))
    candidates.append(Path("/opt/cellphenotyper/third_party/grandqc"))
    for cand in candidates:
        try:
            if cand.exists() and cand.is_dir():
                return cand
        except Exception:
            continue
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run GrandQC artifact analysis on a converted OME-TIFF.")
    ap.add_argument("--image", required=True, help="Input RGB OME-TIFF from CellPhenotyper conversion")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--sample-id", required=True, help="Sample identifier")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Inference device")
    ap.add_argument("--cache-dir", default="", help="Shared cache dir for GrandQC models and Python deps")
    ap.add_argument("--bootstrap-deps", action="store_true", help="Auto-install missing Python deps into the cache dir")
    ap.add_argument("--download-models", action="store_true", help="Auto-download GrandQC checkpoints from Zenodo")
    ap.add_argument("--default-source-mpp", type=float, default=0.25, help="Fallback source MPP when TIFF metadata is missing")
    ap.add_argument("--artifact-mpp-model", default="auto", choices=["auto", "1.0", "1.5", "2.0"], help="GrandQC artifact model magnification surrogate in MPP; 'auto' selects based on source MPP")
    ap.add_argument("--tissue-mpp-model", type=float, default=10.0, help="GrandQC tissue detector working MPP")
    ap.add_argument("--patch-size", type=int, default=512, help="Model patch size")
    ap.add_argument("--artifact-overlap-fraction", type=float, default=0.5, help="Fractional overlap between artifact tiles, used with center-crop merge")
    ap.add_argument("--overlay-factor", type=int, default=10, help="Approximate reduction factor for saved overlay")
    ap.add_argument("--preview-max-side", type=int, default=4096, help="Maximum long side for preview PNG/JPG outputs")
    ap.add_argument("--create-geojson", action="store_true", help="Export artifact GeoJSON scaled to the original image coordinates")
    ap.add_argument("--dry-run", action="store_true", help="Validate setup and metadata only")
    return ap.parse_args()


@contextlib.contextmanager
def file_lock(lock_path: Path):
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def ensure_runtime_deps(cache_dir: Path, enable_bootstrap: bool, require_torch: bool = True) -> None:
    bundled_home = discover_bundled_grandqc_home()
    using_bundled_deps = False
    if bundled_home is not None:
        bundled_deps = bundled_home / "pydeps"
        if bundled_deps.exists():
            if str(bundled_deps) not in sys.path:
                sys.path.insert(0, str(bundled_deps))
            importlib.invalidate_caches()
            using_bundled_deps = True

    deps_dir = cache_dir / "pydeps"
    deps_dir.mkdir(parents=True, exist_ok=True)
    if str(deps_dir) not in sys.path:
        sys.path.insert(0, str(deps_dir))
    importlib.invalidate_caches()

    required = dict(GRANDQC_BASE_DEP_SPECS)
    if require_torch:
        required.update(GRANDQC_INFERENCE_DEP_SPECS)
    missing = [pkg_name for mod_name, pkg_name in required.items() if importlib.util.find_spec(mod_name) is None]
    if using_bundled_deps and not missing:
        print(f"[INFO] Using bundled GrandQC Python deps from {bundled_deps}", flush=True)
        return
    if require_torch and importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is required in the runtime container for GrandQC integration, but torch is not installed.")
    if not missing:
        if using_bundled_deps:
            return
        if not enable_bootstrap:
            return
        stamp_path = deps_dir / ".grandqc_requirements_stamp.json"
        desired_specs = sorted(required.values())
        if stamp_path.exists():
            try:
                current_specs = json.loads(stamp_path.read_text(encoding="utf-8"))
                if current_specs == desired_specs:
                    return
            except Exception:
                pass
    if not enable_bootstrap:
        raise RuntimeError(
            "Missing Python dependencies for GrandQC integration: "
            + ", ".join(missing)
            + ". Re-run with --bootstrap-deps or bake them into the runtime image."
        )
    with file_lock(cache_dir / "locks" / "pip.lock"):
        stamp_path = deps_dir / ".grandqc_requirements_stamp.json"
        desired_specs = sorted(required.values())
        current_specs = None
        if stamp_path.exists():
            try:
                current_specs = json.loads(stamp_path.read_text(encoding="utf-8"))
            except Exception:
                current_specs = None
        really_missing = [pkg_name for mod_name, pkg_name in required.items() if importlib.util.find_spec(mod_name) is None]
        if really_missing or current_specs != desired_specs:
            base_specs = sorted(GRANDQC_BASE_DEP_SPECS.values())
            infer_specs = [
                GRANDQC_INFERENCE_DEP_SPECS["timm"],
                GRANDQC_INFERENCE_DEP_SPECS["pretrainedmodels"],
                GRANDQC_INFERENCE_DEP_SPECS["efficientnet_pytorch"],
                GRANDQC_INFERENCE_DEP_SPECS["munch"],
                GRANDQC_INFERENCE_DEP_SPECS["segmentation_models_pytorch"],
            ] if require_torch else []
            print(f"[INFO] Installing GrandQC Python deps: {', '.join(desired_specs)}", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "--target", str(deps_dir), *base_specs])
            if infer_specs:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "--no-deps", "--target", str(deps_dir), *infer_specs])
            stamp_path.write_text(json.dumps(desired_specs, indent=2), encoding="utf-8")
    importlib.invalidate_caches()
    if str(deps_dir) not in sys.path:
        sys.path.insert(0, str(deps_dir))


def load_modules(require_torch: bool = True):
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401
    import tifffile  # noqa: F401
    import zarr  # noqa: F401
    from PIL import Image  # noqa: F401
    from tqdm import tqdm  # noqa: F401

    torch = None
    smp = None
    if require_torch:
        import segmentation_models_pytorch as smp_mod  # noqa: F401
        import torch as torch_mod  # noqa: F401
        smp = smp_mod
        torch = torch_mod

    return {
        "cv2": cv2,
        "np": np,
        "smp": smp,
        "tifffile": tifffile,
        "torch": torch,
        "zarr": zarr,
        "Image": Image,
        "tqdm": tqdm,
    }


def zenodo_files(record_id: str) -> dict[str, dict]:
    with urlopen_with_certifi(f"https://zenodo.org/api/records/{record_id}", timeout=60) as resp:
        payload = json.load(resp)
    return {f["key"]: f for f in payload.get("files", [])}


def download_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urlopen_with_certifi(url, timeout=300) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)


def urlopen_with_certifi(url: str, timeout: int = 60):
    cafile = None
    try:
        import certifi  # type: ignore
        cafile = certifi.where()
    except Exception:
        cafile = None
    context = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    return urllib.request.urlopen(url, timeout=timeout, context=context)


def ensure_models(cache_dir: Path, artifact_mpp: float, download_models: bool) -> tuple[Path, Path]:
    bundled_home = discover_bundled_grandqc_home()
    if bundled_home is not None:
        bundled_tissue = bundled_home / "models" / "td" / TISSUE_MODEL_FILE
        bundled_artifact = bundled_home / "models" / "qc" / ARTIFACT_MODEL_FILES[artifact_mpp]
        if bundled_tissue.exists() and bundled_artifact.exists():
            print(f"[INFO] Using bundled GrandQC checkpoints from {bundled_home / 'models'}", flush=True)
            return bundled_tissue, bundled_artifact

    model_root = cache_dir / "models"
    td_dir = model_root / "td"
    qc_dir = model_root / "qc"
    td_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    tissue_dest = td_dir / TISSUE_MODEL_FILE
    artifact_dest = qc_dir / ARTIFACT_MODEL_FILES[artifact_mpp]
    if tissue_dest.exists() and artifact_dest.exists():
        return tissue_dest, artifact_dest
    if not download_models:
        raise RuntimeError(
            f"GrandQC checkpoints missing under {model_root}. Re-run with --download-models or pre-populate the cache directory."
        )
    with file_lock(cache_dir / "locks" / "models.lock"):
        if not tissue_dest.exists():
            files = zenodo_files(TISSUE_RECORD_ID)
            meta = files.get(TISSUE_MODEL_FILE)
            if not meta:
                raise FileNotFoundError(f"Could not find {TISSUE_MODEL_FILE} in Zenodo record {TISSUE_RECORD_ID}")
            print(f"[INFO] Downloading GrandQC tissue checkpoint -> {tissue_dest}", flush=True)
            download_file(meta["links"]["self"], tissue_dest)
        if not artifact_dest.exists():
            files = zenodo_files(ARTIFACT_RECORD_ID)
            key = ARTIFACT_MODEL_FILES[artifact_mpp]
            meta = files.get(key)
            if not meta:
                raise FileNotFoundError(f"Could not find {key} in Zenodo record {ARTIFACT_RECORD_ID}")
            print(f"[INFO] Downloading GrandQC artifact checkpoint -> {artifact_dest}", flush=True)
            download_file(meta["links"]["self"], artifact_dest)
    return tissue_dest, artifact_dest


class OmeZarrReader:
    def __init__(self, path: str, tifffile_mod, zarr_mod):
        self.path = str(path)
        self._tifffile = tifffile_mod
        self._zarr = zarr_mod
        self._store = None
        self._root = None
        self._tf = None
        self._level_keys: list[str] = []
        self._axes_by_level: list[str] = []
        self._shape_by_level: list[tuple[int, ...]] = []
        self.level0_h = 0
        self.level0_w = 0
        self.level0_c = 0

    def __enter__(self):
        self._tf = self._tifffile.TiffFile(self.path)
        self._store = self._tifffile.imread(self.path, aszarr=True)
        self._root = self._zarr.open(self._store, mode="r")
        if hasattr(self._root, "keys"):
            self._level_keys = sorted(list(self._root.keys()), key=lambda x: int(x))
        else:
            self._level_keys = [""]
        levels = getattr(self._tf.series[0], "levels", None) or [self._tf.series[0]]
        for idx, _ in enumerate(self._level_keys):
            series = levels[idx] if idx < len(levels) else levels[0]
            self._axes_by_level.append(getattr(series, "axes", "YXS"))
            self._shape_by_level.append(tuple(getattr(series, "shape", ())))
        self.level0_h, self.level0_w, self.level0_c = self._shape_yxc(0)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._store is not None and hasattr(self._store, "close"):
                self._store.close()
        finally:
            if self._tf is not None:
                self._tf.close()

    def _level_arr(self, level: int):
        if self._level_keys == [""]:
            return self._root
        return self._root[self._level_keys[level]]

    def _shape_yxc(self, level: int) -> tuple[int, int, int]:
        axes = self._axes_by_level[level]
        shape = self._shape_by_level[level]
        axis_to_size = {axis: shape[i] for i, axis in enumerate(axes)}
        if "Y" not in axis_to_size or "X" not in axis_to_size:
            raise ValueError(f"Unsupported axes for OME-TIFF level {level}: {axes}")
        h = int(axis_to_size["Y"])
        w = int(axis_to_size["X"])
        c = int(axis_to_size.get("S", axis_to_size.get("C", 1)))
        return h, w, c

    def read_region(self, y0: int, y1: int, x0: int, x1: int, level: int = 0):
        arr = self._level_arr(level)
        axes = self._axes_by_level[level]
        if axes == "YXS":
            data = arr[y0:y1, x0:x1, :]
            return data[:]
        if axes == "CYX":
            data = arr[:, y0:y1, x0:x1][:]
            return data.transpose(1, 2, 0)
        if axes == "SYX":
            data = arr[:, y0:y1, x0:x1][:]
            return data.transpose(1, 2, 0)
        if axes == "YX":
            data = arr[y0:y1, x0:x1][:]
            return data[..., None]
        raise ValueError(f"Unsupported axes for OME-TIFF level {level}: {axes}")

    def read_full_level(self, level: int):
        h, w, _ = self._shape_yxc(level)
        return self.read_region(0, h, 0, w, level=level)

    def read_vips_thumbnail(self, target_h: int, target_w: int, np_mod):
        try:
            import pyvips  # type: ignore
        except Exception:
            return None
        try:
            image = pyvips.Image.thumbnail(self.path, int(target_w), height=int(target_h), size="force")
            if image.bands > 3:
                image = image.extract_band(0, n=3)
            dtype_map = {
                "uchar": np_mod.uint8,
                "char": np_mod.int8,
                "ushort": np_mod.uint16,
                "short": np_mod.int16,
                "uint": np_mod.uint32,
                "int": np_mod.int32,
                "float": np_mod.float32,
                "double": np_mod.float64,
            }
            dtype = dtype_map.get(str(image.format))
            if dtype is None:
                return None
            arr = np_mod.frombuffer(image.write_to_memory(), dtype=dtype)
            arr = arr.reshape(int(image.height), int(image.width), int(image.bands))
            return arr.copy()
        except Exception as exc:
            print(f"[WARN] pyvips thumbnail read failed; falling back to tifffile/zarr reader: {exc}", flush=True)
            return None

    def read_region_decimated(self, y0: int, y1: int, x0: int, x1: int, target_h: int, target_w: int, level: int, np_mod):
        arr = self._level_arr(level)
        axes = self._axes_by_level[level]
        y0 = max(0, int(y0))
        y1 = max(y0 + 1, int(y1))
        x0 = max(0, int(x0))
        x1 = max(x0 + 1, int(x1))
        target_h = max(1, int(target_h))
        target_w = max(1, int(target_w))
        step_y = max(1, int(math.floor((y1 - y0) / float(target_h))))
        step_x = max(1, int(math.floor((x1 - x0) / float(target_w))))
        if axes == "YXS":
            data = arr[y0:y1:step_y, x0:x1:step_x, :]
            return data[:]
        if axes == "CYX":
            data = arr[:, y0:y1:step_y, x0:x1:step_x][:]
            return data.transpose(1, 2, 0)
        if axes == "SYX":
            data = arr[:, y0:y1:step_y, x0:x1:step_x][:]
            return data.transpose(1, 2, 0)
        if axes == "YX":
            data = arr[y0:y1:step_y, x0:x1:step_x][:]
            return data[..., None]
        raise ValueError(f"Unsupported axes for OME-TIFF level {level}: {axes}")

    def read_resampled_full(self, target_h: int, target_w: int, image_mod, np_mod):
        target_h = max(1, int(target_h))
        target_w = max(1, int(target_w))
        if max(self.level0_h / float(target_h), self.level0_w / float(target_w)) > 2.0:
            thumb = self.read_vips_thumbnail(target_h, target_w, np_mod)
            if thumb is not None:
                return normalize_to_rgb_uint8(thumb, np_mod)
        best_level = 0
        best_score = None
        target_scale_h = self.level0_h / float(target_h)
        for level in range(len(self._level_keys)):
            h, w, _ = self._shape_yxc(level)
            scale_h = self.level0_h / float(h)
            score = abs(math.log(max(scale_h, 1e-6) / max(target_scale_h, 1e-6)))
            if best_score is None or score < best_score:
                best_score = score
                best_level = level
        lvl_h, lvl_w, _ = self._shape_yxc(best_level)
        if max(lvl_h / target_h, lvl_w / target_w) > 2.0:
            lvl = self.read_region_decimated(0, lvl_h, 0, lvl_w, target_h, target_w, best_level, np_mod)
        else:
            lvl = self.read_full_level(best_level)
        lvl = normalize_to_rgb_uint8(lvl, np_mod)
        if lvl.shape[0] == target_h and lvl.shape[1] == target_w:
            return lvl
        pil = image_mod.fromarray(lvl[:, :, :3])
        pil = pil.resize((target_w, target_h), image_mod.Resampling.BILINEAR)
        return np_mod.array(pil)

    def read_resampled_region(self, y0: int, y1: int, x0: int, x1: int, target_h: int, target_w: int, image_mod, np_mod, level: int = 0):
        target_h = max(1, int(target_h))
        target_w = max(1, int(target_w))
        if level != 0:
            region_h = max(1, int(y1) - int(y0))
            region_w = max(1, int(x1) - int(x0))
            if max(region_h / target_h, region_w / target_w) > 2.0:
                patch = self.read_region_decimated(y0, y1, x0, x1, target_h, target_w, level, np_mod)
            else:
                patch = self.read_region(y0, y1, x0, x1, level=level)
            patch = normalize_to_rgb_uint8(patch, np_mod)
            if patch.shape[0] == target_h and patch.shape[1] == target_w:
                return patch
            pil = image_mod.fromarray(patch[:, :, :3])
            pil = pil.resize((target_w, target_h), image_mod.Resampling.BILINEAR)
            return np_mod.array(pil)

        region_h0 = max(1, int(y1) - int(y0))
        region_w0 = max(1, int(x1) - int(x0))
        target_scale_h = region_h0 / float(target_h)
        best_level = 0
        best_score = None
        for idx in range(len(self._level_keys)):
            lvl_h, lvl_w, _ = self._shape_yxc(idx)
            level_scale_h = self.level0_h / float(lvl_h)
            score = abs(math.log(max(level_scale_h, 1e-6) / max(target_scale_h, 1e-6)))
            if best_score is None or score < best_score:
                best_score = score
                best_level = idx

        lvl_h, lvl_w, _ = self._shape_yxc(best_level)
        scale_y = self.level0_h / float(lvl_h)
        scale_x = self.level0_w / float(lvl_w)
        ly0 = max(0, min(lvl_h - 1, int(math.floor(int(y0) / scale_y))))
        ly1 = max(ly0 + 1, min(lvl_h, int(math.ceil(int(y1) / scale_y))))
        lx0 = max(0, min(lvl_w - 1, int(math.floor(int(x0) / scale_x))))
        lx1 = max(lx0 + 1, min(lvl_w, int(math.ceil(int(x1) / scale_x))))
        region_h = max(1, ly1 - ly0)
        region_w = max(1, lx1 - lx0)
        if max(region_h / target_h, region_w / target_w) > 2.0:
            patch = self.read_region_decimated(ly0, ly1, lx0, lx1, target_h, target_w, best_level, np_mod)
        else:
            patch = self.read_region(ly0, ly1, lx0, lx1, level=best_level)
        patch = normalize_to_rgb_uint8(patch, np_mod)
        if patch.shape[0] == target_h and patch.shape[1] == target_w:
            return patch
        pil = image_mod.fromarray(patch[:, :, :3])
        pil = pil.resize((target_w, target_h), image_mod.Resampling.BILINEAR)
        return np_mod.array(pil)


def normalize_to_rgb_uint8(arr, np_mod):
    if arr.ndim == 2:
        arr = np_mod.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np_mod.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np_mod.uint8:
        return arr
    arr = arr.astype(np_mod.float32, copy=False)
    out = np_mod.zeros(arr.shape, dtype=np_mod.uint8)
    for ch in range(arr.shape[-1]):
        plane = arr[..., ch]
        vals = plane[np_mod.isfinite(plane)]
        if vals.size == 0:
            continue
        lo = float(np_mod.percentile(vals, 1.0))
        hi = float(np_mod.percentile(vals, 99.0))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo = float(vals.min())
            hi = float(vals.max())
        if hi <= lo:
            scaled = np_mod.zeros_like(plane, dtype=np_mod.float32)
        else:
            scaled = np_mod.clip((plane - lo) / (hi - lo), 0.0, 1.0)
        out[..., ch] = np_mod.round(scaled * 255.0).astype(np_mod.uint8)
    return out


def extract_patch_with_pad(arr, y0: int, x0: int, patch_size: int, np_mod):
    h, w = arr.shape[:2]
    y0 = max(0, min(int(y0), max(0, h - 1)))
    x0 = max(0, min(int(x0), max(0, w - 1)))
    y1 = min(h, y0 + patch_size)
    x1 = min(w, x0 + patch_size)
    patch = arr[y0:y1, x0:x1]
    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
        return patch
    out = np_mod.zeros((patch_size, patch_size, patch.shape[2]), dtype=patch.dtype)
    out[:patch.shape[0], :patch.shape[1], :] = patch
    return out


def read_source_mpp(path: str, tifffile_mod) -> float | None:
    try:
        with tifffile_mod.TiffFile(path) as tf:
            page = tf.pages[0]
            desc = None
            try:
                desc = page.tags["ImageDescription"].value
            except Exception:
                desc = None
            if isinstance(desc, str):
                m = re.search(r'PhysicalSizeX="([0-9]+(?:\.[0-9]+)?)"', desc)
                if m:
                    return float(m.group(1))
                m = re.search(r"MPP\s*=\s*([0-9]+(?:\.[0-9]+)?)", desc)
                if m:
                    return float(m.group(1))
    except Exception:
        return None
    return None


def resolve_artifact_mpp_model(requested: str, source_mpp: float) -> float:
    if requested != "auto":
        return float(requested)
    # GrandQC provides 10x (1.0), 7x (1.5), and 5x (2.0) artifact models.
    # For very high-resolution zoomed-in inputs we prefer the denser 10x grid;
    # for typical 20x/40x WSI crops we keep the 7x default; for coarse inputs
    # we step down to the 5x model.
    if source_mpp <= 0.12:
        return 1.0
    if source_mpp >= 0.30:
        return 2.0
    return 1.5


def resolve_device(requested: str, torch_mod) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch_mod.cuda.is_available():
            raise RuntimeError("GrandQC requested CUDA but torch.cuda.is_available() is False")
        return "cuda"
    if requested == "mps":
        if not getattr(torch_mod.backends, "mps", None) or not torch_mod.backends.mps.is_available():
            raise RuntimeError("GrandQC requested MPS but torch.backends.mps.is_available() is False")
        return "mps"
    if torch_mod.cuda.is_available():
        return "cuda"
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_load_any(path: str, torch_mod, map_location: str):
    install_timm_compat_aliases()
    try:
        return torch_mod.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_mod.load(path, map_location=map_location)


def install_timm_compat_aliases() -> None:
    try:
        import timm.layers as timm_layers  # type: ignore
        sys.modules.setdefault("timm.models.layers", timm_layers)
        try:
            import timm.layers.activations as timm_layers_activations  # type: ignore
            sys.modules.setdefault("timm.models.layers.activations", timm_layers_activations)
        except Exception:
            pass
        try:
            import timm.layers.activations_me as timm_layers_activations_me  # type: ignore
            sys.modules.setdefault("timm.models.layers.activations_me", timm_layers_activations_me)
        except Exception:
            pass
        try:
            import timm.layers.norm_act as timm_layers_norm_act  # type: ignore
            sys.modules.setdefault("timm.models.layers.norm_act", timm_layers_norm_act)
        except Exception:
            pass
        try:
            import timm.models._efficientnet_blocks as timm_efficientnet_blocks  # type: ignore
            sys.modules.setdefault("timm.models.efficientnet_blocks", timm_efficientnet_blocks)
        except Exception:
            pass
        try:
            import timm.models._efficientnet_builder as timm_efficientnet_builder  # type: ignore
            sys.modules.setdefault("timm.models.efficientnet_builder", timm_efficientnet_builder)
        except Exception:
            pass
    except Exception:
        pass


def make_preprocessing_fn(smp_mod):
    return smp_mod.encoders.get_preprocessing_fn("timm-efficientnet-b0", "imagenet")


def preprocess_patch(rgb, preprocessing_fn, np_mod):
    x = preprocessing_fn(rgb)
    return x.transpose(2, 0, 1).astype("float32")


def predict_mask(model, x_tensor, torch_mod, device: str):
    use_autocast = device == "cuda" and hasattr(torch_mod, "autocast")
    with torch_mod.inference_mode():
        if use_autocast:
            with torch_mod.autocast(device_type="cuda", dtype=torch_mod.float16):
                if hasattr(model, "predict"):
                    pred = model.predict(x_tensor)
                else:
                    pred = model(x_tensor)
        else:
            if hasattr(model, "predict"):
                pred = model.predict(x_tensor)
            else:
                pred = model(x_tensor)
    pred = pred.squeeze().detach().cpu().numpy()
    if pred.ndim == 2:
        return pred.astype("int16")
    return pred.argmax(axis=0).astype("int16")


def predict_scores(model, x_tensor, torch_mod, device: str):
    import numpy as _np
    use_autocast = device == "cuda" and hasattr(torch_mod, "autocast")
    with torch_mod.inference_mode():
        if use_autocast:
            with torch_mod.autocast(device_type="cuda", dtype=torch_mod.float16):
                if hasattr(model, "predict"):
                    pred = model.predict(x_tensor)
                else:
                    pred = model(x_tensor)
        else:
            if hasattr(model, "predict"):
                pred = model.predict(x_tensor)
            else:
                pred = model(x_tensor)
    pred = pred.squeeze().detach().float().cpu().numpy()
    if pred.ndim == 2:
        return pred[None, ...].astype("float32")
    pred = pred.astype("float32")
    pred -= pred.max(axis=0, keepdims=True)
    np_exp = _np.exp(pred).astype("float32", copy=False)
    denom = _np.maximum(np_exp.sum(axis=0, keepdims=True), 1e-6)
    return (np_exp / denom).astype("float32")


def make_color_map(mask, np_mod):
    rgb = np_mod.zeros(mask.shape + (3,), dtype=np_mod.uint8)
    for class_id, color in enumerate(QC_COLORS, start=1):
        rgb[mask == class_id] = color
    return rgb


def preview_shape(height: int, width: int, overlay_factor: int, preview_max_side: int) -> tuple[int, int]:
    overlay_factor = max(1, int(overlay_factor))
    preview_max_side = max(256, int(preview_max_side))
    long_side = max(int(height), int(width))
    min_required_factor = max(1, int(math.ceil(long_side / float(preview_max_side))))
    effective_factor = max(overlay_factor, min_required_factor) if long_side > preview_max_side else 1
    coarse_h = max(1, int(math.ceil(height / effective_factor)))
    coarse_w = max(1, int(math.ceil(width / effective_factor)))
    scale = min(1.0, preview_max_side / float(max(coarse_h, coarse_w)))
    return max(1, int(round(coarse_h * scale))), max(1, int(round(coarse_w * scale)))


def make_preview_mask(mask, preview_h: int, preview_w: int, Image, np_mod):
    if mask.shape[0] == preview_h and mask.shape[1] == preview_w:
        return mask
    pil = Image.fromarray(mask)
    resized = pil.resize((preview_w, preview_h), Image.Resampling.NEAREST)
    return np_mod.array(resized, dtype=mask.dtype, copy=False)


def sliding_positions(limit: int, patch_size: int, overlap_px: int) -> list[int]:
    limit = max(1, int(limit))
    patch_size = max(1, int(patch_size))
    overlap_px = max(0, min(int(overlap_px), patch_size - 1))
    if limit <= patch_size:
        return [0]
    stride = max(1, patch_size - overlap_px)
    last = max(0, limit - patch_size)
    positions = []
    cur = 0
    while True:
        positions.append(cur)
        if cur >= last:
            break
        nxt = min(cur + stride, last)
        if nxt == cur:
            break
        cur = nxt
    return positions


def center_merge_bounds(positions: list[int], patch_size: int, limit: int) -> list[tuple[int, int, int, int]]:
    bounds = []
    for idx, pos in enumerate(positions):
        write_start = 0 if idx == 0 else int((positions[idx - 1] + pos + patch_size) // 2)
        write_end = int(limit) if idx == len(positions) - 1 else int((positions[idx + 1] + pos + patch_size) // 2)
        crop_start = int(write_start - pos)
        crop_end = int(write_end - pos)
        bounds.append((int(write_start), int(write_end), crop_start, crop_end))
    return bounds


def overlap_weights_1d(patch_size: int, crop_start: int, crop_end: int, np_mod):
    w = np_mod.ones(int(patch_size), dtype=np_mod.float32)
    left = max(0, int(crop_start))
    right = max(0, int(patch_size - crop_end))
    if left > 0:
        w[:left] = np_mod.arange(left, dtype=np_mod.float32) / float(left)
    if right > 0:
        taper = 1.0 - (np_mod.arange(right, dtype=np_mod.float32) / float(right))
        w[-right:] = np_mod.minimum(w[-right:], taper)
    return np_mod.clip(w, 1e-3, 1.0)


def smooth_blend_weights_1d(length: int, np_mod, edge_floor: float = 1e-3):
    length = int(max(1, length))
    if length <= 2:
        return np_mod.ones(length, dtype=np_mod.float32)
    x = np_mod.linspace(0.0, np_mod.pi, num=length, dtype=np_mod.float32)
    w = np_mod.sin(x) ** 2
    return np_mod.clip(w.astype(np_mod.float32, copy=False), float(edge_floor), 1.0)


def choose_tissue_thumbnail_shape(level0_h: int, level0_w: int,
                                  source_mpp: float, tissue_mpp_model: float,
                                  patch_size: int) -> tuple[int, int, float]:
    target_h = max(1, int(round(level0_h * source_mpp / tissue_mpp_model)))
    target_w = max(1, int(round(level0_w * source_mpp / tissue_mpp_model)))
    min_patch_span = max(256, int(patch_size) * 2)
    scale = max(
        float(min_patch_span) / float(max(1, target_h)),
        float(min_patch_span) / float(max(1, target_w)),
        1.0,
    )
    target_h = min(int(level0_h), max(1, int(round(target_h * scale))))
    target_w = min(int(level0_w), max(1, int(round(target_w * scale))))
    effective_mpp = max(source_mpp, source_mpp * (float(level0_h) / float(max(1, target_h))))
    return target_h, target_w, effective_mpp


def heuristic_tissue_mask(rgb_thumb, cv2_mod, np_mod):
    rgb_thumb = normalize_to_rgb_uint8(rgb_thumb, np_mod)
    gray = cv2_mod.cvtColor(rgb_thumb, cv2_mod.COLOR_RGB2GRAY)
    hsv = cv2_mod.cvtColor(rgb_thumb, cv2_mod.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    tissue_candidate = np_mod.logical_or(gray < 235, sat > 15).astype(np_mod.uint8) * 255
    kernel = cv2_mod.getStructuringElement(cv2_mod.MORPH_ELLIPSE, (5, 5))
    tissue_candidate = cv2_mod.morphologyEx(tissue_candidate, cv2_mod.MORPH_OPEN, kernel)
    tissue_candidate = cv2_mod.morphologyEx(tissue_candidate, cv2_mod.MORPH_CLOSE, kernel)
    tissue_candidate = cv2_mod.medianBlur(tissue_candidate, 5)
    # GrandQC convention in this wrapper: 0=tissue, 1=background.
    return np_mod.where(tissue_candidate > 0, 0, 1).astype(np_mod.uint8)


def mask_to_geojson(mask, output_path: Path, scale_factor: float, cv2_mod):
    features = []
    for class_value in range(2, 7):
        class_mask = ((mask == class_value).astype("uint8") * 255)
        contours, _ = cv2_mod.findContours(class_mask, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = contour.reshape(-1, 2)
            if pts.shape[0] < 4:
                continue
            scaled = (pts.astype("float64") * float(scale_factor)).tolist()
            if scaled[0] != scaled[-1]:
                scaled.append(scaled[0])
            features.append({
                "type": "Feature",
                "properties": {
                    "class_id": int(class_value),
                    "classification": QC_CLASS_MAPPING.get(class_value, "Unknown"),
                    "area": float(cv2_mod.contourArea(contour) * (scale_factor ** 2)),
                },
                "geometry": {"type": "Polygon", "coordinates": [scaled]},
            })
    payload = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {"class_mapping": QC_CLASS_MAPPING, "scale_factor": float(scale_factor)},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_mask_tiff(path: Path, arr, tifffile_mod):
    tifffile_mod.imwrite(str(path), arr, compression="deflate")


def refine_small_fov_foreign_object_mask(full_mask, thumb_rgb, source_mpp: float, artifact_meta: dict, image_mod, cv2_mod, np_mod):
    """
    GrandQC can overcall broad horizontal artifact bands on small high-magnification
    fields. For that regime, keep the GrandQC artifact support but collapse it to
    the dominant very-dark foreign-object component seen in the thumbnail.
    """
    h, w = map(int, full_mask.shape[:2])
    patch_grid = artifact_meta.get("patch_grid_yx", [0, 0]) or [0, 0]
    processed_shape_yx = artifact_meta.get("processed_level0_shape_yx", [0, 0]) or [0, 0]
    artifact_candidate = np_mod.isin(full_mask, [2, 3, 4, 5, 6])
    artifact_fraction = float(artifact_candidate.mean())
    level0_h = int(processed_shape_yx[0]) if len(processed_shape_yx) > 0 else 0
    level0_w = int(processed_shape_yx[1]) if len(processed_shape_yx) > 1 else 0
    fov_h_um = float(level0_h * source_mpp) if level0_h > 0 else 0.0
    fov_w_um = float(level0_w * source_mpp) if level0_w > 0 else 0.0
    max_fov_um = max(fov_h_um, fov_w_um)
    min_fov_um = min(fov_h_um, fov_w_um)
    meta = {
        "enabled": False,
        "reason": "not_applicable",
        "artifact_fraction_before": artifact_fraction,
        "patch_grid_yx": [int(patch_grid[0]), int(patch_grid[1])],
        "fov_um_yx": [float(fov_h_um), float(fov_w_um)],
    }
    if source_mpp > 0.12:
        meta["reason"] = "source_mpp_too_coarse"
        return full_mask, meta
    # Gate the special refinement by physical field-of-view, not just the number
    # of GrandQC tiles. Small high-magnification images can still require more
    # than 8 tiles after overlap while remaining exactly the regime we want here.
    if max_fov_um > 2600.0 or min_fov_um > 1800.0:
        meta["reason"] = "field_of_view_not_small"
        return full_mask, meta
    if artifact_fraction < 0.10:
        meta["reason"] = "artifact_fraction_too_small"
        return full_mask, meta

    thumb_aligned = np_mod.array(
        image_mod.fromarray(thumb_rgb).resize((w, h), image_mod.Resampling.BILINEAR),
        dtype=np_mod.uint8,
        copy=False,
    )
    gray = thumb_aligned.astype(np_mod.float32).mean(axis=2)
    artifact_gray = gray[artifact_candidate]
    if artifact_gray.size < 64:
        meta["reason"] = "too_few_artifact_pixels"
        return full_mask, meta

    dark_threshold = min(90.0, float(np_mod.quantile(artifact_gray, 0.05)) + 8.0)
    seed = ((gray <= dark_threshold) & artifact_candidate).astype(np_mod.uint8)
    num_labels, labels, stats, _ = cv2_mod.connectedComponentsWithStats(seed, connectivity=8)
    if num_labels <= 1:
        meta["reason"] = "no_dark_seed_component"
        meta["dark_threshold"] = dark_threshold
        return full_mask, meta

    component_areas = stats[1:, cv2_mod.CC_STAT_AREA]
    largest_idx = int(np_mod.argmax(component_areas)) + 1
    largest_area = int(component_areas[largest_idx - 1])
    dominant = labels == largest_idx
    ys, xs = np_mod.where(dominant)
    if ys.size == 0:
        meta["reason"] = "empty_dominant_component"
        meta["dark_threshold"] = dark_threshold
        return full_mask, meta

    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    elongation = float(max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w)))
    dominant_fraction = float(largest_area / max(int(artifact_candidate.sum()), 1))
    if largest_area < max(300, int(0.002 * h * w)):
        meta["reason"] = "dominant_component_too_small"
        meta["dark_threshold"] = dark_threshold
        return full_mask, meta
    if elongation < 1.3:
        meta["reason"] = "dominant_component_not_elongated"
        meta["dark_threshold"] = dark_threshold
        meta["dominant_elongation"] = elongation
        return full_mask, meta

    grow_px = max(4, int(round(min(h, w) * 0.0085)))
    kernel = cv2_mod.getStructuringElement(cv2_mod.MORPH_ELLIPSE, (grow_px * 2 + 1, grow_px * 2 + 1))
    refined = cv2_mod.dilate(dominant.astype(np_mod.uint8) * 255, kernel, iterations=1) > 0
    refined &= artifact_candidate

    num2, labels2, stats2, _ = cv2_mod.connectedComponentsWithStats((refined.astype(np_mod.uint8) * 255), connectivity=8)
    if num2 > 1:
        largest2_idx = int(np_mod.argmax(stats2[1:, cv2_mod.CC_STAT_AREA])) + 1
        refined = labels2 == largest2_idx
    contour_canvas = np_mod.zeros((h, w), dtype=np_mod.uint8)
    contours, _ = cv2_mod.findContours((refined.astype(np_mod.uint8) * 255), cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2_mod.drawContours(contour_canvas, contours, -1, 255, thickness=cv2_mod.FILLED)
        refined = contour_canvas > 0

    refined_fraction = float(refined.sum() / max(int(artifact_candidate.sum()), 1))
    if refined_fraction < 0.01:
        meta["reason"] = "refined_fraction_too_small"
        meta["dark_threshold"] = dark_threshold
        return full_mask, meta
    if refined_fraction > 0.50:
        meta["reason"] = "refined_fraction_too_large"
        meta["dark_threshold"] = dark_threshold
        meta["refined_fraction"] = refined_fraction
        return full_mask, meta

    refined_full_mask = full_mask.copy()
    fallback = np_mod.where(full_mask == 7, 7, 1).astype(full_mask.dtype, copy=False)
    remove = artifact_candidate & (~refined)
    refined_full_mask[remove] = fallback[remove]

    meta.update({
        "enabled": True,
        "reason": "applied",
        "dark_threshold": float(dark_threshold),
        "grow_px": int(grow_px),
        "dominant_area": int(largest_area),
        "dominant_fraction_of_artifact": dominant_fraction,
        "dominant_elongation": elongation,
        "artifact_fraction_after": float(np_mod.isin(refined_full_mask, [2, 3, 4, 5, 6]).mean()),
    })
    return refined_full_mask, meta


def run_tissue_detection(reader, source_mpp: float, tissue_mpp_model: float, patch_size: int, device: str, tissue_ckpt: Path, mods):
    np_mod = mods["np"]
    Image = mods["Image"]
    torch_mod = mods["torch"]
    smp_mod = mods["smp"]

    target_h, target_w, effective_mpp = choose_tissue_thumbnail_shape(
        reader.level0_h, reader.level0_w, source_mpp, tissue_mpp_model, patch_size
    )
    print(
        f"[INFO] GrandQC tissue thumbnail target={target_h}x{target_w} "
        f"(requested_mpp={tissue_mpp_model:.3f}, effective_mpp={effective_mpp:.3f})",
        flush=True,
    )
    thumb = reader.read_resampled_full(target_h, target_w, Image, np_mod)
    thumb = normalize_to_rgb_uint8(thumb, np_mod)
    print(f"[INFO] GrandQC tissue thumbnail read complete: {thumb.shape[0]}x{thumb.shape[1]}", flush=True)

    enc_param = [int(mods["cv2"].IMWRITE_JPEG_QUALITY), 80]
    _, enc = mods["cv2"].imencode('.jpg', thumb, enc_param)
    thumb_jpeg = mods["cv2"].imdecode(enc, 1)
    thumb_rgb = mods["cv2"].cvtColor(thumb_jpeg, mods["cv2"].COLOR_BGR2RGB)
    del thumb, enc, thumb_jpeg

    preprocessing_fn = make_preprocessing_fn(smp_mod)
    model = smp_mod.UnetPlusPlus(
        encoder_name="timm-efficientnet-b0",
        encoder_weights=None,
        classes=2,
        activation=None,
    )
    state = torch_load_any(str(tissue_ckpt), torch_mod, map_location="cpu")
    model.load_state_dict(state)
    del state
    model.to(device)
    model.eval()

    height, width = thumb_rgb.shape[:2]
    wi_n = width // patch_size
    he_n = height // patch_size
    overhang_w = width - wi_n * patch_size
    overhang_h = height - he_n * patch_size
    tissue_mask = np_mod.empty((height, width), dtype=np_mod.uint8)
    for h in range(he_n + 1):
        if h == he_n and overhang_h <= 0:
            continue
        src_y0 = h * patch_size if h != he_n else max(0, height - patch_size)
        dst_y0 = h * patch_size if h != he_n else max(0, height - overhang_h)
        dst_y1 = min(height, dst_y0 + (patch_size if h != he_n else max(1, overhang_h)))
        mask_y0 = 0 if h != he_n else max(0, patch_size - (dst_y1 - dst_y0))
        mask_y1 = mask_y0 + (dst_y1 - dst_y0)
        for w in range(wi_n + 1):
            if w == wi_n and overhang_w <= 0:
                continue
            src_x0 = w * patch_size if w != wi_n else max(0, width - patch_size)
            dst_x0 = w * patch_size if w != wi_n else max(0, width - overhang_w)
            dst_x1 = min(width, dst_x0 + (patch_size if w != wi_n else max(1, overhang_w)))
            mask_x0 = 0 if w != wi_n else max(0, patch_size - (dst_x1 - dst_x0))
            mask_x1 = mask_x0 + (dst_x1 - dst_x0)
            if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
                continue
            patch = extract_patch_with_pad(thumb_rgb, src_y0, src_x0, patch_size, np_mod)
            x = preprocess_patch(patch, preprocessing_fn, np_mod)
            x_tensor = torch_mod.from_numpy(x).unsqueeze(0).to(device)
            mask = predict_mask(model, x_tensor, torch_mod, device).astype(np_mod.uint8)
            tissue_mask[dst_y0:dst_y1, dst_x0:dst_x1] = mask[mask_y0:mask_y1, mask_x0:mask_x1]
            del patch, x, x_tensor, mask
    pred_tissue_fraction = float((tissue_mask == 0).astype(np_mod.float32).mean())
    fallback_mask = heuristic_tissue_mask(thumb_rgb, mods["cv2"], np_mod)
    fallback_tissue_fraction = float((fallback_mask == 0).astype(np_mod.float32).mean())
    if pred_tissue_fraction < 0.01 and fallback_tissue_fraction > 0.05:
        print(
            "[WARN] GrandQC tissue detector returned near-zero tissue on a nonblank thumbnail; "
            "falling back to heuristic tissue mask for artifact gating.",
            flush=True,
        )
        tissue_mask = fallback_mask
        tissue_mask_source = "heuristic_fallback"
    else:
        tissue_mask_source = "grandqc_model"
    color = np_mod.zeros(tissue_mask.shape + (3,), dtype=np_mod.uint8)
    color[tissue_mask == 0] = [50, 50, 250]
    color[tissue_mask == 1] = [128, 128, 128]
    overlay = mods["cv2"].addWeighted(thumb_rgb, 0.7, color, 0.3, 0)
    del model
    if device == "cuda":
        torch_mod.cuda.empty_cache()
    return thumb_rgb, tissue_mask, color, overlay, {
        "effective_tissue_mpp": float(effective_mpp),
        "tissue_mask_source": tissue_mask_source,
        "predicted_tissue_fraction": float(pred_tissue_fraction),
        "fallback_tissue_fraction": float(fallback_tissue_fraction),
    }


def run_artifact_detection(reader, source_mpp: float, artifact_mpp_model: float, tissue_mask_thumb, patch_size: int, overlap_fraction: float, device: str, artifact_ckpt: Path, mods):
    np_mod = mods["np"]
    Image = mods["Image"]
    torch_mod = mods["torch"]
    smp_mod = mods["smp"]

    model = torch_load_any(str(artifact_ckpt), torch_mod, map_location="cpu")
    model.to(device)
    model.eval()
    preprocessing_fn = make_preprocessing_fn(smp_mod)

    # Use overlapping tiles and keep only the stable center region from each tile.
    # This removes the hard seam artefacts that appear when neighboring GrandQC
    # predictions disagree near tile edges.
    target_w = max(1, int(round(reader.level0_w * source_mpp / artifact_mpp_model)))
    target_h = max(1, int(round(reader.level0_h * source_mpp / artifact_mpp_model)))
    effective_overlap_fraction = float(overlap_fraction)
    adaptive_overlap_boosted = False
    if source_mpp <= 0.12 and max(target_h, target_w) <= 2500:
        effective_overlap_fraction = max(effective_overlap_fraction, 0.75)
        adaptive_overlap_boosted = effective_overlap_fraction > float(overlap_fraction)
    overlap_px = max(0, min(int(round(patch_size * effective_overlap_fraction)), patch_size - 1))
    y_positions = sliding_positions(target_h, patch_size, overlap_px)
    x_positions = sliding_positions(target_w, patch_size, overlap_px)
    y_bounds = center_merge_bounds(y_positions, patch_size, target_h)
    x_bounds = center_merge_bounds(x_positions, patch_size, target_w)
    tissue_map_art = np_mod.array(Image.fromarray(tissue_mask_thumb).resize((target_w, target_h), Image.Resampling.LANCZOS))

    max_score_blend_pixels = 25_000_000
    use_score_blend = (target_h * target_w) <= max_score_blend_pixels
    full_mask = np_mod.full((target_h, target_w), 7, dtype=np_mod.uint8)
    score_sum = None
    weight_sum = None
    confidence_meta = {
        "enabled": False,
        "suppressed_pixels": 0,
        "artifact_probability_threshold": 0.55,
        "artifact_margin_threshold": 0.10,
    }
    scale_y = reader.level0_h / float(target_h)
    scale_x = reader.level0_w / float(target_w)
    for yi, out_y0 in enumerate(y_positions):
        out_y1 = min(target_h, out_y0 + patch_size)
        read_y0 = int(round(out_y0 * scale_y))
        read_y1 = int(round(out_y1 * scale_y))
        read_y1 = max(read_y0 + 1, min(reader.level0_h, read_y1))
        wy0, wy1, cy0, cy1 = y_bounds[yi]
        for xi, out_x0 in enumerate(x_positions):
            out_x1 = min(target_w, out_x0 + patch_size)
            read_x0 = int(round(out_x0 * scale_x))
            read_x1 = int(round(out_x1 * scale_x))
            read_x1 = max(read_x0 + 1, min(reader.level0_w, read_x1))
            wx0, wx1, cx0, cx1 = x_bounds[xi]
            td_patch = tissue_map_art[out_y0:out_y1, out_x0:out_x1]
            if td_patch.shape != (patch_size, patch_size):
                pad_h = patch_size - td_patch.shape[0]
                pad_w = patch_size - td_patch.shape[1]
                td_patch = np_mod.pad(td_patch, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="constant")
            if np_mod.count_nonzero(td_patch == 0) > 50:
                patch = reader.read_region(read_y0, read_y1, read_x0, read_x1, level=0)
                patch = normalize_to_rgb_uint8(patch, np_mod)
                pil = Image.fromarray(patch[:, :, :3]).resize((patch_size, patch_size), Image.Resampling.LANCZOS).convert("RGB")
                x = preprocess_patch(np_mod.array(pil), preprocessing_fn, np_mod)
                x_tensor = torch_mod.from_numpy(x).unsqueeze(0).to(device)
                if use_score_blend:
                    scores = predict_scores(model, x_tensor, torch_mod, device)
                    del x_tensor
                    if score_sum is None:
                        score_sum = np_mod.zeros((scores.shape[0], target_h, target_w), dtype=np_mod.float32)
                        weight_sum = np_mod.zeros((target_h, target_w), dtype=np_mod.float32)
                    valid = (td_patch == 0).astype(np_mod.float32)
                    h_eff = out_y1 - out_y0
                    w_eff = out_x1 - out_x0
                    wy = smooth_blend_weights_1d(h_eff, np_mod)
                    wx = smooth_blend_weights_1d(w_eff, np_mod)
                    tile_weight = (wy[:, None] * wx[None, :]) * valid[:h_eff, :w_eff]
                    score_sum[:, out_y0:out_y1, out_x0:out_x1] += scores[:, :h_eff, :w_eff] * tile_weight[:h_eff, :w_eff][None, :, :]
                    weight_sum[out_y0:out_y1, out_x0:out_x1] += tile_weight[:h_eff, :w_eff]
                    continue
                mask_raw = predict_mask(model, x_tensor, torch_mod, device).astype(np_mod.uint8)
                del x_tensor
                mask = np_mod.where(td_patch == 1, 7, mask_raw)
            else:
                if use_score_blend:
                    continue
                mask = np_mod.full((patch_size, patch_size), 7, dtype=np_mod.uint8)
            full_mask[wy0:wy1, wx0:wx1] = mask[cy0:cy1, cx0:cx1]
    if use_score_blend and score_sum is not None and weight_sum is not None:
        valid = weight_sum > 0
        score_sum[:, valid] /= weight_sum[valid][None, :]
        full_mask = np_mod.full((target_h, target_w), 7, dtype=np_mod.uint8)
        pred_idx = score_sum[:, valid].argmax(axis=0).astype(np_mod.uint8)
        n_classes = int(score_sum.shape[0])
        artifact_ids = np_mod.array([cid for cid in [2, 3, 4, 5, 6] if cid < n_classes], dtype=np_mod.int64)
        tissue_id = 1 if 1 < n_classes else 0
        background_id = 7 if 7 < n_classes else (n_classes - 1)
        if artifact_ids.size > 0:
            artifact_prob = score_sum[artifact_ids][:, valid].sum(axis=0)
            context_prob = np_mod.maximum(score_sum[tissue_id, valid], score_sum[background_id, valid])
            pred_is_artifact = np_mod.isin(pred_idx, artifact_ids)
            uncertain = pred_is_artifact & (
                (artifact_prob < confidence_meta["artifact_probability_threshold"]) |
                ((artifact_prob - context_prob) < confidence_meta["artifact_margin_threshold"])
            )
            fallback = np_mod.where(score_sum[tissue_id, valid] >= score_sum[background_id, valid], tissue_id, background_id).astype(np_mod.uint8)
            pred_idx[uncertain] = fallback[uncertain]
            confidence_meta["enabled"] = True
            confidence_meta["suppressed_pixels"] = int(uncertain.sum())
        full_mask[valid] = pred_idx
    del model
    if device == "cuda":
        torch_mod.cuda.empty_cache()
    meta = {
        "patch_grid_yx": [int(len(y_positions)), int(len(x_positions))],
        "processed_level0_shape_yx": [int(reader.level0_h), int(reader.level0_w)],
        "processed_qc_shape_yx": [int(full_mask.shape[0]), int(full_mask.shape[1])],
        "overlap_px": int(overlap_px),
        "requested_overlap_fraction": float(overlap_fraction),
        "overlap_fraction": float(effective_overlap_fraction),
        "adaptive_overlap_boosted": bool(adaptive_overlap_boosted),
        "merge_mode": "probability_blend_windowed_overlap" if use_score_blend else "center_crop_overlap",
        "score_blend_enabled": bool(use_score_blend),
        "confidence_filter": confidence_meta,
    }
    artifact_patch_span = max(1, int(round(artifact_mpp_model / source_mpp * patch_size)))
    return full_mask, tissue_map_art, artifact_patch_span, meta


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "grandqc_notice.txt").write_text(NOTICE_TEXT, encoding="utf-8")

    cache_dir = Path(args.cache_dir) if args.cache_dir else (Path(args.outdir).resolve().parent / ".grandqc_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_deps(cache_dir, args.bootstrap_deps, require_torch=not args.dry_run)
    mods = load_modules(require_torch=not args.dry_run)
    np_mod = mods["np"]
    Image = mods["Image"]
    torch_mod = mods["torch"]
    tifffile_mod = mods["tifffile"]
    cv2_mod = mods["cv2"]

    source_mpp = read_source_mpp(args.image, tifffile_mod)
    if not source_mpp or source_mpp <= 0:
        source_mpp = float(args.default_source_mpp)
        print(f"[WARN] Could not resolve source MPP from TIFF metadata; falling back to {source_mpp:.4f}", flush=True)
    else:
        print(f"[INFO] GrandQC source_mpp={source_mpp:.4f}", flush=True)

    with OmeZarrReader(args.image, tifffile_mod, mods["zarr"]) as reader:
        print(f"[INFO] GrandQC level0 shape={reader.level0_h}x{reader.level0_w} channels={reader.level0_c}", flush=True)
        meta = {
            "sample_id": args.sample_id,
            "input_image": str(Path(args.image).resolve()),
            "source_mpp": float(source_mpp),
            "artifact_mpp_model_requested": str(args.artifact_mpp_model),
            "tissue_mpp_model": float(args.tissue_mpp_model),
            "device": args.device,
            "grandqc_license": "CC BY-NC-SA 4.0",
            "grandqc_citation_doi": "10.1038/s41467-024-54769-y",
            "class_mapping": QC_CLASS_MAPPING,
            "level0_shape_yx": [int(reader.level0_h), int(reader.level0_w)],
        }
        if args.dry_run:
            (outdir / f"{args.sample_id}_grandqc_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return 0

        device = resolve_device(args.device, torch_mod)
        print(f"[NOTICE] GrandQC is licensed under CC BY-NC-SA 4.0. Cite Weng et al. 2024 when using these outputs.", flush=True)
        print(f"[INFO] GrandQC device={device}", flush=True)
        artifact_mpp_model = resolve_artifact_mpp_model(str(args.artifact_mpp_model), float(source_mpp))
        print(f"[INFO] GrandQC artifact model mpp={artifact_mpp_model:.1f} (requested={args.artifact_mpp_model})", flush=True)
        meta["artifact_mpp_model"] = float(artifact_mpp_model)

        tissue_ckpt, artifact_ckpt = ensure_models(cache_dir, artifact_mpp_model, args.download_models)
        meta["device"] = device

        print("[INFO] GrandQC running tissue detection", flush=True)
        thumb_rgb, tissue_mask_thumb, tissue_color, tissue_overlay, tissue_meta = run_tissue_detection(
            reader, source_mpp, args.tissue_mpp_model, args.patch_size, device, tissue_ckpt, mods
        )
        print("[INFO] GrandQC running artifact detection", flush=True)
        full_mask, tissue_map_art, artifact_patch_span, artifact_meta = run_artifact_detection(
            reader, source_mpp, artifact_mpp_model, tissue_mask_thumb, args.patch_size, args.artifact_overlap_fraction, device, artifact_ckpt, mods
        )
        full_mask, small_fov_refine_meta = refine_small_fov_foreign_object_mask(
            full_mask, thumb_rgb, source_mpp, artifact_meta, Image, cv2_mod, np_mod
        )
        print("[INFO] GrandQC preparing output masks", flush=True)

        artifact_binary = (np_mod.isin(full_mask, [2, 3, 4, 5, 6]).astype(np_mod.uint8) * 255)
        clean_tissue = ((full_mask == 1).astype(np_mod.uint8) * 255)
        tissue_binary = ((tissue_mask_thumb == 0).astype(np_mod.uint8) * 255)

        qc_target_h = full_mask.shape[0]
        qc_target_w = full_mask.shape[1]
        preview_h, preview_w = preview_shape(qc_target_h, qc_target_w, args.overlay_factor, args.preview_max_side)
        print(f"[INFO] GrandQC writing preview assets at {preview_h}x{preview_w}", flush=True)
        qc_mask_preview = make_preview_mask(full_mask, preview_h, preview_w, Image, np_mod)
        qc_color_preview = make_color_map(qc_mask_preview, np_mod)
        processed_h_l0, processed_w_l0 = artifact_meta["processed_level0_shape_yx"]
        background_for_overlay = reader.read_resampled_region(0, processed_h_l0, 0, processed_w_l0, preview_h, preview_w, Image, np_mod)
        overlay = cv2_mod.addWeighted(background_for_overlay[:, :, :3], 0.65, qc_color_preview[:, :, :3], 0.35, 0)
        artifact_preview = make_preview_mask(artifact_binary, preview_h, preview_w, Image, np_mod)
        artifact_color_preview = np_mod.zeros((preview_h, preview_w, 3), dtype=np_mod.uint8)
        artifact_color_preview[artifact_preview > 0] = [255, 0, 0]
        artifact_overlay = cv2_mod.addWeighted(background_for_overlay[:, :, :3], 0.70, artifact_color_preview, 0.30, 0)

        Image.fromarray(thumb_rgb).save(outdir / f"{args.sample_id}_grandqc_tissue_thumbnail.jpg", quality=85)
        Image.fromarray(tissue_color).save(outdir / f"{args.sample_id}_grandqc_tissue_mask_color.png")
        Image.fromarray(tissue_overlay).save(outdir / f"{args.sample_id}_grandqc_tissue_overlay.jpg", quality=85)
        Image.fromarray(qc_mask_preview).save(outdir / f"{args.sample_id}_grandqc_qc_mask.png")
        Image.fromarray(qc_color_preview).save(outdir / f"{args.sample_id}_grandqc_qc_mask_color.png")
        Image.fromarray(overlay).save(outdir / f"{args.sample_id}_grandqc_overlay.jpg", quality=85)
        Image.fromarray(artifact_overlay).save(outdir / f"{args.sample_id}_grandqc_artifact_overlay.jpg", quality=90)
        del background_for_overlay, qc_mask_preview, qc_color_preview, overlay, artifact_preview, artifact_color_preview, artifact_overlay
        import gc
        gc.collect()
        print("[INFO] GrandQC writing TIFF masks", flush=True)
        save_mask_tiff(outdir / f"{args.sample_id}_grandqc_tissue_detection_mask.tif", tissue_binary, tifffile_mod)
        save_mask_tiff(outdir / f"{args.sample_id}_grandqc_artifact_mask.tif", artifact_binary, tifffile_mod)
        save_mask_tiff(outdir / f"{args.sample_id}_grandqc_clean_tissue_mask.tif", clean_tissue, tifffile_mod)

        if args.create_geojson:
            scale_factor = float(artifact_mpp_model / source_mpp)
            mask_to_geojson(full_mask, outdir / f"{args.sample_id}_grandqc.geojson", scale_factor, cv2_mod)
            meta["geojson_scale_factor"] = scale_factor

        unique, counts = np_mod.unique(full_mask, return_counts=True)
        meta.update({
            "effective_tissue_mpp": tissue_meta["effective_tissue_mpp"],
            "tissue_mask_source": tissue_meta["tissue_mask_source"],
            "predicted_tissue_fraction_raw": tissue_meta["predicted_tissue_fraction"],
            "fallback_tissue_fraction": tissue_meta["fallback_tissue_fraction"],
            "artifact_patch_span_level0_px": int(artifact_patch_span),
            "qc_mask_shape_yx": [int(full_mask.shape[0]), int(full_mask.shape[1])],
            "qc_preview_shape_yx": [int(preview_h), int(preview_w)],
            "tissue_mask_shape_yx": [int(tissue_mask_thumb.shape[0]), int(tissue_mask_thumb.shape[1])],
            "patch_grid_yx": artifact_meta["patch_grid_yx"],
            "processed_level0_shape_yx": artifact_meta["processed_level0_shape_yx"],
            "processed_qc_shape_yx": artifact_meta["processed_qc_shape_yx"],
            "artifact_overlap_px": artifact_meta["overlap_px"],
            "artifact_overlap_fraction": artifact_meta["overlap_fraction"],
            "artifact_merge_mode": artifact_meta["merge_mode"],
            "small_fov_foreign_object_refinement": small_fov_refine_meta,
            "class_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
            "artifact_fraction": float(artifact_binary.astype(np_mod.float32).sum() / 255.0 / max(1, artifact_binary.size)),
            "clean_tissue_fraction": float(clean_tissue.astype(np_mod.float32).sum() / 255.0 / max(1, clean_tissue.size)),
            "tissue_detector_fraction": float(tissue_binary.astype(np_mod.float32).sum() / 255.0 / max(1, tissue_binary.size)),
            "models": {
                "tissue": str(tissue_ckpt.resolve()),
                "artifact": str(artifact_ckpt.resolve()),
            },
        })
        (outdir / f"{args.sample_id}_grandqc_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[OK] GrandQC wrote {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
