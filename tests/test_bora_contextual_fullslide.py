"""Small native-image contract test for the explicit contextual Bora route."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from bora_contextual_fullslide import (  # noqa: E402
    BOUNDARY_SPEC, _manifest, _source_identity, run, sha256,
)


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _ref(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path)}


def _case(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    image = root / "image.tif"
    rgb = np.zeros((3, 1024, 1024), np.uint8)
    rgb[0] = np.arange(1024, dtype=np.uint16)[None, :] // 4
    rgb[1] = 120
    rgb[2] = 175
    with rasterio.open(image, "w", driver="GTiff", width=1024, height=1024,
                       count=3, dtype="uint8", photometric="RGB") as dst:
        dst.write(rgb)
    field = np.zeros((37, 37, 32), np.float32)
    field[..., 0] = (np.linspace(-4, 4, 37, dtype=np.float32)[None, :] +
                     .03*np.sin(np.arange(37, dtype=np.float32)[:, None]*1.7))
    field[..., 1] = 1
    pca = root / "pca.npy"
    token_support = root / "token_support.npy"
    native_support = root / "native_support.npy"
    np.save(pca, field)
    projection = root / "pca_projection.json"
    _json(projection, {"method": "synthetic-fixed-pca", "components": 32})
    np.save(token_support, np.ones((37, 37), bool))
    np.save(native_support, np.ones((1024, 1024), bool))
    base = root / "experiment.cellphenotyper.json"
    _json(base, {"schema": "cellphenotyper-experiment",
        "schema_version": 1, "experiment_id": "synthetic-bora",
        "name": "Synthetic Bora", "settings": {}, "provenance": {},
        "tissues": [{"tissue_id": "slide-a", "name": "Slide A",
            "image": {"path": "image.tif", "sha256": sha256(image),
                "width": 1024, "height": 1024, "mpp_x": .25, "mpp_y": .25,
                "frame_id": "slide-a-frame"},
            "artifacts": [], "feature_sets": []}]})
    tiles = root / "kodama_labeled_tiles.csv"
    with tiles.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("tile_id", "native_x_px", "native_y_px", "kodama_group"))
        for group, x in ((1, 90), (2, 512), (3, 930)):
            for index in range(12):
                writer.writerow((f"g{group}-{index}", x, 20+index*80, group))
    kodama = root / "kodama_binding.json"
    _json(kodama, {"group_count": 3, "class_ids": [1, 2, 3],
        "source_image_sha256": sha256(image),
        "source_frame_id": "slide-a-frame",
        "pca_field_sha256": sha256(pca),
        "pca_projection_sha256": sha256(projection),
        "token_support_sha256": sha256(token_support),
        "labeled_tiles_sha256": sha256(tiles),
        "working_shape_yx": [512, 512]})
    manifest = root / "bora-contextual.json"
    _json(manifest, {"schema": "cellphenotyper-bora-contextual-fullslide-input-v1",
        "head_mode": "fit_from_kodama_tiles",
        "source_frame_id": "slide-a-frame",
        "pca_projection_sha256": sha256(projection),
        "native_shape_yx": [1024, 1024], "native_mpp_xy": [.25, .25],
        "working_shape_yx": [512, 512], "working_mpp_xy": [.5, .5],
        "analysis": {"context_sigma_tokens": 1.0,
            "prior_exponent": 0.0, "training_prior": [1/3]*3,
            "regularization_c": .1, "boundary_spec": BOUNDARY_SPEC,
            "core_pixels": 512, "halo_pixels": 64},
        "inputs": {"pca_field": _ref(pca),
            "pca_projection": _ref(projection),
            "token_support": _ref(token_support),
            "native_support": _ref(native_support),
            "native_he": _ref(image), "base_project": _ref(base),
            "kodama_binding": _ref(kodama),
            "labeled_tiles": _ref(tiles)}})
    return manifest, image


def test_same_slide_training_and_registered_inputs(tmp_path: Path):
    manifest, _ = _case(tmp_path / "case")
    parsed, files, tissue = _manifest(manifest)
    assert parsed["head_mode"] == "fit_from_kodama_tiles"
    assert tissue["tissue_id"] == "slide-a"
    assert len(files) == 8
    bad = json.loads(manifest.read_text())
    bad["native_shape_yx"] = [511, 512]
    _json(manifest, bad)
    with pytest.raises(ValueError, match="Native shape"):
        _manifest(manifest)


def test_head_cannot_transfer_to_other_slide(tmp_path: Path):
    manifest, _ = _case(tmp_path / "case")
    data = json.loads(manifest.read_text())
    model = tmp_path / "model"
    model.mkdir()
    head = model / "numeric_head.npz"
    np.savez(head, scaler_mean=np.zeros(32), scaler_scale=np.ones(32),
             logistic_coef=np.zeros((3, 32)),
             logistic_intercept=np.zeros(3),
             class_ids=np.array([1, 2, 3]))
    receipt = model / "FIT_COMPLETE.json"
    _json(receipt, {"format": "bora-kodama-three-group-head-v1",
        "status": "COMPLETE", "class_ids": [1, 2, 3],
        "model_sha256": sha256(head),
        "source_image_sha256": "a"*64,
        "source_frame_id": "slide-a-frame",
        "pca_projection_sha256": data["pca_projection_sha256"],
        "training_annotation_sha256": data["inputs"]["kodama_binding"]["sha256"]})
    data["head_mode"] = "pinned_same_slide_head"
    del data["inputs"]["labeled_tiles"]
    data["inputs"]["numeric_head"] = _ref(head)
    data["inputs"]["fit_receipt"] = _ref(receipt)
    _json(manifest, data)
    with pytest.raises(ValueError, match="not pinned to this same slide"):
        _manifest(manifest)


def test_full_small_slide_emits_exact_and_editable_predictions(tmp_path: Path):
    source = Path(os.environ.get("BORA_SOURCE_DIR",
        str(Path(__file__).resolve().parents[2] / "bora-main-20261025")))
    if not (source / "src/bora/kodama_bora.py").is_file():
        pytest.skip("Sibling Bora integration source is unavailable")
    manifest, _ = _case(tmp_path / "case")
    result = run(manifest, source, _source_identity(source),
                 tmp_path / "result", disk_budget_bytes=4*1024**3)
    output = tmp_path / "result"
    assert Path(result["annotation"]).is_file()
    assert (output / "contours/global_vectors/exact/mask.uint8.dat").is_file()
    assembly = json.loads((output / "assembly/ASSEMBLY_COMPLETE.json").read_text())
    assert assembly["format"] == "bora-wsi-label-assembly-v1"
    assert assembly["halo_mismatch_pixels"] == 0
    first_shard = np.load(next((output / "native_shards").glob("*.npy")))
    assert first_shard.dtype == np.uint8 and first_shard.ndim == 2
    fit = json.loads((output / "model/FIT_COMPLETE.json").read_text())
    assert fit["source_image_sha256"] == json.loads(manifest.read_text())["inputs"]["native_he"]["sha256"]
    assert (output / "RUN_COMPLETE.json").is_file()
    project = json.loads(Path(result["project"]).read_text())
    artifact = project["tissues"][0]["artifacts"][-1]
    assert artifact["role"] == "tissue_annotations"
    assert artifact["review_only"] is False
    assert artifact["predicted_not_expert_annotation"] is True
    assert artifact["training_eligible"] is False
    assert project["settings"]["bora_annotations_mode"] == "ordinary_editable"
