"""Small native-image contract test for the explicit contextual Bora route."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from bora_contextual_fullslide import (  # noqa: E402
    BOUNDARY_SPEC, _asset_references, _export_project, _manifest,
    _source_identity, _validate_project, run, sha256,
)
from combine_cellvit_bora_project import combine  # noqa: E402
import combine_cellvit_bora_project as combined_export  # noqa: E402

pytestmark = pytest.mark.usefixtures("isolated_bora_source_imports")


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
        str(Path(__file__).resolve().parents[2] / "Bora")))
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


def test_bora_export_preserves_indexed_cells_and_registered_tissue(tmp_path: Path):
    manifest, image = _case(tmp_path / "source")
    base = tmp_path / "source/experiment.cellphenotyper.json"
    project = json.loads(base.read_text())
    tissue = project["tissues"][0]
    indexed = base.parent / "indexed"
    indexed.mkdir()
    source = indexed / "cellvit_partition_manifest.json"
    pointer = indexed / "cellvit_cells.features.json"
    index = indexed / "cellvit_cells.sqlite"
    _json(source, {"schema": "cellphenotyper-cellvit-shards-v1", "cell_count": 2})
    project["provenance"]["cellvit_partitions"] = {
        "path": "indexed/cellvit_partition_manifest.json", "sha256": sha256(source)}
    _json(pointer, {"schema": "cellphenotyper-partitioned-cell-features-v2",
                    "row_count": 2})
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        for key, value in {
            "schema": "cellphenotyper-cell-index", "schema_version": 3,
            "cell_count": 2, "image": tissue["image"],
            "classification": {"sha256": sha256(source)},
        }.items():
            connection.execute("INSERT INTO metadata VALUES(?,?)",
                               (key, json.dumps(value)))
    tissue["cell_classification"] = {
        "path": "indexed/cellvit_partition_manifest.json", "sha256": sha256(source),
        "feature_manifest": {"path": "indexed/cellvit_partition_manifest.json",
                             "sha256": sha256(source)},
        "index": {"path": "indexed/cellvit_cells.sqlite", "sha256": sha256(index),
                  "format": "sqlite"},
        "cell_count": 2}
    tissue["feature_sets"].append({
        "kind": "cellvit-cell", "feature_set_id": "cellvit-vectors",
        "path": "indexed/cellvit_cells.features.json", "sha256": sha256(pointer)})
    sidecar = base.parent / "registered/registration.json"
    sidecar.parent.mkdir()
    _json(sidecar, {"format": "cellphenotyper-dual-mpp-registration-v1"})
    tissue["registered_tissue"] = {
        "registration": {"path": "registered/registration.json", "sha256": sha256(sidecar)},
        "image": {"path": "image.tif", "sha256": sha256(image)},
        "assets": [{"path": "registered/registration.json", "sha256": sha256(sidecar),
                    "role": "kodama_map"}]}
    _json(base, project)
    input_manifest = json.loads(manifest.read_text())
    input_manifest["inputs"]["base_project"]["sha256"] = sha256(base)
    _json(manifest, input_manifest)
    assert _manifest(manifest)[2]["cell_classification"]["cell_count"] == 2
    geojson = tmp_path / "bora.geojson"
    receipt = tmp_path / "BORA_COMPLETE.json"
    _json(geojson, {"type": "FeatureCollection", "features": []})
    _json(receipt, {"status": "COMPLETE"})
    real_output = tmp_path / "separate/output"
    real_output.mkdir(parents=True)
    alias = tmp_path / "output-alias"
    alias.symlink_to(real_output, target_is_directory=True)
    destination = alias / "experiment.cellphenotyper.json"
    original_files = {ref["sha256"]: (base.parent / ref["path"]).resolve()
                      for ref in _asset_references(tissue)}
    _export_project(base, tissue, geojson, receipt, destination)
    combined = _validate_project(destination)
    out_tissue = combined["tissues"][0]
    assert out_tissue["cell_classification"]["cell_count"] == 2
    assert out_tissue["feature_sets"][0]["kind"] == "cellvit-cell"
    assert out_tissue["registered_tissue"]["assets"][0]["role"] == "kodama_map"
    assert (destination.parent / combined["provenance"]["cellvit_partitions"]["path"]).resolve() == source
    for ref in _asset_references(out_tissue):
        if ref["sha256"] in original_files:
            assert (destination.parent / ref["path"]).resolve() == original_files[ref["sha256"]]
    assert out_tissue["artifacts"][-1]["role"] == "tissue_annotations"
    assert out_tissue["artifacts"][-1]["review_only"] is False


def test_bora_export_rejects_broken_index_reference(tmp_path: Path):
    manifest, _ = _case(tmp_path / "source")
    base = tmp_path / "source/experiment.cellphenotyper.json"
    project = json.loads(base.read_text())
    project["tissues"][0]["cell_classification"] = {
        "path": "missing.json", "sha256": "0" * 64,
        "index": {"path": "missing.sqlite", "sha256": "0" * 64,
                  "format": "sqlite"}}
    _json(base, project)
    input_manifest = json.loads(manifest.read_text())
    input_manifest["inputs"]["base_project"]["sha256"] = sha256(base)
    _json(manifest, input_manifest)
    with pytest.raises(FileNotFoundError):
        _manifest(manifest)


def _complete_cellvit_merge_case(root: Path) -> tuple[Path, Path, Path, Path]:
    _case(root)
    base = root / "experiment.cellphenotyper.json"
    project = json.loads(base.read_text())
    tissue = project["tissues"][0]
    source = root / "cellvit_partition_manifest.json"
    plan_sha = "a" * 64
    _json(source, {"schema": "cellphenotyper-cellvit-shards-v1",
        "image": tissue["image"], "plan_sha256": plan_sha,
        "partition_count": 104, "partitions": [{"id": f"r{i:04d}"} for i in range(104)],
        "cell_count": 0})
    source_sha = sha256(source)
    pointer = root / "cellvit_cells.features.json"
    _json(pointer, {"schema": "cellphenotyper-partitioned-cell-features-v2",
                    "row_count": 0})
    feature = {"kind": "cellvit-cell", "feature_set_id": "cellvit-1280",
               "path": pointer.name, "sha256": sha256(pointer)}
    tissue["feature_sets"].append(feature)
    index = root / "cellvit_cells.sqlite"
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        metadata = {"schema": "cellphenotyper-cell-index", "schema_version": 3,
                    "cell_count": 0, "image": tissue["image"],
                    "classification": {"sha256": source_sha},
                    "feature_set": {"kind": "cellvit-cell",
                                    "feature_columns": [f"f{i}" for i in range(1280)],
                                    "descriptor": {"feature_count": 1280}},
                    "models": [{"taxonomy": name, "status": "available"}
                               for name in ("pannuke", "consep", "lizard", "nucls_main",
                                            "nucls_super", "ocelot", "midog", "panoptils")]}
        for key, value in metadata.items():
            connection.execute("INSERT INTO metadata VALUES(?,?)", (key, json.dumps(value)))
    tissue["cell_classification"] = {"path": source.name, "sha256": source_sha,
        "feature_manifest": {"path": source.name, "sha256": source_sha},
        "index": {"path": index.name, "sha256": sha256(index), "format": "sqlite"},
        "cell_count": 0}
    project["provenance"]["cellvit_partitions"] = {
        "path": source.name, "sha256": source_sha}
    _json(base, project)
    cell_receipt = root / "CELLVIT_PROJECT_COMPLETE.json"
    _json(cell_receipt, {"schema": "cellphenotyper-cellvit-fullslide-complete-v2",
        "planned_shards": 104, "complete_shards": 104, "shard_count": 104,
        "cell_count": 0, "plan_sha256": plan_sha,
        "source_image_sha256": tissue["image"]["sha256"],
        "partition_manifest_sha256": source_sha,
        "index_sha256": sha256(index), "project_sha256": sha256(base),
        "every_core_verified": True, "feature_vectors_retained": True,
        "requested_heads": ["pannuke", "consep", "lizard", "nucls_main",
                            "nucls_super", "ocelot", "midog", "panoptils"]})
    geo = root / "BORA_TISSUE_ANNOTATIONS.geojson"
    _json(geo, {"type": "FeatureCollection",
        "source_image_sha256": tissue["image"]["sha256"],
        "source_frame_id": tissue["image"]["frame_id"],
        "native_shape_yx": [1024, 1024], "native_mpp_xy": [.25, .25],
        "predicted_not_expert_annotation": True,
        "vector_export": {"global_contours": True,
                          "minimum_part_and_hole_area_um2": 1000},
        "features": [{"type": "Feature", "geometry": {"type": "Polygon",
                      "coordinates": [[[0, 0], [200, 0], [200, 200], [0, 200], [0, 0]]]},
                      "properties": {"name": "KODAMA Group 1", "area_um2": 2500,
                                     "predicted_not_expert_annotation": True}}]})
    bora_receipt = root / "BORA_ACCEPTED_COMPLETE.json"
    _json(bora_receipt, {"format": "bora-wsi-review-smoothing-v1",
        "status": "PREDICTED_TISSUE_ANNOTATION",
        "review_geojson_sha256": sha256(geo),
        "simplify_outer_boundary": True,
        "biological_accuracy_qualified": False})
    return base, cell_receipt, geo, bora_receipt


def test_accepted_bora_combines_only_complete_104_shards(tmp_path: Path):
    base, cell_receipt, geo, bora_receipt = _complete_cellvit_merge_case(tmp_path / "source")
    output = tmp_path / "combined/experiment.cellphenotyper.json"
    result = combine(base, cell_receipt, geo, bora_receipt, output,
                     expected_bora_sha256=sha256(geo), expected_shards=104)
    assert result["cell_count"] == 0
    assert result["bora_components"] == 1
    merged = _validate_project(output)
    tissue = merged["tissues"][0]
    assert tissue["cell_classification"]["cell_count"] == 0
    assert tissue["artifacts"][-1]["role"] == "tissue_annotations"
    assert tissue["artifacts"][-1]["review_only"] is False
    assert tissue["artifacts"][-1]["training_eligible"] is False
    assert sha256(output.parent / tissue["artifacts"][-1]["path"]) == sha256(geo)
    assert (output.parent / "CELLVIT_PROJECT_COMPLETE.json").is_file()
    assert (output.parent / merged["provenance"]["cellvit_fullslide_complete"]["path"]).resolve() == \
        (output.parent / "CELLVIT_PROJECT_COMPLETE.json").resolve()
    assert sha256(output.parent / "CELLVIT_PROJECT_COMPLETE.json") == sha256(cell_receipt)
    assert (output.parent / tissue["image"]["path"]).resolve() == \
        (base.parent / "image.tif").resolve()
    assert (output.parent / tissue["cell_classification"]["index"]["path"]).resolve() == \
        (base.parent / "cellvit_cells.sqlite").resolve()
    assert len(list(tmp_path.glob(".combined.staging-*"))) == 0


def test_accepted_bora_refuses_partial_2_of_104_cells(tmp_path: Path):
    base, cell_receipt, geo, bora_receipt = _complete_cellvit_merge_case(tmp_path / "source")
    incomplete = json.loads(cell_receipt.read_text())
    incomplete["complete_shards"] = 2
    _json(cell_receipt, incomplete)
    output = tmp_path / "combined/experiment.cellphenotyper.json"
    with pytest.raises(ValueError, match="coverage is incomplete"):
        combine(base, cell_receipt, geo, bora_receipt, output,
                expected_bora_sha256=sha256(geo), expected_shards=104)
    assert not output.exists()
    assert not output.parent.exists()


def test_combined_export_removes_staging_on_validation_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base, cell_receipt, geo, bora_receipt = _complete_cellvit_merge_case(tmp_path / "source")
    output = tmp_path / "combined/experiment.cellphenotyper.json"
    original = combined_export._validate_project
    calls = 0

    def fail_after_staging(path: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated final project validation failure")
        return original(path)

    monkeypatch.setattr(combined_export, "_validate_project", fail_after_staging)
    with pytest.raises(ValueError, match="simulated final project validation failure"):
        combine(base, cell_receipt, geo, bora_receipt, output,
                expected_bora_sha256=sha256(geo), expected_shards=104)
    assert calls == 2
    assert not output.parent.exists()
    assert not list(tmp_path.glob(".combined.staging-*"))


def test_another_imported_bora_is_rejected_before_output(tmp_path: Path, monkeypatch):
    """Test isolation must not relax the production source-provenance guard."""
    from types import ModuleType

    source = tmp_path / "pinned-bora"
    package = source / "src/bora"
    package.mkdir(parents=True)
    (source / "pyproject.toml").write_text("# synthetic source identity\n")
    (package / "__init__.py").write_text("# synthetic source identity\n")
    (package / "placeholder.py").write_text("# no inference is invoked\n")
    unrelated = tmp_path / "unrelated/bora/__init__.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("# deliberately different installation\n")
    imported = ModuleType("bora")
    imported.__file__ = str(unrelated)
    monkeypatch.setitem(sys.modules, "bora", imported)
    manifest, _ = _case(tmp_path / "case")
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="Another Bora installation is already imported"):
        run(manifest, source, _source_identity(source), output,
            disk_budget_bytes=4 * 1024**3)
    assert not output.exists()
