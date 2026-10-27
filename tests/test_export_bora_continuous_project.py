"""Only source-bound native continuous geometry may enter an experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bin import export_bora_continuous_project as module
from bin.bora_fullslide_action import sha256
from bin.export_experiment import export_experiment, validate_experiment


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")))


def _case(root: Path, *, visible_geometry: bool = False,
          native_size: tuple[int, int] | None = None) -> dict:
    native_size = native_size or ((171, 127) if visible_geometry else (16, 12))
    working_size = tuple((value + 1) // 2 for value in native_size)
    width, height = native_size
    native_image = root / "native-0.25.tif"
    working_image = root / "working-0.5.tif"
    support_path = root / "registered-grandqc.tif"
    Image.new("RGB", native_size, "pink").save(native_image)
    Image.new("RGB", working_size, "pink").save(working_image)
    support_pixels = np.zeros((height, width), dtype=np.uint8)
    if visible_geometry:
        support_pixels[30:90, 30:90] = 255
    else:
        support_pixels[:] = 255
    Image.fromarray(support_pixels).save(support_path)
    native = {"frame_id": "native-frame", "sha256": sha256(native_image),
              "width": width, "height": height, "mpp_x": .25, "mpp_y": .25}
    working = {"frame_id": "working-frame", "sha256": sha256(working_image),
               "width": working_size[0], "height": working_size[1],
               "mpp_x": .5, "mpp_y": .5}
    base = root / "base.cellphenotyper.json"
    export_experiment({"name": "Synthetic dual-MPP tissue", "tissues": [{
        "name": "Native tissue", "image": {"path": str(native_image),
            "mpp_x": native["mpp_x"], "mpp_y": native["mpp_y"],
            "frame_id": native["frame_id"]}, "artifacts": [], "feature_sets": []
    }]}, base, base_dir=root)
    registration_path = root / "registration.json"
    _write_json(registration_path, {
        "format": "cellphenotyper-dual-mpp-registration-v1", "verified": True,
        "support_sampling": "absolute pixel centre, nearest, full-frame GrandQC mask",
        "image": working, "cell_full_image": native,
        "full_support_mask": {"width": width, "height": height,
                              "sha256": sha256(support_path)},
    })
    output = root / "bora-continuous"
    for relative in module.OUTPUT_PATHS.values():
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
    support_npy = output / module.OUTPUT_PATHS["native_support_npy"]
    np.save(support_npy, (support_pixels > 0).astype(np.uint8))
    mask = (support_pixels > 0).astype(np.uint8)
    (output / module.OUTPUT_PATHS["semantic_mask"]).write_bytes(mask.tobytes())
    (output / module.OUTPUT_PATHS["exact_mask"]).write_bytes(mask.tobytes())
    model_sha = "a" * 64
    source = {"native_shape_yx": [height, width], "native_mpp_xy": [.25, .25],
              "source_frame_id": native["frame_id"],
              "source_image_sha256": native["sha256"],
              "global_model_binding_sha256": model_sha}
    semantic_sha = sha256(output / module.OUTPUT_PATHS["semantic_mask"])
    exact_mask_sha = sha256(output / module.OUTPUT_PATHS["exact_mask"])
    assembly = {**source, "format": "bora-wsi-probability-assembly-v1",
                "class_ids": [1, 2], "tissue_support_sha256": sha256(support_npy),
                "semantic_mask_sha256": semantic_sha,
                "core_partition_complete": True, "window_count": 1,
                "independently_fitted_windows": False,
                "maximum_halo_total_variation": 0.0,
                "semantic_mask_dtype": "uint8",
                "biological_accuracy_qualified": False}
    _write_json(output / module.OUTPUT_PATHS["assembly_receipt"], assembly)
    polygon = [{"type": "Feature", "properties": {
        "class_id": 1, "classification": {"name": "anonymous group 1"},
        "component_id": "synthetic-reviewed-component",
        "boundary_role": "smoothed_review", "area_um2": 225.0},
        "geometry": {"type": "Polygon", "coordinates": [[
            [30, 30], [90, 30], [90, 90], [30, 90], [30, 30]]]}}] if visible_geometry else []
    for role, key, geometry_role in (
            ("exact", "exact_geojson", "native_exact_after_area_cleanup"),
            ("review", "review_geojson", "smoothed_review")):
        _write_json(output / module.OUTPUT_PATHS[key], {
            **source, "type": "FeatureCollection",
            "coordinate_space": "native_image_pixel_edges", "origin": "top_left",
            "source_semantic_mask_sha256": semantic_sha,
            "predicted_not_expert_annotation": True,
            "tissue_identity_status": "exploratory_transfer",
            "vector_export": {"format": "bora-wsi-global-vector-v1",
                "global_contours": True, "independently_fitted_windows": False,
                "minimum_part_and_hole_area_um2": 100.0,
                "geometry_role": geometry_role,
                "smoothing_tolerance_um": 0.0 if role == "exact" else .5},
            "features": [{**polygon[0], "properties": {
                **polygon[0]["properties"], "boundary_role": geometry_role}}
                ] if visible_geometry else []})
    vector_outputs = {
        "exact/mask.uint8.dat": exact_mask_sha,
        "exact/mask.geojson": sha256(output / module.OUTPUT_PATHS["exact_geojson"]),
        "review/mask.geojson": sha256(output / module.OUTPUT_PATHS["review_geojson"]),
    }
    vectors = {**source, "format": "bora-wsi-global-vector-v1",
               "source_semantic_mask_sha256": semantic_sha,
               "exact_mask_sha256": exact_mask_sha,
               "tissue_identity_status": "exploratory_transfer",
               "minimum_area_um2": 100.0,
               "minimum_area_strictly_less_removed": True,
               "component_ids_preserved_between_exact_and_review": True,
               "exact_vector_matches_cleaned_mask": True,
               "global_contours": True,
               "independently_fitted_vector_windows": False,
               "no_macroregion_cuts": True,
               "biological_accuracy_qualified": False,
               "selected_smoothing_tolerance_um": .5,
               "output_sha256": vector_outputs}
    _write_json(output / module.OUTPUT_PATHS["vector_receipt"], vectors)
    hashes = {key: sha256(output / relative)
              for key, relative in module.OUTPUT_PATHS.items()}
    chain = {**source, "format": module.CHAIN_FORMAT,
             "review_only": True,
             "identity_status": "anonymous_exploratory_review",
             "tissue_identity_status": "exploratory_transfer",
             "biological_accuracy_qualified": False,
             "registration_sha256": sha256(registration_path),
             "registered_support_source_sha256": sha256(support_path),
             "support_conversion":
                "registered-native-GrandQC-TIFF/integer-binary-to-uint8-NPY/v1",
             "minimum_area_um2": 100.0,
             "exact_vector_matches_cleaned_mask": True,
             "global_contours": True, "no_macroregion_cuts": True,
             "independently_fitted_windows": False,
             "class_ids": [1, 2],
             "class_names": {"1": "anonymous group 1", "2": "anonymous group 2"},
             "shard_count": 1, "shard_receipt_sha256": ["b" * 64],
             "output_paths": module.OUTPUT_PATHS,
             "output_sha256": hashes,
             "tissue_support_sha256": hashes["native_support_npy"],
             "assembly_receipt_sha256": hashes["assembly_receipt"],
             "semantic_mask_sha256": hashes["semantic_mask"],
             "vector_receipt_sha256": hashes["vector_receipt"],
             "exact_geojson_sha256": hashes["exact_geojson"],
             "review_geojson_sha256": hashes["review_geojson"],
             "exact_mask_sha256": hashes["exact_mask"]}
    _write_json(output / "CONTINUOUS_CHAIN_COMPLETE.json", chain)
    return {"base": base, "native": native, "working": working,
            "native_image": native_image, "working_image": working_image,
            "registration": registration_path, "support": support_path,
            "output": output, "chain": chain}


def _export(case: dict, destination: Path) -> dict:
    return module.export(case["base"], case["registration"], case["support"],
                         case["output"], destination)


def test_native_continuous_export_is_review_only_and_has_exact_qa_provenance(tmp_path: Path):
    item = _case(tmp_path)
    destination = tmp_path / "with-bora-continuous.cellphenotyper.json"
    receipt = _export(item, destination)
    assert receipt["complete"] is True
    project = validate_experiment(destination)
    artifacts = project["tissues"][0]["artifacts"]
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["role"] == "bora_exploratory_boundary_review"
    assert artifact["review_only"] is True
    assert artifact["sha256"] == item["chain"]["review_geojson_sha256"]
    assert artifact["image_sha256"] == item["native"]["sha256"]
    assert artifact["provenance"]["producer"] == module.CHAIN_FORMAT
    assert artifact["provenance"]["prediction_unit"] == module.PREDICTION_UNIT
    assert artifact["provenance"]["identity_status"] == "anonymous_exploratory_review"
    assert artifact["provenance"]["exact_mask_sha256"] == item["chain"]["exact_mask_sha256"]
    assert artifact["provenance"]["semantic_mask_sha256"] == item["chain"]["semantic_mask_sha256"]
    assert artifact["provenance"]["review_geojson_sha256"] == artifact["sha256"]
    assert not any(row["role"] == "tissue_annotations" for row in artifacts)


def test_visible_boundary_survives_area_floor_but_remains_review_only(tmp_path: Path):
    item = _case(tmp_path, visible_geometry=True)
    destination = tmp_path / "visible-boundary.cellphenotyper.json"
    _export(item, destination)
    artifact = validate_experiment(destination)["tissues"][0]["artifacts"][0]
    assert artifact["review_only"] is True
    assert artifact["role"] == "bora_exploratory_boundary_review"
    source = (destination.parent / artifact["path"]).resolve()
    geometry = json.loads(source.read_text())["features"]
    assert len(geometry) == 1
    assert geometry[0]["properties"]["area_um2"] == 225.0
    assert geometry[0]["geometry"]["coordinates"][0][0] == [30, 30]


def test_rejects_working_resolution_project(tmp_path: Path):
    item = _case(tmp_path)
    working_project = tmp_path / "wrong-frame.cellphenotyper.json"
    export_experiment({"tissues": [{"name": "Wrong frame", "image": {
        "path": str(item["working_image"]), "mpp_x": .5, "mpp_y": .5,
        "frame_id": item["working"]["frame_id"]},
        "artifacts": [], "feature_sets": []}]}, working_project, base_dir=tmp_path)
    item["base"] = working_project
    with pytest.raises(ValueError, match="Bora native image/frame/shape/MPP"):
        _export(item, tmp_path / "rejected.json")


@pytest.mark.parametrize("relative, expected", [
    ("CONTINUOUS_CHAIN_COMPLETE.json", "completion lacks"),
    ("assembly/ASSEMBLY_COMPLETE.json", "output hash changed"),
    ("vectors/GLOBAL_VECTOR_COMPLETE.json", "output hash changed"),
    ("vectors/exact/mask.geojson", "output hash changed"),
    ("vectors/review/mask.geojson", "output hash changed"),
    ("vectors/exact/mask.uint8.dat", "output hash changed"),
    ("support.npy", "output hash changed"),
])
def test_rejects_tampered_chain_or_artifact(tmp_path: Path, relative: str, expected: str):
    item = _case(tmp_path)
    path = item["output"] / relative
    if relative.endswith("CHAIN_COMPLETE.json"):
        payload = json.loads(path.read_text())
        payload["minimum_area_um2"] = 0
        _write_json(path, payload)
    else:
        with path.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match=expected):
        _export(item, tmp_path / "rejected.json")


def test_rejects_changed_registration_or_registered_support(tmp_path: Path):
    item = _case(tmp_path)
    item["registration"].write_text(item["registration"].read_text() + "\n")
    with pytest.raises(ValueError, match="completion lacks"):
        _export(item, tmp_path / "rejected.json")
    # A separate fixture avoids overwriting existing source files.
    other = tmp_path / "other"
    other.mkdir()
    item = _case(other)
    with item["support"].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="support bytes"):
        _export(item, other / "rejected.json")


def test_rejects_relabelled_geojson_even_when_all_hashes_are_rewritten(tmp_path: Path):
    item = _case(tmp_path)
    review = item["output"] / module.OUTPUT_PATHS["review_geojson"]
    payload = json.loads(review.read_text())
    payload["source_frame_id"] = "working-frame"
    _write_json(review, payload)
    vector_path = item["output"] / module.OUTPUT_PATHS["vector_receipt"]
    vector = json.loads(vector_path.read_text())
    vector["output_sha256"]["review/mask.geojson"] = sha256(review)
    _write_json(vector_path, vector)
    chain_path = item["output"] / "CONTINUOUS_CHAIN_COMPLETE.json"
    chain = json.loads(chain_path.read_text())
    chain["output_sha256"]["review_geojson"] = sha256(review)
    chain["review_geojson_sha256"] = sha256(review)
    chain["output_sha256"]["vector_receipt"] = sha256(vector_path)
    chain["vector_receipt_sha256"] = sha256(vector_path)
    _write_json(chain_path, chain)
    with pytest.raises(ValueError, match="review GeoJSON native frame"):
        _export(item, tmp_path / "rejected.json")


def test_existing_destination_is_preserved(tmp_path: Path):
    item = _case(tmp_path)
    destination = tmp_path / "keep.json"
    destination.write_text("untouched")
    with pytest.raises(ValueError, match="new project JSON path"):
        _export(item, destination)
    assert destination.read_text() == "untouched"
