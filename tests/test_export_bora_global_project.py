"""Native Bora GeoJSON must never be relabelled as working-resolution pixels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from bin.bora_fullslide_action import (COLON_HE_ALGORITHM, GENERALIZED_ALGORITHM,
                                       sha256)
from bin.export_experiment import export_experiment, validate_experiment
from bin import export_bora_global_project as module


def case(tmp_path: Path, monkeypatch, identity_status="exploratory_transfer") -> dict:
    working_image = tmp_path / "working-0.5.tif"
    native_image = tmp_path / "native-0.25.tif"
    Image.new("RGB", (4, 4), "white").save(working_image)
    Image.new("RGB", (8, 8), "pink").save(native_image)
    working = {"frame_id": "working-frame", "sha256": sha256(working_image),
               "width": 4, "height": 4, "mpp_x": .5, "mpp_y": .5}
    native = {"frame_id": "native-frame", "sha256": sha256(native_image),
              "width": 8, "height": 8, "mpp_x": .25, "mpp_y": .25}
    base = tmp_path / "base.cellphenotyper.json"
    export_experiment({"name": "One tissue", "tissues": [{"name": "Tissue",
        "image": {"path": str(native_image), "mpp_x": .25, "mpp_y": .25,
                  "frame_id": native["frame_id"]},
        "artifacts": [], "feature_sets": []}]}, base, base_dir=tmp_path)
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"format": "cellphenotyper-dual-mpp-registration-v1",
                                        "verified": True, "image": working,
                                        "cell_full_image": native}))
    bora = tmp_path / "bora"
    (bora / "vectors/exact").mkdir(parents=True)
    (bora / "vectors/review").mkdir(parents=True)
    (bora / "semantic").mkdir(parents=True)
    native_fields = {"native_shape_yx": [8, 8], "native_mpp_xy": [.25, .25],
                     "source_frame_id": native["frame_id"],
                     "source_image_sha256": native["sha256"]}
    assembly = {**native_fields, "working_frame_id": working["frame_id"],
                "working_image_sha256": working["sha256"],
                "source_sha256": {"registration": sha256(registration)},
                "semantic_mask_sha256": "b" * 64,
                "global_model_binding_sha256": "c" * 64}
    vector = {**native_fields, "source_semantic_mask_sha256": "b" * 64,
              "global_model_binding_sha256": "c" * 64}
    geojson = {"type": "FeatureCollection", "coordinate_space": "native_image_pixel_edges",
               "origin": "top_left", **native_fields,
               "source_semantic_mask_sha256": "b" * 64,
               "global_model_binding_sha256": "c" * 64,
               "features": []}
    (bora / "semantic/ASSEMBLY_COMPLETE.json").write_text(json.dumps(assembly))
    (bora / "vectors/GLOBAL_VECTOR_COMPLETE.json").write_text(json.dumps(vector))
    for role in ("exact", "review"):
        (bora / f"vectors/{role}/mask.geojson").write_text(json.dumps(geojson, separators=(",", ":")))
    (bora / "CHAIN_COMPLETE.json").write_text('{"complete":true}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frame_id": working["frame_id"],
                                    "identity_status": identity_status,
                                    "expected": {"image_sha256": working["sha256"]},
                                    "inputs": {"registration": {"path": str(registration),
                                                                "sha256": sha256(registration)}}}))
    chain = {"manifest_sha256": sha256(manifest),
             "tissue_identity_status": identity_status,
             "evidence_class": "verified_gpu_inventory",
             "verified_algorithm_id": GENERALIZED_ALGORITHM,
             "algorithm_evidence_receipt_sha256": "a" * 64}
    monkeypatch.setattr(module, "verify", lambda *_: chain)
    return dict(base=base, bora=bora, manifest=manifest, working_image=working_image,
                native_image=native_image, working=working, native=native,
                registration=registration)


@pytest.mark.parametrize("identity_status", ["exploratory_transfer", "externally_validated"])
def test_project_export_attaches_native_boundary_to_matching_tissue(tmp_path: Path, monkeypatch,
                                                                    identity_status: str):
    item = case(tmp_path, monkeypatch, identity_status)
    output = tmp_path / "with-bora.cellphenotyper.json"
    receipt = module.export(item["base"], item["manifest"], item["bora"], output)
    assert receipt["complete"] is True
    project = validate_experiment(output)
    artifacts = project["tissues"][0]["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["role"] == "bora_exploratory_boundary_review"
    assert not any(artifact["role"] == "tissue_annotations" for artifact in artifacts)
    assert artifacts[0]["review_only"] is True
    assert artifacts[0]["label"] == "Bora anonymous-group boundaries (exploratory review; exact)"
    assert artifacts[0]["sha256"] == sha256(item["bora"] / "vectors/exact/mask.geojson")
    assert artifacts[0]["source_frame_id"] == item["native"]["frame_id"]
    assert artifacts[0]["image_sha256"] == item["native"]["sha256"]
    assert artifacts[0]["provenance"]["native_shape_yx"] == [8, 8]
    assert artifacts[0]["provenance"]["native_mpp_xy"] == [.25, .25]
    assert artifacts[0]["provenance"]["algorithm_id"] == GENERALIZED_ALGORITHM
    assert artifacts[0]["provenance"]["prediction_unit"] == "global_anonymous_Bora-AF_group"
    assert artifacts[0]["provenance"]["identity_status"] == identity_status
    assert artifacts[0]["provenance"]["chain_complete_sha256"] == sha256(item["bora"] / "CHAIN_COMPLETE.json")
    assert artifacts[0]["provenance"]["manifest_sha256"] == sha256(item["manifest"])
    assert receipt["algorithm_id"] == GENERALIZED_ALGORITHM
    assert receipt["artifact_role"] == "bora_exploratory_boundary_review"
    assert receipt["review_only"] is True


def test_project_export_rejects_working_resolution_project(tmp_path: Path, monkeypatch):
    item = case(tmp_path, monkeypatch)
    working_project = tmp_path / "working.cellphenotyper.json"
    export_experiment({"tissues": [{"name": "Tissue",
        "image": {"path": str(item["working_image"]), "mpp_x": .5, "mpp_y": .5,
                  "frame_id": item["working"]["frame_id"]},
        "artifacts": [], "feature_sets": []}]}, working_project, base_dir=tmp_path)
    with pytest.raises(ValueError, match="Bora native image/frame/shape/MPP"):
        module.export(working_project, item["manifest"], item["bora"], tmp_path / "result.json")


def test_project_export_canonicalizes_symlinked_output_and_source_paths(tmp_path: Path, monkeypatch):
    item = case(tmp_path, monkeypatch)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    destination = alias / "with-bora.cellphenotyper.json"
    receipt = module.export(alias / "base.cellphenotyper.json",
                            alias / "manifest.json", alias / "bora", destination)
    assert receipt["project"] == str(tmp_path / destination.name)
    assert (tmp_path / destination.name).is_file()
    project = validate_experiment(destination)
    image_path = (destination.parent / project["tissues"][0]["image"]["path"]).resolve()
    assert image_path == item["native_image"]
    exact_path = (destination.parent / project["tissues"][0]["artifacts"][0]["path"]).resolve()
    assert exact_path == item["bora"] / "vectors/exact/mask.geojson"
    with pytest.raises(ValueError, match="new project JSON path"):
        module.export(item["base"], item["manifest"], item["bora"], destination)


@pytest.mark.parametrize("relative,key,value", [
    ("semantic/ASSEMBLY_COMPLETE.json", "native_shape_yx", [4, 4]),
    ("vectors/GLOBAL_VECTOR_COMPLETE.json", "native_mpp_xy", [.5, .5]),
    ("vectors/exact/mask.geojson", "source_frame_id", "working-frame"),
    ("vectors/review/mask.geojson", "source_image_sha256", "0" * 64),
])
def test_project_export_rejects_mixed_or_tampered_native_metadata(
        tmp_path: Path, monkeypatch, relative: str, key: str, value):
    item = case(tmp_path, monkeypatch)
    path = item["bora"] / relative
    record = json.loads(path.read_text())
    record[key] = value
    path.write_text(json.dumps(record, separators=(",", ":")))
    with pytest.raises(ValueError, match="native frame, image, shape or MPP"):
        module.export(item["base"], item["manifest"], item["bora"], tmp_path / "result.json")


def test_project_export_rejects_changed_registration(tmp_path: Path, monkeypatch):
    item = case(tmp_path, monkeypatch)
    item["registration"].write_text(item["registration"].read_text() + "\n")
    with pytest.raises(ValueError, match="registration differs"):
        module.export(item["base"], item["manifest"], item["bora"], tmp_path / "result.json")


def test_project_export_rejects_colon_method_or_unidentified_result(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    base = tmp_path / "base.cellphenotyper.json"
    base.write_text("{}")
    bora = tmp_path / "bora"
    bora.mkdir()
    monkeypatch.setattr(module, "verify", lambda *args: {
        "verified_algorithm_id": COLON_HE_ALGORITHM,
        "algorithm_evidence_receipt_sha256": "a" * 64})
    with pytest.raises(ValueError, match="hard-group algorithm evidence"):
        module.export(base, manifest, bora, tmp_path / "result.json")


def test_project_export_rejects_identity_status_mismatch(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"identity_status": "exploratory_transfer"}))
    base = tmp_path / "base.cellphenotyper.json"
    base.write_text("{}")
    bora = tmp_path / "bora"
    bora.mkdir()
    monkeypatch.setattr(module, "verify", lambda *_: {
        "verified_algorithm_id": GENERALIZED_ALGORITHM,
        "algorithm_evidence_receipt_sha256": "a" * 64,
        "tissue_identity_status": "externally_validated"})
    with pytest.raises(ValueError, match="identity status differs"):
        module.export(base, manifest, bora, tmp_path / "result.json")
