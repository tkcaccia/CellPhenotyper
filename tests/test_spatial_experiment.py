"""CPU metadata contracts; RDS bytes stay opaque and are never deserialized."""
import builtins
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import export_experiment as exporter
from export_experiment import (ExperimentError, FILENAME, SPATIAL_BUNDLE_SCHEMA,
                               _namespace_id, export_experiment, sha256, validate_experiment)


def spatial_tissue(root, tissue_id="section-a", dataset_id="rna", experiment_id="spatial-study"):
    folder = root / tissue_id
    folder.mkdir(parents=True)
    image = folder / "image.png"
    Image.new("RGB", (64, 64), (190, 150, 170)).save(image)
    # The Python contract only validates references/hashes. A real RDS is tested
    # by the R integration suite; no Python dependency is added to decode it.
    original = folder / "original.rds"
    original.write_bytes(b"opaque original RDS fixture: preserve these bytes\n")
    original_sha, image_sha = sha256(original), sha256(image)
    frame = tissue_id + ":image"
    object_id, review_id = f"spatial:{dataset_id}:object", f"spatial:{dataset_id}:review"
    observations = [dict(id=_namespace_id(experiment_id, tissue_id, dataset_id, local), original_id=local,
                         dataset_id=dataset_id, tissue_id=tissue_id, frame_id=frame, observation_unit="spot",
                         x=8.12345678912345 + index * 8, y=8 + index * 8,
                         cluster_values=dict(rna_cluster="alpha" if index < 2 else "beta", optional=None,
                                             literal_missing="NA"),
                         footprint_width_px=4, footprint_height_px=4)
                    for index, local in enumerate(("BC 01", "\u03b2/02", "NA", "barcode%20literal"))]
    reduction = dict(id=_namespace_id(experiment_id, tissue_id, dataset_id, "KODAMA RNA"),
                     label="KODAMA RNA", dataset_id=dataset_id, frame_id=frame, observation_unit="spot",
                     points=[dict(id=row["id"], x=-i / 3, y=i / 7) for i, row in enumerate(reversed(observations))])
    bundle = dict(schema=SPATIAL_BUNDLE_SCHEMA, schema_version=1, experiment_id=experiment_id,
                  tissue_id=tissue_id, dataset_id=dataset_id, observation_unit="spot", frame_id=frame,
                  image_sha256=image_sha, source_artifact_id=object_id, source_object_sha256=original_sha,
                  coordinate_mapping=dict(confirmed=True, same_section=True, image_sha256=image_sha,
                                          frame_id=frame, coordinate_space="native_pixels",
                                          affine=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                  observations=observations, reductions=[reduction])
    bundle_path = folder / "review.json"
    write_bundle(bundle_path, bundle)
    common = dict(dataset_id=dataset_id, observation_unit="spot", frame_id=frame,
                  image_sha256=image_sha, row_count=len(observations))
    artifacts = [dict(common, artifact_id=object_id, role="spatial_transcriptomics_object", format="rds",
                      path=str(original), review_bundle_artifact_id=review_id),
                 dict(common, artifact_id=review_id, role="spatial_review_bundle", format="json",
                      path=str(bundle_path), source_artifact_id=object_id, source_object_sha256=original_sha)]
    tissue = dict(tissue_id=tissue_id, name=tissue_id, image=dict(path=str(image), frame_id=frame, mpp_x=.5, mpp_y=1),
                  artifacts=artifacts, feature_sets=[])
    return tissue, bundle, bundle_path


def write_bundle(path, bundle):
    path.write_text(json.dumps(bundle, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def spec_for(tissues):
    return dict(experiment_id="spatial-study", name="Same-section spatial study", tissues=tissues)


def test_linked_spatial_artifacts_reopen_relocate_and_keep_original_bytes(tmp_path, monkeypatch):
    root = tmp_path / "project-tree"
    tissues = [spatial_tissue(root, tissue_id)[0] for tissue_id in ("section-a", "section-b", "section-c")]
    source_bytes = [Path(tissue["artifacts"][0]["path"]).read_bytes() for tissue in tissues]
    original_import = builtins.__import__

    def no_deserialization_or_inference(name, *args, **kwargs):
        assert name.split(".")[0] not in {"pyreadr", "rpy2", "pickle", "torch", "timm", "nextflow"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_deserialization_or_inference)
    output = root / "index" / FILENAME
    payload = export_experiment(spec_for(tissues), output)
    assert validate_experiment(output) == payload
    assert all(Path(tissue["artifacts"][0]["path"]).read_bytes() == expected
               for tissue, expected in zip(tissues, source_bytes))
    assert all(not Path(artifact["path"]).is_absolute() for tissue in payload["tissues"] for artifact in tissue["artifacts"])
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    relocated = validate_experiment(moved / "index" / FILENAME)
    second = export_experiment(relocated, moved / "other-index" / FILENAME, base_dir=moved / "index")
    assert second["experiment_id"] == payload["experiment_id"]
    ids = []
    for tissue in second["tissues"]:
        artifact = tissue["artifacts"][1]
        bundle = json.loads((moved / "other-index" / artifact["path"]).read_text())
        ids.extend(row["id"] for row in bundle["observations"])
        assert bundle["observations"][0]["cluster_values"]["literal_missing"] == "NA"
        assert bundle["observations"][0]["cluster_values"]["optional"] is None
    assert len(ids) == len(set(ids)) == 12


def test_namespace_matches_existing_r_encoding_rules():
    assert _namespace_id("study", "t:1", "rna", "a b/\u03b2") == "study::t%3A1::rna::a%20b%2F%CE%B2"
    # Existing R URLencode(repeated=FALSE) preserves the entire component.
    assert _namespace_id("study", "t1", "rna", "a%20b/raw") == "study::t1::rna::a%20b/raw"


@pytest.mark.parametrize("failure", [
    "bad_forward_link", "bad_back_link", "missing_object", "missing_bundle", "wrong_artifact_id",
    "frame", "image_hash", "count", "boolean_count", "oversized_count", "unit", "format",
    "bundle_version", "bundle_source", "bundle_source_hash", "bundle_image_hash", "bundle_tissue",
    "unconfirmed", "cross_section", "mapping_frame", "unknown_space", "missing_affine", "singular",
    "native_transform", "projective", "nonfinite_affine", "nonfinite_extra", "outside", "boolean_x",
    "unscoped_id", "duplicate_id", "unknown_point", "duplicate_point", "missing_point", "reduction_id",
    "reduction_unit", "internal_path", "cluster_array", "invalid_footprint", "dataset_collision",
    "unknown_bundle_field", "unknown_observation_field", "unknown_mapping_field", "unknown_point_field",
])
def test_invalid_spatial_contract_preserves_previous_index(tmp_path, failure):
    tissue, bundle, bundle_path = spatial_tissue(tmp_path)
    output = tmp_path / FILENAME
    export_experiment(spec_for([tissue]), output)
    before = output.read_bytes()
    source, review = tissue["artifacts"]
    mapping = bundle["coordinate_mapping"]
    if failure == "bad_forward_link": source["review_bundle_artifact_id"] = "other"
    elif failure == "bad_back_link": review["source_artifact_id"] = "other"
    elif failure == "missing_object": tissue["artifacts"].remove(source)
    elif failure == "missing_bundle": tissue["artifacts"].remove(review)
    elif failure == "wrong_artifact_id": source["artifact_id"] = "wrong:object"
    elif failure == "frame": review["frame_id"] = "another-image"
    elif failure == "image_hash": source["image_sha256"] = "0" * 64
    elif failure == "count": source["row_count"] += 1
    elif failure == "boolean_count": source["row_count"] = True
    elif failure == "oversized_count": source["row_count"] = 10001
    elif failure == "unit": source["observation_unit"] = "tile"
    elif failure == "format": source["format"] = "json"
    elif failure == "bundle_version": bundle["schema_version"] = 2
    elif failure == "bundle_source": bundle["source_artifact_id"] = "other"
    elif failure == "bundle_source_hash": bundle["source_object_sha256"] = "0" * 64
    elif failure == "bundle_image_hash": bundle["image_sha256"] = "0" * 64
    elif failure == "bundle_tissue": bundle["tissue_id"] = "other"
    elif failure == "unconfirmed": mapping["confirmed"] = False
    elif failure == "cross_section": mapping["same_section"] = False
    elif failure == "mapping_frame": mapping["frame_id"] = "other"
    elif failure == "unknown_space": mapping["coordinate_space"] = {"unknown": True}
    elif failure == "missing_affine": mapping.pop("affine")
    elif failure == "singular": mapping["affine"][1] = [0, 0, 0]
    elif failure == "native_transform": mapping["affine"][0][2] = 5
    elif failure == "projective": mapping["affine"][2][0] = .1
    elif failure == "nonfinite_affine": mapping["affine"][0][0] = float("nan")
    elif failure == "nonfinite_extra": bundle["bad_metadata"] = float("inf")
    elif failure == "outside": bundle["observations"][0]["x"] = 64
    elif failure == "boolean_x": bundle["observations"][0]["x"] = True
    elif failure == "unscoped_id": bundle["observations"][0]["id"] = "BC 01"
    elif failure == "duplicate_id": bundle["observations"][1] = copy.deepcopy(bundle["observations"][0])
    elif failure == "unknown_point": bundle["reductions"][0]["points"][0]["id"] = "unknown"
    elif failure == "duplicate_point": bundle["reductions"][0]["points"][1] = copy.deepcopy(bundle["reductions"][0]["points"][0])
    elif failure == "missing_point": bundle["reductions"][0]["points"].pop()
    elif failure == "reduction_id": bundle["reductions"][0]["id"] = "spatial-study::section-a::rna::raw name"
    elif failure == "reduction_unit": bundle["reductions"][0]["observation_unit"] = "cell"
    elif failure == "internal_path": bundle["provenance"] = {"source_path": "/private/source.rds"}
    elif failure == "cluster_array": bundle["observations"][0]["cluster_values"]["rna_cluster"] = ["a", "b"]
    elif failure == "invalid_footprint": bundle["observations"][0]["footprint_width_px"] = -2
    elif failure == "unknown_bundle_field": bundle["unrecognized"] = "metadata"
    elif failure == "unknown_observation_field": bundle["observations"][0]["unrecognized"] = "metadata"
    elif failure == "unknown_mapping_field": mapping["guess_scale"] = 1
    elif failure == "unknown_point_field": bundle["reductions"][0]["points"][0]["tissue_id"] = "section-a"
    elif failure == "dataset_collision":
        table = tmp_path / "legacy.csv"
        table.write_text("id,x,y\n1,8,8\n")
        tissue["artifacts"].append(dict(artifact_id="legacy:review", role="review_observations", dataset_id="rna", path=str(table)))
    write_bundle(bundle_path, bundle)
    with pytest.raises(ExperimentError):
        export_experiment(spec_for([tissue]), output, overwrite=True)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(f".{FILENAME}.*.tmp"))


@pytest.mark.parametrize("asset", ["object", "review"])
def test_spatial_asset_tampering_is_detected_on_reopen(tmp_path, asset):
    tissue, _, bundle_path = spatial_tissue(tmp_path)
    output = tmp_path / FILENAME
    export_experiment(spec_for([tissue]), output)
    path = Path(tissue["artifacts"][0]["path"]) if asset == "object" else bundle_path
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ExperimentError, match="SHA-256 mismatch"):
        validate_experiment(output)


def test_reopen_checks_semantics_even_when_bundle_hash_is_updated(tmp_path):
    tissue, bundle, bundle_path = spatial_tissue(tmp_path)
    output = tmp_path / FILENAME
    payload = export_experiment(spec_for([tissue]), output)
    bundle["coordinate_mapping"]["same_section"] = False
    write_bundle(bundle_path, bundle)
    payload["tissues"][0]["artifacts"][1]["sha256"] = sha256(bundle_path)
    output.write_text(json.dumps(payload))
    with pytest.raises(ExperimentError, match="same_section"):
        validate_experiment(output)


def test_spatial_bundle_size_and_duplicate_json_keys_are_rejected(tmp_path):
    tissue, bundle, bundle_path = spatial_tissue(tmp_path)
    output = tmp_path / FILENAME
    bundle_path.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    with pytest.raises(ExperimentError, match="16 MiB"):
        export_experiment(spec_for([tissue]), output)
    assert not output.exists()
    text = json.dumps(bundle)
    bundle_path.write_text('{"schema_version":1,' + text[1:])
    with pytest.raises(ExperimentError, match="duplicate JSON key"):
        export_experiment(spec_for([tissue]), output)
    bundle_path.write_text('{"nested":' + '[' * 1200 + '0' + ']' * 1200 + '}')
    with pytest.raises(ExperimentError, match="Cannot read spatial review bundle"):
        export_experiment(spec_for([tissue]), output)


def test_spatial_source_affine_and_empty_clusters_are_explicitly_supported(tmp_path):
    tissue, bundle, bundle_path = spatial_tissue(tmp_path)
    mapping = bundle["coordinate_mapping"]
    mapping.update(coordinate_space="source_pixels", affine=[[.5, 0, -5], [0, .5, -7], [0, 0, 1]])
    # Normalized observation coordinates remain in the target native image.
    bundle["observations"][0]["cluster_values"] = []
    bundle["observations"][0]["footprint"] = dict(x0=-2, y0=6, x1=12, y1=10)
    bundle["observations"][0]["footprint_width_px"] = 14
    write_bundle(bundle_path, bundle)
    output = tmp_path / FILENAME
    export_experiment(spec_for([tissue]), output)
    assert validate_experiment(output)["tissues"][0]["artifacts"][0]["role"] == "spatial_transcriptomics_object"


def test_atomic_failure_keeps_index_and_both_source_assets_unchanged(tmp_path, monkeypatch):
    tissue, _, _ = spatial_tissue(tmp_path)
    output = tmp_path / FILENAME
    export_experiment(spec_for([tissue]), output)
    before = output.read_bytes()
    hashes = [sha256(Path(artifact["path"])) for artifact in tissue["artifacts"]]

    def interrupted_replace(*_args):
        raise OSError("simulated index publication failure")

    monkeypatch.setattr(exporter.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="publication failure"):
        export_experiment(spec_for([tissue]), output, overwrite=True)
    assert output.read_bytes() == before
    assert [sha256(Path(artifact["path"])) for artifact in tissue["artifacts"]] == hashes
    assert not list(tmp_path.glob(f".{FILENAME}.*.tmp"))


def test_directory_symlink_cannot_bypass_original_rds_protection(tmp_path):
    tissue, _, _ = spatial_tissue(tmp_path)
    source = Path(tissue["artifacts"][0]["path"])
    before = source.read_bytes()
    alias = tmp_path / "directory-alias"
    try:
        alias.symlink_to(source.parent, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this platform")
    with pytest.raises(ExperimentError, match="overwrite a source asset"):
        export_experiment(spec_for([tissue]), alias / source.name, overwrite=True)
    assert source.read_bytes() == before


def test_unrelated_generic_artifact_roles_remain_compatible(tmp_path):
    tissue, _, _ = spatial_tissue(tmp_path)
    attachment = tmp_path / "notes.txt"
    attachment.write_text("Unchanged generic display artifact\n")
    tissue["artifacts"] = [dict(role="custom_display_note", path=str(attachment), custom_metadata={"version": 9})]
    output = tmp_path / FILENAME
    payload = export_experiment(spec_for([tissue]), output)
    assert validate_experiment(output)["tissues"][0]["artifacts"][0]["custom_metadata"] == {"version": 9}
    assert payload["schema_version"] == 1
