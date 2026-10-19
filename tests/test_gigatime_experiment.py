"""Small real TIFF/CSV interchange fixtures; no GigaTIME model is executed."""
import builtins
import copy
import csv
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from export_experiment import (ExperimentError, FILENAME, GIGATIME_SCORE_SEMANTICS,
                               export_experiment, sha256, validate_experiment)


def write_scores(path, rows):
    with (gzip.open if path.name.endswith(".gz") else open)(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gigatime_tissue(root, tid="section-a", *, compressed=False, native=False):
    folder = root / tid
    folder.mkdir(parents=True)
    he = folder / "image.png"
    Image.new("RGB", (64, 48), (180, 140, 170)).save(he)
    image_sha = sha256(he)
    frame = tid + ":image"
    marker_image = folder / "virtual_markers.ome.tif"
    planes = np.stack([np.full((24, 32), 12000, dtype=np.uint16), np.full((24, 32), 24000, dtype=np.uint16)])
    tifffile.imwrite(marker_image, planes, ome=True, photometric="minisblack",
                     metadata={"axes": "CYX", "Channel": {"Name": ["CD3", "CD8"]}})
    mask_path = folder / "nuclei.tif"
    labels = np.zeros((48, 64), dtype=np.uint32)
    labels[19:21, 15:17] = 7
    labels[31:33, 39:41] = 100000001
    tifffile.imwrite(mask_path, labels)
    table = folder / ("nuclei_scores.csv.gz" if compressed else "nuclei_scores.csv")
    # Values above one are intentional native exports, not probabilities.
    rows = [dict(label_id="100000001", mask_name="nuclei", centroid_x_px=40 if native else 19.5,
                 centroid_y_px=32 if native else 15.5, area_px=4, CD3__mean=200, CD8__sum=1024),
            dict(label_id="7", mask_name="nuclei", centroid_x_px=16 if native else 7.5,
                 centroid_y_px=20 if native else 9.5, area_px=4, CD3__mean=.25, CD8__sum=255)]
    write_scores(table, rows)
    common = dict(frame_id=frame, image_sha256=image_sha)
    marker = dict(common, artifact_id="gigatime:markers", role="gigatime_marker_image", format="ome-tiff",
                  path=str(marker_image), score_semantics=GIGATIME_SCORE_SEMANTICS,
                  channels=[dict(marker="CD3", page=0), dict(marker="CD8", page=1)],
                  extent=dict(x=0, y=0, width=64, height=48))
    mask = dict(common, artifact_id="segmentation:nuclei", role="cell_segmentation_mask", format="tiff",
                path=str(mask_path), mask_name="nuclei", segmentation_unit="nucleus")
    scores = dict(common, artifact_id="gigatime:nuclei", role="gigatime_cell_scores", format="csv.gz" if compressed else "csv",
                  path=str(table), id_column="label_id", x_column="centroid_x_px", y_column="centroid_y_px",
                  marker_columns=[dict(marker="CD3", column="CD3__mean"), dict(marker="CD8", column="CD8__sum")],
                  mask_name="nuclei", mask_artifact_id=mask["artifact_id"], source_artifact_id=marker["artifact_id"],
                  source_to_image_affine=[[1, 0, 0], [0, 1, 0], [0, 0, 1]] if native else [[2, 0, 1], [0, 2, 1], [0, 0, 1]],
                  source_coordinate_convention="native-image-pixel-centres" if native else "zero-based-pixel-index",
                  score_semantics=GIGATIME_SCORE_SEMANTICS, value_encoding="native_export_values")
    tissue = dict(tissue_id=tid, name=tid, image=dict(path=str(he), frame_id=frame, mpp_x=.5, mpp_y=.5),
                  artifacts=[marker, mask, scores], feature_sets=[])
    return tissue, rows, table


def study(tissues):
    return dict(experiment_id="virtual-marker-study", name="SYNTHETIC virtual marker interchange",
                provenance=dict(synthetic=True, inference_started=False), tissues=tissues)


def test_marker_images_and_shuffled_cells_reopen_without_virchow_features(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    first, _, first_table = gigatime_tissue(root)
    second, _, _ = gigatime_tissue(root, "section-b", compressed=True, native=True)
    source_hashes = {artifact["path"]: sha256(Path(artifact["path"])) for tissue in (first, second) for artifact in tissue["artifacts"]}
    original_import = builtins.__import__

    def no_inference(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "timm", "nextflow", "rpy2"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_inference)
    output = root / "index" / FILENAME
    payload = export_experiment(study([first, second]), output)
    assert validate_experiment(output) == payload
    for tissue in payload["tissues"]:
        assert tissue["feature_sets"] == []
        assert tissue["artifacts"][1]["width"] == 64
        assert tissue["artifacts"][1]["height"] == 48
        assert tissue["artifacts"][2]["row_count"] == 2
    with first_table.open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    assert [row["label_id"] for row in cells] == ["100000001", "7"]
    assert float(cells[0]["CD3__mean"]) == 200
    assert float(cells[0]["CD8__sum"]) == 1024
    assert all(sha256(Path(path)) == digest for path, digest in source_hashes.items())
    moved = tmp_path / "relocated"
    shutil.copytree(root, moved)
    assert validate_experiment(moved / "index" / FILENAME)["experiment_id"] == payload["experiment_id"]


@pytest.mark.parametrize("failure", [
    "channel_name", "page", "duplicate_page", "extent", "image_hash", "frame", "image_semantics",
    "mask_unit", "mask_dimensions", "float_mask", "multiple_mask_planes", "score_source", "score_mask",
    "mask_name", "csv_mask_name", "blank_mask_name", "duplicate_id", "zero_id", "decimal_id", "id_overflow",
    "raw_bounds", "nan_coordinate", "missing_half_pixel", "rotated_affine", "native_nonidentity",
    "negative_score", "nan_score", "missing_column", "unknown_marker", "coordinate_as_marker",
    "count", "semantics", "encoding", "output_convention",
])
def test_invalid_marker_contract_preserves_existing_index_and_assets(tmp_path, failure):
    tissue, rows, table = gigatime_tissue(tmp_path)
    output = tmp_path / FILENAME
    export_experiment(study([tissue]), output)
    before = output.read_bytes()
    marker, mask, scores = tissue["artifacts"]
    if failure == "channel_name": marker["channels"][0]["marker"] = "wrong"
    elif failure == "page": marker["channels"][0]["page"] = 10
    elif failure == "duplicate_page": marker["channels"][1]["page"] = 0
    elif failure == "extent": marker["extent"]["width"] = 65
    elif failure == "image_hash": marker["image_sha256"] = "0" * 64
    elif failure == "frame": scores["frame_id"] = "other-image"
    elif failure == "image_semantics": marker["score_semantics"] = "measured_protein"
    elif failure == "mask_unit": mask["segmentation_unit"] = "grid"
    elif failure == "mask_dimensions": mask["width"] = 32
    elif failure == "float_mask": tifffile.imwrite(mask["path"], np.zeros((48, 64), dtype=np.float32))
    elif failure == "multiple_mask_planes": tifffile.imwrite(mask["path"], np.zeros((2, 48, 64), dtype=np.uint16), photometric="minisblack")
    elif failure == "score_source": scores["source_artifact_id"] = mask["artifact_id"]
    elif failure == "score_mask": scores["mask_artifact_id"] = marker["artifact_id"]
    elif failure == "mask_name": scores["mask_name"] = "cyto"
    elif failure == "csv_mask_name": rows[0]["mask_name"] = "cyto"
    elif failure == "blank_mask_name": rows[0]["mask_name"] = ""
    elif failure == "duplicate_id": rows[1]["label_id"] = rows[0]["label_id"]
    elif failure == "zero_id": rows[0]["label_id"] = "0"
    elif failure == "decimal_id": rows[0]["label_id"] = "7.0"
    elif failure == "id_overflow": rows[0]["label_id"] = str(2 ** 32)
    elif failure == "raw_bounds": rows[0]["centroid_x_px"] = 32
    elif failure == "nan_coordinate": rows[0]["centroid_x_px"] = "nan"
    elif failure == "missing_half_pixel": scores["source_to_image_affine"][0][2] = 0
    elif failure == "rotated_affine": scores["source_to_image_affine"][0][1] = 1
    elif failure == "native_nonidentity": scores["source_coordinate_convention"] = "native-image-pixel-centres"
    elif failure == "negative_score": rows[0]["CD3__mean"] = -1
    elif failure == "nan_score": rows[0]["CD3__mean"] = "nan"
    elif failure == "missing_column": scores["marker_columns"][0]["column"] = "missing"
    elif failure == "unknown_marker": scores["marker_columns"][0]["marker"] = "not-in-image"
    elif failure == "coordinate_as_marker": scores["marker_columns"][0]["column"] = scores["x_column"]
    elif failure == "count": scores["row_count"] = 3
    elif failure == "semantics": scores["score_semantics"] = "calibrated_probability"
    elif failure == "encoding": scores["value_encoding"] = "rescale_to_probability"
    elif failure == "output_convention": scores["coordinate_convention"] = "microns"
    write_scores(table, rows)
    assets_before = {artifact["path"]: sha256(Path(artifact["path"])) for artifact in tissue["artifacts"]}
    with pytest.raises(ExperimentError):
        export_experiment(study([tissue]), output, overwrite=True)
    assert output.read_bytes() == before
    assert all(sha256(Path(path)) == digest for path, digest in assets_before.items())
    assert not list(tmp_path.glob(f".{FILENAME}.*.tmp"))


def test_mif_only_image_is_registered_without_a_cell_table_or_segmentation(tmp_path):
    tissue, _, _ = gigatime_tissue(tmp_path)
    tissue["artifacts"] = tissue["artifacts"][:1]
    output = tmp_path / FILENAME
    export_experiment(study([tissue]), output)
    assert len(validate_experiment(output)["tissues"][0]["artifacts"]) == 1


@pytest.mark.parametrize("asset_index", [0, 1, 2])
def test_marker_asset_tampering_is_detected_on_reopen(tmp_path, asset_index):
    tissue, _, _ = gigatime_tissue(tmp_path)
    output = tmp_path / FILENAME
    export_experiment(study([tissue]), output)
    path = Path(tissue["artifacts"][asset_index]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ExperimentError, match="SHA-256 mismatch"):
        validate_experiment(output)


def test_reopen_checks_marker_semantics_after_manifest_edit(tmp_path):
    tissue, _, _ = gigatime_tissue(tmp_path)
    output = tmp_path / FILENAME
    payload = export_experiment(study([tissue]), output)
    payload["tissues"][0]["artifacts"][2]["source_to_image_affine"][0][2] = 0
    output.write_text(json.dumps(payload))
    with pytest.raises(ExperimentError, match="half-pixel"):
        validate_experiment(output)


def test_float_marker_display_range_is_explicit_and_never_changes_scores(tmp_path):
    tissue, rows, table = gigatime_tissue(tmp_path)
    marker = tissue["artifacts"][0]
    tifffile.imwrite(marker["path"], np.full((2, 24, 32), .4, dtype=np.float32), ome=True,
                     photometric="minisblack", metadata={"axes": "CYX", "Channel": {"Name": ["CD3", "CD8"]}})
    output = tmp_path / FILENAME
    before = table.read_bytes()
    with pytest.raises(ExperimentError, match="display_max"):
        export_experiment(study([tissue]), output)
    for channel in marker["channels"]:
        channel["display_max"] = 1
    export_experiment(study([tissue]), output)
    assert validate_experiment(output)["tissues"][0]["artifacts"][0]["channels"][0]["display_max"] == 1
    assert table.read_bytes() == before
