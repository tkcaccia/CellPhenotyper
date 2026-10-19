"""CPU-only producer contract tests: real CSVs/TIFF headers, no encoder calls."""
import builtins
import copy
import csv
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from export_experiment import (ExperimentError, FILENAME, discover_run, export_experiment,
                               export_runs, sha256, try_export_run, validate_experiment)


REVISION = "a" * 40
CHECKPOINT = "b" * 64


def cached_tissue(root, name="breast", *, kind="virchow2-pooled", compressed=True):
    folder = root / name
    folder.mkdir(parents=True)
    image = folder / "image.tif"
    tifffile.imwrite(image, np.full((450, 450, 3), 180, np.uint8), photometric="rgb")
    width = 2560 if kind == "virchow2-pooled" else 1280
    unit = "tile" if width == 2560 else "token"
    features = folder / ("features.csv.gz" if compressed else "features.csv")
    columns = [f"feat_{index}" for index in range(1, width + 1)]
    with (gzip.open if compressed else open)(features, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observation_id", "x", "y", *columns])
        for index, (x, y) in enumerate(((112, 112), (336, 112), (112, 336)), 1):
            writer.writerow([str(index), x, y, *[index * .1 + channel / 10000 for channel in range(width)]])
    return dict(name=name, image=dict(path=str(image), mpp_x=.5, mpp_y=.5), artifacts=[],
                feature_sets=[dict(kind=kind, observation_unit=unit, path=str(features),
                                   format="csv.gz" if compressed else "csv", feature_columns=columns,
                                   id_column="observation_id", x_column="x", y_column="y", coordinate_frame="image",
                                   encoder=dict(name="Virchow2", revision=REVISION, checkpoint_sha256=CHECKPOINT),
                                   preprocessing=dict(feature_definition="CLS+mean_patch" if width == 2560 else "spatial_patch_tokens"),
                                   grid=dict(tile_width_px=224, tile_height_px=224, spacing_x_px=224, spacing_y_px=224,
                                             origin_x_px=0, origin_y_px=0))])


def run_tree(root, sample="sample", *, offset=(41, 23)):
    """Construct the exact current producer layout without calling its encoder."""
    tissue = cached_tissue(root / "scratch", sample)
    image = root / "03_stardist" / sample / "prepared_crop" / "crop_roi.tif"
    image.parent.mkdir(parents=True)
    shutil.move(tissue["image"]["path"], image)
    (image.parent / "shift.json").write_text(json.dumps(dict(offset_crop_to_original=dict(dx=offset[0], dy=offset[1]),
                                                            full_size=dict(width=800, height=700))))
    grid_dir = root / "09_grid_tiles" / sample
    grid_dir.mkdir(parents=True)
    objects = grid_dir / f"{sample}_uni2_grid_objects.csv"
    objects.write_text("label,x,y\n1,112,112\n2,336,112\n3,112,336\n")
    metadata = grid_dir / f"{sample}_uni2_grid_metadata.json"
    metadata.write_text(json.dumps(dict(image="/old/cache/no-longer-mounted/image.tif", image_width_px=450,
                                        image_height_px=450, source_mpp_x=.5, source_mpp_y=.5,
                                        extraction_tile_size_source_px=224, grid_stride_source_px=224,
                                        tile_origin="top_left", tile_layout="nonoverlapping", context_overlap_source_px=0,
                                        retained_units=3, model_tile_size_px=224, target_mpp=.5)))
    embeddings_dir = root / "09_embeddings" / sample / f"embeddings_{sample}_tile"
    embeddings_dir.mkdir(parents=True)
    embedding = embeddings_dir / "virchow2_embeddings_shard0000.csv.gz"
    with gzip.open(Path(tissue["feature_sets"][0]["path"]), "rt") as source, gzip.open(embedding, "wt") as target:
        target.write(source.read().replace("observation_id,x,y,", "cell_id,cx,cy,", 1))
    marker = dict(encoder="Virchow2", encoder_revision=REVISION, checkpoint_sha256=CHECKPOINT,
                  grid_sha256=sha256(objects), grid_metadata_sha256=sha256(metadata),
                  embeddings_sha256=sha256(embedding), features_per_tile=2560, observations=3,
                  inner_square_used=False, tile_overlap_source_px=0)
    (embeddings_dir / "VIRCHOW2_COMPLETE.json").write_text(json.dumps(marker))
    return root, embedding


def test_multiple_tissues_relocate_with_stable_ids_and_local_row_ids(tmp_path):
    bundle = tmp_path / "bundle"
    a = cached_tissue(bundle, "breast")
    b = cached_tissue(bundle, "colon", kind="virchow2-spatial", compressed=False)
    output = bundle / "project" / FILENAME
    payload = export_experiment(dict(name="Two tissues", tissues=[a, b]), output)
    loaded = validate_experiment(output)
    assert loaded == payload
    assert len({t["tissue_id"] for t in payload["tissues"]}) == 2
    assert all(not Path(t["image"]["path"]).is_absolute() for t in payload["tissues"])
    assert [t["feature_sets"][0]["observation_unit"] for t in loaded["tissues"]] == ["tile", "token"]
    assert all(t["feature_sets"][0]["row_count"] == 3 for t in loaded["tissues"])
    relocated = tmp_path / "relocated"
    shutil.copytree(bundle, relocated)
    after_move = validate_experiment(relocated / "project" / FILENAME)
    assert after_move["experiment_id"] == payload["experiment_id"]
    relative_spec = copy.deepcopy(after_move)
    regenerated = export_experiment(relative_spec, relocated / "other" / FILENAME,
                                    base_dir=relocated / "project")
    assert regenerated["experiment_id"] == payload["experiment_id"]
    assert [t["tissue_id"] for t in regenerated["tissues"]] == [t["tissue_id"] for t in payload["tissues"]]


def test_existing_index_is_preserved_when_export_validation_fails(tmp_path):
    tissue = cached_tissue(tmp_path)
    output = tmp_path / FILENAME
    export_experiment(dict(tissues=[tissue]), output)
    original = output.read_bytes()
    tissue["feature_sets"][0]["sha256"] = "0" * 64
    with pytest.raises(ExperimentError, match="SHA-256 mismatch"):
        export_experiment(dict(tissues=[tissue]), output, overwrite=True)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(f".{FILENAME}.*.tmp"))


def test_shared_feature_id_is_tissue_scoped_but_requires_compatible_definition(tmp_path):
    a = cached_tissue(tmp_path, "a")
    b = cached_tissue(tmp_path, "b")
    output = tmp_path / FILENAME
    payload = export_experiment(dict(tissues=[a, b]), output)
    assert [t["feature_sets"][0]["feature_set_id"] for t in payload["tissues"]] == ["virchow2-pooled"] * 2
    before = output.read_bytes()
    b["feature_sets"][0]["encoder"]["revision"] = "c" * 40
    with pytest.raises(ExperimentError, match="incompatible definitions"):
        export_experiment(dict(tissues=[a, b]), output, overwrite=True)
    assert output.read_bytes() == before


def test_malformed_spec_has_friendly_cli_error(tmp_path):
    tissue = cached_tissue(tmp_path)
    tissue["artifacts"] = None
    spec = tmp_path / "bad_spec.json"
    spec.write_text(json.dumps(dict(tissues=[tissue])))
    result = subprocess.run([sys.executable, str(ROOT / "bin" / "export_experiment.py"),
                             "--spec", str(spec), "--output", str(tmp_path / FILENAME)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "artifacts must be an array" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("failure", ["duplicate_tissue", "duplicate_feature", "columns", "unit", "coordinates", "nan", "duplicate_rows", "hash", "metadata_as_feature"])
def test_invalid_cached_assets_fail_closed(tmp_path, failure):
    tissue = cached_tissue(tmp_path, compressed=False)
    feature = tissue["feature_sets"][0]
    spec = dict(tissues=[tissue])
    if failure == "duplicate_tissue":
        spec["tissues"].append(copy.deepcopy(tissue))
    elif failure == "duplicate_feature":
        tissue["feature_sets"].append(copy.deepcopy(feature))
    elif failure == "columns":
        feature["feature_columns"][-1] = feature["feature_columns"][0]
    elif failure == "unit":
        feature["observation_unit"] = "cell"
    elif failure == "hash":
        tissue["image"]["sha256"] = "0" * 64
    elif failure == "metadata_as_feature":
        feature["feature_columns"][0] = "x"
    else:
        path = Path(feature["path"])
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        if failure == "coordinates":
            rows[1][1] = "450"
        elif failure == "nan":
            rows[1][3] = "NaN"
        else:
            rows[2][0] = rows[1][0]
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
    output = tmp_path / FILENAME
    with pytest.raises(ExperimentError):
        export_experiment(spec, output)
    assert not output.exists()


def test_manifest_hashes_and_versions_are_checked_on_reopen(tmp_path):
    tissue = cached_tissue(tmp_path)
    output = tmp_path / FILENAME
    payload = export_experiment(dict(tissues=[tissue]), output)
    feature_path = Path(tissue["feature_sets"][0]["path"])
    feature_path.write_bytes(feature_path.read_bytes() + b"tampered")
    with pytest.raises(ExperimentError, match="SHA-256 mismatch"):
        validate_experiment(output)
    payload["schema_version"] = 2
    output.write_text(json.dumps(payload))
    with pytest.raises(ExperimentError, match="schema_version=1"):
        validate_experiment(output)


def test_run_discovery_uses_durable_crop_and_records_original_affine(tmp_path):
    run, _ = run_tree(tmp_path / "results")
    run_tree(run, "second", offset=(0, 0))
    output = run / "00_execution" / FILENAME
    payload = export_runs([run], output, name="Paired experiment")
    assert len(payload["tissues"]) == 2
    assert validate_experiment(output)["name"] == "Paired experiment"
    first = next(t for t in payload["tissues"] if t["name"] == "sample")
    assert "03_stardist/sample/prepared_crop" in first["image"]["path"]
    assert first["provenance"]["crop_to_original"]["affine"] == [[1, 0, 41], [0, 1, 23], [0, 0, 1]]
    assert first["feature_sets"][0]["grid"]["origin_x_px"] == 0
    assert first["feature_sets"][0]["x_column"] == "cx"


def test_run_completion_hash_and_observation_alignment_are_required(tmp_path):
    run, embedding = run_tree(tmp_path / "results")
    with gzip.open(embedding, "rt") as stream:
        rows = list(csv.reader(stream))
    rows[1][0] = "unmatched"
    with gzip.open(embedding, "wt", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ExperimentError, match="completion record"):
        discover_run(run)
    marker_path = embedding.parent / "VIRCHOW2_COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["embeddings_sha256"] = sha256(embedding)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ExperimentError, match="IDs/coordinates disagree"):
        discover_run(run)


def test_no_encoder_or_workflow_imported_and_legacy_status_is_explicit(tmp_path, monkeypatch):
    tissue = cached_tissue(tmp_path)
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "timm", "safetensors", "pyvips", "nextflow"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    export_experiment(dict(tissues=[tissue]), tmp_path / FILENAME)
    legacy = tmp_path / "legacy_run"
    status = try_export_run(legacy)
    assert status["status"] == "unavailable"
    assert status["inference_started"] is False
    assert not (legacy / "00_execution" / FILENAME).exists()
    assert json.loads((legacy / "00_execution" / "experiment_export_status.json").read_text()) == status


def test_cli_export_validate_and_friendly_error(tmp_path):
    tissue = cached_tissue(tmp_path)
    tissue["image"]["path"] = "breast/image.tif"
    tissue["feature_sets"][0]["path"] = "breast/features.csv.gz"
    spec = tmp_path / "assets.json"
    spec.write_text(json.dumps(dict(name="Friendly study", tissues=[tissue])))
    output = tmp_path / "project" / FILENAME
    command = [sys.executable, str(ROOT / "bin" / "export_experiment.py")]
    result = subprocess.run([*command, "--spec", str(spec), "--output", str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Ready to open in wsiTools" in result.stdout
    result = subprocess.run([*command, "--validate", str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Friendly study" in result.stdout
    result = subprocess.run([*command, "--run", str(tmp_path / "missing"), "--output", str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "No complete Virchow2" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("broken", [False, True])
def test_execution_reports_include_available_experiment_link_and_status(tmp_path, broken):
    run, embedding = run_tree(tmp_path / "results")
    if broken:
        embedding.write_bytes(b"corrupt cache")
    result = subprocess.run([sys.executable, str(ROOT / "bin" / "write_pipeline_execution_reports.py"),
                             "--outdir", str(run), "--run-name", "Cached two-tissue review",
                             "--success", "true", "--start-point", "grid_tiles", "--end-point", "uni2"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    execution = run / "00_execution"
    status = json.loads((execution / "experiment_export_status.json").read_text())
    if broken:
        assert status["status"] == "error"
        assert "completion record" in status["error"]
        assert f'href="{FILENAME}"' not in (execution / "index.html").read_text()
        assert f"]({FILENAME})" not in (execution / "final_report.md").read_text()
    else:
        assert validate_experiment(execution / FILENAME)["tissues"]
        assert status["status"] == "available"
        assert f'href="{FILENAME}"' in (execution / "index.html").read_text()
        assert f"]({FILENAME})" in (execution / "final_report.md").read_text()
    assert json.loads((execution / "final_report.json").read_text())["interactive_experiment"] == status
    assert json.loads((execution / "project_outputs.json").read_text())["interactive_experiment"] == status


def test_execution_reports_keep_legacy_run_available_without_new_producer_dependencies(tmp_path):
    run = tmp_path / "legacy_results"
    result = subprocess.run([sys.executable, str(ROOT / "bin" / "write_pipeline_execution_reports.py"),
                             "--outdir", str(run), "--run-name", "Existing UNI-2 output",
                             "--success", "true", "--start-point", "convert", "--end-point", "uni2"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    execution = run / "00_execution"
    status = json.loads((execution / "experiment_export_status.json").read_text())
    assert status["status"] == "unavailable"
    assert status["inference_started"] is False
    report = (execution / "final_report.md").read_text()
    assert "Interactive export unavailable" in report
    assert "](experiment_export_status.json)" in report
    assert f"]({FILENAME})" not in report
    assert not (execution / FILENAME).exists()


def test_failed_auto_export_is_visible_without_replacing_previous_index(tmp_path):
    run, embedding = run_tree(tmp_path / "results")
    assert try_export_run(run)["status"] == "available"
    output = run / "00_execution" / FILENAME
    before = output.read_bytes()
    embedding.write_bytes(b"broken")
    status = try_export_run(run)
    assert status["status"] == "error"
    assert status["path"] is None
    assert status["previous_index_retained"] is True
    assert output.read_bytes() == before


def test_synthetic_demo_has_three_tissues_both_units_and_separate_labels(tmp_path):
    destination = tmp_path / "synthetic_demo"
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "create_cached_experiment_demo.py"), str(destination)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "SYNTHETIC" in result.stdout
    payload = validate_experiment(destination / FILENAME)
    assert payload["provenance"]["synthetic"] is True
    assert payload["provenance"]["encoder_executed"] is False
    assert len(payload["tissues"]) == 3
    for tissue in payload["tissues"]:
        assert len(tissue["feature_sets"]) == 2
        assert {f["feature_set_id"] for f in tissue["feature_sets"]} == {"virchow2-pooled", "virchow2-spatial"}
        assert {f["observation_unit"] for f in tissue["feature_sets"]} == {"tile", "token"}
        assert all("feature_columns_prefix" not in f for f in tissue["feature_sets"])
        assert all("class_name" not in f["feature_columns"] for f in tissue["feature_sets"])
        assert {f["preprocessing"]["feature_definition"] for f in tissue["feature_sets"]} == {"CLS+mean_patch", "spatial_patch_tokens"}
    with (destination / "labels.csv").open() as stream:
        labels = list(csv.DictReader(stream))
    assert {row["split"] for row in labels} == {"train", "query"}
    assert {row["class_name"] for row in labels} == {"epithelium", "stroma"}
    assert len([row for row in labels if row["observation_id"] == "1"]) == 6


def test_atomic_replace_failure_preserves_existing_index(tmp_path, monkeypatch):
    import export_experiment as exporter
    tissue = cached_tissue(tmp_path)
    output = tmp_path / FILENAME
    export_experiment(dict(tissues=[tissue]), output)
    before = output.read_bytes()
    def denied_replace(_source, _destination):
        raise OSError("simulated interrupted publication")
    monkeypatch.setattr(exporter.os, "replace", denied_replace)
    with pytest.raises(OSError, match="interrupted"):
        export_experiment(dict(tissues=[tissue]), output, overwrite=True)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(f".{FILENAME}.*.tmp"))
