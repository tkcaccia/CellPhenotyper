"""Registration convenience uses explicit bindings and always writes a new index."""
import builtins
import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from export_experiment import ExperimentError, FILENAME, export_experiment, sha256, validate_experiment
from register_gigatime import inspect_marker_image, register_gigatime
from test_gigatime_experiment import gigatime_tissue, write_scores
from test_spatial_experiment import spatial_tissue
from test_experiment_export import cached_tissue


def registration_fixture(root, *, native=False):
    tissue, rows, table = gigatime_tissue(root / "assets", native=native)
    prepared = copy.deepcopy(tissue["artifacts"])
    tissue["artifacts"] = []
    spatial, _, _ = spatial_tissue(root / "assets", "spatial-section")
    features = cached_tissue(root / "assets", "feature-section")
    source = root / "source" / FILENAME
    spec = dict(experiment_id="spatial-study", name="Preserved multi-image experiment",
                settings=dict(existing=True), provenance=dict(source="synthetic registration fixture"),
                extension_metadata=dict(note="preserve every existing root field"),
                tissues=[tissue, spatial, features])
    spec["tissues"][0]["extension_metadata"] = dict(note="preserve tissue metadata")
    export_experiment(spec, source)
    arguments = dict(tissue_id="section-a", marker_image=prepared[0]["path"],
        channels=[("CD3", 0), ("CD8", 1)], full_image=True, scores=str(table),
        mask=prepared[1]["path"], mask_name="nuclei", segmentation_unit="nucleus",
        score_columns=[("CD3", "CD3__mean"), ("CD8", "CD8__sum")],
        source_coordinate_convention="native-image-pixel-centres" if native else "zero-based-pixel-index")
    return source, arguments, rows, table


@pytest.mark.parametrize("native", [False, True])
def test_new_index_preserves_multitissue_spatial_features_metadata_and_assets(tmp_path, monkeypatch, native):
    source, arguments, _, table = registration_fixture(tmp_path, native=native)
    before = source.read_bytes()
    original = validate_experiment(source)
    hashes = {str(path): sha256(path) for path in (tmp_path / "assets").rglob("*") if path.is_file()}
    original_import = builtins.__import__

    def no_model_imports(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "timm", "nextflow", "pyreadr", "rpy2"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_imports)
    output = tmp_path / "new-project" / FILENAME
    result = register_gigatime(source, output, **arguments)
    assert validate_experiment(output) == result
    assert source.read_bytes() == before
    assert len(result["tissues"]) == 3
    assert result["experiment_id"] == original["experiment_id"]
    assert result["settings"] == original["settings"]
    assert result["provenance"] == original["provenance"]
    assert result["extension_metadata"] == original["extension_metadata"]
    assert result["tissues"][0]["extension_metadata"] == original["tissues"][0]["extension_metadata"]
    assert [a["sha256"] for a in result["tissues"][1]["artifacts"]] == [a["sha256"] for a in original["tissues"][1]["artifacts"]]
    assert result["tissues"][2]["feature_sets"][0]["sha256"] == original["tissues"][2]["feature_sets"][0]["sha256"]
    affine = result["tissues"][0]["artifacts"][2]["source_to_image_affine"]
    assert affine == ([[1, 0, 0], [0, 1, 0], [0, 0, 1]] if native else [[2, 0, 1], [0, 2, 1], [0, 0, 1]])
    assert result["tissues"][0]["artifacts"][2]["row_count"] == 2
    assert all(sha256(Path(path)) == digest for path, digest in hashes.items())
    assert "200" in table.read_text()


def test_inspection_and_real_cli_registration_need_no_manual_hash(tmp_path):
    source, arguments, _, _ = registration_fixture(tmp_path)
    facts = inspect_marker_image(arguments["marker_image"])
    assert [(row["marker"], row["page"], row["width"], row["height"], row["dtype"]) for row in facts] == [
        ("CD3", 0, 32, 24, "uint16"), ("CD8", 1, 32, 24, "uint16")]
    command = [sys.executable, str(ROOT / "bin" / "register_gigatime.py")]
    inspection = subprocess.run([*command, "--inspect-marker-image", arguments["marker_image"]], capture_output=True, text=True, timeout=30)
    assert inspection.returncode == 0, inspection.stderr
    assert "page 0: CD3" in inspection.stdout and "uint16" in inspection.stdout
    output = tmp_path / "from-cli" / FILENAME
    result = subprocess.run([*command, "--experiment", str(source), "--tissue-id", "section-a", "--marker-image", arguments["marker_image"],
        "--channel", "CD3", "0", "--channel", "CD8", "1", "--full-image", "--scores", arguments["scores"],
        "--mask", arguments["mask"], "--mask-name", "nuclei", "--segmentation-unit", "nucleus",
        "--score-column", "CD3", "CD3__mean", "--score-column", "CD8", "CD8__sum",
        "--source-coordinate-convention", "zero-based-pixel-index", "--output", str(output)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Ready to open in wsiTools" in result.stdout and "Retained 3 tissues" in result.stdout
    assert validate_experiment(output)["tissues"][0]["artifacts"][2]["row_count"] == 2


def test_explicit_crop_extent_and_uppercase_compressed_csv_keep_raw_values(tmp_path):
    source, arguments, _, table = registration_fixture(tmp_path)
    compressed = table.with_name("scores.CSV.GZ")
    with gzip.open(compressed, "wb") as stream:
        stream.write(table.read_bytes())
    before = compressed.read_bytes()
    arguments.update(full_image=False, extent=[8, 4, 48, 36], scores=str(compressed))
    result = register_gigatime(source, tmp_path / "crop-project.json", **arguments)
    artifacts = result["tissues"][0]["artifacts"]
    assert artifacts[0]["extent"] == dict(x=8, y=4, width=48, height=36)
    assert artifacts[2]["source_to_image_affine"] == [[1.5, 0, 8.75], [0, 1.5, 4.75], [0, 0, 1]]
    assert artifacts[2]["format"] == "csv.gz"
    assert compressed.read_bytes() == before
    assert validate_experiment(tmp_path / "crop-project.json") == result


def test_float_marker_only_registration_requires_explicit_display_range(tmp_path):
    import numpy as np
    import tifffile
    source, arguments, _, _ = registration_fixture(tmp_path)
    marker = Path(arguments["marker_image"])
    planes = np.full((2, 24, 32), .25, dtype=np.float32)
    tifffile.imwrite(marker, planes, ome=True, photometric="minisblack",
                     metadata={"axes": "CYX", "Channel": {"Name": ["CD3", "CD8"]}})
    for key in ("scores", "mask", "mask_name", "segmentation_unit", "score_columns", "source_coordinate_convention"):
        arguments.pop(key)
    output = tmp_path / "float-project.json"
    assert all(item["requires_display_max"] for item in inspect_marker_image(marker))
    with pytest.raises(ExperimentError, match="Floating marker 'CD3'.*display-max"):
        register_gigatime(source, output, **arguments)
    assert not output.exists()
    arguments["display_max"] = [("CD3", 1), ("CD8", 1)]
    result = register_gigatime(source, output, **arguments)
    assert len(result["tissues"][0]["artifacts"]) == 1
    assert all(channel["display_max"] == 1 for channel in result["tissues"][0]["artifacts"][0]["channels"])
    assert validate_experiment(output) == result


@pytest.mark.parametrize("failure", ["existing", "inplace", "unknown_tissue", "no_extent", "no_coordinate_mode", "wrong_marker", "missing_mask", "id_collision"])
def test_incomplete_or_ambiguous_registration_never_changes_source(tmp_path, failure):
    source, arguments, _, _ = registration_fixture(tmp_path)
    before = source.read_bytes()
    output = tmp_path / "fresh.json"
    if failure == "existing": output.write_text("prior output")
    elif failure == "inplace": output = source
    elif failure == "unknown_tissue": arguments["tissue_id"] = "missing"
    elif failure == "no_extent": arguments["full_image"] = False
    elif failure == "no_coordinate_mode": arguments["source_coordinate_convention"] = None
    elif failure == "wrong_marker": arguments["channels"] = [("wrong", 0)]
    elif failure == "missing_mask": arguments["mask"] = None
    elif failure == "id_collision":
        register_gigatime(source, output, **arguments)
        source = output
        before = source.read_bytes()
        output = tmp_path / "second-fresh.json"
    old_output = output.read_bytes() if output.exists() else None
    with pytest.raises(ExperimentError):
        register_gigatime(source, output, **arguments)
    assert source.read_bytes() == before
    if old_output is None:
        assert not output.exists()
    else:
        assert output.read_bytes() == old_output


def test_existing_path_aliases_and_dangling_symlinks_are_refused(tmp_path):
    source, arguments, _, _ = registration_fixture(tmp_path)
    before = source.read_bytes()
    alias = tmp_path / "alias.json"
    dangling = tmp_path / "dangling.json"
    try:
        alias.symlink_to(source)
        dangling.symlink_to(tmp_path / "missing-target")
    except OSError:
        pytest.skip("Symlinks unavailable")
    for output in (alias, dangling):
        with pytest.raises(ExperimentError, match="NEW filename"):
            register_gigatime(source, output, **arguments)
    assert source.read_bytes() == before
    assert dangling.is_symlink()


def test_concurrent_new_output_is_preserved_at_atomic_publication(tmp_path, monkeypatch):
    import export_experiment as exporter
    source, arguments, _, _ = registration_fixture(tmp_path)
    output = tmp_path / "race" / FILENAME
    real_link = exporter.os.link

    def racing_link(staged, destination):
        Path(destination).write_bytes(b"another writer owns this output")
        return real_link(staged, destination)

    monkeypatch.setattr(exporter.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        register_gigatime(source, output, **arguments)
    assert output.read_bytes() == b"another writer owns this output"
    assert not list(output.parent.glob(f".{FILENAME}.*.tmp"))


def test_conflicting_image_convention_fails_without_replacing_index(tmp_path):
    source, _, _, _ = registration_fixture(tmp_path)
    before = source.read_bytes()
    spec = validate_experiment(source)
    spec["tissues"][0]["image"]["coordinate_convention"] = "world_coordinates"
    with pytest.raises(ExperimentError, match="coordinate_convention conflicts"):
        export_experiment(spec, source, base_dir=source.parent, overwrite=True)
    assert source.read_bytes() == before


def test_validation_error_names_tissue_artifact_file_line_and_column(tmp_path):
    source, arguments, rows, table = registration_fixture(tmp_path)
    output = tmp_path / "new.json"
    payload = register_gigatime(source, output, **arguments)
    rows[0]["CD3__mean"] = -1
    write_scores(table, rows)
    payload["tissues"][0]["artifacts"][2]["sha256"] = sha256(table)
    output.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, str(ROOT / "bin" / "export_experiment.py"), "--validate", str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    for expected in ("Cannot validate experiment", "section-a", "gigatime:scores", table.name, "CSV line 2", "CD3__mean"):
        assert expected in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("action", ["validate", "export"])
def test_asset_hash_error_keeps_tissue_artifact_file_context_and_action(tmp_path, action):
    source, arguments, _, table = registration_fixture(tmp_path)
    output = tmp_path / "registered.json"
    register_gigatime(source, output, **arguments)
    table.write_text(table.read_text() + "\n")
    command = [sys.executable, str(ROOT / "bin" / "export_experiment.py")]
    if action == "validate":
        command += ["--validate", str(output)]
    else:
        command += ["--spec", str(output), "--output", str(tmp_path / "not-published.json")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    for expected in (f"Cannot {action} experiment", "section-a", "gigatime:scores", table.name, "SHA-256 mismatch"):
        assert expected in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "not-published.json").exists()
