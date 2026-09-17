"""Synthetic contract QA: genuine Python CLI environments, no model inference.

The tiny test package/checkpoint/graph are deliberately not CellViT weights or
learned outputs. Only interpreter provenance and portable-data contracts run.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import cellvit_embedding_io as io
import cellvit_embeddings as embeddings
import run_cellvitpp as wrapper


def synthetic_cli(directory, *, package_version="1.0.9", env_shebang=False):
    """Build a no-pip stdlib-only environment, not an installed learned runtime."""
    venv.EnvBuilder(with_pip=False, symlinks=True).create(directory)
    interpreter = directory / "bin" / "python"
    site = Path(subprocess.check_output([str(interpreter), "-c",
        "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    package = site / "cellvit"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('Provenance must not import this synthetic package')\n")
    (package / "cli.py").write_text("def main():\n    raise RuntimeError('No model execution in this fixture')\n")
    distribution = site / f"cellvit-{package_version}.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(f"Metadata-Version: 2.1\nName: cellvit\nVersion: {package_version}\n")
    (distribution / "entry_points.txt").write_text("[console_scripts]\ncellvit-inference = cellvit.cli:main\n")
    (distribution / "WHEEL").write_text("Wheel-Version: 1.0\nGenerator: synthetic-contract-test\n")
    executable = directory / "bin" / "cellvit-inference"
    shebang = "/usr/bin/env python" if env_shebang else str(interpreter)
    executable.write_text(f"#!{shebang}\nfrom cellvit.cli import main\nif __name__ == '__main__':\n    raise SystemExit(main())\n")
    executable.chmod(0o700)
    environment = {**os.environ, "PATH": str(directory / "bin") + os.pathsep + os.environ.get("PATH", ""),
                   "PYTHONDONTWRITEBYTECODE": "1"}
    return SimpleNamespace(executable=executable, interpreter=interpreter, package=package, env=environment)


@pytest.mark.parametrize("env_shebang", [False, True])
def test_identity_uses_configured_interpreter_not_wrapper_metadata(tmp_path, env_shebang):
    cli = synthetic_cli(tmp_path / "runtime", env_shebang=env_shebang)
    # A package import would raise, and the environment has no Torch installed.
    with patch("importlib.metadata.version", return_value="WRAPPER-NOT-CELLVIT"):
        identity = wrapper.executable_runtime_identity(cli.executable, cli.env)
    assert identity["status"] == "verified_configured_python_entrypoint"
    assert identity["package_version"] == "1.0.9"
    assert Path(identity["interpreter_prefix"]).resolve() == cli.interpreter.parent.parent.resolve()
    assert identity["portable_identity"]["dependency_versions"]["torch"] is None
    assert identity["portable_identity"]["interpreter_sha256"] == io.sha256(cli.interpreter)
    assert identity["portable_identity"]["executable_sha256"] == io.sha256(cli.executable)
    assert str(tmp_path) not in json.dumps(identity["portable_identity"])


def test_runtime_source_change_with_same_size_and_mtime_changes_identity(tmp_path):
    cli = synthetic_cli(tmp_path / "runtime")
    before = wrapper.executable_runtime_identity(cli.executable, cli.env)
    source = cli.package / "cli.py"
    stat = source.stat()
    source.write_text(source.read_text().replace("No model", "NO model"))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = wrapper.executable_runtime_identity(cli.executable, cli.env)
    assert source.stat().st_size == stat.st_size
    assert before["portable_identity"]["package_source_sha256"] != after["portable_identity"]["package_source_sha256"]


@pytest.mark.parametrize("kind", ["wrong_entrypoint", "shell", "package_escape", "package_cycle", "binary"])
def test_unknown_or_escaping_runtime_cannot_claim_verified_version(tmp_path, kind):
    cli = synthetic_cli(tmp_path / "runtime")
    executable = cli.executable
    if kind == "wrong_entrypoint":
        executable.write_text(executable.read_text().replace("from cellvit.cli", "from not_cellvit.cli"))
    elif kind == "shell":
        executable.write_text("#!/bin/sh\nexit 0\n")
    elif kind == "package_escape":
        (cli.package / "outside.py").symlink_to(cli.executable)
    elif kind == "package_cycle":
        (cli.package / "cycle").symlink_to(cli.package, target_is_directory=True)
    else:
        executable = cli.interpreter
    identity = wrapper.executable_runtime_identity(executable, cli.env)
    assert identity["status"] == "unverified"
    assert identity["package_version"] is None and identity["portable_identity"] is None


def make_bound_embedding_fixture(directory, *, runtime=None, amp=False, mpp=.25, complete=True):
    """Use real capture/export/completion, mocking only numeric graph decoding."""
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / "source.tif"
    tifffile.imwrite(image, np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3), photometric="rgb")
    shift, resolution = directory / "shift.json", directory / "resolution.json"
    shift.write_text(json.dumps({"source_mpp": mpp, "crop_size": {"width": 12, "height": 8},
                                "offset_crop_to_original": {"dx": 100, "dy": 200}}))
    resolution.write_text(json.dumps({"status": "pass", "mpp_x": mpp, "mpp_y": mpp}))
    execution, paths = io.capture_inputs(image, shift, resolution)
    checkpoint = directory / "SYNTHETIC_NOT_WEIGHTS.pth"
    checkpoint.write_bytes(b"Synthetic contract fixture only; not learned weights")
    execution.update({"synthetic_contract_fixture": True, "used_model": False,
        "model": "HIPT", "taxonomy": "pannuke", "amp": amp,
        "runtime_identity": runtime or {"status": "unverified", "package_version": None, "portable_identity": None},
        "checkpoint": {"status": "verified", **io.file_record(checkpoint)},
        "prepared_image": io.file_record(image),
        "preprocessing": {"method": "synthetic_mock_of_libvips_tiffsave", "libvips_version": "synthetic-test",
            "native_size_px": [12, 8], "source_bands": 3, "source_pixel_format": "uchar", "bands": 3,
            "channel_conversion": "none", "tile_size_px": [512, 512], "compression": "jpeg",
            "jpeg_quality": 92, "pyramid": True, "bigtiff": True}})
    outdir = directory / "bundle"
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True)
    raw = {"synthetic_contract_fixture": True, "cells": [
        {"id": "007", "centroid": [2., 3.]}, {"id": "NA", "centroid": [7., 4.]},
        {"id": "discarded", "centroid": [10., 6.]}]}
    raw_json, retained = rawdir / "cells.json", outdir / "cellvit_cells.json"
    raw_json.write_text(json.dumps(raw))
    retained.write_text(json.dumps({"synthetic_contract_fixture": True, "cells": raw["cells"][1::-1]}))
    (rawdir / "cells.pt").write_bytes(b"Synthetic graph decoder fixture; not a pickle")
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    positions = np.asarray([cell["centroid"] for cell in raw["cells"]])
    with patch.object(embeddings, "load_graph_arrays", return_value=(vectors, positions)):
        metadata = embeddings.export_embeddings(raw_json, retained, outdir,
            {"synthetic_contract_fixture": True}, execution=execution)
    receipt = io.complete_embedding_bundle(outdir, execution, source_paths=paths) if complete else None
    return SimpleNamespace(outdir=outdir, execution=execution, paths=paths, metadata=metadata,
        receipt=receipt, vectors=vectors[[1, 0]], image=image, shift=shift, resolution=resolution)


def load_fixture(fixture, source=None):
    return io.load_cellvit_embedding_bundle(source or fixture.outdir,
        expected_inputs={key: value["sha256"] for key, value in fixture.execution["inputs"].items()},
        expected_geometry=fixture.execution["geometry"])


def test_real_completion_producer_and_loader_are_portable_without_upstream_sources(tmp_path):
    cli = synthetic_cli(tmp_path / "runtime")
    identity = wrapper.executable_runtime_identity(cli.executable, cli.env)
    fixture = make_bound_embedding_fixture(tmp_path / "example", runtime=identity)
    portable = tmp_path / "portable"
    portable.mkdir()
    for name in [*io.FILES.values(), io.COMPLETION]:
        shutil.copy2(fixture.outdir / name, portable / name)
    with patch.object(embeddings, "load_graph_arrays", side_effect=AssertionError("Consumer must not load graph")):
        matrix, ids, contract, sources = load_fixture(fixture, portable)
    np.testing.assert_array_equal(matrix, fixture.vectors)
    assert ids.cellvitpp_id.tolist() == ["NA", "007"]
    assert ids.source_graph_row.tolist() == [1, 0]
    assert contract["reference_compatible"] is True
    assert contract["status"] == "verified_exact_inputs_and_population"
    assert contract["retained_population_sha256"] == io.sha256(portable / "cellvit_cells.json")
    assert set(sources) == {*io.FILES, "receipt"}
    assert all(path.parent == portable for path in sources.values())


def test_unknown_runtime_is_visible_but_not_reference_compatible(tmp_path):
    fixture = make_bound_embedding_fixture(tmp_path)
    _, _, contract, _ = load_fixture(fixture)
    assert contract["reference_compatible"] is False
    assert contract["feature_definition"]["runtime"] is None


@pytest.mark.parametrize("source", ["image", "shift", "resolution_json"])
def test_expected_input_hashes_are_mandatory_and_exact(tmp_path, source):
    fixture = make_bound_embedding_fixture(tmp_path)
    expected = {key: value["sha256"] for key, value in fixture.execution["inputs"].items()}
    expected[source] = "0" * 64
    with pytest.raises(ValueError, match="source input hash mismatch"):
        io.load_cellvit_embedding_bundle(fixture.outdir, expected_inputs=expected,
                                        expected_geometry=fixture.execution["geometry"])


@pytest.mark.parametrize("key,value", [("source_mpp", .5), ("crop_size_px", [13, 8]), ("crop_origin_px", [101, 200])])
def test_expected_geometry_is_exact(tmp_path, key, value):
    fixture = make_bound_embedding_fixture(tmp_path)
    expected = {**fixture.execution["geometry"], key: value}
    with pytest.raises(ValueError, match="geometry/calibration mismatch"):
        io.load_cellvit_embedding_bundle(fixture.outdir,
            expected_inputs={key: value["sha256"] for key, value in fixture.execution["inputs"].items()},
            expected_geometry=expected)


@pytest.mark.parametrize("key", list(io.FILES))
def test_corrupt_payload_is_rejected(tmp_path, key):
    fixture = make_bound_embedding_fixture(tmp_path)
    with (fixture.outdir / io.FILES[key]).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        load_fixture(fixture)


@pytest.mark.parametrize("mutation", ["metadata_count", "metadata_retained_hash", "duplicate_id", "raw_centroid", "row_order", "nonfinite", "out_of_bounds"])
def test_completion_rejects_inconsistent_payloads_without_receipt(tmp_path, mutation):
    fixture = make_bound_embedding_fixture(tmp_path, complete=False)
    if mutation.startswith("metadata"):
        path = fixture.outdir / io.FILES["metadata"]
        value = json.loads(path.read_text())
        if mutation == "metadata_count":
            value["raw_cell_count"] += 1
        else:
            value["upstream_artifacts"]["retained_cells_json"]["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    elif mutation == "raw_centroid":
        path = fixture.outdir / io.FILES["raw_population"]
        path.write_text(path.read_text().replace("7.0,4.0", "7.1,4.0"))
    elif mutation == "nonfinite":
        data = fixture.vectors.copy()
        data[0, 0] = np.nan
        np.save(fixture.outdir / io.FILES["embeddings"], data)
    elif mutation == "out_of_bounds":
        fixture.execution["geometry"]["crop_size_px"] = [2, 2]
        fixture.execution["preprocessing"]["native_size_px"] = [2, 2]
    else:
        path = fixture.outdir / io.FILES["ids"]
        value = path.read_text()
        value = value.replace("007", "NA") if mutation == "duplicate_id" else value.replace("0,NA", "1,NA")
        path.write_text(value)
    with pytest.raises(ValueError):
        io.complete_embedding_bundle(fixture.outdir, fixture.execution, source_paths=fixture.paths)
    assert not (fixture.outdir / io.COMPLETION).exists()


def test_source_mutation_during_completion_leaves_no_receipt(tmp_path):
    fixture = make_bound_embedding_fixture(tmp_path, complete=False)
    original = io._payloads
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with fixture.image.open("ab") as handle:
            handle.write(b"source changed")
        return result
    with patch.object(io, "_payloads", side_effect=mutate), pytest.raises(ValueError, match="source inputs changed"):
        io.complete_embedding_bundle(fixture.outdir, fixture.execution, source_paths=fixture.paths)
    assert not (fixture.outdir / io.COMPLETION).exists()


def test_interrupted_new_bundle_is_not_downgraded_to_legacy(tmp_path):
    fixture = make_bound_embedding_fixture(tmp_path, complete=False)
    with pytest.raises(ValueError, match="lack their completion"):
        load_fixture(fixture)


def test_no_existing_embedding_output_is_overwritten(tmp_path):
    fixture = make_bound_embedding_fixture(tmp_path)
    before = {path: io.sha256(path) for path in fixture.outdir.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        embeddings.export_embeddings(fixture.outdir / "raw/cells.json", fixture.outdir / "cellvit_cells.json",
                                     fixture.outdir, {}, execution=fixture.execution)
    with pytest.raises(FileExistsError):
        io.complete_embedding_bundle(fixture.outdir, fixture.execution)
    assert {path: io.sha256(path) for path in before} == before


@pytest.mark.parametrize("key", ["embeddings", "ids", "metadata", "retained_population"])
def test_legacy_escape_is_rejected_before_any_payload_read(tmp_path, key):
    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("not a valid payload")
    (root / io.FILES[key]).symlink_to(outside)
    with patch.object(io, "read_json", side_effect=AssertionError("No read before containment check")):
        with pytest.raises(ValueError, match="escapes"):
            io.load_cellvit_embedding_bundle(root, expected_inputs={}, expected_geometry={})


def test_feature_definition_retains_physical_conversion_and_requested_amp(tmp_path):
    fixture = make_bound_embedding_fixture(tmp_path)
    baseline, _ = io.feature_definition(fixture.execution)
    for field, value in [("amp", True), ("source_mpp", .5), ("channel_conversion", "replicate_grayscale_to_rgb")]:
        other = copy.deepcopy(fixture.execution)
        if field == "amp":
            other[field] = value
        elif field == "source_mpp":
            other["geometry"][field] = value
        else:
            other["preprocessing"][field] = value
        definition, _ = io.feature_definition(other)
        assert definition != baseline
    assert "not measured autocast" in baseline["amp_semantics"]


@pytest.mark.parametrize("change_source", [False, True])
def test_wrapper_runs_synthetic_configured_cli_and_completes_only_stable_inputs(tmp_path, monkeypatch, change_source):
    fixture = make_bound_embedding_fixture(tmp_path / "source")
    cli = synthetic_cli(tmp_path / "runtime")
    (cli.package / "__init__.py").write_text("# Explicit synthetic CLI, not a learned runtime.\n")
    program = '''import json, sys
from pathlib import Path
def main():
    if '--help' in sys.argv:
        print('synthetic contract-test CLI [--graph]')
        return 0
    output = Path(sys.argv[sys.argv.index('--outdir') + 1]) / 'cellvit_input'
    output.mkdir(parents=True)
    payload = {'synthetic_contract_fixture': True, 'cells': [
        {'id': '007', 'centroid': [2., 3.]}, {'id': 'NA', 'centroid': [7., 4.]},
        {'id': 'discarded', 'centroid': [10., 6.]}]}
    (output / 'cells.json').write_text(json.dumps(payload))
    (output / 'cells.pt').write_bytes(b'SYNTHETIC_GRAPH_NOT_PICKLE')
    return 0
'''
    if change_source:
        # The explicitly requested synthetic CLI changes one source after output.
        program = program.replace("    (output / 'cells.pt').write_bytes(b'SYNTHETIC_GRAPH_NOT_PICKLE')",
            "    (output / 'cells.pt').write_bytes(b'SYNTHETIC_GRAPH_NOT_PICKLE')\n"
            + f"    Path({str(fixture.shift)!r}).write_text('source changed during fixture execution')")
    (cli.package / "cli.py").write_text(program)
    cache = tmp_path / "cache"
    cache.mkdir()
    wrapper.required_model_path(cache, "HIPT").write_bytes(b"SYNTHETIC_NOT_LEARNED_WEIGHTS")
    outdir = tmp_path / "wrapper_output"
    monkeypatch.setenv("CELLVIT_CACHE", str(cache))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(sys, "argv", ["run_cellvitpp.py", "--image", str(fixture.image),
        "--shift", str(fixture.shift), "--resolution-json", str(fixture.resolution),
        "--outdir", str(outdir), "--executable", str(cli.executable),
        "--batch-size", "2", "--export-embeddings", "--amp"])
    def prepare(src, dst):
        shutil.copyfile(src, dst)
        return fixture.execution["preprocessing"]
    monkeypatch.setattr(wrapper, "make_pyramid", prepare)
    monkeypatch.setattr(embeddings, "require_graph_loader", lambda: None)
    monkeypatch.setattr(embeddings, "load_graph_arrays", lambda _: (
        np.arange(12, dtype=np.float32).reshape(3, 4), np.asarray([[2., 3.], [7., 4.], [10., 6.]])))
    if change_source:
        with pytest.raises(ValueError, match="source inputs changed"):
            wrapper.main()
        assert not (outdir / io.COMPLETION).exists()
    else:
        wrapper.main()
        _, ids, contract, _ = load_fixture(fixture, outdir)
        assert ids.cellvitpp_id.tolist() == ["007", "NA", "discarded"]
        execution = contract["receipt"]["execution"]
        assert execution["amp"] is True
        assert execution["runtime_identity"]["package_version"] == "1.0.9"
        assert execution["runtime_identity"]["portable_identity"]["dependency_versions"]["torch"] is None
        assert execution["checkpoint"]["sha256"] == io.sha256(wrapper.required_model_path(cache, "HIPT"))


@pytest.mark.parametrize("kind", ["failed_report", "anisotropic", "shift_mpp", "crop_size", "missing_report"])
def test_source_capture_requires_passed_consistent_calibration(tmp_path, kind):
    fixture = make_bound_embedding_fixture(tmp_path)
    resolution = json.loads(fixture.resolution.read_text())
    shift = json.loads(fixture.shift.read_text())
    if kind == "failed_report":
        resolution["status"] = "fail"
    elif kind == "anisotropic":
        resolution["mpp_y"] = .5
    elif kind == "shift_mpp":
        shift["source_mpp"] = .5
    elif kind == "crop_size":
        shift["crop_size"]["width"] += 1
    fixture.shift.write_text(json.dumps(shift))
    fixture.resolution.write_text(json.dumps(resolution))
    with pytest.raises(ValueError):
        io.capture_inputs(fixture.image, fixture.shift, None if kind == "missing_report" else fixture.resolution)


@pytest.mark.parametrize("taxonomy,compatible", [("pannuke", True), ("binary", True), ("lizard", False), ("nucls_main", False)])
def test_external_classifier_weights_are_not_claimed_bound(tmp_path, taxonomy, compatible):
    cli = synthetic_cli(tmp_path / "runtime")
    fixture = make_bound_embedding_fixture(tmp_path / "example", complete=False,
        runtime=wrapper.executable_runtime_identity(cli.executable, cli.env))
    fixture.execution["taxonomy"] = taxonomy
    io.complete_embedding_bundle(fixture.outdir, fixture.execution, source_paths=fixture.paths)
    _, _, contract, _ = load_fixture(fixture)
    assert contract["reference_compatible"] is compatible
    assert "external taxonomy classifier weights not captured" in contract["feature_definition"]["checkpoint_scope"]
