from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "run_sample_batch.py"
SPEC = importlib.util.spec_from_file_location("run_sample_batch", SCRIPT)
assert SPEC and SPEC.loader
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def make_project(path: Path) -> Path:
    project = path / "project"
    for directory in ("bin", "lib", "modules", "subworkflows"):
        (project / directory).mkdir(parents=True)
    (project / "main.nf").write_text("workflow {}\n", encoding="utf-8")
    (project / "nextflow.config").write_text("params {}\n", encoding="utf-8")
    (project / "nextflow_schema.json").write_text("{}\n", encoding="utf-8")
    (project / "bin" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    return project


def make_fake_nextflow(path: Path, counter: Path, fail_sample: str = "") -> Path:
    script = path / "fake_nextflow.py"
    script.write_text(
        """#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
work = pathlib.Path(value('-work-dir'))
out = pathlib.Path(value('--outdir_base'))
work.mkdir(parents=True, exist_ok=True)
(work / 'cache.bin').write_bytes(b'cache')
out.mkdir(parents=True, exist_ok=True)
(out / 'pipeline.ok').write_text('ok')
counter = pathlib.Path(%s)
rows = json.loads(counter.read_text()) if counter.exists() else []
rows.append(out.name)
counter.write_text(json.dumps(rows))
raise SystemExit(9 if out.name == %s else 0)
""" % (repr(str(counter)), repr(fail_sample)),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def run_batch(tmp_path: Path, *, fail_sample: str = "", extra: list[str] | None = None):
    project = make_project(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("alpha.tif", "beta.tif"):
        (inputs / name).write_bytes(name.encode())
    params = tmp_path / "params.yml"
    params.write_text("folder_input: null\nimage_input: null\nroi_geojson: null\n", encoding="utf-8")
    counter = tmp_path / "calls.json"
    fake = make_fake_nextflow(tmp_path, counter, fail_sample)
    output = tmp_path / "results"
    work = tmp_path / "work"
    command = [
        sys.executable, str(SCRIPT),
        "--input-dir", str(inputs),
        "--output-root", str(output),
        "--work-root", str(work),
        "--project-dir", str(project),
        "--params-file", str(params),
        "--nextflow", str(fake),
        *(extra or []),
    ]
    return subprocess.run(command, text=True, capture_output=True), output, work, counter, params


def test_success_cleans_each_work_dir_and_reuses_matching_completion(tmp_path: Path) -> None:
    first, output, work, counter, _params = run_batch(tmp_path)
    assert first.returncode == 0, first.stderr
    assert json.loads(counter.read_text()) == ["alpha", "beta"]
    assert not (work / "alpha").exists()
    assert not (work / "beta").exists()
    for sample in ("alpha", "beta"):
        marker = json.loads((output / sample / "00_execution" / "batch_complete.json").read_text())
        assert marker["status"] == "completed"

    second = subprocess.run([
        sys.executable, str(SCRIPT),
        "--input-dir", str(tmp_path / "inputs"),
        "--output-root", str(output),
        "--work-root", str(work),
        "--project-dir", str(tmp_path / "project"),
        "--params-file", str(tmp_path / "params.yml"),
        "--nextflow", str(tmp_path / "fake_nextflow.py"),
    ], text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert json.loads(counter.read_text()) == ["alpha", "beta"]
    manifest = json.loads((output / "batch_status.json").read_text())
    assert all(row["reused_completion"] for row in manifest["samples"].values())


def test_failed_sample_keeps_work_and_later_sample_remains_pending(tmp_path: Path) -> None:
    completed, output, work, counter, _params = run_batch(tmp_path, fail_sample="alpha")
    assert completed.returncode == 9
    assert (work / "alpha" / "cache.bin").is_file()
    assert not (output / "alpha" / "00_execution" / "batch_complete.json").exists()
    assert json.loads(counter.read_text()) == ["alpha"]
    manifest = json.loads((output / "batch_status.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["samples"]["alpha"]["status"] == "failed"
    assert manifest["samples"]["beta"]["status"] == "pending"


def test_keep_work_on_success_is_explicit_opt_out(tmp_path: Path) -> None:
    completed, _output, work, _counter, _params = run_batch(
        tmp_path, extra=["--keep-work-on-success"],
    )
    assert completed.returncode == 0
    assert (work / "alpha" / "cache.bin").is_file()
    assert (work / "beta" / "cache.bin").is_file()


def test_vsi_inventory_includes_companion_payload(tmp_path: Path) -> None:
    image = tmp_path / "sample.vsi"
    image.write_bytes(b"header")
    companion = tmp_path / "_sample_" / "stack1"
    companion.mkdir(parents=True)
    (companion / "frame_t.ets").write_bytes(b"pixels")
    inventory = batch.input_inventory(image)
    assert [row["path"] for row in inventory] == ["_sample_/stack1/frame_t.ets", "sample.vsi"]


def test_sample_params_can_bind_generated_preflight_metadata(tmp_path: Path) -> None:
    source = tmp_path / "params.yml"
    source.write_text("folder_input: /old\nimage_input: null\nroi_geojson: old\n", encoding="utf-8")
    destination = tmp_path / "resolved.yml"
    image = tmp_path / "sample.vsi"
    metadata = tmp_path / "series.json"
    batch.write_sample_params(
        source, destination, image,
        {"storage_preflight_input_metadata": str(metadata)},
    )
    text = destination.read_text()
    assert f'image_input: "{image}"' in text
    assert "folder_input: null" in text
    assert "roi_geojson: null" in text
    assert f'storage_preflight_input_metadata: "{metadata}"' in text


def test_work_cleanup_refuses_work_root(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    with pytest.raises(RuntimeError, match="Refusing unsafe"):
        batch.remove_sample_work(root, root)
