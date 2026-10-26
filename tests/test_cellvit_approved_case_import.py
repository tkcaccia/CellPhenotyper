"""CPU-only bound-case import tests; no frozen launcher or GPU is invoked."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import cellvit_approved_case_import as bridge
from cellvit_models import digest_file
from cellvit_shards import atomic_json
from test_cellvit_archive_consumer import _bundle


def _case(tmp_path, monkeypatch, *, cleanup=True):
    original, image, plan, shard, source = _bundle(
        tmp_path, "cellvit_cell_classification.csv.gz")
    run_root = tmp_path / "approved-run"
    canonical = run_root / "results" / "PM016_TLN1_HE" / "03_cellvit_fullslide"
    canonical.mkdir(parents=True)
    plan_path = canonical / "cellvit_shard_plan.json"
    atomic_json(plan_path, plan)
    case = canonical / "cases" / shard["id"]
    case.mkdir(parents=True)
    shutil.copytree(source, case / "output")
    source_manifest = tmp_path / "frozen-source-manifest.json"
    atomic_json(source_manifest, {"frozen": True})
    source_sha = digest_file(source_manifest)
    scope_sha = "d" * 64
    baseline_sha = "e" * 64
    binding = {
        "schema": bridge.BINDING_SCHEMA, "host": "Chiamaka",
        "uct_excluded": True, "case_id": "PM016-cellvit-fullrun-" + shard["id"],
        "shard_id": shard["id"], "case_dir": str(case),
        "run_root": str(run_root), "scope": "bound-case.scope",
        "allowed_host": "icgeb-bioinformatics-unit", "cpu_ids": [0, 1],
        "output_dir": str(case / "native-output"),
        "plan_sha256": plan["identity_sha256"],
        "plan_file_sha256": digest_file(plan_path),
        "image_sha256": plan["image"]["sha256"],
        "gpu_source_sha256": source_sha, "approved_scope_sha256": scope_sha,
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": source_sha, "plan_path": str(plan_path),
        "runtime_sha256": "f" * 64,
    }
    binding_path = case / "GPU_CASE_BINDING.json"
    atomic_json(binding_path, binding)
    marker = case / "output" / "complete.json"
    checkpoint_path = case / "GPU_CASE_CHECKPOINT.json"
    atomic_json(checkpoint_path, {
        "schema": bridge.CHECKPOINT_SCHEMA, "complete": True,
        "case_id": binding["case_id"], "binding_sha256": digest_file(binding_path),
        "plan_sha256": plan["identity_sha256"],
        "complete_sha256": digest_file(marker), "cell_count": 1,
    })
    supervisor_path = case / "SUPERVISOR_RESULT.json"
    atomic_json(supervisor_path, {
        "success": True, "cleanup_verified": cleanup, "payload_returncode": 0,
        "cpu_test": False, "events": [
            {"event": "two_empty_owned_snapshots"},
            {"event": "final_assigned_contexts", "contexts": [{"owner": "baseline"}]},
            {"event": "final_assigned_contexts", "contexts": [{"owner": "baseline"}]},
        ],
    })
    result_path = case / "GPU_CASE_RESULT.json"
    atomic_json(result_path, {
        "schema": bridge.RESULT_SCHEMA, "case_id": binding["case_id"],
        "shard_id": shard["id"], "plan_sha256": plan["identity_sha256"],
        "binding_sha256": digest_file(binding_path),
        "approved_scope_sha256": scope_sha, "gpu_source_sha256": source_sha,
        "runtime_sha256": binding["runtime_sha256"],
        "scope": binding["scope"], "success": True, "payload_returncode": 0,
        "cleanup_verified": cleanup, "failures": [],
        "cleanup": {"owned_processes_absent": True, "scope_inactive": True,
                    "assigned_gpu_contexts_baseline_only": True,
                    "gpu_context_baseline_sha256": baseline_sha,
                    "bounded_checks_complete": True},
        "gpu_context_baseline_sha256": baseline_sha,
        "resources": {"allocated_gpus": 1, "cpus": 2, "peak_memory_bytes": 1024},
        "complete_sha256": digest_file(marker),
        "checkpoint_sha256": digest_file(checkpoint_path),
        "supervisor_evidence_sha256": digest_file(supervisor_path),
        "cell_count": 1,
    })
    options = bridge.parser().parse_args([
        "--bundle-dir", str(canonical), "--case-dir", str(case),
        "--image", str(image), "--shift", str(original / "shift.json"),
        "--shard-id", shard["id"], "--case-result-sha256", digest_file(result_path),
        "--binding-sha256", digest_file(binding_path),
        "--gpu-source-sha256", source_sha,
        "--plan-file-sha256", digest_file(plan_path),
        "--reserve-gib", "0", "--case-budget-gib", "0.00001",
    ])
    # This fixture has one synthetic head; production route enforces SAM/AMP/eight.
    monkeypatch.setattr(bridge, "_require_declared_coverage", lambda *_: None)
    return options, canonical, shard


def test_bound_successful_case_imports_only_one_canonical_shard(tmp_path, monkeypatch):
    options, root, shard = _case(tmp_path, monkeypatch)
    result = bridge.import_case(options)
    assert result["gpu_started"] is False and result["cell_count"] == 1
    assert (root / "shards" / shard["id"] / "complete.json").is_file()
    saved = json.loads((root / "shards" / shard["id"] /
                        "CELLVIT_GPU_CLEANUP.json").read_text())
    assert saved["attestation_state"] == "published" and saved["draft"] is False
    assert saved["case_result_sha256"] == options.case_result_sha256
    with pytest.raises(FileExistsError, match="new canonical shard"):
        bridge.import_case(options)


def test_failed_or_changed_case_never_imports(tmp_path, monkeypatch):
    options, root, shard = _case(tmp_path, monkeypatch, cleanup=False)
    with pytest.raises(ValueError, match="failed or differs"):
        bridge.import_case(options)
    assert not (root / "shards").exists()
    options.case_result_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 pin"):
        bridge.import_case(options)
    assert not (root / "shards").exists()


def test_changed_frozen_source_never_imports(tmp_path, monkeypatch):
    options, root, _ = _case(tmp_path, monkeypatch)
    binding = json.loads((options.case_dir / "GPU_CASE_BINDING.json").read_text())
    Path(binding["source_manifest_path"]).write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="source manifest or plan bytes"):
        bridge.import_case(options)
    assert not (root / "shards").exists()


def test_changed_supervisor_journal_never_imports(tmp_path, monkeypatch):
    options, root, _ = _case(tmp_path, monkeypatch)
    journal = options.case_dir / "SUPERVISOR_RESULT.json"
    saved = json.loads(journal.read_text())
    saved["events"].append({"event": "changed-after-terminal-result"})
    atomic_json(journal, saved)
    with pytest.raises(ValueError, match="failed or differs"):
        bridge.import_case(options)
    assert not (root / "shards").exists()


def _rebind_case(options, mutate):
    """Keep the synthetic terminal chain internally consistent after a binding edit."""
    case = options.case_dir
    binding_path = case / "GPU_CASE_BINDING.json"
    binding = json.loads(binding_path.read_text())
    mutate(binding)
    atomic_json(binding_path, binding)
    options.binding_sha256 = digest_file(binding_path)
    checkpoint_path = case / "GPU_CASE_CHECKPOINT.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["binding_sha256"] = options.binding_sha256
    atomic_json(checkpoint_path, checkpoint)
    result_path = case / "GPU_CASE_RESULT.json"
    result = json.loads(result_path.read_text())
    result["binding_sha256"] = options.binding_sha256
    result["checkpoint_sha256"] = digest_file(checkpoint_path)
    atomic_json(result_path, result)
    options.case_result_sha256 = digest_file(result_path)


@pytest.mark.parametrize("run_root", ["same_as_bundle", "unrelated", "relative"])
def test_case_run_root_must_be_strict_ancestor(tmp_path, monkeypatch, run_root):
    options, root, _ = _case(tmp_path, monkeypatch)
    values = {"same_as_bundle": str(root),
              "unrelated": str(tmp_path / "unrelated"),
              "relative": "approved-run"}
    if run_root == "unrelated":
        (tmp_path / "unrelated").mkdir()
    _rebind_case(options, lambda binding: binding.update(run_root=values[run_root]))
    with pytest.raises(ValueError, match="campaign run root|canonical CellPhenotyper result root"):
        bridge.import_case(options)
    assert not (root / "shards").exists()


def test_case_cannot_import_into_another_bundle_under_same_run(tmp_path, monkeypatch):
    options, root, _ = _case(tmp_path, monkeypatch)
    other = root.parent / "another_cellvit_fullslide"
    other.mkdir()
    shutil.copyfile(root / "cellvit_shard_plan.json", other / "cellvit_shard_plan.json")
    options.bundle_dir = other
    with pytest.raises(ValueError, match="canonical CellPhenotyper result root"):
        bridge.import_case(options)
    assert not (root / "shards").exists()
    assert not (other / "shards").exists()
