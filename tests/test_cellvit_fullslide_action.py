"""CPU-only admission and one-action checks; no inference or GPU process."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import cellvit_fullslide_action as action


def _pilot_attestation(tmp_path, root, plan, shard, marker):
    from cellvit_models import digest_file
    from cellvit_shards import atomic_json

    hashes = {"binding": "a" * 64, "supervisor_result": "b" * 64,
              "application_checkpoint": "c" * 64,
              "complete_marker": digest_file(marker)}
    audit_path = tmp_path / "READONLY_VERIFICATION.json"
    atomic_json(audit_path, {
        "host": "icgeb-bioinformatics-unit", "finished_utc": "2026-10-25T12:00:00Z",
        "plan_identity_sha256": plan["identity_sha256"],
        "plan_file_sha256": digest_file(root / "cellvit_shard_plan.json"),
        "assigned_gpu_uuid": "GPU-test", "assigned_gpu_compute_contexts": [],
        "nvidia_smi_exit_code": 0,
        "cases": [{
            "shard_id": shard["id"], "case_id": "historical-" + shard["id"],
            "scope": "historical.scope", "binding_sha256": hashes["binding"],
            "supervisor_result_sha256": hashes["supervisor_result"],
            "checkpoint_file_sha256": hashes["application_checkpoint"],
            "complete_marker_sha256": hashes["complete_marker"],
            "bound_file_hash_mismatches": [], "declared_output_hash_mismatches": [],
            "checkpoint_output_hash_mismatches": [],
            "saved_supervisor_success": True,
            "saved_supervisor_cleanup_verified": True,
            "saved_payload_returncode": 0, "saved_supervisor_failures": [],
            "saved_two_empty_owned_snapshots_event": True,
            "saved_final_assigned_contexts": [[], []],
            "same_recorded_births_alive": [], "active_marker_present": False,
            "failure_marker_present": False,
            "scope_properties": {"ControlGroup": "", "ActiveState": "inactive"},
        }],
    })
    return {
        "schema": action.PILOT_SCHEMA, "attestation_state": "published",
        "draft": False, "shard_id": shard["id"],
        "plan_sha256": plan["identity_sha256"],
        "plan_file_sha256": digest_file(root / "cellvit_shard_plan.json"),
        "checkpoint_sha256": hashes["complete_marker"],
        "cleanup_verified": True, "owned_processes_absent": True,
        "scope_inactive": True, "assigned_gpu_contexts_empty": True,
        "bounded_checks_complete": True,
        "source_case_id": "historical-" + shard["id"],
        "source_scope": "historical.scope",
        "source_host": "icgeb-bioinformatics-unit",
        "assigned_gpu_uuid": "GPU-test",
        "source_artifacts_sha256": hashes,
        "remote_evidence_paths": {key: str(tmp_path / "remote" / key)
                                  for key in (*hashes, "plan")},
        "readonly_verification_path": str(audit_path),
        "readonly_verification_sha256": digest_file(audit_path),
        "attestor": "synthetic-test-publisher",
        "published_utc": "2026-10-25T12:01:00Z",
        "final_recheck": {"hash_mismatches": 0,
                          "saved_supervisor_success": True,
                          "saved_supervisor_cleanup_verified": True,
                          "scope_inactive": True,
                          "assigned_gpu_compute_contexts": 0,
                          "r7_administrator_gate_resolved": False},
    }


def _options(tmp_path):
    image = tmp_path / "slide.tif"
    tifffile.imwrite(image, np.zeros((16, 16, 3), np.uint8), photometric="rgb")
    shift = tmp_path / "shift.json"
    shift.write_text('{"source_mpp":0.25}')
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"checked-mask")
    cache = tmp_path / "classifiers"
    cache.mkdir()
    return action.parser().parse_args([
        "--action", "plan", "--image", str(image), "--shift", str(shift),
        "--clean-tissue-mask", str(mask), "--bundle-dir", str(tmp_path / "bundle"),
        "--classifier-cache", str(cache), "--python", sys.executable,
        "--reserve-gib", "0", "--case-budget-gib", "0.00001", "--amp",
    ])


def test_plan_is_cpu_only_and_incomplete_shard_cannot_advance(tmp_path, monkeypatch):
    options = _options(tmp_path)
    monkeypatch.setattr(action, "preflight_models", lambda *_: [
        {"model_id": name, "status": "available"} for name in action.HEADS])
    receipt = action.run_action(options)
    assert receipt["gpu_started"] is False
    assert receipt["shard_count"] == 1
    bundle = options.bundle_dir
    plan = json.loads((bundle / "cellvit_shard_plan.json").read_text())
    options.action = "shard"
    options.shard_id = plan["shards"][0]["id"]
    with pytest.raises(RuntimeError, match="lifecycle supervisor"):
        action.run_action(options)
    assert not (bundle / action.ACTIVE_FILE).exists()
    assert not (bundle / "shards").exists()
    frozen_adapter = tmp_path / "one_case_chiamaka_adapter.py"
    frozen_adapter.write_text("#!/bin/sh\nexit 0\n")
    frozen_adapter.chmod(0o755)
    options.supervisor = frozen_adapter
    with pytest.raises(RuntimeError, match="No compatible direct-supervisor"):
        action.run_action(options)
    assert not (bundle / action.ACTIVE_FILE).exists()
    assert not (bundle / action.FAILURE_FILE).exists()
    action.atomic_json(bundle / action.ACTIVE_FILE, {
        "plan_sha256": plan["identity_sha256"], "shard_id": options.shard_id})
    options.action = "finalize"
    with pytest.raises(RuntimeError, match="cleanup"):
        action.run_action(options)
    assert not (bundle / "experiment.cellphenotyper.json").exists()


def test_disk_admission_fails_before_model_or_gpu(tmp_path, monkeypatch):
    options = _options(tmp_path)
    options.reserve_gib = 10**9
    monkeypatch.setattr(action, "preflight_models", lambda *_: [
        {"model_id": name, "status": "available"} for name in action.HEADS])
    with pytest.raises(RuntimeError, match="disk admission failed"):
        action.run_action(options)
    assert not (options.bundle_dir / "cellvit_shard_plan.json").exists()


def test_pm016_plan_cannot_be_reduced_to_completed_pilots():
    image = {"sha256": action.PM016_IMAGE_SHA256, "width": 23904,
             "height": 14762}
    plan = {"image": image, "core_size": 1920, "halo": 960,
            "run_contract": {"model": "SAM", "amp": True,
                             "classification_device": "cpu",
                             "requested_taxonomies": ["all"]},
            "shards": [{"id": f"core-{i}"} for i in range(104)]}
    action._require_declared_coverage(plan, 104)
    with pytest.raises(ValueError, match="expected core count"):
        action._require_declared_coverage(plan, 2)
    plan["shards"] = plan["shards"][:2]
    with pytest.raises(ValueError, match="complete 104-core"):
        action._require_declared_coverage(plan, 0)
    plan["shards"] = [{"id": f"core-{i}"} for i in range(104)]
    plan["run_contract"]["requested_taxonomies"] = ["pannuke"]
    with pytest.raises(ValueError, match="eight-head"):
        action._require_declared_coverage(plan, 104)


def test_nextflow_rejects_implicit_gpu_shard(tmp_path):
    if not __import__("shutil").which("nextflow"):
        pytest.skip("Nextflow unavailable")
    args = ["nextflow", "run", str(ROOT / "cellvit_fullslide.nf"), "-preview",
            "--cellvit_fullslide_action", "shard",
            "--cellvit_fullslide_image", str(tmp_path / "image.tif"),
            "--cellvit_fullslide_shift", str(tmp_path / "shift.json"),
            "--cellvit_fullslide_clean_mask", str(tmp_path / "mask.png"),
            "--cellvit_fullslide_bundle", str(tmp_path / "bundle"),
            "--cellvit_classifier_cache", str(tmp_path / "classifiers")]
    result = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True,
                            timeout=60, env={**__import__("os").environ,
                                             "NXF_OFFLINE": "true", "NXF_ANSI_LOG": "false"})
    assert result.returncode != 0
    assert "No compatible direct GPU supervisor" in result.stdout + result.stderr


def test_adopt_checked_pilot_into_one_canonical_bundle(tmp_path, monkeypatch):
    from test_cellvit_archive_consumer import _bundle
    from cellvit_models import digest_file
    from cellvit_shards import atomic_json, compact_shard, verify_complete_shard

    original, image, plan, shard, source = _bundle(tmp_path, "cellvit_cell_classification.csv.gz")
    monkeypatch.setattr(action, "_require_declared_coverage", lambda *_: None)
    compact_shard(plan, shard, source)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    atomic_json(canonical / "cellvit_shard_plan.json", plan)
    checkpoint_hash = digest_file(source / "complete.json")
    attestation = tmp_path / "pilot-terminal-cleanup.json"
    atomic_json(attestation, _pilot_attestation(
        tmp_path, canonical, plan, shard, source / "complete.json"))
    mask = tmp_path / "unused-mask.png"
    mask.write_bytes(b"mask")
    cache = tmp_path / "cache"
    cache.mkdir()
    options = action.parser().parse_args([
        "--action", "adopt", "--image", str(image),
        "--shift", str(original / "shift.json"),
        "--clean-tissue-mask", str(mask), "--bundle-dir", str(canonical),
        "--classifier-cache", str(cache), "--shard-id", shard["id"],
        "--adopt-from", str(source),
        "--expected-checkpoint-sha256", checkpoint_hash,
        "--adoption-cleanup-receipt", str(attestation),
        "--adoption-cleanup-receipt-sha256", digest_file(attestation),
        "--reserve-gib", "0", "--case-budget-gib", "0.00001",
    ])
    receipt = action.run_action(options)
    adopted = canonical / "shards" / shard["id"]
    assert receipt["source_preserved"] and not receipt["gpu_started"]
    assert verify_complete_shard(plan, shard, adopted)["cell_count"] == 1
    assert digest_file(adopted / "complete.json") == checkpoint_hash
    assert (source / "cellvit_records.tar.gz").is_file()
    with pytest.raises(FileExistsError, match="new canonical shard"):
        action.run_action(options)


@pytest.mark.parametrize("cleanup_verified,draft,match", [
    (False, False, "terminal cleanup"),
    (True, True, "draft or unpublished"),
])
def test_adoption_rejects_unverified_cleanup_before_copy(
        tmp_path, monkeypatch, cleanup_verified, draft, match):
    from test_cellvit_archive_consumer import _bundle
    from cellvit_models import digest_file
    from cellvit_shards import atomic_json

    original, image, plan, shard, source = _bundle(tmp_path, "cellvit_cell_classification.csv.gz")
    monkeypatch.setattr(action, "_require_declared_coverage", lambda *_: None)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    atomic_json(canonical / "cellvit_shard_plan.json", plan)
    attestation = tmp_path / "unsafe-cleanup.json"
    claim = _pilot_attestation(tmp_path, canonical, plan, shard,
                               source / "complete.json")
    claim["draft"] = draft
    claim["cleanup_verified"] = cleanup_verified
    atomic_json(attestation, claim)
    mask = tmp_path / "unused-mask.png"
    mask.write_bytes(b"mask")
    cache = tmp_path / "cache"
    cache.mkdir()
    options = action.parser().parse_args([
        "--action", "adopt", "--image", str(image),
        "--shift", str(original / "shift.json"),
        "--clean-tissue-mask", str(mask), "--bundle-dir", str(canonical),
        "--classifier-cache", str(cache), "--shard-id", shard["id"],
        "--adopt-from", str(source),
        "--expected-checkpoint-sha256", digest_file(source / "complete.json"),
        "--adoption-cleanup-receipt", str(attestation),
        "--adoption-cleanup-receipt-sha256", digest_file(attestation),
        "--reserve-gib", "0", "--case-budget-gib", "0.00001",
    ])
    with pytest.raises(ValueError, match=match):
        action.run_action(options)
    assert not (canonical / "shards").exists()


def test_published_pilot_requires_pinned_audit_plan_paths_and_publisher(tmp_path):
    from test_cellvit_archive_consumer import _bundle
    from cellvit_shards import atomic_json

    _, _, plan, shard, source = _bundle(tmp_path, "cellvit_cell_classification.csv.gz")
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    atomic_json(canonical / "cellvit_shard_plan.json", plan)
    marker = source / "complete.json"
    valid = _pilot_attestation(tmp_path, canonical, plan, shard, marker)
    action._published_pilot_provenance(canonical, plan, shard, valid,
                                       valid["checkpoint_sha256"])
    for field, changed in (
            ("attestor", ""),
            ("plan_file_sha256", "0" * 64),
            ("remote_evidence_paths", {}),
            ("readonly_verification_sha256", "0" * 64),
            ("source_artifacts_sha256", {**valid["source_artifacts_sha256"],
                                         "supervisor_result": "0" * 64})):
        receipt = {**valid, field: changed}
        with pytest.raises(ValueError, match="pilot|verification"):
            action._published_pilot_provenance(canonical, plan, shard, receipt,
                                               valid["checkpoint_sha256"])


def test_finalize_resumes_checked_manifest_and_index_after_interruption(tmp_path, monkeypatch):
    """Synthetic two-core fixture exercises CPU recovery, not SAM inference."""
    from test_cellvit_partition_project import _source
    from cellvit_models import digest_file
    from cellvit_shards import atomic_json
    from index_cellvit_partitions import build_partition_index

    root, image, manifest = _source(tmp_path)
    plan = json.loads((root / "cellvit_shard_plan.json").read_text())
    for shard in plan["shards"]:
        marker = root / "shards" / shard["id"] / "complete.json"
        atomic_json(root / "shards" / shard["id"] / action.CLEANUP_FILE, {
            "schema": "cellvit-bound-case-cleanup-v1",
            "attestation_state": "published", "draft": False,
            "shard_id": shard["id"], "plan_sha256": plan["identity_sha256"],
            "checkpoint_sha256": digest_file(marker), "cleanup_verified": True,
            "owned_processes_absent": True, "scope_inactive": True,
            "assigned_gpu_contexts_empty": True, "bounded_checks_complete": True,
        })
    index = root / "cellvit_cells.sqlite"
    build_partition_index(manifest, image, index)
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"unused")
    cache = tmp_path / "classifiers"
    cache.mkdir()
    options = action.parser().parse_args([
        "--action", "finalize", "--image", str(image),
        "--shift", str(root / "shift.json"), "--clean-tissue-mask", str(mask),
        "--bundle-dir", str(root), "--classifier-cache", str(cache),
        "--reserve-gib", "0", "--case-budget-gib", "0.00001",
    ])
    monkeypatch.setattr(action, "_require_declared_coverage", lambda *_: None)
    receipt = action.run_action(options)
    assert receipt["complete_shards"] == len(plan["shards"])
    assert (root / "experiment.cellphenotyper.json").is_file()
    assert action.run_action(options) == receipt


def test_finalize_rejects_existing_image_only_project_without_overwriting(tmp_path, monkeypatch):
    from test_cellvit_partition_project import _source
    from cellvit_models import digest_file
    from cellvit_shards import atomic_json
    from export_experiment import export_experiment

    root, image, manifest = _source(tmp_path)
    plan = json.loads((root / 'cellvit_shard_plan.json').read_text())
    for shard in plan['shards']:
        folder = root / 'shards' / shard['id']
        atomic_json(folder / action.CLEANUP_FILE, {
            'schema': 'cellvit-bound-case-cleanup-v1', 'attestation_state': 'published',
            'draft': False, 'shard_id': shard['id'],
            'plan_sha256': plan['identity_sha256'],
            'checkpoint_sha256': digest_file(folder / 'complete.json'),
            'cleanup_verified': True, 'owned_processes_absent': True,
            'scope_inactive': True, 'assigned_gpu_contexts_empty': True,
            'bounded_checks_complete': True})
    project_path = root / 'experiment.cellphenotyper.json'
    export_experiment({'name': 'Existing image-only project', 'tissues': [{
        'tissue_id': 'other', 'name': 'Other',
        'image': {'path': str(image), 'mpp_x': .25, 'mpp_y': .25},
        'artifacts': [], 'feature_sets': []}]}, project_path, base_dir=root)
    previous = project_path.read_bytes()
    options = action.parser().parse_args([
        '--action', 'finalize', '--image', str(image),
        '--shift', str(root / 'shift.json'), '--clean-tissue-mask', str(image),
        '--bundle-dir', str(root), '--classifier-cache', str(root),
        '--reserve-gib', '0', '--case-budget-gib', '0.00001'])
    monkeypatch.setattr(action, '_require_declared_coverage', lambda *_: None)
    with pytest.raises(ValueError, match='Existing CellViT project'):
        action.run_action(options)
    assert project_path.read_bytes() == previous
    assert not (root / 'CELLVIT_PROJECT_COMPLETE.json').exists()


def test_nextflow_previews_full_run_without_starting_any_case(tmp_path):
    if not __import__('shutil').which('nextflow'):
        pytest.skip('Nextflow unavailable')
    result = subprocess.run([
        'nextflow', 'run', str(ROOT / 'cellvit_fullslide.nf'), '-preview',
        '--cellvit_fullslide_action', 'run',
        '--cellvit_fullslide_image', str(tmp_path / 'image.tif'),
        '--cellvit_fullslide_shift', str(tmp_path / 'shift.json'),
        '--cellvit_fullslide_clean_mask', str(tmp_path / 'mask.png'),
        '--cellvit_fullslide_bundle', str(tmp_path / 'never-started'),
        '--cellvit_classifier_cache', str(tmp_path / 'classifiers'),
        '--cellvit_fullslide_gpu_uuid', 'GPU-00000000-0000-0000-0000-000000000000',
        '--cellvit_fullslide_runtime_sif', str(tmp_path / 'pinned.sif'),
        '--cellvit_fullslide_runtime_sif_sha256', 'a'*64,
        '--cellvit_fullslide_runtime_python', '/runtime/python',
        '--cellvit_fullslide_runtime_executable', '/runtime/cellvit-inference',
        '--cellvit_fullslide_expected_shards', '104'],
        cwd=tmp_path, text=True, capture_output=True, timeout=60,
        env={**__import__('os').environ, 'NXF_OFFLINE': 'true', 'NXF_ANSI_LOG': 'false'})
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / 'never-started').exists()
