"""The Nextflow route selects only evaluated token-local H&E-refined Bora."""

from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _invocation(tmp_path: Path) -> list[str]:
    source = tmp_path / "reviewed-bora"
    package = source / "src/bora"
    package.mkdir(parents=True)
    for name in ("wsi_semantic_refined_chain.py", "wsi_semantic_probability.py"):
        (package / name).write_text("# stub admission only\n")
    evaluation = tmp_path / "expert-evaluation"
    evaluation.mkdir()
    roi_rows = [("train", 1, "s-train", 100), ("train", 2, "t-train", 100),
                ("holdout", 1, "s-a", 80), ("holdout", 1, "s-b", 80),
                ("holdout", 2, "t-a", 80), ("holdout", 2, "t-b", 80)]
    (evaluation / "EVALUATION.json").write_text(json.dumps({
        "complete": True,
        "status": "same_slide_holdout_passed_independent_patient_validation_pending",
        "same_slide_holdout": True,
        "annotation_source": "declared_independent_expert",
        "class_ids": [1, 2], "class_names": {"1": "stroma", "2": "tumor"},
        "confusion": [[150, 10], [10, 150]], "holdout_tokens": 320,
        "token_shape_yx": [25, 25], "macro_iou": 150 / 170,
        "per_polygon": [{"split": split, "class_id": cid, "roi_id": roi,
            "retained_supported_interior_tokens": count}
            for split, cid, roi, count in roi_rows],
        "per_class": {
            "1": {"name": "stroma", "support": 160,
                "holdout_roi_counts": {"s-a": 80, "s-b": 80},
                "precision": 150 / 160, "recall": 150 / 160, "iou": 150 / 170},
            "2": {"name": "tumor", "support": 160,
                "holdout_roi_counts": {"t-a": 80, "t-b": 80},
                "precision": 150 / 160, "recall": 150 / 160, "iou": 150 / 170}},
        "gates": {key: True for key in ("holdout_material", "macro_iou",
            "per_class_iou", "per_class_recall", "no_single_class_collapse")},
        "gate_thresholds": {"holdout_tokens_per_class": 100,
            "holdout_rois_per_class": 2, "macro_iou": .7,
            "class_iou": .6, "class_recall": .7},
    }) + "\n")
    (evaluation / "local_semantic_model.npz").write_bytes(b"stub admission only")
    materialization = tmp_path / "materialized-tokens"
    materialization.mkdir()
    (materialization / "MATERIALIZATION_COMPLETE.json").write_text("{}\n")
    for name in ("roi-spec.json", "registration.json", "support.tif", "base-project.json"):
        (tmp_path / name).write_bytes(b"stub admission only")
    return ["nextflow", "run", str(ROOT / "bora_refined_semantic_fullslide.nf"),
            "-stub-run", "--bora_refined_algorithm",
            "bora-wsi-native-he-refined-semantic-chain-v1", "--compute_device", "cpu",
            "--bora_refined_roi_spec", str(tmp_path / "roi-spec.json"),
            "--bora_refined_evaluation_folder", str(evaluation),
            "--bora_refined_materialization_folder", str(materialization),
            "--bora_refined_registration", str(tmp_path / "registration.json"),
            "--bora_refined_support_tiff", str(tmp_path / "support.tif"),
            "--bora_refined_base_project", str(tmp_path / "base-project.json"),
            "--bora_refined_source", str(source),
            "--bora_refined_source_sha256", "a" * 64,
            "--bora_refined_runtime_sha256", "b" * 64,
            "--bora_refined_output", str(tmp_path / "new-refined-result"),
            "-work-dir", str(tmp_path / "work")]


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, NXF_OFFLINE="true", NXF_ANSI_LOG="false",
               NXF_OPTS="-Xms64m -Xmx512m")
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=90, check=False)


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
def test_refined_route_stub_is_one_cpu_process_and_never_creates_accepted_result(tmp_path: Path):
    run = _run(_invocation(tmp_path), tmp_path)
    output = run.stdout + run.stderr
    assert run.returncode == 0, output
    assert output.count("Submitted process > BORA_REFINED_SEMANTIC_FULLSLIDE_CPU") == 1
    assert "BORA_GLOBAL_FULLSLIDE" not in output
    assert "REFINE_VIRCHOW2_BORA" not in output
    assert not (tmp_path / "new-refined-result").exists()


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
def test_legacy_algorithm_cannot_fall_back_into_refined_route(tmp_path: Path):
    args = _invocation(tmp_path)
    slot = args.index("bora-wsi-native-he-refined-semantic-chain-v1")
    args[slot] = "bora-global-hard-group-v1"
    run = _run(args, tmp_path)
    assert run.returncode != 0
    assert "legacy hard-group route cannot substitute" in run.stdout + run.stderr
    assert not (tmp_path / "new-refined-result").exists()


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
def test_missing_expert_holdout_cannot_launch_refined_process(tmp_path: Path):
    args = _invocation(tmp_path)
    (tmp_path / "expert-evaluation/EVALUATION.json").unlink()
    run = _run(args, tmp_path)
    assert run.returncode != 0
    assert "same-slide holdout evaluation" in run.stdout + run.stderr
    assert "Submitted process" not in run.stdout + run.stderr
    assert not (tmp_path / "new-refined-result").exists()


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
def test_failed_holdout_cannot_launch_refined_process(tmp_path: Path):
    args = _invocation(tmp_path)
    evaluation = tmp_path / "expert-evaluation/EVALUATION.json"
    report = json.loads(evaluation.read_text())
    report["status"] = "same_slide_holdout_failed"
    report["gates"]["no_single_class_collapse"] = False
    evaluation.write_text(json.dumps(report) + "\n")
    run = _run(args, tmp_path)
    assert run.returncode != 0
    assert "unsupported PM016 or transferred labels" in run.stdout + run.stderr
    assert "Submitted process" not in run.stdout + run.stderr
    assert not (tmp_path / "new-refined-result").exists()


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
@pytest.mark.parametrize("tamper", ["missing_confusion", "missing_per_class",
                                     "forged_macro_iou", "stale_roi_count",
                                     "nonfinite_macro_iou"])
def test_inconsistent_holdout_counts_fail_nextflow_preflight(tmp_path: Path, tamper: str):
    args = _invocation(tmp_path)
    evaluation = tmp_path / "expert-evaluation/EVALUATION.json"
    report = json.loads(evaluation.read_text())
    if tamper == "missing_confusion":
        del report["confusion"]
    elif tamper == "missing_per_class":
        del report["per_class"]
    elif tamper == "forged_macro_iou":
        report["macro_iou"] = .99
    elif tamper == "stale_roi_count":
        report["per_class"]["2"]["holdout_roi_counts"]["t-a"] += 1
    elif tamper == "nonfinite_macro_iou":
        report["macro_iou"] = float("nan")
    evaluation.write_text(json.dumps(report) + "\n")
    run = _run(args, tmp_path)
    assert run.returncode != 0
    assert "expert holdout" in (run.stdout + run.stderr).lower()
    assert "Submitted process" not in run.stdout + run.stderr
    assert not (tmp_path / "new-refined-result").exists()


def test_real_controller_holdout_verifier_rejects_unsupported_tissue_labels(tmp_path: Path):
    # The Nextflow preflight is a fast UX guard. The controller uses this
    # source/model/registration verifier again before any output is created.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_export_bora_semantic_project import _fixture, SOURCE_ROOT
    from bin.export_bora_semantic_project import _evaluation

    item = _fixture(tmp_path, regional_signal=True)
    evaluation = item["evaluation"] / "EVALUATION.json"
    report = json.loads(evaluation.read_text())
    report["status"] = "same_slide_holdout_failed"
    report["gates"]["no_single_class_collapse"] = False
    evaluation.write_text(json.dumps(report, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="Same-slide expert evaluation"):
        _evaluation(item["spec"], item["evaluation"], SOURCE_ROOT,
                    item["registration"])


def test_refined_runtime_identity_pins_semantic_solver_and_tiff_modules():
    from bin import bora_refined_semantic_fullslide_action as action
    identity = action._semantic_dependency_identity()
    assert set(identity) == {"scikit-learn", "tifffile", "threadpoolctl", "joblib"}
    for record in identity.values():
        assert record["version"]
        for module in record["modules"].values():
            path = Path(module["path"])
            assert path.is_file()
            assert action.sha256(path) == module["sha256"]


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
def test_main_workflow_refuses_refined_inputs_instead_of_using_legacy_bora(tmp_path: Path):
    command = ["nextflow", "run", str(ROOT / "main.nf"), "-stub-run",
               "--bora_refined_algorithm", "bora-wsi-native-he-refined-semantic-chain-v1",
               "--outdir_base", str(tmp_path / "not-created"),
               "-work-dir", str(tmp_path / "work")]
    run = _run(command, tmp_path)
    assert run.returncode != 0
    assert "Run bora_refined_semantic_fullslide.nf" in run.stdout + run.stderr
    assert "Submitted process" not in run.stdout + run.stderr
    assert not list((tmp_path / "not-created").rglob("RUN_COMPLETE.json"))
    assert not list((tmp_path / "not-created").rglob("experiment.cellphenotyper.json"))
