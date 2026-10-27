"""The global Bora handoff cannot silently accept a mixed KODAMA fit or partial output."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from bin.bora_fullslide_action import (COLON_HE_ALGORITHM, GENERALIZED_ALGORITHM,
                                       REQUIRED_OUTPUTS, prepare, sha256, verify)


ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    csv = tmp_path / "groups.csv"
    csv.write_text("id,group\n1,1\n2,2\n")
    receipt = tmp_path / "direct-fit.json"
    receipt.write_text(json.dumps({"native_fit_reused": True, "posthoc_merge": False,
                                   "requested_groups": 2, "obtained_groups": 2,
                                   "csv_sha256": sha256(csv)}))
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({
        "format": "cellphenotyper-dual-mpp-registration-v1", "verified": True,
        "image": {"frame_id": "frame", "sha256": "b" * 64,
                  "width": 4, "height": 4, "mpp_x": .5, "mpp_y": .5},
        "cell_full_image": {"frame_id": "native-frame", "sha256": "1" * 64,
                            "width": 8, "height": 8, "mpp_x": .25, "mpp_y": .25}}))
    manifest = tmp_path / "manifest.json"
    spec = {"schema": "bora-wsi-global-v1", "sample_id": "PM016", "frame_id": "frame",
            "algorithm_id": GENERALIZED_ALGORITHM,
            "identity_status": "exploratory_transfer", "evidence_class": "verified_gpu_inventory",
            "expected": {"scale_factors": [1, 2, 4], "case_count": 3, "group_count": 2,
                         "inventory_sha256": "a" * 64, "image_sha256": "b" * 64,
                         "classifier_sha256": "d" * 64},
            "inputs": {"cluster_receipt": {"path": str(receipt), "sha256": sha256(receipt)},
                       "cluster_csv": {"path": str(csv), "sha256": sha256(csv)},
                       "classifier_model": {"sha256": "d" * 64},
                       "registration": {"path": str(registration), "sha256": sha256(registration)},
                       "support_mask": {"sha256": "f" * 64}},
            "limits": {"memory_budget_bytes": 1024 ** 3}}
    manifest.write_text(json.dumps(spec))
    return manifest, tmp_path / "new-output", spec


def test_hard_group_algorithm_cannot_be_implicit(tmp_path: Path):
    manifest, output, spec = fixture(tmp_path)
    spec.pop("algorithm_id")
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="implicit default"):
        prepare(manifest, output)


def test_nextflow_routes_require_explicit_legacy_intent():
    fullslide = (ROOT / "bora_fullslide.nf").read_text()
    main = (ROOT / "main.nf").read_text()
    assert "bora_fullslide_algorithm" in fullslide
    assert "manifest.algorithm_id != algorithm" in fullslide
    assert "legacy_per_window_kodama_mask_v1" in main
    assert "corrected semantic H&E-refined Bora" in main


def test_direct_fit_and_three_scales_required(tmp_path: Path):
    manifest, output, spec = fixture(tmp_path)
    assert prepare(manifest, output, required_group_count=2)["sample_id"] == "PM016"
    with pytest.raises(ValueError, match="group count"):
        prepare(manifest, output, required_group_count=3)
    spec["expected"]["scale_factors"] = [1]
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="three-scale"):
        prepare(manifest, output)


def test_mixed_or_merged_kodama_rejected(tmp_path: Path):
    manifest, output, spec = fixture(tmp_path)
    receipt = Path(spec["inputs"]["cluster_receipt"]["path"])
    data = json.loads(receipt.read_text())
    data["posthoc_merge"] = True
    receipt.write_text(json.dumps(data))
    spec["inputs"]["cluster_receipt"]["sha256"] = sha256(receipt)
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="direct fit"):
        prepare(manifest, output)


def test_exact_vector_parity_and_completion_required(tmp_path: Path):
    manifest, output, spec = fixture(tmp_path)
    for relative in REQUIRED_OUTPUTS:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
    native = {"native_shape_yx": [8, 8], "native_mpp_xy": [.25, .25],
              "source_frame_id": "native-frame", "source_image_sha256": "1" * 64}
    geo = {"type": "FeatureCollection", "coordinate_space": "native_image_pixel_edges",
           "origin": "top_left", **native,
           "source_semantic_mask_sha256": sha256(output / "semantic/semantic-mask.uint8.dat"),
           "global_model_binding_sha256": "c" * 64, "features": []}
    for role in ("exact", "review"):
        (output / f"vectors/{role}/mask.geojson").write_text(json.dumps(geo, separators=(",", ":")))
    vector_hashes = {name: sha256(output / "vectors" / name) for name in
                     ("exact/mask.uint8.dat", "exact/mask.geojson", "review/mask.geojson")}
    semantic_hash = sha256(output / "semantic/semantic-mask.uint8.dat")
    assembly = {"format": "bora-global-group-materialization-v1",
                **native,
                "prediction_unit": "global_anonymous_Bora-AF_group",
                "probability_shards_persisted": False,
                "global_fit_count_declared": 1, "window_fit_count_declared": 0,
                "independently_fitted_windows": False,
                "working_frame_id": "frame", "working_image_sha256": "b" * 64,
                "semantic_mask_sha256": semantic_hash,
                "global_model_binding_sha256": "c" * 64,
                "tissue_identity_status": "exploratory_transfer",
                "source_sha256": {
                    "model": "d" * 64, "registration": spec["inputs"]["registration"]["sha256"],
                    "support": "f" * 64,
                    "global_group_map": sha256(output / "af/readout/group_map.npy"),
                    "global_appearance_fit": sha256(output / "af/readout/APPEARANCE_FIT_COMPLETE.json"),
                    "global_descriptors": sha256(output / "descriptors.npz")}}
    assembly_path = output / "semantic/ASSEMBLY_COMPLETE.json"
    assembly_path.write_text(json.dumps(assembly))
    vector = {"minimum_area_um2": 100.0, "exact_vector_matches_cleaned_mask": True,
              **native,
              "source_semantic_mask_sha256": semantic_hash,
              "global_model_binding_sha256": "c" * 64,
              "output_sha256": vector_hashes}
    vector_path = output / "vectors/GLOBAL_VECTOR_COMPLETE.json"
    vector_path.write_text(json.dumps(vector))
    (output / "ADMISSION.json").write_text(json.dumps({"inventory_sha256": "a" * 64}))
    chain = {"schema": "bora-wsi-global-chain-result-v1", "complete": True,
             "manifest_sha256": sha256(manifest), "inventory_sha256": "a" * 64,
             "sample_id": "PM016", "frame_id": "frame",
             "evidence_class": "verified_gpu_inventory", "gpu_execution_qualified": True,
             "tissue_identity_status": "exploratory_transfer",
             "exact_100_um2_floor": True, "exact_raster_parity": True,
             "vector_receipt_sha256": sha256(vector_path), "output_sha256": vector_hashes}
    (output / "CHAIN_COMPLETE.json").write_text(json.dumps(chain))
    result = verify(manifest, output, spec)
    assert result["complete"] is True
    assert result["verified_algorithm_id"] == GENERALIZED_ALGORITHM
    assert result["algorithm_evidence_receipt_sha256"] == sha256(assembly_path)
    vector["source_frame_id"] = "frame"
    vector_path.write_text(json.dumps(vector))
    chain["vector_receipt_sha256"] = sha256(vector_path)
    (output / "CHAIN_COMPLETE.json").write_text(json.dumps(chain))
    with pytest.raises(ValueError, match="native frame"):
        verify(manifest, output, spec)
    vector["source_frame_id"] = "native-frame"
    vector_path.write_text(json.dumps(vector))
    chain["vector_receipt_sha256"] = sha256(vector_path)
    (output / "CHAIN_COMPLETE.json").write_text(json.dumps(chain))
    assembly["prediction_unit"] = "native_H&E_guided_pixel"
    assembly_path.write_text(json.dumps(assembly))
    with pytest.raises(ValueError, match="hard-group prediction"):
        verify(manifest, output, spec)
    assembly["prediction_unit"] = "global_anonymous_Bora-AF_group"
    assembly_path.write_text(json.dumps(assembly))
    assembly["source_sha256"]["model"] = "0" * 64
    assembly_path.write_text(json.dumps(assembly))
    with pytest.raises(ValueError, match="hard-group prediction"):
        verify(manifest, output, spec)
    assembly["source_sha256"]["model"] = "d" * 64
    assembly_path.write_text(json.dumps(assembly))
    (output / "vectors/exact/mask.geojson").write_bytes(b"different")
    with pytest.raises(ValueError, match="vector hash"):
        verify(manifest, output, spec)


def test_colon_he_request_rejected_before_launch_or_publication(tmp_path: Path):
    manifest, output, spec = fixture(tmp_path)
    with pytest.raises(ValueError, match="colon H&E-guided Bora method"):
        prepare(manifest, output, requested_algorithm=COLON_HE_ALGORITHM)
    spec["algorithm_id"] = COLON_HE_ALGORITHM
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="colon H&E-guided Bora method"):
        prepare(manifest, output)
    with pytest.raises(ValueError, match="colon H&E-guided Bora method"):
        verify(manifest, output, spec)


def test_no_overwrite_or_fixture_gpu_claim(tmp_path: Path):
    manifest, output, spec = fixture(tmp_path)
    spec["evidence_class"] = "cpu_synthetic_fixture"
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="verified GPU"):
        prepare(manifest, output)
    spec["evidence_class"] = "verified_gpu_inventory"
    manifest.write_text(json.dumps(spec))
    output.mkdir()
    with pytest.raises(ValueError, match="no overwrite"):
        prepare(manifest, output)


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
def test_nextflow_dag_is_one_cpu_stub_without_gpu_successor(tmp_path: Path):
    manifest, output, _ = fixture(tmp_path)
    sif = tmp_path / "cpu.sif"
    sif.write_bytes(b"stub-only")
    command = ["nextflow", "run", str(ROOT / "bora_fullslide.nf"), "-stub-run",
               "--bora_fullslide_algorithm", GENERALIZED_ALGORITHM,
               "--bora_fullslide_manifest", str(manifest),
               "--bora_fullslide_output", str(output),
               "--bora_fullslide_cpu_sif", str(sif),
               "--bora_fullslide_memory_gb", "2", "--compute_device", "cpu",
               "-work-dir", str(tmp_path / "nf-work")]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True,
                            timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Submitted process > BORA_GLOBAL_FULLSLIDE") == 1
    assert "GPU" not in result.stdout
    assert not output.exists(), "A stub must not masquerade as completed Bora output"
