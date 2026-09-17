import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from build_model_inventory import build_inventory, immutable_revision, write_inventory  # noqa: E402
from model_provenance import hf_model_provenance, model_bundle_records  # noqa: E402


def write_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "component_id": "complete_model",
                        "stage_id": "example",
                        "display_name": "Complete",
                        "source_repository": "https://example.org/model",
                        "artifact_source": "https://example.org/release/1",
                        "license": "Apache-2.0",
                        "license_status": "verified_upstream",
                        "training_domain": "histology",
                        "metadata_globs": ["stage/**/metadata.json"],
                        "stage_presence_globs": ["stage/**/prediction.csv"],
                    },
                    {
                        "component_id": "missing_metadata",
                        "stage_id": "missing",
                        "display_name": "Missing metadata",
                        "source_repository": "https://example.org/missing",
                        "artifact_source": None,
                        "license": "MIT",
                        "license_status": "verified_upstream",
                        "training_domain": "histology",
                        "metadata_globs": ["missing/**/metadata.json"],
                        "stage_presence_globs": ["missing/**/prediction.csv"],
                    },
                    {
                        "component_id": "not_run",
                        "stage_id": "unused",
                        "display_name": "Unused",
                        "source_repository": "https://example.org/unused",
                        "artifact_source": None,
                        "license": "MIT",
                        "license_status": "verified_upstream",
                        "training_domain": "histology",
                        "metadata_globs": ["unused/**/metadata.json"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_inventory_distinguishes_complete_missing_metadata_and_not_run(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    write_registry(registry)
    complete = tmp_path / "stage" / "sample"
    complete.mkdir(parents=True)
    (complete / "prediction.csv").write_text("score\n1\n")
    (complete / "metadata.json").write_text(
        json.dumps(
            {
                "model_provenance": {
                    "source_repository": "https://example.org/model",
                    "requested_revision": "main",
                    "resolved_revision": "a" * 40,
                    "cache_path": "/models/example",
                    "checkpoints": [{"sha256": "b" * 64, "cache_path": "/models/example/model.pth"}],
                }
            }
        )
    )
    missing = tmp_path / "missing" / "sample"
    missing.mkdir(parents=True)
    (missing / "prediction.csv").write_text("score\n1\n")

    payload = build_inventory(tmp_path, registry)

    assert payload["status"] == "fail"
    assert payload["used_model_count"] == 2
    complete_row = next(row for row in payload["models"] if row["component_id"] == "complete_model")
    assert complete_row["status"] == "complete"
    assert complete_row["release_ready"] is True
    assert complete_row["resolved_revisions"] == ["a" * 40]
    assert complete_row["checkpoint_sha256"] == ["b" * 64]
    assert complete_row["cache_paths"] == ["/models/example"]
    missing_row = next(row for row in payload["models"] if row["component_id"] == "missing_metadata")
    assert missing_row["status"] == "partial"
    assert "runtime_model_metadata" in missing_row["missing_release_fields"]
    unused_row = next(row for row in payload["models"] if row["component_id"] == "not_run")
    assert unused_row["status"] == "not_run"


def test_inventory_writes_json_and_tsv(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    write_registry(registry)
    json_path, tsv_path, payload = write_inventory(tmp_path, registry)
    assert json_path.is_file()
    assert tsv_path.is_file()
    assert payload["status"] == "not_applicable"
    assert "component_id\tstage_id" in tsv_path.read_text()


def test_model_bundle_hashes_configuration_as_well_as_weights(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"rays": 32}')
    (model / "weights.h5").write_bytes(b"weights")
    rows = model_bundle_records(model)
    assert [row["logical_name"] for row in rows] == ["config.json", "weights.h5"]
    assert rows[1]["sha256"] == hashlib.sha256(b"weights").hexdigest()


def test_hf_provenance_uses_snapshot_commit_and_checkpoint_hash(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--org--name" / "snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"model")
    provenance = hf_model_provenance("org/name", snapshot)
    assert provenance["resolved_revision"] == "c" * 40
    assert provenance["cache_path"] == str(snapshot.resolve())
    assert provenance["checkpoints"][0]["sha256"] == hashlib.sha256(b"model").hexdigest()


def test_immutable_revision_rejects_mutable_branches() -> None:
    assert immutable_revision("main") is False
    assert immutable_revision("latest") is False
    assert immutable_revision("d" * 40) is True
    assert immutable_revision('sha256:' + 'a' * 64)
    assert not immutable_revision('sha256:main')
    assert not immutable_revision('sha256:' + 'a' * 63)


def test_hierarchy_uni2_is_included_in_real_model_registry(tmp_path):
    directory = tmp_path / '22_tissue_hierarchy/s1/features'
    directory.mkdir(parents=True)
    (directory / 'hierarchy_features_summary.json').write_text(json.dumps({'model_provenance': {
        'source_repository': 'https://huggingface.co/MahmoodLab/UNI2-h', 'requested_revision': 'local_snapshot',
        'resolved_revision': 'sha256:' + 'e' * 64, 'cache_path': '/local/snapshot',
        'checkpoints': [{'logical_name': 'model.safetensors', 'sha256': 'f' * 64}]}}))
    inventory = build_inventory(tmp_path, ROOT / 'resources/model_registry.json')
    model = next(row for row in inventory['models'] if row['component_id'] == 'uni2_h')
    assert model['used_in_run']
    assert model['instance_count'] == 1
    assert model['checkpoint_sha256'] == ['f' * 64]
    assert model['resolved_revisions'] == ['sha256:' + 'e' * 64]
    assert immutable_revision("zenodo:14041538") is True
    assert immutable_revision("release:stardist-models-v0.1") is True
    assert immutable_revision("package:cellvit==1.0.9") is True


def test_explicit_not_used_metadata_overrides_stub_prediction_presence(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    write_registry(registry)
    stage = tmp_path / "stage" / "sample"
    stage.mkdir(parents=True)
    (stage / "prediction.csv").write_text("score\n")
    (stage / "metadata.json").write_text(
        json.dumps({"model_provenance": {"used_model": False}}),
        encoding="utf-8",
    )
    payload = build_inventory(tmp_path, registry)
    row = next(row for row in payload["models"] if row["component_id"] == "complete_model")
    assert row["used_in_run"] is False
    assert row["status"] == "not_run"
