#!/usr/bin/env python3
"""Build a conservative run-level inventory of every learned model."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
MUTABLE_REVISIONS = {"", "main", "master", "latest", "head"}


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_checkpoint_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if "sha256" in value or "cache_path" in value:
            return [value]
        rows = []
        for name, item in value.items():
            if isinstance(item, dict):
                rows.append({"logical_name": name, **item})
        return rows
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def extract_runtime(component_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("model_provenance") or payload.get("pipeline_model_provenance") or {}
    if isinstance(provenance, dict) and component_id in provenance and isinstance(provenance[component_id], dict):
        provenance = provenance[component_id]
    if not isinstance(provenance, dict):
        provenance = {}
    runtime = {
        "source_repository": provenance.get("source_repository"),
        "requested_revision": provenance.get("requested_revision"),
        "resolved_revision": provenance.get("resolved_revision"),
        "cache_path": provenance.get("cache_path"),
        "checkpoints": _as_checkpoint_rows(provenance.get("checkpoints") or provenance.get("checkpoint")),
        "used_model": provenance.get("used_model", True),
        "license": provenance.get("license") or payload.get("license"),
        "training_domain": provenance.get("training_domain") or payload.get("training_domain"),
    }

    if component_id == "hovernet_monusac":
        runtime["resolved_revision"] = runtime["resolved_revision"] or payload.get("upstream_revision")
        digest = payload.get("checkpoint_sha256")
        if digest and not runtime["checkpoints"]:
            runtime["checkpoints"] = [{"logical_name": "MoNuSAC", "sha256": digest, "cache_path": payload.get("checkpoint_path")}]
    elif component_id == "titan":
        runtime["requested_revision"] = runtime["requested_revision"] or payload.get("revision")
        runtime["resolved_revision"] = runtime["resolved_revision"] or payload.get("resolved_revision")
        runtime["cache_path"] = runtime["cache_path"] or payload.get("local_model_snapshot")
        if not runtime["checkpoints"]:
            runtime["checkpoints"] = _as_checkpoint_rows(payload.get("checkpoints"))
    elif component_id in {"grandqc_tissue", "grandqc_artifact"}:
        key = "tissue" if component_id.endswith("tissue") else "artifact"
        nested = payload.get("model_provenance") or {}
        if isinstance(nested, dict) and isinstance(nested.get(key), dict):
            selected = nested[key]
            runtime["used_model"] = selected.get("used_model", runtime["used_model"])
            runtime["source_repository"] = selected.get("source_repository") or runtime["source_repository"]
            runtime["resolved_revision"] = selected.get("resolved_revision") or runtime["resolved_revision"]
            runtime["cache_path"] = selected.get("cache_path") or runtime["cache_path"]
            runtime["checkpoints"] = _as_checkpoint_rows(selected.get("checkpoints") or selected.get("checkpoint"))
    return runtime


def immutable_revision(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(
        COMMIT_RE.fullmatch(text)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", text)
        or text.startswith(("zenodo:", "release:", "package:"))
    )


def build_inventory(outdir: Path, registry_path: Path) -> dict[str, Any]:
    registry = read_json(registry_path)
    rows: list[dict[str, Any]] = []
    for static in registry.get("models") or []:
        component_id = str(static["component_id"])
        metadata_paths = sorted({
            path
            for pattern in static.get("metadata_globs") or []
            for path in outdir.glob(str(pattern))
            if path.is_file()
        })
        presence_paths = sorted({
            path
            for pattern in static.get("stage_presence_globs") or static.get("metadata_globs") or []
            for path in outdir.glob(str(pattern))
            if path.is_file()
        })
        instances = []
        explicitly_not_used = False
        for metadata_path in metadata_paths:
            payload = read_json(metadata_path)
            runtime = extract_runtime(component_id, payload)
            if runtime.get("used_model") is False:
                explicitly_not_used = True
                continue
            checkpoints = runtime.get("checkpoints") or []
            digests = sorted({str(row.get("sha256")) for row in checkpoints if row.get("sha256")})
            missing: list[str] = []
            source_repository = runtime.get("source_repository") or static.get("source_repository")
            license_value = runtime.get("license") or static.get("license")
            training_domain = runtime.get("training_domain") or static.get("training_domain")
            if not source_repository:
                missing.append("source_repository")
            if not immutable_revision(runtime.get("resolved_revision")):
                missing.append("immutable_resolved_revision")
            if not digests:
                missing.append("checkpoint_sha256")
            if not license_value:
                missing.append("license")
            if not training_domain:
                missing.append("training_domain")
            if not runtime.get("cache_path"):
                missing.append("cache_path")
            license_status = str(static.get("license_status") or "")
            if license_status.endswith("unverified"):
                missing.append("verified_checkpoint_license")
            if license_status.endswith("not_recorded") and not runtime.get("license"):
                missing.append("verified_checkpoint_license")
            if "conflict" in license_status:
                missing.append("resolved_license_conflict")
            instances.append({
                "metadata_path": str(metadata_path.absolute()),
                **runtime,
                "license": license_value,
                "training_domain": training_domain,
                "checkpoint_sha256": digests,
                "missing_release_fields": sorted(set(missing)),
                "release_ready": not missing,
            })
        used = bool(instances or (presence_paths and not explicitly_not_used))
        if used and not instances:
            missing = [
                "source_repository" if not static.get("source_repository") else None,
                "immutable_resolved_revision",
                "checkpoint_sha256",
                "license" if not static.get("license") else None,
                "training_domain" if not static.get("training_domain") else None,
                "cache_path",
                "runtime_model_metadata",
            ]
            instances.append({
                "metadata_path": None,
                "source_repository": static.get("source_repository"),
                "requested_revision": None,
                "resolved_revision": None,
                "cache_path": None,
                "checkpoints": [],
                "checkpoint_sha256": [],
                "missing_release_fields": sorted(field for field in missing if field),
                "release_ready": False,
                "presence_paths": [str(path.absolute()) for path in presence_paths],
            })
        missing_union = sorted({field for instance in instances for field in instance["missing_release_fields"]})
        status = "not_run" if not used else ("complete" if not missing_union else "partial")
        runtime_sources = sorted({
            str(instance.get("source_repository"))
            for instance in instances if instance.get("source_repository")
        })
        requested_revisions = sorted({
            str(instance.get("requested_revision"))
            for instance in instances if instance.get("requested_revision")
        })
        resolved_revisions = sorted({
            str(instance.get("resolved_revision"))
            for instance in instances if instance.get("resolved_revision")
        })
        checkpoint_digests = sorted({
            digest
            for instance in instances
            for digest in instance.get("checkpoint_sha256") or []
        })
        cache_paths = sorted({
            str(instance.get("cache_path"))
            for instance in instances if instance.get("cache_path")
        })
        runtime_licenses = sorted({
            str(instance.get("license"))
            for instance in instances if instance.get("license")
        })
        rows.append({
            **static,
            "used_in_run": used,
            "status": status,
            "release_ready": bool(used and status == "complete"),
            "instance_count": len(instances),
            "missing_release_fields": missing_union,
            "runtime_source_repositories": runtime_sources,
            "requested_revisions": requested_revisions,
            "resolved_revisions": resolved_revisions,
            "checkpoint_sha256": checkpoint_digests,
            "cache_paths": cache_paths,
            "runtime_licenses": runtime_licenses,
            "instances": instances,
        })
    used_rows = [row for row in rows if row["used_in_run"]]
    blockers = [
        {"component_id": row["component_id"], "missing_release_fields": row["missing_release_fields"]}
        for row in used_rows if not row["release_ready"]
    ]
    return {
        "schema_version": "1.0",
        "status": "not_applicable" if not used_rows else ("pass" if not blockers else "fail"),
        "release_ready": bool(used_rows and not blockers),
        "used_model_count": len(used_rows),
        "registered_model_count": len(rows),
        "release_blockers": blockers,
        "models": rows,
        "interpretation": (
            "A successful pipeline execution is not release-ready unless every used learned model has "
            "an immutable resolved revision, checkpoint digest, cache path, source, training domain and verified license."
        ),
    }


def write_inventory(outdir: Path, registry_path: Path, execution_dir: Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
    target = execution_dir or (outdir / "00_execution")
    target.mkdir(parents=True, exist_ok=True)
    payload = build_inventory(outdir, registry_path)
    json_path = target / "model_inventory.json"
    tsv_path = target / "model_inventory.tsv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "component_id", "stage_id", "display_name", "used_in_run", "status", "release_ready",
        "instance_count", "source_repository", "artifact_source", "license", "license_status",
        "training_domain", "missing_release_fields",
        "runtime_source_repositories", "requested_revisions", "resolved_revisions",
        "checkpoint_sha256", "cache_paths", "runtime_licenses",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in payload["models"]:
            writer.writerow({
                field: ";".join(str(item) for item in row.get(field, []))
                if isinstance(row.get(field), list) else row.get(field)
                for field in fields
            })
    return json_path, tsv_path, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--registry", default=str(Path(__file__).resolve().parents[1] / "resources" / "model_registry.json"))
    parser.add_argument("--execution-dir", default="")
    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    execution_dir = Path(args.execution_dir).resolve() if args.execution_dir else None
    json_path, tsv_path, payload = write_inventory(outdir, Path(args.registry), execution_dir)
    print(json.dumps({"status": payload["status"], "used_model_count": payload["used_model_count"], "json": str(json_path), "tsv": str(tsv_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
