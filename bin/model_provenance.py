#!/usr/bin/env python3
"""Small, dependency-free helpers for learned-model provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


WEIGHT_SUFFIXES = {".bin", ".ckpt", ".h5", ".hdf5", ".npz", ".pth", ".pt", ".safetensors", ".tar"}


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_record(path: str | Path, *, logical_name: str = "") -> dict[str, Any]:
    source = Path(path).resolve()
    record: dict[str, Any] = {
        "logical_name": logical_name or source.name,
        "cache_path": str(source),
        "filename": source.name,
        "size_bytes": None,
        "sha256": None,
        "status": "missing",
    }
    if source.is_file():
        record.update(
            size_bytes=int(source.stat().st_size),
            sha256=sha256_file(source),
            status="verified",
        )
    return record


def checkpoint_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [checkpoint_record(path) for path in paths]


def model_bundle_records(root: str | Path) -> list[dict[str, Any]]:
    """Hash every regular file in a small model bundle, including its config."""
    source = Path(root).resolve()
    if not source.is_dir():
        return []
    records = []
    for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
        record = checkpoint_record(path, logical_name=path.relative_to(source).as_posix())
        records.append(record)
    return records


def git_revision(repo: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repo)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hf_cache_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_snapshot_identity(snapshot_path: str | Path) -> tuple[str | None, Path | None]:
    current = Path(snapshot_path).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if candidate.parent.name == "snapshots":
            return candidate.name, candidate
    return None, None


def resolve_hf_snapshot(repo_id: str, revision: str = "main") -> tuple[str | None, Path | None]:
    repo_dir = _hf_cache_root() / f"models--{repo_id.replace('/', '--')}"
    if not repo_dir.is_dir():
        return None, None
    revision_text = str(revision or "main").strip()
    commit = revision_text if _looks_like_commit(revision_text) else None
    ref_path = repo_dir / "refs" / revision_text
    if commit is None and ref_path.is_file():
        commit = ref_path.read_text(encoding="utf-8").strip() or None
    if commit:
        snapshot = repo_dir / "snapshots" / commit
        if snapshot.is_dir():
            return commit, snapshot.resolve()
    snapshots = sorted(
        (path for path in (repo_dir / "snapshots").glob("*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if len(snapshots) == 1:
        return snapshots[0].name, snapshots[0].resolve()
    return None, None


def weight_files(root: str | Path) -> list[Path]:
    source = Path(root)
    if source.is_file():
        return [source] if source.suffix.lower() in WEIGHT_SUFFIXES else []
    if not source.is_dir():
        return []
    return sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )


def hf_model_provenance(repo_id: str, snapshot_path: str | Path, *, requested_revision: str = "main") -> dict[str, Any]:
    commit, snapshot = hf_snapshot_identity(snapshot_path)
    local_candidate = Path(snapshot_path).expanduser()
    if snapshot is None and local_candidate.is_dir() and _looks_like_commit(local_candidate.name):
        commit, snapshot = local_candidate.name, local_candidate.resolve()
    if snapshot is None:
        commit, snapshot = resolve_hf_snapshot(repo_id, requested_revision)
    checkpoints = checkpoint_records(weight_files(snapshot)) if snapshot is not None else []
    return {
        "source_repository": f"https://huggingface.co/{repo_id}",
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": commit,
        "cache_path": str(snapshot) if snapshot is not None else str(snapshot_path),
        "checkpoints": checkpoints,
    }


def _looks_like_commit(value: str) -> bool:
    lowered = value.lower()
    return len(lowered) >= 7 and len(lowered) <= 64 and all(char in "0123456789abcdef" for char in lowered)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
