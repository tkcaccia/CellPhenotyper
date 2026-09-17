#!/usr/bin/env python3
"""Record provenance for the protected installed PathoFMPred package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_provenance import checkpoint_record


MODEL_SUFFIXES = {".bin", ".model", ".qs", ".rda", ".rdata", ".rdb", ".rds", ".rdx"}


def read_dcf(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    current = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw[:1].isspace() and current:
            fields[current] = f"{fields[current]} {raw.strip()}".strip()
        elif ":" in raw:
            current, value = raw.split(":", 1)
            current = current.strip()
            fields[current] = value.strip()
    return fields


def package_model_files(package_dir: Path) -> list[Path]:
    files = [
        path for path in package_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ]
    for name in ("DESCRIPTION", "NAMESPACE"):
        candidate = package_dir / name
        if candidate.is_file():
            files.append(candidate)
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    package_dir = (Path(args.library_dir).resolve() / "PathoFMPred")
    description = package_dir / "DESCRIPTION"
    if not description.is_file():
        raise FileNotFoundError(f"Installed PathoFMPred DESCRIPTION is missing: {description}")
    fields = read_dcf(description)
    version = fields.get("Version")
    source = (fields.get("URL") or fields.get("BugReports") or "").split(",", 1)[0].strip() or None
    checkpoints = [
        checkpoint_record(path, logical_name=path.relative_to(package_dir).as_posix())
        for path in package_model_files(package_dir)
    ]
    payload = {
        "package": fields.get("Package", "PathoFMPred"),
        "version": version,
        "license": fields.get("License") or None,
        "model_provenance": {
            "source_repository": source,
            "requested_revision": version,
            "resolved_revision": f"package:PathoFMPred=={version}" if version else None,
            "cache_path": str(package_dir),
            "checkpoints": checkpoints,
        },
        "note": (
            "Hashes cover installed serialized package/model databases plus DESCRIPTION and NAMESPACE. "
            "Private source-repository identity remains a release blocker when DESCRIPTION omits URL/BugReports."
        ),
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "pathofmpred_model_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "checkpoint_files": len(checkpoints), "version": version}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
