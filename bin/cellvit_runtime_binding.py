"""Pin extra Python dependencies supplied beside the CellViT Singularity image."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


def pydeps_tree_sha256(folder: Path) -> str:
    """Hash relative filenames and contents; reject links and mutable special files."""
    folder = Path(folder)
    if not folder.is_absolute() or folder.is_symlink() or not folder.is_dir():
        raise ValueError("CellViT runtime pydeps must be an absolute directory without symlinks")
    digest = hashlib.sha256(b"cellvit-runtime-pydeps-tree-v1\n")
    count = 0
    for path in sorted(folder.rglob("*"), key=lambda item: item.relative_to(folder).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError(f"Unsupported CellViT runtime pydeps entry: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(folder).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                content.update(block)
        digest.update(content.digest())
        count += 1
    if count == 0:
        raise ValueError("CellViT runtime pydeps directory is empty")
    return digest.hexdigest()


def verified_pydeps(folder: Path | None, expected_sha256: str) -> Path | None:
    if folder is None and not expected_sha256:
        return None
    if folder is None or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("CellViT runtime pydeps directory and SHA-256 must be supplied together")
    if pydeps_tree_sha256(folder) != expected_sha256:
        raise RuntimeError("Pinned CellViT runtime pydeps changed")
    return folder.resolve(strict=True)
