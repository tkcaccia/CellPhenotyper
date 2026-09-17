"""Direct-object neighborhood inspection parity and source checks; no server."""
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import tifffile

from test_cell_inspector import specimen as base_specimen
from test_cohort_streaming import as_store, tree_hashes
from assemble_spatial_cell_profiles import assemble
from cell_profile_io import sha256_file
from neighborhood_feature_io import FeatureColumns
import cell_inspector as module


@pytest.fixture
def specimens(tmp_path):
    inputs, cells, _, labels = base_specimen.__wrapped__(tmp_path)
    base = inputs["profile_dir"]
    cells = cells.drop(columns=[name for name in cells if name.startswith("niche_")])
    cells.to_parquet(base / "cell_profiles.parquet", index=False)
    cells.to_csv(base / "cell_profiles.csv", index=False)
    path = base / "cell_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["inputs"] = {"image_sha256": sha256_file(inputs["image"]),
                          "labels_sha256": sha256_file(inputs["labels"])}
    for name in ("cell_profiles.csv", "cell_profiles.parquet"):
        manifest["files"][name] = sha256_file(base / name)
    (base / "feature_blocks").mkdir()
    for block in manifest["feature_blocks"].values():
        source = base / block["path"]
        block["path"] = "feature_blocks/" + source.name
        source.rename(base / block["path"])
        block.update(dtype="float32", feature_names=["first", "second"])
    path.write_text(json.dumps(manifest))
    support = np.ones(labels.shape, dtype=np.uint8)
    support[:, 125:] = 0
    support_path = tmp_path / "support.tif"
    tifffile.imwrite(support_path, support)
    dense = tmp_path / "dense"
    assemble(base, support_path, inputs["shift"], None, dense,
             radii_um=(20., 50.), feature_groups=("context",), repeats=2)
    store = as_store(dense, tmp_path / "store")
    return {"dense": dict(inputs, profile_dir=dense),
            "store": dict(inputs, profile_dir=store), "cells": cells}


def mean_path(inspector):
    return inspector.root / inspector.neighborhood_reader.groups["context"]["segments"][0]["path"]


def mutate_value(path):
    before = path.stat()
    values = np.load(path, mmap_mode="r+")
    values[0, 0] += 1
    values.flush()
    del values
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns


def test_all_spatial_values_match_dense_without_hydrating_profiles(specimens):
    roots = [specimens[name]["profile_dir"] for name in ("dense", "store")]
    before = {str(root): tree_hashes(root) for root in roots}
    dense = module.CellInspector(**specimens["dense"])
    store = module.CellInspector(**specimens["store"])
    columns = store.neighborhood_display_columns
    assert columns and all(name.startswith(("r20um_", "r50um_")) for name in columns)
    assert not set(columns) & set(store.cells.columns)
    for uid in specimens["cells"].cell_uid:
        actual, expected = store.detail(uid), dense.detail(uid)
        for key in expected:
            if key != "neighborhood_features":
                assert actual[key] == expected[key], key
        assert not any(name.startswith("own_") for name in actual["spatial"])
        assert store.similar(uid, ["context", "local"]) == dense.similar(uid, ["context", "local"])
    outside = store.detail(specimens["cells"].cell_uid.iloc[-1])["spatial"]
    assert all(outside[name] is None for name in columns)
    assert store.queue() == dense.queue()
    assert dense.metadata()["neighborhood_features"] == {"status": "scalar_profile_fields"}
    assert store.metadata()["neighborhood_features"]["status"] == "verified_array_backed"
    assert {str(root): tree_hashes(root) for root in roots} == before


def test_single_row_reads_and_no_full_hash_on_unchanged_requests(specimens, monkeypatch):
    inspector = module.CellInspector(**specimens["store"])
    calls = []
    original = FeatureColumns.read
    def bounded(self, rows, columns):
        assert isinstance(rows, np.ndarray) and rows.shape == (1,)
        assert list(columns) == inspector.neighborhood_display_columns
        calls.append(rows.tolist())
        return original(self, rows, columns)
    monkeypatch.setattr(FeatureColumns, "read", bounded)
    monkeypatch.setattr(FeatureColumns, "recheck", lambda *a: pytest.fail("Full digest per unchanged request"))
    monkeypatch.setattr("neighborhood_feature_io._sha", lambda *a: pytest.fail("Full digest per unchanged request"))
    for index in (0, 3, 0):
        inspector.detail(specimens["cells"].cell_uid.iloc[index])
        inspector.metadata()
    assert calls == [[0], [3], [0]]
    metadata = inspector.metadata()["neighborhood_features"]
    metadata["source_files"].clear()
    metadata["groups"].clear()
    metadata["displayed_columns"].clear()
    assert inspector.neighborhood_reader.source_files
    assert inspector.neighborhood_reader.groups and inspector.neighborhood_display_columns


def test_unchanged_bytes_with_changed_stat_rehash_once(specimens, monkeypatch):
    inspector = module.CellInspector(**specimens["store"])
    path = mean_path(inspector)
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    original, calls = FeatureColumns.recheck, []
    def record(self):
        calls.append(True)
        return original(self)
    monkeypatch.setattr(FeatureColumns, "recheck", record)
    uid = specimens["cells"].cell_uid.iloc[0]
    inspector.detail(uid)
    inspector.detail(uid)
    inspector.metadata()
    assert calls == [True]


@pytest.mark.parametrize("action", ["detail", "metadata"])
def test_same_stat_payload_corruption_is_rejected(specimens, action):
    inspector = module.CellInspector(**specimens["store"])
    mutate_value(mean_path(inspector))
    with pytest.raises(ValueError, match="changed|SHA|hash"):
        if action == "detail":
            inspector.detail(specimens["cells"].cell_uid.iloc[0])
        else:
            inspector.metadata()


@pytest.mark.parametrize("replacement", [False, True])
def test_change_during_read_rejects_current_result(specimens, monkeypatch, replacement):
    inspector = module.CellInspector(**specimens["store"])
    path = mean_path(inspector)
    original = FeatureColumns.read
    def changed(self, rows, columns):
        result = original(self, rows, columns)
        if replacement:
            other = path.with_name("replacement.npy")
            shutil.copyfile(path, other)
            os.replace(other, path)
        else:
            mutate_value(path)
        return result
    monkeypatch.setattr(FeatureColumns, "read", changed)
    with pytest.raises(ValueError, match="changed|SHA|hash"):
        inspector.detail(specimens["cells"].cell_uid.iloc[0])


def test_contained_symlink_retarget_refreshes_mmap_not_old_inode(specimens):
    initial = module.CellInspector(**specimens["store"])
    path = mean_path(initial)
    original = path.with_name("original.npy")
    path.rename(original)
    path.symlink_to(original.name)
    inspector = module.CellInspector(**specimens["store"])
    uid = specimens["cells"].cell_uid.iloc[0]
    expected = inspector.detail(uid)["spatial"]
    replacement = path.with_name("replacement.npy")
    shutil.copyfile(original, replacement)
    link = path.with_name("next.npy")
    link.symlink_to(replacement.name)
    os.replace(link, path)
    previous_reader = inspector.neighborhood_reader
    assert inspector.detail(uid)["spatial"] == expected
    assert inspector.neighborhood_reader is not previous_reader
    mutate_value(original)
    assert inspector.detail(uid)["spatial"] == expected


def test_source_escape_and_startup_corruption_fail_closed(specimens, tmp_path):
    inspector = module.CellInspector(**specimens["store"])
    path = mean_path(inspector)
    external = tmp_path / "outside.npy"
    shutil.copyfile(path, external)
    link = path.with_name("escape.npy")
    link.symlink_to(external)
    os.replace(link, path)
    with pytest.raises(ValueError, match="escapes"):
        inspector.detail(specimens["cells"].cell_uid.iloc[0])
    with pytest.raises(ValueError, match="escaping"):
        module.CellInspector(**specimens["store"])


def test_startup_checks_payload_and_expert_annotations_remain_display_only(specimens, tmp_path):
    inspector = module.CellInspector(**specimens["store"])
    uid = specimens["cells"].cell_uid.iloc[0]
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps({"annotations": [{"cell_uid": uid, "label": "expert-only"}]}))
    annotated = module.CellInspector(**specimens["store"], expert_annotations=notes)
    assert annotated.detail(uid)["expert_annotation"]["label"] == "expert-only"
    assert annotated.detail(uid)["expert_annotation_used_for_inference"] is False
    assert annotated.detail(uid)["spatial"] == inspector.detail(uid)["spatial"]
    assert annotated.similar(uid, ["context"]) == inspector.similar(uid, ["context"])
    mutate_value(mean_path(inspector))
    with pytest.raises(ValueError, match="SHA|hash"):
        module.CellInspector(**specimens["store"])
