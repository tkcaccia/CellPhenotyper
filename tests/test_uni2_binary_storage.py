"""No weights, torch installation or network required: real storage code only."""
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import uni2_embedding_io as io


def producer_functions(*extra):
    names = {"init_embedding_writer", "append_embedding_rows", "flush_embedding_writer",
             "embedding_shard_inventory", "validate_extraction_grid_cache", "validate_extraction_completion_cache", *extra}
    path = ROOT / "bin/extract_uni2_embeddings.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"np": np, "pd": pd, "Path": Path, "json": json, "hashlib": hashlib, "__file__": str(path),
                 **{key: getattr(io, key) for key in ("write_binary_shard", "binary_shard_paths", "iter_binary_blocks", "discover_embedding_shards", "sha256_file")},
                 "BINARY_SUFFIX": io.SUFFIX}
    exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + nodes,
                            type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def fixture(tmp_path, dtype="float32"):
    rows = pd.DataFrame({"cell_id": ["0001", "9007199254740993", "NA", "cell,4"],
                         "cx": [1, 2, 3, 4], "source_mpp": [.273774374855905] * 4,
                         "observation_type": ["cell"] * 4, "tile_path": [""] * 4})
    values = np.asarray([[0., -0., 1.1], [2., -3.456, 9.], [1e-30, 1e20, -1e-10], [4., 5., 6.]], dtype=dtype)
    path = io.write_binary_shard(tmp_path / "uni2_embeddings_shard0000", rows, values, embedding_mode="tile")
    return path, rows, values


@pytest.mark.parametrize("dtype", ["float32", "float64", ">f4", ">f8"])
def test_exact_binary_roundtrip_and_streamed_metadata(tmp_path, dtype, monkeypatch):
    path, rows, values = fixture(tmp_path, dtype)
    actual, matrix, manifest = io.read_binary_shard(path)
    assert actual.cell_id.tolist() == rows.cell_id.tolist()
    assert matrix.flags.writeable is False and isinstance(matrix, np.memmap)
    expected = np.asarray(values, dtype=f"<f{values.dtype.itemsize}")
    assert matrix.tobytes() == expected.tobytes()  # includes signed zero
    assert manifest["embedding_mode"] == "tile" and manifest["shape"] == [4, 3]
    assert actual.source_mpp.tolist() == rows.source_mpp.tolist()
    calls, read_csv = [], pd.read_csv
    def bounded_read(*args, **kwargs):
        calls.append(kwargs.get("chunksize"))
        return read_csv(*args, **kwargs)
    monkeypatch.setattr(io.pd, "read_csv", bounded_read)
    blocks = list(io.iter_binary_blocks(path, block_rows=2))
    assert calls == [2] and [len(block[0]) for block in blocks] == [2, 2]
    assert np.concatenate([block[1] for block in blocks]).tobytes() == expected.tobytes()
    assert io.discover_embedding_shards(tmp_path) == [path]


@pytest.mark.parametrize("change", ["truncated", "flipped", "row_reorder", "missing", "unsafe", "shape", "dtype", "hash", "row_header", "duplicate_id", "nonfinite"])
def test_corruptions_fail_closed(tmp_path, change):
    path, _, _ = fixture(tmp_path)
    manifest = json.loads(path.read_text())
    rows, features = io.binary_shard_paths(path)[1:]
    if change == "truncated":
        features.write_bytes(features.read_bytes()[:-1])
    elif change == "flipped":
        data = bytearray(features.read_bytes()); data[1] ^= 1; features.write_bytes(data)
    elif change == "row_reorder":
        table = pd.read_csv(rows, dtype={"cell_id": str}, keep_default_na=False)
        table.iloc[::-1].to_csv(rows, index=False)
    elif change == "missing":
        features.unlink()
    elif change == "unsafe":
        manifest["features_file"] = "../escape.features.bin"
    elif change == "shape":
        manifest["shape"] = [3, 4]
    elif change == "dtype":
        manifest["dtype"] = ">f4"
    elif change == "hash":
        manifest["rows_sha256"] = "x" * 64
    elif change in {"row_header", "duplicate_id"}:
        table = pd.read_csv(rows, dtype={"cell_id": str}, keep_default_na=False)
        if change == "row_header":
            table = table.rename(columns={"cx": "wrong"})
        else:
            table.loc[3, "cell_id"] = table.loc[0, "cell_id"]
        table.to_csv(rows, index=False)
        manifest.update(rows_sha256=io.sha256_file(rows), rows_size_bytes=rows.stat().st_size)
    else:
        values = np.memmap(features, mode="r+", dtype="<f4", shape=(4, 3)); values[1, 1] = np.inf; values.flush(); del values
        manifest["features_sha256"] = io.sha256_file(features)
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        io.read_binary_shard(path)


@pytest.mark.parametrize("values", [np.ones((2, 2), dtype="int32"), np.ones((2, 2), dtype="float16"), np.array([[np.nan], [1]], dtype="float32")])
def test_writer_rejects_lossy_or_invalid_values_before_creating_files(tmp_path, values):
    with pytest.raises(ValueError):
        io.write_binary_shard(tmp_path / "uni2_embeddings_shard0000", pd.DataFrame({"cell_id": ["1", "2"]}), values)
    assert not list(tmp_path.iterdir())


def test_no_overwrite_and_exact_discovery(tmp_path):
    path, rows, values = fixture(tmp_path)
    before = {p: p.read_bytes() for p in io.binary_shard_paths(path)}
    with pytest.raises(FileExistsError):
        io.write_binary_shard(path, rows, values)
    assert before == {p: p.read_bytes() for p in before}
    extra = tmp_path / "uni2_embeddings_shard0001.features.bin"
    extra.write_bytes(b"orphan")
    with pytest.raises(ValueError, match="Orphan"):
        io.discover_embedding_shards(tmp_path)
    extra.unlink()
    csv_path = tmp_path / "uni2_embeddings_shard0001.csv.gz"
    csv_path.write_bytes(b"mixed")
    with pytest.raises(ValueError, match="Mixed"):
        io.discover_embedding_shards(tmp_path)
    csv_path.unlink()
    stage = tmp_path / "stage"; stage.mkdir()
    (stage / "grid").symlink_to(tmp_path / "source", target_is_directory=True)
    source = tmp_path / "source"; source.mkdir()
    for payload in before:
        payload.rename(source / payload.name)
    assert io.discover_embedding_shards(stage) == [stage / "grid" / path.name]


@pytest.mark.parametrize("storage", ["csv", "binary"])
def test_producer_writer_is_bounded_and_preserves_paired_rows(tmp_path, storage, monkeypatch):
    functions = producer_functions()
    states = []
    values = np.random.default_rng(23).normal(size=(11, 7)).astype("float32")
    metadata = pd.DataFrame({"cell_id": [f"00{i}" for i in range(11)], "cx": np.arange(11), "observation_type": "cell"})
    for mode in ("tile", "inner_square"):
        directory = tmp_path / mode; directory.mkdir()
        state = functions["init_embedding_writer"](directory, "uni2", storage, 3, mode)
        for start, stop in ((0, 2), (2, 9), (9, 11)):
            functions["append_embedding_rows"](state, metadata.iloc[start:stop], values[start:stop])
            assert state["row_count"] < 3
            assert all(not any(str(c).startswith("feat_") for c in part.columns) for part in state["row_parts"])
        functions["flush_embedding_writer"](state)
        assert state["total_rows"] == 11 and state["shard_idx"] == 4
        states.append(state)
        arrays, ids = [], []
        for shard in io.discover_embedding_shards(directory):
            if storage == "binary":
                rows, data, manifest = io.read_binary_shard(shard)
                assert manifest["embedding_mode"] == mode
                arrays.append(data); ids.extend(rows.cell_id)
            else:
                rows = pd.read_csv(shard, dtype={"cell_id": str})
                arrays.append(rows.filter(like="feat_").to_numpy(dtype="float32")); ids.extend(rows.cell_id)
        np.testing.assert_array_equal(np.concatenate(arrays), values)
        assert ids == metadata.cell_id.tolist()
    assert states[0]["ids"] == states[1]["ids"]


def test_actual_grid_and_completion_cache_reject_row_reorder_and_extra_payload(tmp_path):
    functions = producer_functions()
    directory = tmp_path / "grid_r00_c00"; directory.mkdir()
    fixture(directory)
    inventory = functions["embedding_shard_inventory"](directory, "binary")
    marker = {"cache_contract_sha256": "a" * 64, "embedding_storage": "binary", "shards": 1,
              "embedding_mode": "tile", "shard_files": inventory, "rows_written": 4, "index_start": 0, "index_end": 4}
    validate = functions["validate_extraction_grid_cache"]
    assert validate(marker, "a" * 64, directory)
    assert not validate(marker, "b" * 64, directory)
    receipt = directory / ".uni2_grid_complete.json"; receipt.write_text(json.dumps(marker))
    all_payloads = [receipt, *io.binary_shard_paths(directory / inventory[0]["name"])]
    completion = {"cache_contract_sha256": "a" * 64, "embedding_storage": "binary", "payload_inventory": [
        {"path": p.relative_to(tmp_path).as_posix(), "sha256": io.sha256_file(p), "size_bytes": p.stat().st_size} for p in all_payloads]}
    (tmp_path / ".uni2_embedding_complete.json").write_text(json.dumps(completion))
    functions["validate_extraction_completion_cache"](tmp_path, "a" * 64, "binary", "uni2")
    # Even a self-consistently rehashed shard remains bound by the original grid/root receipts.
    path = directory / inventory[0]["name"]
    meta = json.loads(path.read_text()); row_path = directory / meta["rows_file"]
    rows = pd.read_csv(row_path, dtype={"cell_id": str}, keep_default_na=False)
    rows.iloc[::-1].to_csv(row_path, index=False)
    meta.update(rows_sha256=io.sha256_file(row_path), rows_size_bytes=row_path.stat().st_size)
    path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate(marker, "a" * 64, directory)
    with pytest.raises(RuntimeError, match="checksum/size mismatch"):
        functions["validate_extraction_completion_cache"](tmp_path, "a" * 64, "binary", "uni2")


def test_cli_storage_default_and_helper_cache_identity(tmp_path, monkeypatch):
    import argparse
    import re
    functions = producer_functions("parse_args", "build_extraction_cache_contract")
    functions.update(argparse=argparse, re=re, package_version=lambda name: "test")
    for name in ("image", "mask"):
        (tmp_path / name).write_bytes(b"fixture")
    monkeypatch.setattr(sys, "argv", ["extract", "--image", str(tmp_path / "image"), "--mask", str(tmp_path / "mask"), "--outdir", str(tmp_path / "output")])
    args = functions["parse_args"]()
    assert args.embedding_storage == "csv"
    assert args.embedding_mode == "tile"
    kwargs = dict(model_state_sha256="a" * 64, backend="test", pooling="cls", calibration={}, global_scaling={}, preprocessing={})
    csv_contract, csv_hash = functions["build_extraction_cache_contract"](args, **kwargs)
    assert "uni2_embedding_io.py" in csv_contract["code_sha256"]
    args.embedding_storage = "binary"
    assert functions["build_extraction_cache_contract"](args, **kwargs)[1] != csv_hash
    args.embedding_storage = "csv"
    args.embedding_mode = "nuclei"
    assert functions["build_extraction_cache_contract"](args, **kwargs)[1] != csv_hash


def main_environment(tmp_path, monkeypatch, storage, paired_mode=None, primary_mode="tile"):
    """Execute actual main/writer/cache logic with a deterministic array encoder.

    Replaces only encoder/tensor/image backends, not writer, inventory, argument,
    observation-loading, grid assignment or completion-validation code.
    """
    import argparse
    import contextlib
    import math
    import re
    import shutil
    import time
    from types import SimpleNamespace
    from PIL import Image
    from uni2_grid import assign_rounded_centers_to_grid
    functions = producer_functions("main", "parse_args", "build_extraction_cache_contract", "tile_folder",
                                   "load_observations_csv", "resolve_extraction_tile_size",
                                   "compute_inner_square_bounds", "build_inner_square_mask")
    class ArrayTensor:
        def __init__(self, array): self.array = np.asarray(array)
        def to(self, *args, **kwargs): return self
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.array
    class Transform:
        def __call__(self, image): return np.asarray(image)
        def __str__(self): return "synthetic_rgb_uint8_transform"
    class Encoder:
        def __init__(self): self.calls = 0
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, images):
            self.calls += 1
            array = images.array.astype("float32")
            return ArrayTensor(np.stack([array.mean((1, 2, 3)), array.std((1, 2, 3)),
                                         array[:, 1, 1, 0], -array[:, 1, 1, 1]], axis=1).astype("float32"))
        forward_features = __call__
    pixels = np.arange(8 * 8 * 3, dtype="uint8").reshape(8, 8, 3)
    class Reader:
        shape = pixels.shape
        backend = "synthetic_array"
        def __init__(self, *args, **kwargs): pass
        def read(self, x, y, w, h): return pixels[y:y + h, x:x + w].copy()
    encoder = Encoder()
    functions.update(argparse=argparse, math=math, re=re, shutil=shutil, time=time, Image=Image,
                     torch=SimpleNamespace(device=lambda value: value, cuda=SimpleNamespace(is_available=lambda: False),
                                           set_num_threads=lambda n: None, inference_mode=contextlib.nullcontext,
                                           stack=lambda values, dim=0: ArrayTensor(np.stack(values, axis=dim))),
                     tqdm=lambda values, **kwargs: values, RegionReader=Reader,
                     load_encoder=lambda **kwargs: (encoder, Transform(), "timm_hf", "cls", "uni2"),
                     fingerprint_encoder_state=lambda model: "a" * 64, package_version=lambda name: "synthetic_test",
                     infer_source_mpp=lambda path: .5, compute_global_percentiles=lambda *args, **kwargs: (np.zeros(3), np.full(3, 255)),
                     to_rgb_uint8_global=lambda image, **kwargs: image,
                     extract_cls_and_patch_tokens=lambda value: (value, object()),
                     pool_from_token_parts=lambda cls, patch, pooling: cls,
                     derive_inner_square_style_embeddings=lambda cls, **kwargs: cls.numpy() + np.float32(.125),
                     assign_rounded_centers_to_grid=assign_rounded_centers_to_grid)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in ("image", "mask"):
        (tmp_path / name).write_bytes(b"synthetic_input")
    pd.DataFrame({"label": [11, 23, 31, 47], "x": [2, 3, 4, 5], "y": [2, 3, 4, 5], "area_px": [2] * 4}).to_csv(tmp_path / "objects.csv", index=False)
    argv = ["extract", "--image", str(tmp_path / "image"), "--mask", str(tmp_path / "mask"),
            "--objects-csv", str(tmp_path / "objects.csv"), "--outdir", str(tmp_path / "primary"),
            "--encoder", "synthetic", "--grid", "1x2", "--tile-size", "4", "--img-size", "4",
            "--inner-square-fixed-px", "2", "--inner-square-min-px", "1", "--batch", "3", "--rows-per-csv", "1",
            "--embedding-storage", storage, "--embedding-mode", primary_mode]
    if paired_mode:
        argv += ["--paired-inner-square-outdir", str(tmp_path / "secondary"), "--paired-inner-square-mode", paired_mode]
    monkeypatch.setattr(sys, "argv", argv)
    return functions, encoder


@pytest.mark.parametrize("storage", ["csv", "binary"])
@pytest.mark.parametrize("paired_mode", ["token_subset", "masked_forward"])
def test_actual_main_paired_writer_receipts_and_resume(tmp_path, monkeypatch, storage, paired_mode):
    functions, encoder = main_environment(tmp_path, monkeypatch, storage, paired_mode)
    functions["main"]()
    calls = encoder.calls
    before = {}
    for name, mode in (("primary", "tile"), ("secondary", "inner_square")):
        root = tmp_path / name
        completion = json.loads((root / ".uni2_embedding_complete.json").read_text())
        assert completion["embedding_mode"] == mode and completion["embedding_storage"] == storage
        assert completion["rows_written"] == 4 and completion["missing_cells"] == 0
        assert completion["model_provenance"]["resolved_revision"] is None  # storage never invents encoder provenance
        assert len(completion["payload_inventory"]) == (14 if storage == "binary" else 6)
        ids = []
        for shard in io.discover_embedding_shards(root):
            if storage == "binary":
                rows, values, manifest = io.read_binary_shard(shard)
                assert manifest["embedding_mode"] == mode and values.dtype == np.dtype("float32")
            else:
                rows = pd.read_csv(shard, dtype={"cell_id": str})
            ids.extend(rows.cell_id)
        assert ids == ["11", "23", "31", "47"]
        for record in completion["payload_inventory"]:
            path = root / record["path"]
            assert io.sha256_file(path) == record["sha256"] and path.stat().st_size == record["size_bytes"]
            before[path] = path.read_bytes()
    functions["main"]()
    assert encoder.calls == calls  # resume ran zero synthetic forward passes
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize("mode", ["nuclei", "cyto", "inner_square"])
def test_actual_main_single_primary_representation_identity(tmp_path, monkeypatch, mode):
    functions, _ = main_environment(tmp_path, monkeypatch, "binary", primary_mode=mode)
    functions["main"]()
    root = tmp_path / "primary"
    assert json.loads((root / ".uni2_embedding_complete.json").read_text())["embedding_mode"] == mode
    assert {json.loads(p.read_text())["embedding_mode"] for p in io.discover_embedding_shards(root)} == {mode}


def test_actual_main_rejects_incompatible_pair_before_output_creation(tmp_path, monkeypatch):
    functions, encoder = main_environment(tmp_path, monkeypatch, "binary", "token_subset", "nuclei")
    with pytest.raises(ValueError, match="requires --embedding-mode tile"):
        functions["main"]()
    assert not (tmp_path / "primary").exists() and encoder.calls == 0


@pytest.mark.parametrize("corruption", ["row_reorder", "extra_shard", "incomplete_bundle"])
def test_actual_main_preserves_corrupt_or_incomplete_outputs(tmp_path, monkeypatch, corruption):
    functions, encoder = main_environment(tmp_path, monkeypatch, "binary")
    functions["main"]()
    root = tmp_path / "primary"
    shard = io.discover_embedding_shards(root)[0]
    if corruption == "row_reorder":
        # One-row shard: changing its ID is the smallest possible lineage error.
        record = json.loads(shard.read_text()); rows = shard.parent / record["rows_file"]
        rows.write_text(rows.read_text().replace("11,", "99,", 1))
    elif corruption == "extra_shard":
        (shard.parent / "uni2_embeddings_shard9999.features.bin").write_bytes(b"orphan")
    else:
        (root / ".uni2_embedding_complete.json").unlink()
        next(root.rglob(".uni2_grid_complete.json")).unlink()
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    calls = encoder.calls
    with pytest.raises((RuntimeError, ValueError)):
        functions["main"]()
    assert encoder.calls == calls and before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("root_receipts", [True, False])
def test_actual_main_rejects_transplanted_paired_representation(tmp_path, monkeypatch, root_receipts):
    functions, encoder = main_environment(tmp_path, monkeypatch, "binary", "token_subset")
    functions["main"]()
    if not root_receipts:
        (tmp_path / "primary/.uni2_embedding_complete.json").unlink()
        (tmp_path / "secondary/.uni2_embedding_complete.json").unlink()
    (tmp_path / "primary").rename(tmp_path / "swap")
    (tmp_path / "secondary").rename(tmp_path / "primary")
    (tmp_path / "swap").rename(tmp_path / "secondary")
    calls = encoder.calls
    with pytest.raises(RuntimeError, match="embedding mode differs"):
        functions["main"]()
    assert encoder.calls == calls


@pytest.mark.parametrize("paired_mode", ["token_subset", "masked_forward"])
def test_actual_main_csv_binary_paired_output_value_equivalence(tmp_path, monkeypatch, paired_mode):
    collected = {}
    for storage in ("csv", "binary"):
        location = tmp_path / storage
        functions, _ = main_environment(location, monkeypatch, storage, paired_mode)
        functions["main"]()
        for family in ("primary", "secondary"):
            ids, arrays = [], []
            for shard in io.discover_embedding_shards(location / family):
                if storage == "binary":
                    rows, values, _ = io.read_binary_shard(shard)
                else:
                    rows = pd.read_csv(shard, dtype={"cell_id": str}, float_precision="round_trip")
                    values = rows.filter(like="feat_").to_numpy(dtype="float32")
                ids.extend(rows.cell_id)
                arrays.append(values)
            collected[storage, family] = ids, np.concatenate(arrays)
    for family in ("primary", "secondary"):
        csv_ids, csv_values = collected["csv", family]
        binary_ids, binary_values = collected["binary", family]
        assert csv_ids == binary_ids
        np.testing.assert_array_equal(csv_values, binary_values)
