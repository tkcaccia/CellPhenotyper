"""Actual small Zarr writer/requantification tests; no learned inference or GPUs."""
import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import tifffile

pytest.importorskip("torch")
pytest.importorskip("zarr")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import torch
import run_gigatime_on_crop as producer
import quantify_gigatime_intensity as consumer


def write_fixture(tmp_path, monkeypatch, *, dtype="float32", channel_indices=None, fail_after=None):
    shape = (33, 41)
    yy, xx = np.indices(shape)
    image = np.stack([xx, yy, (xx + yy) % 255], axis=-1).astype(np.uint8)
    image_path = tmp_path / "he.tif"
    tifffile.imwrite(image_path, image, photometric="rgb", compression="deflate", tile=(16, 16))
    nuclei = np.zeros(shape, np.uint32)
    nuclei[4:10, 3:8] = 1
    nuclei[17:29, 23:36] = 17
    whole = nuclei.copy()
    whole[2:12, 1:10] = 1
    whole[15:31, 21:38] = 17
    ring = np.where(nuclei == 0, whole, 0).astype(np.uint32)
    masks = {}
    for name, values in (("nuclei", nuclei), ("cyto", whole), ("ring", ring)):
        path = tmp_path / f"{name}.tif"
        tifffile.imwrite(path, values, compression="deflate", tile=(16, 16))
        masks[name] = str(path)
    metadata = {"inference_shape_yx": list(shape), "original_shape_yx": list(shape),
        "model_channels": list(consumer.CANONICAL_CHANNEL_NAMES), "output_dtype": dtype,
        "resolution_contract": "exact_mpp_v2", "source_mpp": .25, "effective_mpp": .25,
        "downsample_factor": 1., "model_arithmetic": "test_analytic_field_not_model_inference",
        "model_provenance": {"checkpoints": [{"status": "verified", "sha256": "a"*64}]}}
    if dtype != "float32":
        metadata["storage_scale_max"] = int(np.iinfo(dtype).max)
    metadata["marker_schema"] = consumer.build_marker_schema(metadata, masks)
    selected = list(range(23)) if channel_indices is None else channel_indices
    names = [consumer.CANONICAL_CHANNEL_NAMES[index] for index in selected]
    metadata["store_channels"] = names
    quantifiers = producer.build_quantifiers(nuclei_mask_path=masks["nuclei"], cyto_mask_path=masks["cyto"],
        ring_mask_path=masks["ring"], target_shape=shape, channel_names=consumer.CANONICAL_CHANNEL_NAMES, block_size=8)
    outdir = tmp_path / "result"
    qc = producer.MarkerScoreQC(outdir=outdir, target_shape=shape, channel_names=consumer.CANONICAL_CHANNEL_NAMES, max_samples=100)
    calls = []
    def analytic_region(image_rgb, positions, model, device, patch_size, batch_size):
        calls.append(image_rgb.shape)
        if fail_after is not None and len(calls) > fail_after:
            raise RuntimeError("synthetic interrupted analytic writer")
        base = (image_rgb[..., 0].astype(np.float32) + image_rgb[..., 1].astype(np.float32)) / np.float32(256)
        scores = np.stack([base + np.float32(index / 100) for index in range(23)])
        return scores, np.ones((1, *scores.shape[1:]), np.float32)
    monkeypatch.setattr(producer, "run_region_inference", analytic_region)
    options = dict(source_path=str(image_path), page=0, outdir=outdir, model=None, device=torch.device("cpu"),
        patch_size=8, stride=4, batch_size=2, factor=1., tile_size=16, output_dtype=dtype, metadata=metadata,
        skip_background_blocks=False, skip_background_mask_path="", skip_background_downsample=8,
        skip_background_min_fraction=0, skip_background_close_radius=0, skip_background_min_obj_area=0,
        skip_background_hole_area=0, output_channel_indices=selected, output_channel_names=names,
        quantifiers=quantifiers, score_qc=qc, quant_dir=outdir / "quant", jpg_exporter=None,
        jpg_channel_indices=[], sample_id="fixture")
    return options, masks, metadata, calls


def validate(store, masks, **kwargs):
    with consumer.LazyImageReader(str(store)) as reader:
        return consumer.validate_restart_contract(reader, "nuclei", masks["nuclei"], **kwargs)


@pytest.mark.parametrize("storage_format", [2, 3])
def test_actual_zarr_writer_and_requantification_match_all_compartments(tmp_path, monkeypatch, storage_format):
    if int(consumer.zarr.__version__.split(".")[0]) < 3 and storage_format == 3:
        pytest.skip("Storage format 3 requires a Zarr 3 runtime")
    original_group = producer.zarr.group
    def group(*args, **kwargs):
        kwargs["zarr_format" if int(consumer.zarr.__version__.split(".")[0]) >= 3 else "zarr_version"] = storage_format
        return original_group(*args, **kwargs)
    monkeypatch.setattr(producer.zarr, "group", group)
    options, masks, metadata, calls = write_fixture(tmp_path, monkeypatch)
    producer.blockwise_write_zarr_outputs(**options)
    store = options["outdir"] / "gigatime_probs.zarr"
    assert len(calls) == 9
    # Copy only the self-contained Zarr, without adjacent TIFF/channel/schema files.
    portable = tmp_path / "portable.zarr"
    shutil.copytree(store, portable)
    original_hashes = consumer.zarr_file_inventory(portable)
    for compartment in ("nuclei", "cyto", "ring"):
        with consumer.LazyImageReader(str(portable)) as reader, consumer.LazyMaskReader(masks[compartment]) as mask:
            contract = consumer.validate_restart_contract(reader, compartment, masks[compartment], expected_schema=metadata["marker_schema"])
            assert contract["equivalent_to_authoritative_integrated"]
            assert contract["storage_integrity"]["status"] == "exact_encoded_file_inventory_verified"
            yy, xx = np.indices((reader.height, reader.width))
            base = (xx.astype(np.float32) + yy.astype(np.float32)) / np.float32(256)
            expected_field = np.stack([base + np.float32(index / 100) for index in range(23)])
            np.testing.assert_array_equal(reader.read_block(0, reader.height, 0, reader.width), expected_field)
            rows, _, _ = consumer.quantify_blockwise(reader, mask, reader.channel_names, mask_name=compartment, block_size=7)
        with (options["quant_dir"] / f"fixture_{compartment}_gigatime_quantification.csv").open() as handle:
            integrated = list(csv.DictReader(handle))
        assert len(rows) == len(integrated) == 2
        for expected, actual in zip(integrated, rows):
            for name, value in actual.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    assert float(expected[name]) == pytest.approx(value, abs=1e-12, rel=0), (compartment, name)
                else:
                    assert expected[name] == str(value)
    assert consumer.zarr_file_inventory(portable) == original_hashes
    cli_out = tmp_path / "cli_requantification"
    command = [sys.executable, str(ROOT / "bin/quantify_gigatime_intensity.py"), "--image", str(portable),
        "--mask", masks["nuclei"], "--mask-name", "nuclei", "--out-quant-csv", str(cli_out / "quant.csv"),
        "--out-mean-csv", str(cli_out / "mean.csv"), "--out-stats-csv", str(cli_out / "stats.csv"),
        "--out-summary-json", str(cli_out / "summary.json")]
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads((cli_out / "summary.json").read_text())["restart_contract"]["storage_integrity"]["status"] == "exact_encoded_file_inventory_verified"


def test_native_tiff_fallback_is_bounded_and_preserves_integer_downsample(tmp_path, monkeypatch):
    yy, xx = np.indices((37, 45))
    values = np.stack([xx, yy, xx + yy], axis=-1).astype(np.uint8)
    path = tmp_path / "compressed.tif"
    tifffile.imwrite(path, values, photometric="rgb", compression="deflate", tile=(16, 16))
    monkeypatch.setattr(producer, "pyvips", None)
    def incompatible(*args, **kwargs):
        raise ImportError("synthetic adapter API mismatch")
    monkeypatch.setattr(tifffile.TiffPage, "aszarr", incompatible)
    windows = []
    original = producer.WindowReader.read
    def read(self, x0, y0, x1, y1):
        windows.append((x1-x0, y1-y0))
        return original(self, x0, y0, x1, y1)
    monkeypatch.setattr(producer.WindowReader, "read", read)
    with producer.LazyCropReader(str(path), 0, 2) as reader:
        np.testing.assert_array_equal(reader.read_region(2, 6, 3, 9), values[4:12:2, 6:18:2])
        assert reader.backend == "tiff_segment_windows"
    assert windows == [(12, 8)]


@pytest.mark.parametrize("mutation", ["missing_chunk", "changed_value", "extra_file", "stale_sidecar", "native_incomplete", "reordered_names", "missing_receipt"])
def test_zarr_corruption_or_stale_metadata_cannot_be_equivalent(tmp_path, monkeypatch, mutation):
    options, masks, metadata, _ = write_fixture(tmp_path, monkeypatch)
    producer.blockwise_write_zarr_outputs(**options)
    store = options["outdir"] / "gigatime_probs.zarr"
    root = consumer.zarr.open_group(str(store), mode="r+")
    if mutation == "missing_chunk":
        receipt = json.loads((store / consumer.ZARR_STORAGE_MANIFEST).read_text())
        chunk = next(name for name in receipt["files"] if name.startswith("0/") and not name.endswith(".json") and not name.rsplit("/", 1)[-1].startswith("."))
        (store / chunk).unlink()
    elif mutation == "changed_value":
        root["0"][0, 4, 3] = .999
    elif mutation == "extra_file":
        (store / "unexpected.bin").write_bytes(b"unexpected")
    elif mutation == "stale_sidecar":
        sidecar = copy.deepcopy(metadata)
        sidecar["marker_schema"]["checkpoint_sha256"] = "b"*64
        (options["outdir"] / "gigatime_metadata.json").write_text(json.dumps(sidecar))
    elif mutation == "native_incomplete":
        root.attrs["gigatime_storage"] = {"state": "in_progress"}
    elif mutation == "reordered_names":
        root["0"].attrs["channel_names"] = consumer.CANONICAL_CHANNEL_NAMES[::-1]
    else:
        (store / consumer.ZARR_STORAGE_MANIFEST).unlink()
    with pytest.raises(ValueError, match="Incompatible GigaTIME restart"):
        validate(store, masks)


@pytest.mark.parametrize("dtype,indices", [("uint8", None), ("float32", [0, 3, 8, 16]), ("float32", list(reversed(range(23))))])
def test_real_encoded_subset_or_reordered_zarr_remains_non_equivalent(tmp_path, monkeypatch, dtype, indices):
    options, masks, metadata, _ = write_fixture(tmp_path, monkeypatch, dtype=dtype, channel_indices=indices)
    producer.blockwise_write_zarr_outputs(**options)
    store = options["outdir"] / "gigatime_probs.zarr"
    with pytest.raises(ValueError, match="Incompatible GigaTIME restart"):
        validate(store, masks)
    assert validate(store, masks, allow_legacy_research=True)["equivalent_to_authoritative_integrated"] is False


def test_interrupted_writer_cannot_reuse_completed_sidecar(tmp_path, monkeypatch):
    options, masks, metadata, _ = write_fixture(tmp_path, monkeypatch, fail_after=1)
    options["outdir"].mkdir(parents=True, exist_ok=True)
    stale = {**metadata, "inference_complete": True}
    (options["outdir"] / "gigatime_metadata.json").write_text(json.dumps(stale))
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            producer.blockwise_write_zarr_outputs(**options)
        store = options["outdir"] / "gigatime_probs.zarr"
        with pytest.raises(ValueError, match="native Zarr|completion"):
            validate(store, masks)
        assert not (store / consumer.ZARR_STORAGE_MANIFEST).exists()
        before = consumer.zarr_file_inventory(store)
        with pytest.raises(FileExistsError, match="Preserving existing"):
            producer.blockwise_write_zarr_outputs(**options)
        assert consumer.zarr_file_inventory(store) == before
    finally:
        for quantifier in options["quantifiers"]:
            quantifier.close()
