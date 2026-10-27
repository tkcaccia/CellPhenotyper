"""Physical-scale and color-depth safety at the crop-to-GigaTIME handoff."""

import importlib.util
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile


BIN = Path(__file__).parents[1] / "bin"
sys.path.insert(0, str(BIN))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_ome(path: Path, *, mpp_x: float, mpp_y: float, unit: str = "nm",
              dtype=np.uint8) -> None:
    tifffile.imwrite(
        path, np.zeros((32, 48, 3), dtype=dtype), photometric="rgb", ome=True,
        metadata={"axes": "YXS", "PhysicalSizeX": mpp_x,
                  "PhysicalSizeY": mpp_y, "PhysicalSizeXUnit": unit,
                  "PhysicalSizeYUnit": unit},
    )


def test_crop_preserves_ome_nanometer_resolution_in_microns(tmp_path):
    crop = load("prepare_analysis_crop_scale", "prepare_analysis_crop.py")
    image = tmp_path / "source.ome.tif"
    write_ome(image, mpp_x=250, mpp_y=250)
    assert crop.read_shape_and_mpp(image) == (32, 48, 0.25)
    if importlib.util.find_spec("zarr") is not None:
        stardist = load("run_stardist_scale", "run_stardist_roi_segmentation.py")
        assert stardist.read_mpp_from_tiff(image) == pytest.approx(0.25)


def test_crop_rejects_anisotropic_scalar_resolution(tmp_path):
    crop = load("prepare_analysis_crop_anisotropy", "prepare_analysis_crop.py")
    image = tmp_path / "anisotropic.ome.tif"
    write_ome(image, mpp_x=250, mpp_y=300)
    with pytest.raises(ValueError, match="scalar MPP"):
        crop.read_shape_and_mpp(image)


def test_gigatime_uses_unit_aware_resolution_and_rejects_16bit(tmp_path):
    pytest.importorskip("torch")
    gigatime = load("run_gigatime_input_contract", "run_gigatime_on_crop.py")
    image = tmp_path / "source.ome.tif"
    write_ome(image, mpp_x=250, mpp_y=250)
    assert gigatime._read_mpp_from_tiff(str(image)) == pytest.approx(0.25)

    deep = tmp_path / "deep.ome.tif"
    write_ome(deep, mpp_x=250, mpp_y=250, dtype=np.uint16)
    with pytest.raises(ValueError, match="8-bit RGB"):
        gigatime.inspect_crop_image(
            str(deep), 0, 100, 4096, 0.25, 8.0,
            num_output_channels=1, bytes_per_sample=1, strict_target_mpp=True,
        )


def test_gigatime_rejects_anisotropic_direct_input(tmp_path):
    pytest.importorskip("torch")
    gigatime = load("run_gigatime_anisotropy", "run_gigatime_on_crop.py")
    image = tmp_path / "anisotropic.ome.tif"
    write_ome(image, mpp_x=250, mpp_y=300)
    with pytest.raises(ValueError, match="near-isotropic"):
        gigatime._read_mpp_from_tiff(str(image))


def test_gigatime_requires_canonical_entry_report_when_supplied(tmp_path):
    pytest.importorskip("torch")
    gigatime = load("run_gigatime_entry_report", "run_gigatime_on_crop.py")
    payload = {
        "schema_version": 1, "canonical_channel_order": "rgb",
        "source_channel_order": "bgr", "action": "whole_slide_bgr_to_rgb_lossless_lzw",
        "intensity_mapping": "identity_uint8", "mpp_x": .25, "mpp_y": .25,
        "source_resolution_report": "source.json",
        "output_layout": {"shape_yxc": [32, 48, 3], "dtype": "uint8"},
    }
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps(payload))
    report = gigatime.load_input_normalization_report(str(path))
    gigatime.validate_input_normalization_spacing(report, {"source_mpp": .25})
    with pytest.raises(ValueError, match="spacing disagrees"):
        gigatime.validate_input_normalization_spacing(report, {"source_mpp": .325})
    payload["canonical_channel_order"] = "bgr"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="canonical uint8 RGB"):
        gigatime.load_input_normalization_report(str(path))


def test_auto_parameters_lock_model_geometry_and_preserve_requested_record():
    pytest.importorskip("torch")
    gigatime = load("run_gigatime_auto_contract", "run_gigatime_on_crop.py")
    args = argparse.Namespace(
        auto_parameters=True, repo_id="prov-gigatime/GigaTIME",
        repo_revision=gigatime.DEFAULT_GIGATIME_REVISION,
        patch_size=512, stride=256, target_mpp=0.5,
        strict_target_mpp=False, auto_hardware=False,
        blockwise=False, skip_background_blocks=True,
    )
    record = gigatime.apply_auto_parameters(args)
    assert record["enabled"] is True
    assert record["requested"]["target_mpp"] == 0.5
    assert record["effective"] == {
        "patch_size": 256, "stride": 128, "target_mpp": 0.25,
        "strict_target_mpp": True, "auto_hardware": True,
        "blockwise": True, "skip_background_blocks": False,
    }
    args.repo_revision = "unverified"
    with pytest.raises(ValueError, match="pinned official CNN"):
        gigatime.apply_auto_parameters(args)


def test_cell_quantification_requires_persisted_native_dapi_and_nuclear_mask():
    pytest.importorskip("torch")
    gigatime = load("run_gigatime_quant_qc_contract", "run_gigatime_on_crop.py")
    args = argparse.Namespace(quant_dir="cell_scores", nuclei_mask="stardist.tif",
                              output_format="none")
    with pytest.raises(ValueError, match="persisted native DAPI"):
        gigatime.validate_quantification_qc_contract(args, ["DAPI"])
    args.output_format = "zarr"
    with pytest.raises(ValueError, match="persisted native DAPI"):
        gigatime.validate_quantification_qc_contract(args, ["CK"])
    args.nuclei_mask = ""
    with pytest.raises(ValueError, match="independent H&E nuclei mask"):
        gigatime.validate_quantification_qc_contract(args, ["DAPI"])
    args.nuclei_mask = "stardist.tif"
    gigatime.validate_quantification_qc_contract(args, ["DAPI"])
    args.output_format = "ome_tiff"
    args.compression = "jpeg"
    with pytest.raises(ValueError, match="lossless native DAPI"):
        gigatime.validate_quantification_qc_contract(args, ["DAPI"])
    args.compression = "deflate"
    gigatime.validate_quantification_qc_contract(args, ["DAPI"])


def test_standalone_quantification_remains_unvalidated_when_nuclear_qc_fails(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    gigatime = load("run_gigatime_quant_qc_gate", "run_gigatime_on_crop.py")
    summary = tmp_path / "sample_gigatime_integrated_quantification_summary.json"
    summary.write_text(json.dumps({"technical_qc_passed": False}))
    module = types.ModuleType("gigatime_nuclear_qc")
    module.assess = lambda image, nuclei, output: {"status": "fail"}
    monkeypatch.setitem(sys.modules, "gigatime_nuclear_qc", module)
    args = argparse.Namespace(outdir=str(tmp_path), nuclei_mask="independent_mask.tif", nuclear_qc_mode="fail")
    with pytest.raises(RuntimeError, match="must not be classified"):
        gigatime.run_quantification_nuclear_qc(args, tmp_path / "image.zarr", "sample", tmp_path)
    assert json.loads(summary.read_text())["technical_qc_passed"] is False
    module.assess = lambda image, nuclei, output: {"status": "pass"}
    gigatime.run_quantification_nuclear_qc(args, tmp_path / "image.zarr", "sample", tmp_path)
    assert json.loads(summary.read_text())["technical_qc_passed"] is True
