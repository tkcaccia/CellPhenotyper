"""Physical-scale and color-depth safety at the crop-to-GigaTIME handoff."""

import importlib.util
import sys
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
