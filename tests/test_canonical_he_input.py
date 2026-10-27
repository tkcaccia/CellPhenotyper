"""Pipeline-entry RGB canonicalization preserves pixels and physical scale."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile


BIN = Path(__file__).parents[1] / "bin"
sys.path.insert(0, str(BIN))
SPEC = importlib.util.spec_from_file_location("canonicalize_he_input_test", BIN / "canonicalize_he_input.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def source(tmp_path: Path, dtype=np.uint8) -> tuple[Path, Path]:
    image = tmp_path / "source.ome.tif"
    data = np.zeros((64, 64, 3), dtype=dtype)
    data[..., 0], data[..., 1], data[..., 2] = 20, 40, 80
    tifffile.imwrite(image, data, photometric="rgb", ome=True,
                     metadata={"axes": "YXS", "PhysicalSizeX": .25, "PhysicalSizeY": .25})
    resolution = tmp_path / "resolution.json"
    resolution.write_text(json.dumps({"mpp_x": .25, "mpp_y": .25}))
    return image, resolution


def test_rgb_entry_is_verified_without_changing_image_bytes(tmp_path: Path) -> None:
    image, resolution = source(tmp_path)
    before = image.read_bytes()
    report = MODULE.canonicalize(image, tmp_path / "canonical.json",
                                 channel_order="rgb", source_resolution_report=resolution)
    assert image.read_bytes() == before
    assert report["action"] == "declared_rgb_no_pixel_change"
    assert report["canonical_channel_order"] == "rgb"
    assert report["intensity_mapping"] == "identity_uint8"


def test_non_uint8_is_rejected_before_gigatime(tmp_path: Path) -> None:
    image, resolution = source(tmp_path, np.uint16)
    with pytest.raises(ValueError, match="three-channel uint8"):
        MODULE.canonicalize(image, tmp_path / "canonical.json",
                            channel_order="rgb", source_resolution_report=resolution)


def test_declared_bgr_is_swapped_losslessly_with_spacing(tmp_path: Path) -> None:
    pytest.importorskip("pyvips")
    image, resolution = source(tmp_path)
    report = MODULE.canonicalize(image, tmp_path / "canonical.json",
                                 channel_order="bgr", source_resolution_report=resolution)
    assert tifffile.imread(image)[0, 0].tolist() == [80, 40, 20]
    assert report["output_layout"]["compression"] == "LZW"
    from validate_input_resolution import read_resolution

    assert read_resolution(image).mpp_x == pytest.approx(.25, abs=.001)
