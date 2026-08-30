import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from ome_tiff_metadata import (  # noqa: E402
    create_tiff_memmap,
    label_storage_dtype,
    read_mpp_json,
    tiff_resolution_kwargs,
    validate_ome_tiff,
)


def test_resolution_sidecar_and_tiff_tags_round_trip(tmp_path: Path) -> None:
    sidecar = tmp_path / "shift.json"
    sidecar.write_text(json.dumps({"source_mpp": 0.2737746490978351}))
    mpp = read_mpp_json(sidecar)
    assert mpp == pytest.approx((0.2737746490978351, 0.2737746490978351))

    path = tmp_path / "labels.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((64, 96), dtype=np.uint16),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": mpp[0],
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeY": mpp[1],
            "PhysicalSizeYUnit": "um",
        },
        **{k: v for k, v in tiff_resolution_kwargs(*mpp, "YX").items() if k != "metadata"},
    )
    summary = validate_ome_tiff(
        path,
        expected_shape=(64, 96),
        expected_mpp=mpp,
        require_pyramid=False,
    )
    assert summary["spatial_shape"] == [64, 96]


def test_validation_rejects_wrong_physical_size(tmp_path: Path) -> None:
    path = tmp_path / "wrong.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((16, 16), dtype=np.uint8),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 1.0,
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeY": 1.0,
            "PhysicalSizeYUnit": "um",
        },
    )
    with pytest.raises(RuntimeError, match="PhysicalSizeX mismatch"):
        validate_ome_tiff(
            path,
            expected_shape=(16, 16),
            expected_mpp=(0.25, 0.25),
            require_pyramid=False,
        )


def test_memmap_preserves_resolution_tags(tmp_path: Path) -> None:
    path = tmp_path / "streaming-labels.tif"
    mpp = (0.2737746490978351, 0.28125)
    labels = tifffile.memmap(
        path,
        shape=(32, 48),
        dtype=np.uint16,
        bigtiff=True,
        **tiff_resolution_kwargs(*mpp, "YX"),
    )
    labels[4:12, 8:20] = 3
    labels.flush()
    del labels

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.tags["ResolutionUnit"].value == 3
        x_num, x_den = page.tags["XResolution"].value
        y_num, y_den = page.tags["YResolution"].value
        assert 10_000.0 / (x_num / x_den) == pytest.approx(mpp[0], rel=1e-5)
        assert 10_000.0 / (y_num / y_den) == pytest.approx(mpp[1], rel=1e-5)


def test_label_storage_dtype_avoids_uint16_byte_order_ambiguity() -> None:
    assert label_storage_dtype(7) == np.dtype("u1")
    assert label_storage_dtype(256) == np.dtype(">u2")
    assert label_storage_dtype(65536) == np.dtype(">u4")
    with pytest.raises(ValueError, match="non-negative"):
        label_storage_dtype(-1)


def test_big_endian_tiff_memmap_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "labels-be.tif"
    labels = create_tiff_memmap(
        path,
        shape=(24, 32),
        dtype=np.dtype(">u2"),
        mpp_x=0.25,
        mpp_y=0.5,
    )
    labels[2:8, 3:11] = 1024
    labels.flush()
    del labels
    with tifffile.TiffFile(path) as tif:
        assert tif.byteorder == ">"
        observed = tif.asarray()
    assert observed[2:8, 3:11].min() == 1024
