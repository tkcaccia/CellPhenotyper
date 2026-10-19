"""Tiny CPU-only export checks using the actual optional Zarr/TIFF runtime."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

zarr = pytest.importorskip("zarr", reason="Zarr is supplied by the GigaTIME storage runtime")

MODULE_PATH = Path(__file__).parents[1] / "bin" / "export_gigatime_store_to_ometiff.py"
SPEC = importlib.util.spec_from_file_location("export_gigatime_ometiff_test", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(exporter)


def marker_image(tmp_path, channels=2, shape=(32, 32), dtype="uint16", pyramid_shapes=()):
    path = tmp_path / "markers.ome.tif"
    names = ["CD3", "CD8"][:channels]
    data = np.arange(channels * shape[0] * shape[1], dtype=np.dtype(dtype)).reshape(channels, *shape)
    metadata = {"axes": "CYX", "Channel": {"Name": names},
                "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
                "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"}
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(data, subifds=len(pyramid_shapes), photometric="minisblack", metadata=metadata)
        for height, width in pyramid_shapes:
            writer.write(np.zeros((channels, height, width), dtype=dtype),
                         subfiletype=1, photometric="minisblack", metadata=None)
    return path, data, names


def verify(path, data, names, **overrides):
    options = dict(expected_shape_cyx=data.shape, expected_dtype=data.dtype,
                   expected_channels=names, require_pyramid=False,
                   expected_physical_size_um=0.5)
    options.update(overrides)
    return exporter.verify_written_ome_tiff(path, **options)


def run_export(monkeypatch, source, destination, *, tile_size=2048, no_pyramid=False, output_channels=""):
    (source.parent / "gigatime_metadata.json").write_text(json.dumps({"effective_mpp": 0.5, "output_dtype": "uint16"}))
    args = SimpleNamespace(input=str(source), output=str(destination), output_channels=output_channels,
                           tile_size=tile_size, compression="deflate", output_dtype="auto", predictor=False,
                           no_pyramid=no_pyramid, pyramid_cache_max_gib=0.001, verification_json="")
    monkeypatch.setattr(exporter, "parse_args", lambda: args)
    exporter.main()
    return json.loads(destination.with_suffix(destination.suffix + ".verify.json").read_text())


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_verifier_accepts_scalar_singleton_and_multichannel_ome(tmp_path, channels, dtype):
    path, data, names = marker_image(tmp_path, channels=channels, dtype=dtype)
    report = verify(path, data, names, expected_pyramid_shapes=[])
    assert report["size_c"] == channels
    assert report["axes"] == ("YX" if channels == 1 else "CYX")
    assert report["channel_names"] == names
    assert report["levels"] == 1


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("no_pyramid,tile_size", [(False, 2048), (True, 16)])
def test_small_export_writes_verified_pixels_with_default_or_disabled_pyramid(
    tmp_path, monkeypatch, channels, no_pyramid, tile_size
):
    source, data, names = marker_image(tmp_path, channels=channels)
    output = tmp_path / "export.ome.tif"
    assert exporter.compute_pyramid_level_shapes(32, 32, 2048) == []
    report = run_export(monkeypatch, source, output, tile_size=tile_size, no_pyramid=no_pyramid)
    assert report["levels"] == 1
    assert report["size_c"] == channels
    assert report["channel_names"] == names
    assert report["dtype"] == "uint16"
    np.testing.assert_array_equal(tifffile.imread(output), data[0] if channels == 1 else data)


@pytest.mark.parametrize("channels", [1, 2])
def test_export_verifies_all_computed_odd_sized_pyramid_levels(tmp_path, monkeypatch, channels):
    source, data, names = marker_image(tmp_path, channels=channels, shape=(33, 49))
    output = tmp_path / "export.ome.tif"
    expected_levels = [(17, 25), (9, 13)]
    assert exporter.compute_pyramid_level_shapes(33, 49, 16) == expected_levels
    report = run_export(monkeypatch, source, output, tile_size=16)
    assert report["levels"] == 3
    assert [shape[-2:] for shape in report["level_shapes"]] == [[33, 49], [17, 25], [9, 13]]
    with tifffile.TiffFile(output) as handle:
        for index, level in enumerate(handle.series[0].levels):
            expected = data[:, ::2**index, ::2**index]
            np.testing.assert_array_equal(level.asarray(), expected[0] if channels == 1 else expected)


def test_export_preserves_requested_marker_order_and_pixel_identity(tmp_path, monkeypatch):
    source, data, _ = marker_image(tmp_path)
    output = tmp_path / "export.ome.tif"
    report = run_export(monkeypatch, source, output, output_channels="CD8,CD3")
    assert report["channel_names"] == ["CD8", "CD3"]
    np.testing.assert_array_equal(tifffile.imread(output), data[::-1])


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("axes", ["CYX", "YXC"])
def test_explicit_zarr_channel_axis_round_trips_across_tiles_and_pyramid(tmp_path, monkeypatch, channels, axes):
    data = np.arange(channels * 33 * 49, dtype=np.uint16).reshape(channels, 33, 49)
    stored = data if axes == "CYX" else np.moveaxis(data, 0, -1)
    source = tmp_path / "source.zarr"
    array = zarr.open(str(source), mode="w", shape=stored.shape, dtype="uint16")
    array[:] = stored
    array.attrs.update({"axes": axes, "channel_names": ["CD3", "CD8"][:channels]})
    output = tmp_path / "export.ome.tif"
    report = run_export(monkeypatch, source, output, tile_size=16)
    assert report["levels"] == 3
    with tifffile.TiffFile(output) as handle:
        for index, level in enumerate(handle.series[0].levels):
            expected = data[:, ::2**index, ::2**index]
            np.testing.assert_array_equal(level.asarray(), expected[0] if channels == 1 else expected)


def test_single_channel_rank_three_row_fallback_retains_channel_axis(tmp_path):
    source, data, _ = marker_image(tmp_path, channels=1, shape=(33, 49))
    with exporter.LazyGigaTIMEReader(source) as reader:
        # ndarray has no .oindex, exercising the documented bounded fallback.
        reader.arr = data
        reader.axes = "CYX"
        actual = reader._read_decimated_block_cyx_orthogonal(
            np.array([0, 2, 4]), np.array([0, 3, 6]), channel_indices=[0], export_dtype=data.dtype)
    np.testing.assert_array_equal(actual, data[:, [0, 2, 4]][:, :, [0, 3, 6]])


def test_require_pyramid_api_still_rejects_a_flat_image(tmp_path):
    path, data, names = marker_image(tmp_path)
    with pytest.raises(ValueError, match="pyramid is required"):
        verify(path, data, names, require_pyramid=True)


@pytest.mark.parametrize("overrides,message", [
    ({"expected_shape_cyx": (1, 32, 32)}, "SizeC mismatch"),
    ({"expected_shape_cyx": (2, 31, 32)}, "shape mismatch"),
    ({"expected_dtype": np.dtype("uint8")}, "dtype mismatch"),
    ({"expected_channels": ["CD8", "CD3"]}, "channel names mismatch"),
    ({"expected_physical_size_um": 0.25}, "PhysicalSizeX mismatch"),
    ({"expected_pyramid_shapes": [(16, 16)]}, "pyramid level count mismatch"),
])
def test_verifier_rejects_incorrect_metadata_or_missing_expected_levels(tmp_path, overrides, message):
    path, data, names = marker_image(tmp_path)
    with pytest.raises(ValueError, match=message):
        verify(path, data, names, **overrides)


def test_verifier_rejects_incorrect_pyramid_dimensions(tmp_path):
    path, data, names = marker_image(tmp_path, pyramid_shapes=[(16, 16)])
    with pytest.raises(ValueError, match="Pyramid level 1 shape mismatch"):
        verify(path, data, names, require_pyramid=True, expected_pyramid_shapes=[(15, 16)])


def test_verifier_rejects_unexpected_extra_pyramid_levels(tmp_path):
    path, data, names = marker_image(tmp_path, pyramid_shapes=[(16, 16)])
    with pytest.raises(ValueError, match="pyramid level count mismatch"):
        verify(path, data, names, expected_pyramid_shapes=[])


def test_singleton_channel_exception_does_not_accept_rgb_samples(tmp_path):
    path = tmp_path / "rgb.ome.tif"
    tifffile.imwrite(path, np.zeros((16, 16, 3), dtype=np.uint16), photometric="rgb", ome=True,
                     metadata={"axes": "YXS", "Channel": {"Name": ["RGB"]}})
    with pytest.raises(ValueError, match="axes must include C,Y,X"):
        exporter.verify_written_ome_tiff(path, expected_shape_cyx=(3, 16, 16),
            expected_dtype=np.dtype("uint16"), expected_channels=["RGB"], require_pyramid=False)


def test_verifier_rejects_multiple_timepoints_in_scalar_marker_export(tmp_path):
    path = tmp_path / "timepoints.ome.tif"
    tifffile.imwrite(path, np.zeros((2, 2, 16, 16), dtype=np.uint16), photometric="minisblack", ome=True,
                     metadata={"axes": "TCYX", "Channel": {"Name": ["CD3", "CD8"]}})
    with pytest.raises(ValueError, match="scalar channel planes"):
        exporter.verify_written_ome_tiff(path, expected_shape_cyx=(2, 16, 16),
            expected_dtype=np.dtype("uint16"), expected_channels=["CD3", "CD8"], require_pyramid=False)
