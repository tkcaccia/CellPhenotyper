import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tifffile


SCRIPT = Path(__file__).parents[1] / "bin" / "validate_input_resolution.py"
SPEC = importlib.util.spec_from_file_location("validate_input_resolution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_tiff(path: Path, mpp_x: float | None, mpp_y: float | None = None) -> None:
    kwargs = {}
    if mpp_x is not None:
        mpp_y = mpp_x if mpp_y is None else mpp_y
        kwargs.update(
            resolution=(10_000.0 / mpp_x, 10_000.0 / mpp_y),
            resolutionunit="CENTIMETER",
        )
    tifffile.imwrite(path, np.zeros((32, 48, 3), dtype=np.uint8), photometric="rgb", **kwargs)


def args_for(path: Path, report: Path, **overrides):
    values = {
        "image": str(path),
        "report": str(report),
        "min_mpp": 0.05,
        "max_mpp": 0.50,
        "cell_target_mpp": 0.25,
        "max_anisotropy_fraction": 0.05,
        "max_conversion_drift_fraction": 0.02,
        "max_metadata_conflict_fraction": 0.02,
        "override_mpp": 0.0,
        "reference_report": "",
        "require_rgb": False,
        "require_pyramid": False,
        "hash_file": True,
        "strict": True,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_accepts_high_resolution_isotropic_tiff(tmp_path):
    image = tmp_path / "high_resolution.tif"
    write_tiff(image, 0.25)

    report, accepted = MODULE.validate(args_for(image, tmp_path / "report.json"))

    assert accepted
    assert report["status"] == "pass"
    assert report["metadata_source"] == "tiff-resolution-tags"
    assert abs(report["effective_mpp"] - 0.25) < 1e-6
    assert report["linear_upsample_factor_to_cell_target"] == 1.0
    assert report["file_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert report["image_layout"]["dtype"] == "uint8"
    assert report["image_layout"]["bits_per_sample"] == 8
    assert report["image_layout"]["channel_count"] == 3
    assert report["image_layout"]["color_interpretation"] == "RGB"
    assert report["image_layout"]["pyramid_level_count"] == 1
    assert report["mpp_conflicts"] == []


def test_ome_physical_size_units_are_normalized_to_micrometres():
    mpp_x, mpp_y, source = MODULE._parse_description(
        '<Pixels PhysicalSizeX="250" PhysicalSizeXUnit="nm" '
        'PhysicalSizeY="0.00025" PhysicalSizeYUnit="mm"/>'
    )
    assert source == "ome-xml"
    assert abs(mpp_x - 0.25) < 1.0e-12
    assert abs(mpp_y - 0.25) < 1.0e-12


def test_rejects_coarse_native_resolution(tmp_path):
    image = tmp_path / "coarse.tif"
    write_tiff(image, 0.5473)

    report, accepted = MODULE.validate(args_for(image, tmp_path / "report.json"))

    assert not accepted
    assert report["status"] == "fail"
    assert any("too coarse" in message for message in report["failures"])
    assert report["linear_upsample_factor_to_cell_target"] > 2.0


def test_rejects_anisotropic_pixels(tmp_path):
    image = tmp_path / "anisotropic.tif"
    write_tiff(image, 0.25, 0.30)

    report, accepted = MODULE.validate(args_for(image, tmp_path / "report.json"))

    assert not accepted
    assert any("anisotropy" in message for message in report["failures"])


def test_override_allows_image_without_physical_metadata(tmp_path):
    image = tmp_path / "missing_metadata.tif"
    write_tiff(image, None)

    report, accepted = MODULE.validate(
        args_for(image, tmp_path / "report.json", override_mpp=0.25)
    )

    assert accepted
    assert report["metadata_source"] == "explicit-override"
    assert report["effective_mpp"] == 0.25


def test_rejects_conversion_mpp_drift(tmp_path):
    source = tmp_path / "source.tif"
    converted = tmp_path / "converted.tif"
    write_tiff(source, 0.25)
    write_tiff(converted, 0.30)

    source_report, accepted = MODULE.validate(args_for(source, tmp_path / "source.json"))
    assert accepted
    (tmp_path / "source.json").write_text(json.dumps(source_report))

    report, accepted = MODULE.validate(
        args_for(
            converted,
            tmp_path / "converted.json",
            reference_report=str(tmp_path / "source.json"),
        )
    )

    assert not accepted
    assert any("Conversion changed physical resolution" in message for message in report["failures"])


def test_rejects_conflicting_mpp_metadata_without_override(tmp_path):
    image = tmp_path / "conflicting.tif"
    tifffile.imwrite(
        image,
        np.zeros((32, 48, 3), dtype=np.uint8),
        photometric="rgb",
        description='PhysicalSizeX="0.25" PhysicalSizeY="0.25"',
        resolution=(20_000.0, 20_000.0),
        resolutionunit="CENTIMETER",
    )

    report, accepted = MODULE.validate(args_for(image, tmp_path / "report.json"))

    assert not accepted
    assert report["mpp_conflicts"]
    assert any("conflicting physical-resolution" in message for message in report["failures"])


def test_explicit_override_resolves_conflicting_mpp_metadata(tmp_path):
    image = tmp_path / "conflicting_override.tif"
    tifffile.imwrite(
        image,
        np.zeros((32, 48, 3), dtype=np.uint8),
        photometric="rgb",
        description='PhysicalSizeX="0.25" PhysicalSizeY="0.25"',
        resolution=(20_000.0, 20_000.0),
        resolutionunit="CENTIMETER",
    )

    report, accepted = MODULE.validate(
        args_for(image, tmp_path / "report.json", override_mpp=0.25)
    )

    assert accepted
    assert report["status"] == "pass"
    assert report["metadata_source"] == "explicit-override"
    assert any("explicit override" in message for message in report["warnings"])


def test_converted_contract_rejects_flat_or_non_rgb_image(tmp_path):
    flat_rgb = tmp_path / "flat_rgb.tif"
    write_tiff(flat_rgb, 0.25)
    report, accepted = MODULE.validate(
        args_for(flat_rgb, tmp_path / "flat.json", require_rgb=True, require_pyramid=True)
    )
    assert not accepted
    assert any("must be pyramidal" in message for message in report["failures"])

    grayscale = tmp_path / "grayscale.tif"
    tifffile.imwrite(
        grayscale,
        np.zeros((32, 48), dtype=np.uint8),
        resolution=(40_000.0, 40_000.0),
        resolutionunit="CENTIMETER",
    )
    report, accepted = MODULE.validate(
        args_for(grayscale, tmp_path / "gray.json", require_rgb=True)
    )
    assert not accepted
    assert any("three-channel RGB-compatible" in message for message in report["failures"])


def test_converted_contract_accepts_tiled_rgb_pyramid(tmp_path):
    image = tmp_path / "pyramid.tif"
    base = np.zeros((32, 48, 3), dtype=np.uint8)
    with tifffile.TiffWriter(image) as writer:
        writer.write(
            base,
            photometric="rgb",
            tile=(16, 16),
            subifds=1,
            resolution=(40_000.0, 40_000.0),
            resolutionunit="CENTIMETER",
        )
        writer.write(
            base[::2, ::2],
            photometric="rgb",
            tile=(16, 16),
            subfiletype=1,
            resolution=(20_000.0, 20_000.0),
            resolutionunit="CENTIMETER",
        )

    report, accepted = MODULE.validate(
        args_for(image, tmp_path / "report.json", require_rgb=True, require_pyramid=True)
    )

    assert accepted
    assert report["image_layout"]["is_tiled"] is True
    assert report["image_layout"]["pyramid_level_count"] == 2
