import importlib.util
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
        "override_mpp": 0.0,
        "reference_report": "",
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
