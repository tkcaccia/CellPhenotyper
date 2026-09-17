import importlib.util
import json
import sys
from pathlib import Path

import pytest


pytest.importorskip("shapely")


SCRIPT = Path(__file__).parents[1] / "bin" / "validate_roi_geojson.py"
SPEC = importlib.util.spec_from_file_location("validate_roi_geojson", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_case(tmp_path: Path, payload: dict):
    image = tmp_path / "sample.ome.tif"
    image.write_bytes(b"image-placeholder")
    image_report = tmp_path / "sample.converted_resolution.json"
    image_report.write_text(
        json.dumps(
            {
                "status": "pass",
                "image": str(image),
                "width_px": 100,
                "height_px": 80,
                "file_sha256": "a" * 64,
            }
        )
    )
    roi = tmp_path / "sample.geojson"
    roi.write_text(json.dumps(payload))
    return image, image_report, roi


def args_for(image: Path, image_report: Path, roi: Path, **overrides):
    values = {
        "geojson": str(roi),
        "image": str(image),
        "image_report": str(image_report),
        "report": str(roi.with_suffix(".qc.json")),
        "sample_id": "sample",
        "source_kind": "provided",
        "source_name": roi.name,
        "mode": "fail",
        "bounds_tolerance_px2": 1.0e-6,
    }
    values.update(overrides)
    return type("Args", (), values)()


def feature(coordinates, properties=None):
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {"type": "Polygon", "coordinates": coordinates},
    }


def valid_payload():
    return {
        "type": "FeatureCollection",
        "pipeline_metadata": {
            "source_image": "sample.ome.tif",
            "coordinate_space": "level0_pixels",
        },
        "features": [
            feature(
                [
                    [[0, 0], [100, 0], [100, 80], [0, 80], [0, 0]],
                    [[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]],
                ]
            )
        ],
    }


def test_valid_roi_preserves_holes_and_records_hashes(tmp_path):
    image, image_report, roi = write_case(tmp_path, valid_payload())

    report, accepted = MODULE.validate(args_for(image, image_report, roi))

    assert accepted
    assert report["status"] == "pass"
    assert report["geometry_modified"] is False
    assert report["hole_count"] == 1
    assert report["valid_in_bounds_feature_count"] == 1
    assert len(report["roi_sha256"]) == 64
    assert report["image_sha256"] == "a" * 64


def test_self_intersection_is_not_repaired_silently(tmp_path):
    payload = valid_payload()
    payload["features"] = [
        feature([[[10, 10], [60, 60], [10, 60], [60, 10], [10, 10]]])
    ]
    image, image_report, roi = write_case(tmp_path, payload)

    report, accepted = MODULE.validate(args_for(image, image_report, roi))

    assert not accepted
    assert report["geometry_modified"] is False
    assert any("not repaired" in message or "zero area" in message for message in report["failures"])


def test_out_of_bounds_geometry_is_not_clipped(tmp_path):
    payload = valid_payload()
    payload["features"] = [
        feature([[[-5, 10], [50, 10], [50, 60], [-5, 60], [-5, 10]]])
    ]
    image, image_report, roi = write_case(tmp_path, payload)

    report, accepted = MODULE.validate(args_for(image, image_report, roi))

    assert not accepted
    assert report["outside_image_area_px2"] > 0
    assert any("was not clipped" in message for message in report["failures"])


def test_mismatched_image_association_fails(tmp_path):
    payload = valid_payload()
    payload["pipeline_metadata"]["source_image"] = "different_slide.svs"
    image, image_report, roi = write_case(tmp_path, payload)

    report, accepted = MODULE.validate(args_for(image, image_report, roi))

    assert not accepted
    assert report["mismatched_image_associations"] == ["different_slide.svs"]


def test_geographic_crs_fails_pixel_coordinate_contract(tmp_path):
    payload = valid_payload()
    payload["crs"] = {"type": "name", "properties": {"name": "EPSG:4326"}}
    image, image_report, roi = write_case(tmp_path, payload)

    report, accepted = MODULE.validate(args_for(image, image_report, roi))

    assert not accepted
    assert any("Geographic/world" in message for message in report["failures"])


def test_warn_mode_keeps_legacy_roi_but_reports_failure(tmp_path):
    payload = valid_payload()
    payload["features"] = [
        feature([[[-5, 10], [50, 10], [50, 60], [-5, 60], [-5, 10]]])
    ]
    image, image_report, roi = write_case(tmp_path, payload)

    report, accepted = MODULE.validate(
        args_for(image, image_report, roi, mode="warn")
    )

    assert accepted
    assert report["status"] == "warning"
    assert report["failures"]


def test_prepare_roi_module_escapes_runtime_source_kind() -> None:
    module = (Path(__file__).parents[1] / "modules" / "prepare_roi_geojson.nf").read_text()

    assert '--source-kind "\\$ROI_SOURCE_KIND"' in module
