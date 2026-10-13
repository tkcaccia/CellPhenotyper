from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from geojson_coordinates import (  # noqa: E402
    CROP_FRAME,
    ORIGINAL_FRAME,
    original_to_crop_feature_collection,
    read_crop_transform,
    rebase_feature_collection,
)
from select_neoplastic_section import load_sections, write_geojson  # noqa: E402


def shift_file(tmp_path: Path) -> Path:
    path = tmp_path / "shift.json"
    path.write_text(json.dumps({
        "offset_crop_to_original": {"dx": 100, "dy": 200},
        "crop_size": {"width": 20, "height": 10},
        "full_size": {"width": 500, "height": 600},
        "crop_bbox_xyxy": {"x0": 100, "y0": 200, "x1": 120, "y1": 210},
    }))
    return path


def collection() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"value": 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[1, 2], [19, 2], [19, 9], [1, 9], [1, 2]]],
            },
        }],
    }


def test_crop_geojson_round_trips_through_original_image_coordinates(tmp_path: Path) -> None:
    shift = shift_file(tmp_path)
    transform = read_crop_transform(shift)
    published = rebase_feature_collection(collection(), transform)
    assert published["cellphenotyper_coordinate_system"]["coordinate_space"] == ORIGINAL_FRAME
    assert published["features"][0]["geometry"]["coordinates"][0][0] == [101.0, 202.0]
    assert published["features"][0]["properties"]["coordinate_space"] == ORIGINAL_FRAME
    restored = original_to_crop_feature_collection(published, shift)
    assert restored["cellphenotyper_coordinate_system"]["coordinate_space"] == CROP_FRAME
    assert restored["features"][0]["geometry"] == collection()["features"][0]["geometry"]


def test_rebase_rejects_geometry_outside_declared_crop(tmp_path: Path) -> None:
    payload = collection()
    payload["features"][0]["geometry"]["coordinates"][0][1][0] = 21
    with pytest.raises(ValueError, match="outside the analysis crop"):
        rebase_feature_collection(payload, read_crop_transform(shift_file(tmp_path)))


def test_shift_contract_fails_on_inconsistent_bbox(tmp_path: Path) -> None:
    path = shift_file(tmp_path)
    payload = json.loads(path.read_text())
    payload["crop_bbox_xyxy"]["x1"] = 121
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disagree"):
        read_crop_transform(path)


def test_downstream_section_selection_inverts_only_for_internal_crop_use(tmp_path: Path) -> None:
    pytest.importorskip("shapely")
    shift = shift_file(tmp_path)
    published = rebase_feature_collection(collection(), read_crop_transform(shift))
    source = tmp_path / "domains.geojson"
    source.write_text(json.dumps(published))
    sections, frame = load_sections(source, "standard", shift)
    assert frame == ORIGINAL_FRAME
    assert sections[0]["geometry"].bounds == (1.0, 2.0, 19.0, 9.0)
    selected = tmp_path / "selected.geojson"
    write_geojson(sections[0], selected, original_offset=(100, 200))
    output = json.loads(selected.read_text())
    assert output["cellphenotyper_coordinate_system"]["coordinate_space"] == ORIGINAL_FRAME
    assert output["features"][0]["geometry"]["coordinates"][0][0] == [101.0, 202.0]
