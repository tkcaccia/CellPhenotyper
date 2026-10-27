"""Fail-closed checks for converting manually drawn viewer ROIs."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from prepare_bora_expert_geojson import digest, prepare


def save(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def case(tmp_path):
    image = tmp_path / "native.tif"
    image.write_bytes(b"fixed synthetic image bytes")
    frame = dict(frame_id="frame-1", sha256=digest(image), width=5000, height=5000,
                 mpp_x=0.25, mpp_y=0.25, origin_x=0, origin_y=0,
                 coordinate_space="crop-native-pixels",
                 coordinate_convention="pixel-edge-origin_centres-in-native-pixels")
    project = save(tmp_path / "experiment.cellphenotyper.json",
                   {"tissues": [{"image": dict(frame, path="native.tif")}]})
    registration = save(tmp_path / "registration.json", {"cell_full_image": frame})
    inventory = save(tmp_path / "CHUNK_INVENTORY.json", {"complete": True})
    boxes = [(100, 100, 400, 400), (800, 100, 1100, 400),
             (2500, 2500, 2800, 2800), (3400, 2500, 3700, 2800)]
    features = []
    assignments = {}
    for n, ((x0, y0, x1, y1), cid, split) in enumerate(zip(
            boxes, (1, 2, 1, 2), ("train", "train", "holdout", "holdout")), 1):
        fid = f"drawn_roi_{n}"
        klass = {1: "lymphoid", 2: "stroma"}[cid]
        features.append({"type": "Feature", "id": fid,
                         "properties": {"source": "drawn", "class": klass,
                                        "classification": {"name": klass},
                                        "wsiTools": {"coordinate_space": "level0_slide_pixels",
                                                     "display_transform_applied": False,
                                                     "slide_width": 5000, "slide_height": 5000}},
                         "geometry": {"type": "Polygon", "coordinates": [[
                             [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}})
        assignments[fid] = {"class_id": cid, "split": split, "roi_id": f"region-{n}",
                            "annotator_id": "reviewer-1", "annotation_source": "independent_expert"}
    export = save(tmp_path / "reviewed.geojson", {"type": "FeatureCollection", "features": features})
    review = save(tmp_path / "review.json", {
        "format": "cellphenotyper-wsitools-expert-review-v1",
        "source_geojson_sha256": digest(export), "project_sha256": digest(project),
        "class_names": {"1": "lymphoid", "2": "stroma"},
        "reviewed_features": assignments})
    return dict(project=project, registration=registration, image=image,
                inventory=inventory, export=export, review=review)


def convert(c):
    return prepare(c["project"], c["registration"], c["image"], c["inventory"],
                   c["export"], c["review"])


def resave_export(c, update):
    document = json.loads(c["export"].read_text())
    update(document)
    save(c["export"], document)
    review = json.loads(c["review"].read_text())
    review["source_geojson_sha256"] = digest(c["export"])
    save(c["review"], review)


def test_preserves_native_frame_and_manual_geometries(case):
    output = convert(case)
    source = output["source"]
    assert output["schema"] == "cellphenotyper-expert-tissue-v1"
    assert source["image_sha256"] == digest(case["image"])
    assert source["token_inventory_sha256"] == digest(case["inventory"])
    assert source["native_mpp_xy"] == [0.25, 0.25]
    assert source["export_geojson_sha256"] == digest(case["export"])
    assert output["features"][0]["geometry"] == json.loads(case["export"].read_text())["features"][0]["geometry"]
    assert {f["properties"]["split"] for f in output["features"]} == {"train", "holdout"}


def test_rejects_predicted_source_even_if_review_manifest_declares_expert(case):
    resave_export(case, lambda a: a["features"][0]["properties"].update(source="Bora_prediction"))
    with pytest.raises(ValueError, match="imported/predicted"):
        convert(case)


def test_rejects_model_provenance_on_manually_named_shape(case):
    resave_export(case, lambda a: a["features"][0]["properties"].update(source_role="bora_local_exact_semantic_vector"))
    with pytest.raises(ValueError, match="model/import"):
        convert(case)


def test_rejects_missing_annotator_or_split(case):
    review = json.loads(case["review"].read_text())
    del review["reviewed_features"]["drawn_roi_1"]["annotator_id"]
    save(case["review"], review)
    with pytest.raises(ValueError, match="explicit split, annotator"):
        convert(case)


def test_rejects_class_not_matching_viewer(case):
    review = json.loads(case["review"].read_text())
    review["reviewed_features"]["drawn_roi_1"]["class_id"] = 2
    save(case["review"], review)
    with pytest.raises(ValueError, match="class does not match"):
        convert(case)


def test_rejects_changed_export_without_updated_review_digest(case):
    document = json.loads(case["export"].read_text())
    document["features"][0]["properties"]["name"] = "changed"
    save(case["export"], document)
    with pytest.raises(ValueError, match="exact exported GeoJSON"):
        convert(case)


def test_rejects_wrong_native_frame(case):
    resave_export(case, lambda a: a["features"][0]["properties"]["wsiTools"].update(slide_width=4999))
    with pytest.raises(ValueError, match="registered level-0"):
        convert(case)


def test_rejects_train_holdout_gap_violation(case):
    def move(a):
        a["features"][2]["geometry"]["coordinates"] = [[[1200, 600], [1500, 600],
            [1500, 900], [1200, 900], [1200, 600]]]
    resave_export(case, move)
    with pytest.raises(ValueError, match="224 µm spatial gap"):
        convert(case)
