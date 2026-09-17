"""Native-window and real localhost API checks; these do not open a browser."""
import http.client
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "bin"))
import cell_inspector as module
from cell_profile_io import sha256_file


MARKER = "predicted__nucleus__CD3__mean"


@pytest.fixture
def specimen(tmp_path):
    h, w = 128, 160
    pixels = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    labels = np.zeros((h, w), dtype=np.uint32)
    xy = np.array([[15, 30], [45, 30], [75, 30], [105, 30], [135, 30]])
    ids = ["001", "7", "9", "12", "15"]
    for (x, y), cell_id in zip(xy, ids):
        labels[y-4:y+5, x-4:x+5] = int(cell_id)
    image_path, label_path = tmp_path / "he.tif", tmp_path / "labels.tif"
    tifffile.imwrite(image_path, pixels, photometric="rgb", tile=(16, 16), compression="deflate")
    tifffile.imwrite(label_path, labels, tile=(16, 16), compression="deflate")
    shift_path = tmp_path / "shift.json"
    shift_path.write_text(json.dumps({"source_mpp": 0.5, "offset_crop_to_original": {"dx": 100, "dy": 200},
                                     "crop_size": {"width": w, "height": h}}))
    root = tmp_path / "profiles"
    root.mkdir()
    cells = pd.DataFrame({"sample_id": ["specimen A"] * 5, "cell_id": ids,
        "cell_uid": ["specimen A:<script>&/ +:001"] + ["specimen A:seg:" + value for value in ids[1:]],
        "x_crop_px": xy[:, 0], "y_crop_px": xy[:, 1],
        "x_um": (xy[:, 0] + 100) * 0.5, "y_um": (xy[:, 1] + 200) * 0.5,
        MARKER: [0., .7, .1, np.nan, .9], "agreement_tier": ["high", "low", "high", "high", "high"],
        "cellvitpp_id": ["004", "9", "6", "7", "8"], "stardist_iou": [0.9, 0.1, .7, .8, .9],
        "morphology__morphology_status": ["ok", "disconnected_label", "ok", "ok", "ok"],
        "morphology__touches_image_edge": [False, True, False, False, False],
        "tissue_domain": [1, 0, 2, 2, 2],
        "tissue_domain_status": ["assigned", "unknown_or_outside_support", "assigned", "assigned", "assigned"],
        "niche_id": [1, 1, 0, 2, 2],
        "niche_status": ["assigned_exploratory", "assigned_fixed_k", "outside_tissue_support",
                          "single_niche_no_supported_subdivision", "assigned_exploratory"],
        "niche_stability": [1., .5, np.nan, np.nan, .9], "phenotype": ["Epithelial"] * 5})
    cells.to_csv(root / "cell_profiles.csv", index=False)
    cells[["sample_id", "cell_id", "cell_uid"]].to_csv(root / "feature_rows.csv", index=False)
    blocks = {"context": np.array([[0, 0], [1, 0], [-1, 0], [2, 0], [np.nan, np.nan]], dtype=np.float32),
              "local": np.array([[0, 0], [10, 0], [0, 0], [1, 0], [.1, .1]], dtype=np.float32)}
    for name, values in blocks.items():
        np.save(root / (name + ".npy"), values)
    manifest = {"schema_version": "1.0.0", "observation_unit": "cell", "sample_id": "specimen A",
        "cell_count": len(cells), "source_mpp": 0.5, "crop_origin_um": [50., 100.],
        "crop_size_px": [w, h], "biological_marker_features": [MARKER],
        "feature_blocks": {name: {"path": name + ".npy", "shape": list(values.shape),
            "sha256": sha256_file(root / (name + ".npy")), "reference_compatible": name == "context",
            "feature_definition": {"model": "synthetic-test", "context": name}} for name, values in blocks.items()},
        "files": {name: sha256_file(root / name) for name in ("cell_profiles.csv", "feature_rows.csv")}}
    (root / "cell_profiles_manifest.json").write_text(json.dumps(manifest))
    return {"profile_dir": root, "image": image_path, "labels": label_path, "shift": shift_path}, cells, pixels, labels


@pytest.fixture
def inspector(specimen):
    return module.CellInspector(**specimen[0])


def test_detail_preserves_exact_identity_and_missing_values(inspector, specimen):
    _, cells, _, _ = specimen
    detail = inspector.detail(cells.cell_uid.iloc[0], 64)
    assert detail["cell_id"] == "001"
    assert detail["cell_uid"] == "specimen A:<script>&/ +:001"
    assert detail["detector_evidence"]["cellvitpp_id"] == "004"
    assert detail["coordinates_um"] == [57.5, 115.]
    assert detail["crop_xy"] == [15., 30.]
    assert detail["window"] == {"x0": 0, "y0": 0, "x1": 64, "y1": 64}
    assert detail["feature_blocks"]["context"] == {"dimension": 2, "available": True,
        "missing_dimensions": 0, "reference_compatible": True}
    assert inspector.detail(cells.cell_uid.iloc[3])["markers"][MARKER] is None
    missing = inspector.detail(cells.cell_uid.iloc[4])["feature_blocks"]["context"]
    assert missing["available"] is False and missing["missing_dimensions"] == 2
    assert detail["expert_annotation_used_for_inference"] is False
    assert inspector.search("<script>&/ +")[0]["cell_id"] == "001"
    assert inspector.search("001")[0]["cell_uid"] == cells.cell_uid.iloc[0]
    json.dumps(detail, allow_nan=False)


def test_native_png_and_only_selected_nuclear_boundary_are_bounded(inspector, specimen, monkeypatch):
    _, cells, pixels, labels = specimen
    calls = []
    original_window = module.RasterReader.window
    def bounded(self, x0, y0, x1, y1):
        calls.append((self.path, x1-x0, y1-y0))
        assert x1-x0 <= 64 and y1-y0 <= 64
        return original_window(self, x0, y0, x1, y1)
    monkeypatch.setattr(module.RasterReader, "window", bounded)
    monkeypatch.setattr(tifffile, "imread", lambda *args, **kwargs: pytest.fail("Whole-image decoder used"))
    uid = cells.cell_uid.iloc[0]
    base = np.asarray(Image.open(io.BytesIO(inspector.png(uid, 64))))
    outline = np.asarray(Image.open(io.BytesIO(inspector.png(uid, 64, overlay=True))))
    np.testing.assert_array_equal(base, pixels[:64, :64])
    assert outline.shape == (64, 64, 4)
    assert outline[:, :, 3].sum() > 0
    assert not (outline[labels[:64, :64] != 1, 3] > 0).any()
    assert outline[30, 15, 3] == 0  # nucleus interior remains transparent
    assert outline[26, 15, 3] == 255
    assert calls == [(inspector.image, 64, 64), (inspector.labels, 64, 64)]


def test_pick_exact_labels_and_physical_radius(inspector, specimen):
    cells = specimen[1]
    assert inspector.pick(15, 30) == {"status": "selected", "cell_uid": cells.cell_uid.iloc[0], "method": "exact_label"}
    assert inspector.pick(45.2, 30.8)["cell_uid"] == cells.cell_uid.iloc[1]
    assert inspector.pick(15, 40, radius_um=6)["method"] == "nearest_centroid"
    assert inspector.pick(15, 40, radius_um=4)["status"] == "no_nearby_cell"
    with pytest.raises(ValueError, match="outside"):
        inspector.pick(-1, 0)
    with pytest.raises(ValueError, match="finite"):
        inspector.pick(np.nan, 1)
    with pytest.raises(ValueError, match="radius"):
        inspector.pick(1, 1, radius_um=101)


def test_foreign_clicked_label_fails_clearly(inspector, monkeypatch):
    monkeypatch.setattr(module.RasterReader, "window", lambda *args: np.array([[999]], dtype=np.uint32))
    with pytest.raises(ValueError, match="no canonical profile"):
        inspector.pick(10, 10)


def test_review_queue_flags_issues_not_normal_niche_labels(inspector, specimen):
    cells = specimen[1]
    queue = inspector.queue()
    assert queue["total"] == 2
    assert queue["rows"][0]["cell_uid"] == cells.cell_uid.iloc[1]
    assert set(queue["rows"][0]["reasons"]) == {"detector_disagreement", "morphology_issue", "image_edge",
        "unknown_domain", "uncertain_domain", "unstable_niche"}
    assert queue["rows"][1]["reasons"] == ["niche_review"]
    assert inspector.queue(reason="morphology_issue")["total"] == 1
    assert inspector.queue(offset=1, limit=1)["rows"][0]["cell_uid"] == cells.cell_uid.iloc[2]
    assert not inspector.detail(cells.cell_uid.iloc[3])["review_reasons"]
    assert module.review_reasons({"phenotype": "unknown", "reference_status": "outside_reference"}) == ["unknown_phenotype", "unknown_reference"]


def test_compartment_availability_and_technical_controls_are_explicit(inspector, specimen):
    control = "predicted__nucleus__TRITC__mean"
    inspector.cells[control] = [.1] * len(inspector.cells)
    inspector.cells["compartment__cytoplasm_status"] = "proxy_not_membrane_segmentation"
    inspector.cells["uni2_context_available"] = False
    inspector.cells["r25um_phenotype_fraction:epithelial"] = .5
    inspector.manifest["tables"] = {"nucleus": {"background_columns": [control]}}
    detail = inspector.detail(specimen[1].cell_uid.iloc[0])
    assert control not in detail["markers"]
    assert detail["technical_marker_controls"][control] == .1
    assert detail["compartment_evidence"]["compartment__cytoplasm_status"] == "proxy_not_membrane_segmentation"
    assert detail["feature_availability"]["uni2_context_available"] is False
    assert detail["spatial"]["r25um_phenotype_fraction:epithelial"] == .5
    assert detail["detector_evidence"]["phenotype"] == "Epithelial"


def test_similarity_separate_blocks_ties_weights_missing_vectors(inspector, specimen):
    cells = specimen[1]
    query = cells.cell_uid.iloc[0]
    context = inspector.similar(query, ["context"])
    assert [row["cell_id"] for row in context["results"]] == ["7", "9", "12"]
    assert context["results"][0]["distance"] == context["results"][1]["distance"]
    assert context["results"][0]["distance"] == pytest.approx(np.sqrt(.4))
    assert all(row["sample_id"] == "specimen A" for row in context["results"])
    assert "not calibrated" in context["distance_semantics"]
    assert context["expert_annotations_used"] is False
    local = inspector.similar(query, ["local"])
    assert local["results"][0]["cell_id"] == "9"
    assert local["results"][0]["distance"] == 0.
    weighted = inspector.similar(query, ["context", "local"], [1., 10.])
    assert [r["cell_id"] for r in weighted["results"]] == ["9", "12", "7"]
    assert not any(row["cell_id"] == "15" for row in weighted["results"])
    assert inspector.similar(cells.cell_uid.iloc[4], ["context"])["status"] == "missing_features"
    assert inspector.similar(cells.cell_uid.iloc[4], ["local"])["status"] == "matched"
    assert len(inspector.similar(query, ["context"], k=1)["results"]) == 1


def test_predicted_marker_discordance_is_optional_and_preserves_unknown(inspector, specimen):
    cells = specimen[1]
    result = inspector.similar(cells.cell_uid.iloc[0], ["context"], marker=MARKER, min_difference=.5)
    assert [row["cell_id"] for row in result["results"]] == ["7"]
    assert result["results"][0]["predicted_marker_difference"] == .7
    assert inspector.similar(cells.cell_uid.iloc[3], ["context"], marker=MARKER, min_difference=.5)["status"] == "missing_marker"
    assert inspector.similar(cells.cell_uid.iloc[0], ["context"], marker=MARKER, min_difference=5)["status"] == "no_matching_cells"


@pytest.mark.parametrize("options", [
    {"groups": []}, {"groups": ["not_a_feature"]}, {"groups": ["context", "context"]},
    {"groups": ["context"], "weights": [0]}, {"groups": ["context"], "weights": [np.nan]},
    {"groups": ["context"], "weights": [1, 2]}, {"groups": ["context"], "k": 0},
    {"groups": ["context"], "marker": MARKER},
    {"groups": ["context"], "marker": "agreement_tier", "min_difference": .5},
    {"groups": ["context"], "marker": MARKER, "min_difference": -1},
])
def test_similarity_rejects_ambiguous_or_invalid_requests(inspector, specimen, options):
    with pytest.raises(ValueError):
        inspector.similar(specimen[1].cell_uid.iloc[0], **options)


def test_expert_annotations_are_display_only(inspector, specimen, tmp_path):
    uid = specimen[1].cell_uid.iloc[0]
    note = {"cell_uid": uid, "reviewed_label": "Unrelated expert description", "reviewer": "Expert A"}
    path = tmp_path / "expert.json"
    path.write_text(json.dumps({"annotations": [note]}))
    annotated = module.CellInspector(**specimen[0], expert_annotations=path)
    assert annotated.detail(uid)["expert_annotation"] == note
    assert annotated.similar(uid, ["context"]) == inspector.similar(uid, ["context"])
    assert annotated.queue() == inspector.queue()
    assert annotated.metadata()["expert_annotations"] == 1
    path.write_text(json.dumps({"annotations": [note, note]}))
    with pytest.raises(ValueError, match="duplicate"):
        module.CellInspector(**specimen[0], expert_annotations=path)
    path.write_text(json.dumps({"annotations": [{"cell_uid": "foreign"}]}))
    with pytest.raises(ValueError, match="foreign"):
        module.CellInspector(**specimen[0], expert_annotations=path)


def test_feature_path_escape_and_shape_rejected(specimen, tmp_path):
    inputs = specimen[0]
    path = inputs["profile_dir"] / "cell_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["feature_blocks"]["context"]["path"] = "../outside.npy"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        module.CellInspector(**inputs)
    manifest["feature_blocks"]["context"]["path"] = "context.npy"
    manifest["feature_blocks"]["context"]["shape"] = [999, 2]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row-aligned"):
        module.CellInspector(**inputs)


def test_hash_and_row_identity_mismatch_rejected(specimen):
    inputs = specimen[0]
    values = np.load(inputs["profile_dir"] / "context.npy")
    values[0, 0] = 99
    np.save(inputs["profile_dir"] / "context.npy", values)
    with pytest.raises(ValueError, match="hash mismatch"):
        module.CellInspector(**inputs)
    rows_path = inputs["profile_dir"] / "feature_rows.csv"
    rows = pd.read_csv(rows_path, dtype=str)
    rows.iloc[::-1].to_csv(rows_path, index=False)
    with pytest.raises(ValueError, match="do not align"):
        module.CellInspector(**inputs)


def test_native_dimensions_and_display_dtype_are_explicit(specimen, tmp_path):
    inputs = dict(specimen[0])
    wrong = tmp_path / "wrong.tif"
    tifffile.imwrite(wrong, np.zeros((10, 10, 3), dtype=np.uint8), photometric="rgb")
    inputs["image"] = wrong
    with pytest.raises(ValueError, match="dimensions"):
        module.CellInspector(**inputs)
    tifffile.imwrite(wrong, np.zeros((128, 160, 3), dtype=np.uint16), photometric="rgb")
    with pytest.raises(ValueError, match="uint8"):
        module.CellInspector(**inputs)


def bind_raster_hashes(specimen, names=("image", "labels")):
    inputs = specimen[0]
    path = inputs["profile_dir"] / "cell_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["inputs"] = {name + "_sha256": sha256_file(inputs[name]) for name in names}
    path.write_text(json.dumps(manifest))
    return manifest


def test_legacy_raster_provenance_is_explicit_not_verified(inspector, specimen):
    assert inspector.metadata()["raster_identity_status"] == "legacy_partial_or_unverified"
    assert inspector.metadata()["raster_provenance"] == {
        name: {"status": "legacy_missing_sha256", "sha256": None} for name in ("image", "labels")}
    assert inspector.detail(specimen[1].cell_uid.iloc[0])["detector_evidence"]["source_raster_identity"] == "legacy_partial_or_unverified"


@pytest.mark.parametrize("changed", ["image", "labels"])
def test_same_dimensions_and_cell_ids_do_not_override_mismatched_source_hash(specimen, changed):
    bind_raster_hashes(specimen)
    checked = module.CellInspector(**specimen[0])
    assert checked.metadata()["raster_identity_status"] == "verified_sha256"
    assert all(record["status"] == "verified_sha256" for record in checked.metadata()["raster_provenance"].values())
    if changed == "image":
        pixels = specimen[2].copy()
        pixels[0, 0, 0] ^= 1
        tifffile.imwrite(specimen[0][changed], pixels, photometric="rgb", tile=(16, 16), compression="deflate")
    else:
        labels = specimen[3].copy()
        labels[26, 11] = 0
        assert set(np.unique(labels)) == set(np.unique(specimen[3]))
        tifffile.imwrite(specimen[0][changed], labels, tile=(16, 16), compression="deflate")
    with pytest.raises(ValueError, match=f"Inspector {changed} SHA256 does not match"):
        module.CellInspector(**specimen[0])


def test_partial_raster_hash_provenance_does_not_upgrade_legacy_image(specimen):
    bind_raster_hashes(specimen, names=("labels",))
    checked = module.CellInspector(**specimen[0])
    assert checked.metadata()["raster_identity_status"] == "legacy_partial_or_unverified"
    assert checked.metadata()["raster_provenance"]["labels"]["status"] == "verified_sha256"
    assert checked.metadata()["raster_provenance"]["image"]["status"] == "legacy_missing_sha256"


@pytest.mark.parametrize("value", [None, "", "not-a-hash", "g" * 64, ["a" * 64]])
def test_declared_invalid_raster_hash_is_not_treated_as_legacy(specimen, value):
    manifest = bind_raster_hashes(specimen)
    manifest["inputs"]["image_sha256"] = value
    (specimen[0]["profile_dir"] / "cell_profiles_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not a valid source hash"):
        module.CellInspector(**specimen[0])


@pytest.fixture
def running_server(inspector):
    server = module.make_server(inspector, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def request(server, path, *, method="GET", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_real_localhost_server_identity_png_and_frontend(running_server, specimen):
    assert running_server.server_address[0] == "127.0.0.1"
    code, headers, content = request(running_server, "/")
    assert code == 200 and headers["Content-Type"].startswith("text/html")
    assert b"CellPhenotyper" in content and b"/api/similar" in content
    code, headers, content = request(running_server, "/api/metadata")
    assert code == 200 and json.loads(content)["cell_count"] == 5
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    uid = specimen[1].cell_uid.iloc[0]
    code, _, content = request(running_server, "/api/cell?" + urlencode({"uid": uid, "size": 64}))
    assert code == 200 and json.loads(content)["cell_uid"] == uid
    code, headers, content = request(running_server, "/api/window.png?" + urlencode({"uid": uid, "size": 64}))
    assert code == 200 and headers["Content-Type"] == "image/png"
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(content))), specimen[2][:64, :64])
    code, _, content = request(running_server, "/api/similar?" + urlencode({"uid": uid, "groups": "context"}))
    assert code == 200 and json.loads(content)["results"][0]["cell_id"] == "7"
    code, _, content = request(running_server, "/api/pick?x=15&y=30")
    assert code == 200 and json.loads(content)["cell_uid"] == uid


@pytest.mark.parametrize("path,method,headers,expected", [
    ("/etc/passwd", "GET", {}, 404), ("/../../etc/passwd", "GET", {}, 404),
    ("/api/cell?path=/etc/passwd", "GET", {}, 400),
    ("/api/search?q=a&q=b", "GET", {}, 400),
    ("/api/metadata", "GET", {"Host": "attacker.example"}, 403),
    ("/api/metadata", "GET", {"Origin": "https://attacker.example"}, 403),
    ("/api/cell?uid=unknown", "GET", {}, 404),
    ("/api/pick?x=-1&y=0", "GET", {}, 400),
    ("/api/search?limit=50000", "GET", {}, 400),
    ("/api/metadata", "POST", {}, 405), ("/api/metadata", "DELETE", {}, 405),
])
def test_http_paths_are_fixed_local_only_and_read_only(running_server, path, method, headers, expected):
    code, _, content = request(running_server, path, method=method, headers=headers)
    assert code == expected
    assert "error" in json.loads(content)


def test_frontend_has_safe_text_rendering_and_valid_javascript():
    html = module.HTML_PATH.read_text()
    assert ".innerHTML" not in html and "document.write" not in html
    assert "expert_annotation_used_for_inference" not in html  # no frontend mutation of this guarantee
    assert "textContent" in html and "URLSearchParams" in html
    assert all('id="' + element + '"' in html for element in (
        "cell-window", "cell-map", "search-form", "review-results", "feature-groups", "expert-note"))
    node = shutil.which("node")
    if node:
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        result = subprocess.run([node, "--check", "-"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_cli_help_has_only_explicit_startup_paths():
    completed = subprocess.run([sys.executable, str(module.__file__), "--help"], text=True, capture_output=True)
    assert completed.returncode == 0
    for arg in ("--profile-dir", "--image", "--labels", "--shift", "--expert-annotations", "--port"):
        assert arg in completed.stdout


def test_portable_atlas_launch_and_check_only_without_a_server(specimen, tmp_path):
    execution = tmp_path / "00_execution"
    execution.mkdir()
    atlas_path = execution / "specimen_atlas.json"
    launch = {"status": "available", "path_base": "specimen_atlas_json_directory",
              "datasets": {key: os.path.relpath(value, execution) for key, value in specimen[0].items()}}
    atlas_path.write_text(json.dumps({"specimens": [{"sample_id": "specimen A", "cell_inspector": launch}]}))
    sources = module.atlas_launch_inputs(atlas_path, "specimen A")
    assert sources == {name: path.absolute() for name, path in specimen[0].items()}
    command = [sys.executable, module.__file__, "--atlas-manifest", str(atlas_path),
               "--sample-id", "specimen A", "--check-only"]
    checked = subprocess.run(command, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["cell_count"] == 5
    assert "http://" not in checked.stdout
    with pytest.raises(ValueError, match="exact sample"):
        module.atlas_launch_inputs(atlas_path, "another sample")
    launch["datasets"]["image"] = "../../outside.tif"
    atlas_path.write_text(json.dumps({"specimens": [{"sample_id": "specimen A", "cell_inspector": launch}]}))
    with pytest.raises(ValueError, match="escapes"):
        module.atlas_launch_inputs(atlas_path, "specimen A")


def test_atlas_cli_cannot_be_mixed_with_untracked_sources():
    command = [sys.executable, module.__file__, "--atlas-manifest", "not_read.json", "--sample-id", "A",
               "--image", "arbitrary.tif", "--check-only"]
    checked = subprocess.run(command, capture_output=True, text=True)
    assert checked.returncode == 2 and "cannot be mixed" in checked.stderr


@pytest.fixture
def region_specimen(specimen, tmp_path):
    inputs, cells, _, labels = specimen
    inputs = dict(inputs)
    manifest = bind_raster_hashes(specimen)
    root = tmp_path / "hierarchy"
    directory = root / "region_profiles"
    directory.mkdir(parents=True)
    regions, parents, status = (np.zeros(labels.shape, np.uint32) for _ in range(3))
    parents[:80, :60] = 1
    parents[:80, 60:] = 2
    regions[:80, :60], regions[:80, 60:120], regions[:80, 120:140] = 1, 2, 3
    status[parents > 0] = 2
    status[regions > 0] = 1
    for name, data in (("region_mask.ome.tif", regions), ("parent_domains.ome.tif", parents), ("hierarchy_status.ome.tif", status)):
        tifffile.imwrite(root / name, data, tile=(16, 16), compression="deflate")
    resolution = tmp_path / "source.converted_resolution.json"
    resolution.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5, "width_px": 1000, "height_px": 1000}))
    inputs.update(hierarchy_dir=root, resolution_json=resolution)
    manifest["inputs"].update(domain_mask_sha256=sha256_file(root / "parent_domains.ome.tif"),
        shift_sha256=sha256_file(inputs["shift"]), resolution_json_sha256=sha256_file(resolution))
    (inputs["profile_dir"] / "cell_profiles_manifest.json").write_text(json.dumps(manifest))
    rows = []
    for value, x in ((1, 20), (2, 90), (3, 130)):
        yy, xx = np.nonzero(regions == value)
        rows.append({"region_uid": "specimen A::hierarchy_v1::region_" + str(value), "region_id": str(value),
            "sample_id": "specimen A", "parent_domain_id": 1 if value == 1 else 2, "subdomain_id": value,
            "area_um2": len(xx) * .25, "x_um": (xx.mean()+.5+100)*.5, "y_um": (yy.mean()+.5+200)*.5,
            "representative_grid_id": value, "representative_x_um": (x+100)*.5, "representative_y_um": 115.})
    table = pd.DataFrame(rows)
    table.to_csv(directory / "region_profiles.csv", index=False)
    table[["region_uid", "region_id", "sample_id"]].to_csv(directory / "feature_rows.csv", index=False)
    grid = pd.DataFrame({"label": [1, 2, 3], "sample_id": ["specimen A"]*3, "parent_domain_id": [1, 2, 2],
        "subdomain_id": [1, 2, 3], "status_code": [1, 1, 1], "x_um": table.representative_x_um,
        "y_um": table.representative_y_um, "seed_stability": [.9, 1., 1.], "scale_agreement": [1., 1., 1.], "centroid_margin": [.4, .8, .6]})
    grid.to_csv(root / "grid_subdomains.csv", index=False)
    blocks = {"context": np.array([[0., 0.], [1., 0.], [np.nan, np.nan]], np.float32),
              "local": np.array([[0., 0.], [10., 1.], [.5, 0.]], np.float32)}
    region_manifest = {"schema_version": "1.0.0", "observation_unit": "tissue_region", "region_count": 3,
        "hierarchy_id": "hierarchy_v1", "coordinate_space": "original_slide_micrometres", "feature_blocks": {}, "files": {}}
    for name, values in blocks.items():
        np.save(directory / (name + ".npy"), values)
        region_manifest["feature_blocks"][name] = {"path": name + ".npy", "shape": list(values.shape),
            "feature_names": [name+"_0", name+"_1"], "feature_definition": {"model": "test", "context": name, "observation_unit": "tissue_region"}}
    for name in ("region_profiles.csv", "feature_rows.csv", "local.npy", "context.npy"):
        region_manifest["files"][name] = sha256_file(directory / name)
    (directory / "region_profiles_manifest.json").write_text(json.dumps(region_manifest))
    summary = {"sample_id": "specimen A", "hierarchy_id": "hierarchy_v1", "geometry": {
        "shape_yx": list(labels.shape), "mpp_xy": [.5, .5], "origin_um_xy": [50., 100.], "coordinate_space": "original_slide_micrometres"},
        "inputs": {"image": {"sha256": sha256_file(inputs["image"])}, "parent_mask": {"sha256": manifest["inputs"]["domain_mask_sha256"]}, "shift_json": {"sha256": sha256_file(inputs["shift"])},
                   "resolution_json": {"sha256": sha256_file(resolution)}}, "status_codes": {0: "background", 1: "accepted_subdomain", 2: "parent_only_no_observation"}}
    (root / "hierarchy_summary.json").write_text(json.dumps(summary))
    refresh_region_hashes(root)
    return inputs, table, regions, specimen


def refresh_region_hashes(root):
    path = root / "hierarchy_summary.json"
    summary = json.loads(path.read_text())
    summary["outputs"] = {name: sha256_file(root / name) for name in ("region_mask.ome.tif", "parent_domains.ome.tif",
        "hierarchy_status.ome.tif", "grid_subdomains.csv", "region_profiles/region_profiles_manifest.json")}
    if (root / "parent_uncertainty.ome.tif").exists():
        summary["outputs"]["parent_uncertainty.ome.tif"] = sha256_file(root / "parent_uncertainty.ome.tif")
    path.write_text(json.dumps(summary))


def test_region_native_geometry_membership_distributions_and_similarity(region_specimen):
    inputs, table, _, specimen = region_specimen
    inspector = module.CellInspector(**inputs)
    regions = inspector.regions
    uid = table.region_uid.iloc[0]
    detail = regions.detail(uid, 64)
    assert detail["crop_xy"] == [20., 30.]
    assert detail["coordinates_um"] == [65., 120.]  # native pixel centres, matching the producer
    assert detail["cell_count"] == 2
    assert [r["cell_id"] for r in detail["representative_cells"]] == ["001", "7"]
    assert detail["predicted_marker_distributions"][MARKER]["mean"] == .35
    assert detail["subdomain_grid_stability"]["seed_stability"]["quantiles_p05_p50_p95"] == [.9, .9, .9]
    assert detail["feature_blocks"]["context"]["within_region_feature_distribution"] is None
    assert detail["uncertainty"]["boundary_uncertainty"] is None
    assert regions.detail(table.region_uid.iloc[1])["predicted_marker_distributions"][MARKER]["missing"] == 1
    unresolved = regions.pick(150, 30)
    assert unresolved["status"] == "unresolved_parent_tissue"
    assert unresolved["reason"] == "parent_only_no_observation"
    assert unresolved["parent_domain_id"] > 0 and unresolved["hierarchy_status_code"] == 2
    assert regions.pick(100, 110)["status"] == "no_parent_assignment_uncertainty_unavailable"
    assert regions.pick(20, 30)["region_uid"] == uid
    assert regions.similar(uid, ["context"])["results"][0]["region_id"] == "2"
    assert regions.similar(uid, ["local"])["results"][0]["region_id"] == "3"
    assert regions.similar(table.region_uid.iloc[2], ["context"])["status"] == "missing_features"
    assert regions.similar(uid, ["local"], marker=MARKER, min_difference=.4)["results"][0]["region_id"] == "3"
    assert "cell_uid" not in regions.similar(uid, ["local"])["results"][0]
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(regions.png(uid, 64)))), specimen[2][:64, :64])
    outline = np.asarray(Image.open(io.BytesIO(regions.png(uid, 64, overlay=True))))
    assert outline[30, 59, 3] == 255 and outline[30, 15, 3] == 45
    assert outline[:, 60:, 3].sum() == 0
    assert regions.bounds(uid, 64, x=150, y=90) == (118, 58, 160, 122)
    assert inspector.metadata()["regions"]["status"] == "verified_sha256"
    assert inspector.detail(specimen[1].cell_uid.iloc[0])["cell_id"] == "001"


@pytest.mark.parametrize("mutation,match", [("mask_hash", "SHA256 mismatch"), ("missing_hash", "lacks a valid output"),
    ("sample", "sample ID"), ("origin", "geometry differs"), ("parent_source", "parent_mask source"),
    ("row_shuffle", "Feature-row identities"), ("feature_hash", "file SHA256 mismatch"), ("raster_id", "ID absent"),
    ("area", "area/centroid"), ("legacy", "verified canonical"), ("representative", "representative grid identity")])
def test_region_lineage_fails_closed(region_specimen, mutation, match):
    inputs, _, _, _ = region_specimen
    root = inputs["hierarchy_dir"]
    summary_path = root / "hierarchy_summary.json"
    summary = json.loads(summary_path.read_text())
    profile_path = root / "region_profiles/region_profiles_manifest.json"
    profile = json.loads(profile_path.read_text())
    if mutation == "mask_hash":
        with (root / "region_mask.ome.tif").open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "missing_hash":
        summary.pop("outputs")
    elif mutation == "sample":
        summary["sample_id"] = "another specimen"
    elif mutation == "origin":
        summary["geometry"]["origin_um_xy"][0] += 10
    elif mutation == "parent_source":
        summary["inputs"]["parent_mask"]["sha256"] = "a"*64
    elif mutation == "feature_hash":
        np.save(root / "region_profiles/local.npy", np.zeros((3, 2), np.float32))
    elif mutation == "legacy":
        path = inputs["profile_dir"] / "cell_profiles_manifest.json"
        manifest = json.loads(path.read_text()); manifest["inputs"].pop("image_sha256")
        path.write_text(json.dumps(manifest))
    elif mutation == "raster_id":
        data = tifffile.imread(root / "region_mask.ome.tif"); data[30, 10] = 99
        tifffile.imwrite(root / "region_mask.ome.tif", data)
    elif mutation in ("row_shuffle", "area", "representative"):
        name = "feature_rows.csv" if mutation == "row_shuffle" else "region_profiles.csv"
        table = pd.read_csv(root / "region_profiles" / name)
        if mutation == "row_shuffle":
            table = table.iloc[::-1]
        elif mutation == "area":
            table.loc[0, "area_um2"] += 1
        else:
            table.loc[0, "representative_grid_id"] = 2
        table.to_csv(root / "region_profiles" / name, index=False)
        profile["files"][name] = sha256_file(root / "region_profiles" / name)
        profile_path.write_text(json.dumps(profile))
    summary_path.write_text(json.dumps(summary))
    if mutation in ("row_shuffle", "area", "raster_id", "representative"):
        refresh_region_hashes(root)
    with pytest.raises(ValueError, match=match):
        module.CellInspector(**inputs)


def test_region_notes_are_display_only_and_bounded_windows(region_specimen, tmp_path, monkeypatch):
    inputs, rows, _, _ = region_specimen
    baseline = module.CellInspector(**inputs)
    uid = rows.region_uid.iloc[0]
    notes = tmp_path / "region_notes.json"
    notes.write_text(json.dumps({"annotations": [{"region_uid": uid, "reviewed_label": "<script>Not an inference</script>"}]}))
    reviewed = module.CellInspector(**inputs, region_annotations=notes)
    assert reviewed.regions.similar(uid, ["local"]) == baseline.regions.similar(uid, ["local"])
    assert reviewed.regions.detail(uid)["expert_annotation_used_for_inference"] is False
    assert reviewed.regions.detail(uid)["expert_annotation"]["reviewed_label"].startswith("<script>")
    window = module.RasterReader.window
    def bounded(self, x0, y0, x1, y1):
        assert x1-x0 <= 64 and y1-y0 <= 64
        return window(self, x0, y0, x1, y1)
    monkeypatch.setattr(module.RasterReader, "window", bounded)
    reviewed.regions.png(uid, 64)
    reviewed.regions.png(uid, 64, overlay=True)
    with pytest.raises(ValueError, match=r"\[64,1024\]"):
        reviewed.regions.png(uid, 100000)


def test_region_parent_uncertainty_lineage_and_exclusion(region_specimen):
    inputs, _, regions, _ = region_specimen
    root = inputs["hierarchy_dir"]
    path = root / "parent_uncertainty.ome.tif"
    uncertainty = np.zeros(regions.shape, np.uint8)
    uncertainty[:80, 140:] = 7
    def bind():
        tifffile.imwrite(path, uncertainty)
        summary_path = root / "hierarchy_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["inputs"]["parent_uncertainty"] = {"sha256": sha256_file(path)}
        summary_path.write_text(json.dumps(summary))
        manifest_path = inputs["profile_dir"] / "cell_profiles_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["inputs"]["domain_uncertainty_sha256"] = sha256_file(path)
        manifest_path.write_text(json.dumps(manifest))
        refresh_region_hashes(root)
    bind()
    assert module.CellInspector(**inputs).regions.metadata()["parent_uncertainty_status"] == "hash_verified_and_excluded_from_regions"
    uncertainty[30, 15] = 3  # deliberately place unresolved provenance in an accepted region
    bind()
    with pytest.raises(ValueError, match="overlaps unresolved/inferred"):
        module.CellInspector(**inputs)


def test_real_region_http_routes_and_path_restrictions(region_specimen):
    inputs, rows, _, _ = region_specimen
    server = module.make_server(module.CellInspector(**inputs), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        uid = rows.region_uid.iloc[0]
        for route in ("/api/region", "/api/region-window.png", "/api/region-overlay.png"):
            code, headers, body = request(server, route + "?" + urlencode({"uid": uid, "size": 64}))
            assert code == 200
            if route.endswith("png"):
                assert Image.open(io.BytesIO(body)).size == (64, 64)
            else:
                assert json.loads(body)["region_uid"] == uid
        code, _, body = request(server, "/api/region-similar?" + urlencode({"uid": uid, "groups": "local"}))
        assert code == 200 and json.loads(body)["results"][0]["region_id"] == "3"
        assert request(server, "/api/region?path=/etc/passwd")[0] == 400
        assert request(server, "/api/regions?limit=999999")[0] == 400
        assert request(server, "/api/region-pick?x=150&y=30")[0] == 200
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


@pytest.mark.skipif(os.environ.get("RUN_INSPECTOR_BROWSER_TESTS") != "1", reason="Opt-in actual headless Chrome render/interaction QA")
def test_headless_region_and_cell_ui_render(region_specimen, tmp_path, monkeypatch):
    chrome = os.environ.get("CHROME_BIN", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not Path(chrome).is_file():
        pytest.skip("No Chrome executable available")
    inputs, rows, _, _ = region_specimen
    # Exercise the unchanged UI handlers in an actual headless browser. This
    # startup test driver is only in the temporary test-served HTML, never the
    # shipped inspector. It does not launch a visible app or Codex panel.
    driver = """<script>
    (async()=>{try{
      for(let i=0;i<100 && (!meta || document.querySelectorAll('#region-results button').length===0);i++)await new Promise(r=>setTimeout(r,50));
      const button=document.querySelector('#region-results button');if(!button)throw Error('No region navigation');button.click();
      for(let i=0;i<100 && !currentRegion;i++)await new Promise(r=>setTimeout(r,50));
      if(!currentRegion||currentRegion.region_id!=='1')throw Error('Region selection failed');
      const canvas=document.getElementById('region-window'),box=canvas.getBoundingClientRect();
      if(box.width<100||box.height<50||document.getElementById('region-panel').hidden)throw Error('Region window not rendered');
      if(canvas.getContext('2d').getImageData(10,10,1,1).data[3]!==255)throw Error('Region H&E not painted');
      const check=document.querySelector('.region-feature-check[value="local"]');check.click();
      await findSimilarRegions();if(!document.querySelector('#region-similar-results button'))throw Error('Region similarity missing');
      await panRegion(1,0);
      const cells=document.querySelectorAll('#region-cells button');if(cells.length!==2)throw Error('Representative cell count');cells[0].click();
      for(let i=0;i<100 && (!current||current.cell_id!=='001');i++)await new Promise(r=>setTimeout(r,50));
      if(!current||current.cell_id!=='001')throw Error('Cell navigation lost identity');
      document.body.dataset.qa='passed';
    }catch(error){document.body.dataset.qa='failed: '+error.message;}})();
    </script>"""
    page = tmp_path / "test_inspector.html"
    page.write_text(module.HTML_PATH.read_text().replace("</body>", driver+"</body>"))
    monkeypatch.setattr(module, "HTML_PATH", page)
    server = module.make_server(module.CellInspector(**inputs), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        command = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                   "--disable-background-networking", "--disable-component-update", "--disable-sync", "--disable-extensions",
                   "--user-data-dir="+str(tmp_path / "chrome-profile"), "--virtual-time-budget=15000", "--window-size=1500,1100",
                   "--screenshot="+str(tmp_path / "inspector.png"), "--dump-dom", f"http://127.0.0.1:{server.server_port}"]
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=25)
            assert completed.returncode == 0, completed.stderr[-2000:]
            rendered = completed.stdout
        except subprocess.TimeoutExpired as exc:
            # Some macOS headless Chrome versions retain their process after
            # exporting both DOM and screenshot. subprocess.run has killed the
            # exact child at this point. Accept only complete successful UI
            # artifacts, never a timeout alone as proof of a successful render.
            rendered = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        (tmp_path / "rendered.html").write_text(rendered)
        assert 'data-qa="passed"' in rendered, rendered[:1000] + rendered[-3000:]
        assert Image.open(tmp_path / "inspector.png").size == (1500, 1100)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)
