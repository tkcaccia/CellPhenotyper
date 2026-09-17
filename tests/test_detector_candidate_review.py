"""Rejected predictions stay reviewable without changing canonical cells."""
import csv
import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "bin"))
from build_cell_consensus import Cell
from detector_candidate_review import export_candidates, candidate_geometry, VERIFICATION_KEYS
import cell_inspector as inspector_module
from test_cell_inspector import specimen, request


@pytest.fixture
def candidates(specimen, tmp_path):
    paths, _, _, _ = specimen
    cells = [Cell("cellvitpp", "004", 15, 30, [[11, 26], [19, 26], [19, 34], [11, 34]]),
        Cell("stardist", "002", 45, 70, []),
        Cell("hovernet", "<script>&/ +", 75, 70, [[70, 65], [80, 65], [80, 75], [70, 75]],
             type_id=2, cell_type="lymphocyte", probability=.7),
        Cell("cellvitpp", "bad-outline", 105, 70, [[100, 65], [110, 75], [100, 75], [110, 65]])]
    source_paths = {key: tmp_path / (key + ".json") for key in ("stardist_objects", "hovernet_cells", "cellvit_cells")}
    source_paths["stardist_objects"] = tmp_path / "stardist.csv"
    source_paths["stardist_objects"].write_text("label,x,y\n002,45,70\n")
    for source, name in (("hovernet", "hovernet_cells"), ("cellvitpp", "cellvit_cells")):
        source_paths[name].write_text(json.dumps({"cells": [{"id": c.source_id,
            "centroid": [c.x, c.y], "contour": c.contour, "type_id": c.type_id,
            "type_name": c.cell_type, "type_prob": c.probability} for c in cells if c.source == source]}))
    accepted = {("cellvitpp", "004")}
    decisions = {(c.source, c.source_id): {"decision": "accepted_instance_fusion" if (c.source, c.source_id) in accepted
        else "abstained_broad_pair_missing", "agreement_score": .25} for c in cells}
    alignment = tmp_path / "alignment.csv"
    with alignment.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "source_id", "x", "y", "accepted", "consensus_label", "decision", "agreement_score"])
        writer.writeheader()
        for c in cells:
            ok = (c.source, c.source_id) in accepted
            writer.writerow({"source": c.source, "source_id": c.source_id, "x": c.x, "y": c.y,
                "accepted": ok, "consensus_label": "1" if ok else "", **decisions[(c.source, c.source_id)]})
    inputs = {**source_paths, **{key: paths[key] for key in ("image", "labels", "shift")}, "alignment_csv": alignment}
    output = tmp_path / "rejected.geojson"
    metadata = export_candidates(cells, decisions, accepted, inputs, output)
    return cells, decisions, accepted, inputs, output, metadata


def test_export_original_geometry_identity_and_population(candidates, tmp_path):
    cells, decisions, accepted, inputs, output, metadata = candidates
    assert metadata["candidate_count"] == 3
    assert metadata["input_prediction_count"] == 4
    assert metadata["accepted_prediction_count"] == 1
    assert metadata["canonical_population_changed"] is False
    assert metadata["geometry_counts"] == {"invalid_source_outline": 1, "source_polygon": 1, "centroid_only": 1}
    first = json.loads(output.read_text())
    second = tmp_path / "second.geojson"
    export_candidates(list(reversed(cells)), decisions, accepted, inputs, second)
    assert json.loads(second.read_text()) == first
    for feature in first["features"]:
        p = feature["properties"]
        assert feature["id"] == p["candidate_uid"]
        assert p["candidate_uid"].startswith("detector_candidate:")
        assert "cell_uid" not in p and "cell_id" not in p
        assert p["canonical_assignment"] is None
        assert p["source_id"] != "004"
        assert p["detector_scope"]["instance_fusion_role"] in ("broad_scope", "scoped_support")
    with pytest.raises(ValueError, match="fresh output"):
        export_candidates(cells, decisions, accepted, inputs, output)


@pytest.mark.parametrize("contour,issue", [([], "not_available"), ([[1, 2]], "fewer_than"),
    ([[1, 2], [3, float("nan")], [1, 4]], "nonfinite"), ([[1, 2], [3], [1, 4]], "malformed")])
def test_invalid_missing_contours_are_centroids_not_repaired(contour, issue):
    geometry, status, reason = candidate_geometry(Cell("stardist", "1", 8, 9, contour))
    assert geometry == {"type": "Point", "coordinates": [8., 9.]}
    assert status == "centroid_only" and issue in reason


def test_source_contract_rejects_unmatched_decisions_and_duplicate_ids(candidates, tmp_path):
    cells, decisions, accepted, inputs, _, _ = candidates
    with pytest.raises(ValueError, match="unique"):
        export_candidates(cells + [cells[0]], decisions, accepted, inputs, tmp_path / "x.json")
    with pytest.raises(ValueError, match="exactly"):
        export_candidates(cells, {}, accepted, inputs, tmp_path / "x.json")
    with pytest.raises(ValueError, match="accepted fusion"):
        export_candidates(cells, decisions, set(), inputs, tmp_path / "x.json")


def test_inspector_retains_population_and_reads_bounded_source_windows(specimen, candidates, monkeypatch):
    paths, cells, pixels, _ = specimen
    baseline = inspector_module.CellInspector(**paths)
    inspected = inspector_module.CellInspector(**paths, rejected_candidates=candidates[4])
    pd.testing.assert_frame_equal(inspected.cells, baseline.cells)
    assert inspected.queue() == baseline.queue()
    assert inspected.search("004") == baseline.search("004")
    assert inspected.similar(cells.cell_uid.iloc[0], ["context"]) == baseline.similar(cells.cell_uid.iloc[0], ["context"])
    layer = inspected.rejected_candidates
    assert layer.metadata()["status"] == "raster_bound_payload_unverified"
    assert layer.metadata()["raster_binding_status"] == "verified_sha256"
    assert layer.metadata()["candidate_payload_status"] == "unverified_source_payload"
    assert layer.queue(limit=1)["total"] == 3
    assert len(layer.queue(source="stardist")["rows"]) == 1
    uid = layer.queue(source="stardist")["rows"][0]["candidate_uid"]
    detail = layer.detail(uid, 64)
    assert detail["geometry_status"] == "centroid_only"
    assert detail["coordinates_um"] == [72.5, 135.]
    assert detail["profile_available"] is False and detail["used_for_inference"] is False
    with pytest.raises(KeyError):
        inspected.detail(uid)
    original = inspector_module.RasterReader.window
    calls = []
    def bounded(reader, x0, y0, x1, y1):
        assert x1-x0 <= 64 and y1-y0 <= 64
        calls.append((x0, y0, x1, y1))
        return original(reader, x0, y0, x1, y1)
    monkeypatch.setattr(inspector_module.RasterReader, "window", bounded)
    image = np.asarray(Image.open(io.BytesIO(layer.png(uid, 64))))
    x0, y0, x1, y1 = calls[0]
    np.testing.assert_array_equal(image, pixels[y0:y1, x0:x1])
    for row in layer.queue()["rows"]:
        overlay = np.asarray(Image.open(io.BytesIO(layer.png(row["candidate_uid"], 64, True))))
        assert overlay.shape == (64, 64, 4)
        assert np.count_nonzero(overlay[..., 3]) > 0
    assert len(calls) == 1


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["metadata"]["input_sha256"].update(image="0"*64), "SHA256"),
    (lambda p: p["metadata"].update(coordinate_frame="original_pixels"), "coordinate frame"),
    (lambda p: p["metadata"].update(source_mpp=.25), "calibration"),
    (lambda p: p["metadata"].update(candidate_count=999), "accounting"),
    (lambda p: p["features"][0]["properties"].update(cell_uid="forbidden"), "identity"),
    (lambda p: p["features"][0]["properties"].update(coordinates_um=[0, 0]), "centroid"),
    (lambda p: p["features"][0]["properties"].update(decision="accepted_instance_fusion"), "exclusion"),
    (lambda p: p["features"][0]["properties"].update(geometry_issue=None), "explicit reason"),
])
def test_inspector_rejects_mismatched_or_contradictory_layers(specimen, candidates, mutation, match):
    output = candidates[4]
    payload = json.loads(output.read_text()); mutation(payload); output.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        inspector_module.CellInspector(**specimen[0], rejected_candidates=output)


def test_actual_localhost_review_api_and_no_mutations(specimen, candidates):
    inspected = inspector_module.CellInspector(**specimen[0], rejected_candidates=candidates[4])
    server = inspector_module.make_server(inspected, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        code, _, body = request(server, "/api/rejected-candidates?source=hovernet")
        assert code == 200
        row = json.loads(body)["rows"][0]
        assert row["source_id"] == "<script>&/ +"
        query = urlencode({"uid": row["candidate_uid"], "size": 64})
        code, _, body = request(server, "/api/rejected-candidate?" + query)
        assert code == 200 and json.loads(body)["profile_available"] is False
        code, _, body = request(server, "/api/rejected-candidate-window.png?" + query)
        assert code == 200 and Image.open(io.BytesIO(body)).size == (64, 64)
        for suffix, method, expected in [("?path=/etc/passwd", "GET", 400), ("", "POST", 405), ("?limit=1000", "GET", 400)]:
            assert request(server, "/api/rejected-candidates" + suffix, method=method)[0] == expected
    finally:
        server.shutdown(); server.server_close(); thread.join(2)


def test_standalone_existing_artifact_export_without_inference(candidates, tmp_path):
    _, _, _, inputs, original, _ = candidates
    output = tmp_path / "standalone.geojson"
    script = Path(__file__).parents[1] / "bin" / "detector_candidate_review.py"
    command = [sys.executable, str(script), "--output", str(output)]
    for key, value in inputs.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    actual, expected = json.loads(output.read_text()), json.loads(original.read_text())
    assert actual["metadata"] == expected["metadata"]
    assert [f["id"] for f in actual["features"]] == [f["id"] for f in expected["features"]]
    assert [f["geometry"] for f in actual["features"]] == [f["geometry"] for f in expected["features"]]


def explicit_sources(candidates):
    return {key: candidates[3][key] for key in VERIFICATION_KEYS}


def test_explicit_original_sources_verify_full_geometry_and_evidence(specimen, candidates):
    layer = inspector_module.CellInspector(**specimen[0], rejected_candidates=candidates[4],
        candidate_sources=explicit_sources(candidates)).rejected_candidates
    assert layer.metadata()["status"] == "source_verified"
    assert layer.metadata()["candidate_payload_status"] == "verified_against_explicit_sources"
    for row in layer.queue()["rows"]:
        assert layer.detail(row["candidate_uid"])["candidate_payload_status"] == "verified_against_explicit_sources"
    assert "not an independent trusted payload receipt" in layer.metadata()["review_file_hash_role"]


@pytest.mark.parametrize("change", ["geometry", "decision", "fusion_evidence", "detector_evidence", "missing_candidate"])
def test_self_declared_hashes_never_verify_tampered_candidate_payload(specimen, candidates, change):
    output = candidates[4]
    payload = json.loads(output.read_text())
    feature = next(f for f in payload["features"] if f["geometry"]["type"] == "Polygon")
    props = feature["properties"]
    if change == "geometry":
        feature["geometry"]["coordinates"][0] = [[x+25, y] for x, y in feature["geometry"]["coordinates"][0]]
    elif change == "decision":
        props["decision"] = "invented_exclusion"
    elif change == "fusion_evidence":
        props["fusion_evidence"]["agreement_score"] = "999"
    elif change == "detector_evidence":
        props["detector_phenotype_evidence"]["type"] = "invented_phenotype"
    else:
        payload["features"].remove(feature)
        payload["metadata"]["candidate_count"] -= 1
        payload["metadata"]["input_prediction_count"] -= 1
    from collections import Counter
    payload["metadata"]["decision_counts"] = dict(Counter(f["properties"]["decision"] for f in payload["features"]))
    payload["metadata"]["geometry_counts"] = dict(Counter(f["properties"]["geometry_status"] for f in payload["features"]))
    output.write_text(json.dumps(payload))
    # Plausible self-declared geometry cannot be authenticated without sources.
    unverified = inspector_module.CellInspector(**specimen[0], rejected_candidates=output).rejected_candidates
    assert unverified.metadata()["candidate_payload_status"] == "unverified_source_payload"
    assert unverified.metadata()["status"] != "source_verified"
    with pytest.raises(ValueError, match="(explicit|rejected population)"):
        inspector_module.CellInspector(**specimen[0], rejected_candidates=output, candidate_sources=explicit_sources(candidates))


def test_verification_inputs_are_explicit_complete_and_hash_matched(specimen, candidates):
    sources = explicit_sources(candidates)
    with pytest.raises(ValueError, match="all four"):
        inspector_module.CellInspector(**specimen[0], rejected_candidates=candidates[4], candidate_sources={"alignment_csv": sources["alignment_csv"]})
    with pytest.raises(ValueError, match="requires --rejected"):
        inspector_module.CellInspector(**specimen[0], candidate_sources=sources)
    sources["hovernet_cells"].write_text(sources["hovernet_cells"].read_text() + "\n")
    with pytest.raises(ValueError, match="hovernet_cells SHA256"):
        inspector_module.CellInspector(**specimen[0], rejected_candidates=candidates[4], candidate_sources=sources)


def test_cli_explicit_source_verification_without_server(specimen, candidates):
    command = [sys.executable, str(Path(inspector_module.__file__)), "--check-only", "--rejected-candidates", str(candidates[4])]
    for key, value in specimen[0].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    for key, value in explicit_sources(candidates).items():
        command += ["--candidate-" + key.replace("_", "-"), str(value)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rejected_candidates"]["candidate_payload_status"] == "verified_against_explicit_sources"


def test_legacy_pixel_only_export_never_guesses_physical_scale(specimen, candidates, tmp_path):
    cells, decisions, accepted, inputs, _, _ = candidates
    shift = json.loads(inputs["shift"].read_text()); shift.pop("source_mpp")
    inputs["shift"].write_text(json.dumps(shift))
    report = tmp_path / "resolution.json"
    report.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
    output = tmp_path / "pixel_only.json"
    meta = export_candidates(cells, decisions, accepted, inputs, output)
    assert meta["source_mpp"] is None and meta["crop_origin_um"] is None
    assert meta["calibration_source"] == "physical_scale_unavailable"
    layer = inspector_module.CellInspector(**specimen[0], resolution_json=report, rejected_candidates=output).rejected_candidates
    uid = layer.queue(source="stardist")["rows"][0]["candidate_uid"]
    detail = layer.detail(uid)
    assert detail["coordinates_um"] is None
    assert detail["display_coordinates_um"] == [72.5, 135.]
    assert detail["producer_calibration"] == "physical_scale_unavailable"
    assert detail["display_coordinate_calibration"] == "passed_resolution_report"
    calibrated = tmp_path / "calibrated.json"
    export_candidates(cells, decisions, accepted, {**inputs, "resolution_json": report}, calibrated)
    report.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5, "changed": True}))
    with pytest.raises(ValueError, match="resolution SHA256"):
        inspector_module.CellInspector(**specimen[0], resolution_json=report, rejected_candidates=calibrated)


@pytest.mark.skipif(os.environ.get("RUN_INSPECTOR_BROWSER_TESTS") != "1", reason="Opt-in synthetic headless UI interaction")
@pytest.mark.parametrize("verified_sources", [False, True])
def test_headless_rejected_review_stays_separate_from_canonical(specimen, candidates, tmp_path, monkeypatch, verified_sources):
    chrome = os.environ.get("CHROME_BIN", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not Path(chrome).is_file():
        pytest.skip("No existing Chrome executable")
    driver = """<script>(async()=>{try{
      for(let i=0;i<120&&(!meta||!document.querySelector('#rejected-results button'));i++)await new Promise(r=>setTimeout(r,50));
      if(meta.cell_count!==5||meta.rejected_candidates.candidate_count!==3)throw Error('Population changed');
      if(meta.rejected_candidates.candidate_payload_status!=='EXPECTED_CANDIDATE_STATUS')throw Error('Wrong provenance status');
      const old=current.cell_uid;
      document.querySelector('#rejected-results button').click();
      for(let i=0;i<100&&!currentCandidate;i++)await new Promise(r=>setTimeout(r,50));
      if(!currentCandidate||current.cell_uid!==old)throw Error('Candidate changed canonical selection');
      if(document.getElementById('rejected-detail').hidden)throw Error('Candidate detail not visible');
      const canvas=document.getElementById('rejected-window');
      if(canvas.getBoundingClientRect().width<100||canvas.getContext('2d').getImageData(10,10,1,1).data[3]!==255)throw Error('H&E not painted');
      document.getElementById('rejected-source').value='stardist';await loadCandidateQueue();
      document.querySelector('#rejected-results button').click();
      for(let i=0;i<100&&currentCandidate.source!=='stardist';i++)await new Promise(r=>setTimeout(r,50));
      if(!document.getElementById('rejected-caption').textContent.includes('centroid only'))throw Error('Missing point limitation');
      if(!document.getElementById('rejected-caption').textContent.includes('EXPECTED_CAPTION_STATUS'))throw Error('Missing payload verification distinction');
      if(document.querySelectorAll('#rejected-results button').length!==1)throw Error('Detector filter failed');
      document.body.dataset.qa='passed';
    }catch(e){document.body.dataset.qa='failed: '+e.message;}})();</script>"""
    driver = driver.replace("EXPECTED_CANDIDATE_STATUS", "verified_against_explicit_sources" if verified_sources else "unverified_source_payload")
    driver = driver.replace("EXPECTED_CAPTION_STATUS", "Source payload reverified." if verified_sources else "UNVERIFIED CANDIDATE PAYLOAD.")
    page = tmp_path / "candidate_ui.html"
    page.write_text(inspector_module.HTML_PATH.read_text().replace("</body>", driver + "</body>"))
    monkeypatch.setattr(inspector_module, "HTML_PATH", page)
    server = inspector_module.make_server(inspector_module.CellInspector(**specimen[0], rejected_candidates=candidates[4],
        candidate_sources=explicit_sources(candidates) if verified_sources else None), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        command = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--disable-background-networking", "--disable-component-update", "--disable-sync", "--disable-extensions",
            "--user-data-dir=" + str(tmp_path / "chrome"), "--virtual-time-budget=15000", "--window-size=1500,1100",
            "--dump-dom", f"http://127.0.0.1:{server.server_port}"]
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=25)
            assert result.returncode == 0, result.stderr[-2000:]
            rendered = result.stdout
        except subprocess.TimeoutExpired as exc:
            rendered = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        assert 'data-qa="passed"' in rendered, rendered[-2500:]
    finally:
        server.shutdown(); server.server_close(); thread.join(2)
