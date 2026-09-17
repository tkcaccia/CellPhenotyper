"""Real tiny cohort/inspector handoffs; synthetic features, no learned inference."""
import http.client
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

sys.path.insert(0, str(Path(__file__).parents[1] / "bin"))
import cell_inspector as inspector_module
import cohort_niche_io
from assemble_spatial_cell_profiles import assemble
from cell_profile_io import sha256_file
from fit_cohort_niches import fit_cohort
from specimen_atlas import build_specimen_atlas, discover_cell_inspector, render_specimen_atlas
from test_cell_inspector import specimen as base_specimen
from test_specimen_atlas import record as indexed_record


@pytest.fixture
def cohort_specimens(tmp_path):
    results = tmp_path / "results"
    execution = results / "00_execution"
    execution.mkdir(parents=True)
    specimens, records = [], []
    for sample in ("specimen A", "specimen B"):
        work = results / "19_cell_profiles" / sample
        work.mkdir(parents=True)
        inputs, cells, pixels, labels = base_specimen.__wrapped__(work)
        image = work / (sample + ".tif")
        inputs["image"].rename(image)
        inputs["image"] = image
        root = inputs["profile_dir"]
        cells["sample_id"] = sample
        cells["cell_uid"] = [sample + ":exact:<script>&/ +:" + value for value in cells.cell_id]
        cells["reference_status"] = "assigned_reference"
        cells["reference_id"] = "frozen_reference_distinct_from_niches"
        cells = cells.drop(columns=[name for name in cells if name.startswith("niche_")])
        cells.to_csv(root / "cell_profiles.csv", index=False)
        cells.to_parquet(root / "cell_profiles.parquet", index=False)
        cells[["sample_id", "cell_id", "cell_uid"]].to_csv(root / "feature_rows.csv", index=False)
        objects = work / "objects.csv"
        cells[["cell_id", "x_crop_px", "y_crop_px"]].to_csv(objects, index=False)
        manifest = json.loads((root / "cell_profiles_manifest.json").read_text())
        manifest["sample_id"] = sample
        manifest["inputs"] = {key + "_sha256": sha256_file(path) for key, path in
            {"image": image, "labels": inputs["labels"], "shift": inputs["shift"], "objects": objects}.items()}
        manifest["feature_blocks"]["context"].update(feature_names=["first", "second"], dtype="float32")
        (root / "feature_blocks").mkdir()
        for block in manifest["feature_blocks"].values():
            original = root / block["path"]
            destination = root / "feature_blocks" / original.name
            original.rename(destination)
            block["path"] = destination.relative_to(root).as_posix()
        manifest["files"] = {name: sha256_file(root / name) for name in
                             set(manifest["files"]) | {"cell_profiles.parquet"}}
        (root / "cell_profiles_manifest.json").write_text(json.dumps(manifest))
        support = np.ones(labels.shape, np.uint8)
        support[:, 125:] = 0
        support_path = work / "support.tif"
        tifffile.imwrite(support_path, support)
        spatial = work / "cell_profiles"
        assemble(root, support_path, inputs["shift"], None, spatial,
                 radii_um=(20.,), feature_groups=("context",), repeats=2)
        inputs["profile_dir"] = spatial
        specimens.append(inputs)
        for path, stage in ((spatial / "cell_profiles_manifest.json", "cell_profiles"),
                            (objects, "cell_consensus"), (inputs["labels"], "cell_consensus"),
                            (inputs["shift"], "stardist"), (image, "stardist")):
            records.append(indexed_record(path, results, stage, sample + ":" + path.name))
    bundle = results / "25_cohort_niches" / "cohort_niches"
    assignments, model, summary = fit_cohort([item["profile_dir"] for item in specimens], bundle,
                                            fixed_k=2, repeats=2)
    for path in sorted(bundle.iterdir()):
        records.append(indexed_record(path, results, "cohort_niches", path.name))
    return {"inputs": specimens, "bundle": bundle, "assignments": assignments,
            "model": model, "summary": summary, "records": records, "execution": execution}


def test_exact_shared_niches_are_separate_from_canonical_and_reference_fields(cohort_specimens):
    fixture = cohort_specimens
    for inputs in fixture["inputs"]:
        before = pd.read_parquet(inputs["profile_dir"] / "cell_profiles.parquet")
        inspector = inspector_module.CellInspector(**inputs, cohort_niches=fixture["bundle"])
        pd.testing.assert_frame_equal(inspector.cells, before)
        expected, record = cohort_niche_io.load_cohort_bundle(fixture["bundle"], profile_dir=inputs["profile_dir"])
        assert inspector.metadata()["cohort_niches"]["record"] == record
        assert inspector.metadata()["cohort_niches"]["status"] == "verified_source_bound"
        returned_metadata = inspector.metadata()
        returned_metadata["cohort_niches"]["record"]["source_files"].clear()
        assert inspector.metadata()["cohort_niches"]["record"] == record
        assert len(expected) == 5
        for index, source in before.iterrows():
            detail = inspector.detail(source.cell_uid)
            assert detail["cell_uid"] == source.cell_uid
            assert detail["cell_id"] == source.cell_id
            assert detail["spatial"]["niche_id"] == inspector_module.clean_json(source.niche_id)
            assert detail["spatial"]["reference_id"] == "frozen_reference_distinct_from_niches"
            assert not any(name.startswith("cohort_") for name in detail["spatial"])
            assert detail["cohort_niche"] == inspector_module.clean_json(
                expected.iloc[index][cohort_niche_io.COHORT_COLUMNS].to_dict())
            assert set(inspector_module.review_reasons(source.to_dict())) <= set(detail["review_reasons"])
        last = inspector.detail(before.cell_uid.iloc[-1])
        assert last["cohort_niche"]["cohort_niche_id"] is None
        assert last["cohort_niche"]["cohort_niche_status"] == "outside_tissue_support"
        assert "cohort_niche_review" in last["review_reasons"]
        assert inspector.queue(reason="cohort_niche_review")["rows"][-1]["cell_uid"] == before.cell_uid.iloc[-1]
        assert inspector.search("001")[0]["cell_id"] == "001"
    assert fixture["assignments"].groupby("sample_id").cohort_niche_model_id.first().nunique() == 1


def test_missing_attachment_and_cohort_review_flags_are_explicit(cohort_specimens):
    inspector = inspector_module.CellInspector(**cohort_specimens["inputs"][0])
    assert inspector.metadata()["cohort_niches"] == {"status": "not_provided"}
    assert inspector.detail(inspector.cells.cell_uid.iloc[0])["cohort_niche"] == {}
    assert not any("cohort" in reason for row in inspector.queue()["rows"] for reason in row["reasons"])
    assert inspector_module.review_reasons({"cohort_niche_status": "assigned_fixed_k",
        "cohort_niche_stability": .5, "niche_stability": 1}) == ["unstable_cohort_niche"]
    assert inspector_module.review_reasons({"cohort_niche_status": "single_niche_no_supported_subdivision",
        "cohort_niche_stability": np.nan}) == []


@pytest.mark.parametrize("changed", ["profile", "bundle"])
def test_cohort_rejects_same_ids_with_foreign_source_or_changed_assignment(cohort_specimens, changed):
    inputs = cohort_specimens["inputs"][0]
    if changed == "profile":
        path = inputs["profile_dir"] / "cell_profiles_manifest.json"
        value = json.loads(path.read_text())
        value["same_ids_do_not_prove_source"] = "different producer"
        path.write_text(json.dumps(value))
    else:
        path = cohort_specimens["bundle"] / "cohort_niche_assignments.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        inspector_module.CellInspector(**inputs, cohort_niches=cohort_specimens["bundle"])


def test_cohort_source_changes_fail_before_cached_detail_or_flags(cohort_specimens, monkeypatch):
    inputs = cohort_specimens["inputs"][0]
    inspector = inspector_module.CellInspector(**inputs, cohort_niches=cohort_specimens["bundle"])
    calls = []
    original = cohort_niche_io.verify_cohort_sources
    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(cohort_niche_io, "verify_cohort_sources", tracked)
    inspector.detail(inspector.cells.cell_uid.iloc[0])
    inspector.metadata()
    inspector.queue()
    assert not calls, "Unchanged cell clicks must not reread all source features"
    path = inputs["profile_dir"] / "cell_profiles_manifest.json"
    path.touch()
    inspector.metadata()
    assert len(calls) == 1
    path.write_bytes(path.read_bytes() + b" ")
    for method in (lambda: inspector.detail(inspector.cells.cell_uid.iloc[0]), inspector.metadata,
                   inspector.queue, lambda: inspector.search("001"),
                   lambda: inspector.similar(inspector.cells.cell_uid.iloc[0], ["context"])):
        with pytest.raises(ValueError):
            method()


def test_cohort_source_symlink_retarget_is_not_silently_trusted(cohort_specimens, tmp_path):
    inputs = cohort_specimens["inputs"][0]
    inspector = inspector_module.CellInspector(**inputs, cohort_niches=cohort_specimens["bundle"])
    path = inputs["profile_dir"] / "cell_profiles_manifest.json"
    outside = tmp_path / "foreign_manifest.json"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        inspector.metadata()


def atlas_payload(fixture, **kwargs):
    return build_specimen_atlas(execution_dir=fixture["execution"], run_name="explicit-selection-test",
        success=True, analysis_intent="exploratory", uni2_sampling_mode="grid", cell_detection_mode="consensus",
        claim_ceiling="synthetic_contract_only", project_records=fixture["records"], quality_signals=[],
        uncertainty_register=[], **kwargs)


def test_indexed_cohort_is_not_guessed_to_be_the_current_run(cohort_specimens):
    payload = atlas_payload(cohort_specimens)
    assert [row["sample_id"] for row in payload["specimens"]] == ["specimen A", "specimen B"]
    for specimen in payload["specimens"]:
        launch = specimen["cell_inspector"]
        assert launch["status"] == "available"
        assert launch["cohort_niches"]["status"] == "not_attached_current_run_unverified"
        assert "cohort_niches" not in launch["datasets"]
    assert "producing run" in render_specimen_atlas(payload)


def test_explicit_portable_atlas_selection_validates_each_exact_profile(cohort_specimens, tmp_path):
    fixture = cohort_specimens
    payload = atlas_payload(fixture, cohort_niches=fixture["bundle"])
    path = fixture["execution"] / "specimen_atlas.json"
    path.write_text(json.dumps(payload))
    for specimen in payload["specimens"]:
        launch = specimen["cell_inspector"]
        assert launch["cohort_niches"]["status"] == "verified_explicit_source_bound"
        assert launch["cohort_niches"]["cohort_niche_model_id"] == fixture["model"]["cohort_niche_model_id"]
        inputs = inspector_module.atlas_launch_inputs(path, specimen["sample_id"])
        assert inputs["cohort_niches"] == fixture["bundle"]
        actual = inspector_module.CellInspector(**inputs)
        assert actual.manifest["sample_id"] == specimen["sample_id"]
        assert actual.metadata()["cohort_niches"]["status"] == "verified_source_bound"
    moved = tmp_path / "moved_results"
    shutil.copytree(fixture["execution"].parent, moved)
    inputs = inspector_module.atlas_launch_inputs(moved / "00_execution/specimen_atlas.json", "specimen A")
    assert inspector_module.CellInspector(**inputs).cohort_record["sample_id"] == "specimen A"


@pytest.mark.parametrize("mode", ["explicit_addition", "same_declared", "conflict", "wrong_source"])
def test_atlas_cli_accepts_only_nonconflicting_explicit_cohort(cohort_specimens, tmp_path, mode):
    fixture = cohort_specimens
    declared = mode in ("same_declared", "conflict")
    payload = atlas_payload(fixture, **({"cohort_niches": fixture["bundle"]} if declared else {}))
    atlas = fixture["execution"] / "specimen_atlas.json"
    atlas.write_text(json.dumps(payload))
    bundle = fixture["bundle"] / "cohort_niches_completion.json"
    if mode == "conflict":
        bundle = tmp_path / "different_bundle"
        shutil.copytree(fixture["bundle"], bundle)
    elif mode == "wrong_source":
        path = fixture["inputs"][0]["profile_dir"] / "cell_profiles_manifest.json"
        content = json.loads(path.read_text())
        content["same_ids_foreign_profile"] = True
        path.write_text(json.dumps(content))
    command = [sys.executable, str(Path(inspector_module.__file__)), "--atlas-manifest", str(atlas),
               "--sample-id", "specimen A", "--cohort-niches", str(bundle), "--check-only"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if mode in ("explicit_addition", "same_declared"):
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["cohort_niches"]["status"] == "verified_source_bound"
    else:
        assert result.returncode != 0
        assert ("conflicts" if mode == "conflict" else "SHA256") in result.stderr


@pytest.mark.parametrize("change", ["missing_index", "duplicate_index", "wrong_stage", "other_directory"])
def test_explicit_atlas_does_not_attach_ambiguous_or_noncurrent_stage(cohort_specimens, change, tmp_path):
    fixture = cohort_specimens
    records = [row for row in fixture["records"] if row["stage_id"] == "cohort_niches"
               or Path(row["relative_path"]).parts[0] == "specimen A"]
    cohort_records = [row for row in records if row["stage_id"] == "cohort_niches"]
    selected = fixture["bundle"]
    if change == "missing_index":
        records.remove(cohort_records[0])
    elif change == "duplicate_index":
        records.append(cohort_records[0])
    elif change == "wrong_stage":
        records = [{**row, "stage_folder": "old_25_cohort_niches"} if row in cohort_records else row for row in records]
    else:
        selected = tmp_path / "old_cohort"
        shutil.copytree(fixture["bundle"], selected)
    launch = discover_cell_inspector(records, fixture["execution"], "specimen A", cohort_niches=selected)
    assert launch["status"] == "available"
    assert launch["cohort_niches"]["status"] == "unavailable"
    assert "cohort_niches" not in launch["datasets"]


def test_actual_check_only_never_imports_training_dependencies(cohort_specimens):
    inputs = cohort_specimens["inputs"][0]
    command = [sys.executable, "-c", """
import importlib.abc, runpy, sys
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('sklearn', 'torch'):
            raise RuntimeError('Training dependency imported during read-only inspection: '+fullname)
sys.meta_path.insert(0, NoTraining())
script = sys.argv.pop(1)
sys.path.insert(0, str(__import__('pathlib').Path(script).parent))
runpy.run_path(script, run_name='__main__')
""", str(Path(inspector_module.__file__))]
    for key, value in inputs.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    command += ["--cohort-niches", str(cohort_specimens["bundle"]), "--check-only"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    meta = json.loads(result.stdout)
    assert meta["cohort_niches"]["status"] == "verified_source_bound"
    assert meta["cell_count"] == 5


def test_real_loopback_cohort_detail_metadata_and_frontend(cohort_specimens):
    inspector = inspector_module.CellInspector(**cohort_specimens["inputs"][0], cohort_niches=cohort_specimens["bundle"])
    server = inspector_module.make_server(inspector, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def get(path):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            connection.request("GET", path)
            response = connection.getresponse()
            body = response.read()
            connection.close()
            return response.status, body
        code, body = get("/api/cell?" + urlencode({"uid": inspector.cells.cell_uid.iloc[0]}))
        detail = json.loads(body)
        assert code == 200 and detail["cell_id"] == "001"
        assert detail["cohort_niche"]["cohort_niche_model_id"] == cohort_specimens["model"]["cohort_niche_model_id"]
        code, body = get("/api/metadata")
        assert code == 200 and json.loads(body)["cohort_niches"]["record"]["sample_id"] == "specimen A"
        code, body = get("/")
        assert code == 200 and b'id="cohort-niche"' in body
        assert b"does not replace the per-slide niche" in body
        assert get("/api/cell?cohort_niches=/tmp/foreign")[0] == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.skipif(os.environ.get("RUN_INSPECTOR_BROWSER_TESTS") != "1",
                    reason="Opt-in actual headless Chrome interaction QA; no visible panels")
def test_headless_cohort_fields_flags_and_literal_identity(cohort_specimens, tmp_path, monkeypatch):
    chrome = os.environ.get("CHROME_BIN", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not Path(chrome).is_file():
        pytest.skip("No Chrome executable available")
    inspector = inspector_module.CellInspector(**cohort_specimens["inputs"][0], cohort_niches=cohort_specimens["bundle"])
    expected_uid = inspector.cells.cell_uid.iloc[-1]
    driver = """<script>
    (async()=>{try{
      for(let i=0;i<100&&!current;i++)await new Promise(r=>setTimeout(r,50));
      if(!current||current.cell_id!=='001')throw Error('Initial exact ID missing');
      const cohort=document.getElementById('cohort-niche'),spatial=document.getElementById('spatial');
      if(!cohort.textContent.includes('cohort_niche_model_id'))throw Error('Cohort model not rendered');
      if(!spatial.textContent.includes('reference_id')||spatial.textContent.includes('cohort_niche_id'))throw Error('Separate layers mixed');
      if(document.getElementById('selected-uid').textContent!==current.cell_uid)throw Error('UID was not rendered literally');
      await selectCell(EXPECTED_UID);
      if(current.cohort_niche.cohort_niche_id!==null||!cohort.textContent.includes('Unknown / unavailable'))throw Error('Unassigned converted into class');
      if(!document.getElementById('flags').textContent.includes('cohort_niche_review'))throw Error('Cohort flag missing');
      if(!document.getElementById('cohort-provenance').textContent.includes('source_identity'))throw Error('Source evidence missing');
      document.body.dataset.cohortQa='passed';
    }catch(error){document.body.dataset.cohortQa='failed: '+error.message;}})();
    </script>""".replace("EXPECTED_UID", json.dumps(expected_uid))
    page = tmp_path / "cohort_test.html"
    page.write_text(inspector_module.HTML_PATH.read_text().replace("</body>", driver + "</body>"))
    monkeypatch.setattr(inspector_module, "HTML_PATH", page)
    server = inspector_module.make_server(inspector, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                   "--disable-background-networking", "--disable-component-update", "--disable-sync", "--disable-extensions",
                   "--user-data-dir=" + str(tmp_path / "chrome-profile"), "--virtual-time-budget=15000",
                   "--window-size=1500,1100", "--dump-dom", f"http://127.0.0.1:{server.server_port}"]
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=25)
            assert completed.returncode == 0, completed.stderr[-2000:]
            rendered = completed.stdout
        except subprocess.TimeoutExpired as exc:
            rendered = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        assert 'data-cohort-qa="passed"' in rendered, rendered[-3500:]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
