import json
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from specimen_atlas import build_specimen_atlas, render_specimen_atlas, discover_cell_inspector  # noqa: E402


def record(path: Path, outdir: Path, stage_id: str, output_id: str) -> dict:
    return {
        "output_id": output_id,
        "stage_id": stage_id,
        "stage_title": stage_id.replace("_", " ").title(),
        "stage_folder": path.relative_to(outdir).parts[0],
        "relative_path": path.relative_to(outdir / path.relative_to(outdir).parts[0]).as_posix(),
        "absolute_path": str(path),
        "resolved_target_path": str(path),
        "size_bytes": path.stat().st_size,
    }


def test_atlas_groups_primary_and_cell_routes_and_preserves_semantics(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    execution = outdir / "00_execution"
    execution.mkdir(parents=True)
    sample = "sample&one"

    converted = outdir / "01_input" / sample / f"{sample}.converted_resolution.json"
    converted.parent.mkdir(parents=True)
    converted.write_text(
        json.dumps({"status": "pass", "width_px": 2000, "height_px": 1000, "effective_mpp": 0.5}),
        encoding="utf-8",
    )
    grid = outdir / "09_grid_tiles" / sample / f"{sample}_uni2_grid_preview.png"
    grid.parent.mkdir(parents=True)
    grid.write_bytes(b"grid")
    consensus = outdir / "03d_cell_consensus" / f"{sample}__cells" / "consensus_preview.png"
    consensus.parent.mkdir(parents=True)
    consensus.write_bytes(b"cells")
    uncertainty = outdir / "11_clustering" / sample / f"{sample}_cluster_kodama_uncertainty.png"
    uncertainty.parent.mkdir(parents=True)
    uncertainty.write_bytes(b"uncertainty")

    records = [
        record(converted, outdir, "input", "input_1"),
        record(grid, outdir, "grid_tiles", "grid_1"),
        record(consensus, outdir, "cell_consensus", "cells_1"),
        record(uncertainty, outdir, "clustering", "uncertainty_1"),
    ]
    signal_path = outdir / "02_grandqc" / sample / "qc.json"
    payload = build_specimen_atlas(
        execution_dir=execution,
        run_name="unsafe <run>",
        success=True,
        analysis_intent="tissue_domain_discovery",
        uni2_sampling_mode="both",
        cell_detection_mode="consensus",
        claim_ceiling="exploratory_description_only",
        project_records=records,
        quality_signals=[{"kind": "grandqc", "status": "pass", "details": "safe <detail>", "path": str(signal_path)}],
        uncertainty_register=[
            {
                "stage_id": "clustering",
                "stage_title": "Clustering",
                "stage_present": True,
                "uncertainty_implementation": "quantified_with_abstention",
                "decision_or_abstention": "abstain on ambiguity",
                "calibration_status": "not_calibrated",
                "interpretation_limit": "stability is not identity",
            }
        ],
    )

    assert [row["sample_id"] for row in payload["specimens"]] == [sample]
    specimen = payload["specimens"][0]
    assets = {
        layer["id"]: layer["assets"]
        for layer in specimen["layers"]
    }
    assert assets["domains"][0]["route"] == "grid tissue-domain primary; cell-centred auxiliary"
    assert assets["cells"][0]["route"] == "cell-centred"
    assert assets["uncertainty"][0]["output_id"] == "uncertainty_1"
    assert not assets["virtual_markers"]
    assert assets["cells"][0]["semantic"].startswith("predicted nuclear instances")
    assert specimen["quality_signals"][0]["evidence_href"].startswith("../02_grandqc/")
    assert str(tmp_path) not in json.dumps(payload)

    page = render_specimen_atlas(payload)
    assert "unsafe &lt;run&gt;" in page
    assert "sample&amp;one" in page
    assert "safe &lt;detail&gt;" in page
    assert "Uncertainty Coverage" in page
    assert "Not produced in this stage window" in page
    assert "uncalibrated virtual-marker score; not measured protein abundance" in page


def test_execution_report_writes_and_indexes_specimen_atlas(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    converted = outdir / "01_input" / "sample_a" / "sample_a.converted_resolution.json"
    converted.parent.mkdir(parents=True)
    converted.write_text(json.dumps({"status": "pass", "effective_mpp": 0.5}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin" / "write_pipeline_execution_reports.py"),
            "--outdir", str(outdir),
            "--run-name", "atlas integration",
            "--success", "true",
            "--start-point", "convert",
            "--end-point", "convert",
            "--analysis-intent", "exploratory",
            "--uni2-sampling-mode", "grid",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    execution = outdir / "00_execution"
    assert (execution / "specimen_atlas.html").exists()
    assert (execution / "specimen_atlas.json").exists()
    assert "Specimen atlas" in (execution / "index.html").read_text(encoding="utf-8")
    report = json.loads((execution / "final_report.json").read_text(encoding="utf-8"))
    assert report["specimen_atlas_html"] == str(execution / "specimen_atlas.html")
    manifest = json.loads((execution / "project_outputs.json").read_text(encoding="utf-8"))
    atlas_records = [
        row for row in manifest["records"]
        if row["relative_path"] in {"specimen_atlas.html", "specimen_atlas.json"}
    ]
    assert len(atlas_records) == 2
    assert all(row["output_id"].startswith("execution_") for row in atlas_records)


def test_quality_signals_are_specimen_scoped_with_global_fallback(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    execution = outdir / "00_execution"
    execution.mkdir(parents=True)
    records = []
    for index, sample in enumerate(("sample_a", "sample_b"), start=1):
        source = outdir / "01_input" / sample / f"{sample}.converted_resolution.json"
        source.parent.mkdir(parents=True)
        source.write_text("{}", encoding="utf-8")
        records.append(record(source, outdir, "input", f"input_{index}"))

    payload = build_specimen_atlas(
        execution_dir=execution,
        run_name="two specimens",
        success=True,
        analysis_intent="exploratory",
        uni2_sampling_mode="grid",
        cell_detection_mode="consensus",
        claim_ceiling="exploratory_description_only",
        project_records=records,
        quality_signals=[
            {"kind": "qc_a", "status": "pass", "details": "A", "path": str(outdir / "02_grandqc" / "sample_a" / "qc.json")},
            {"kind": "qc_b", "status": "review", "details": "B", "path": str(outdir / "02_grandqc" / "sample_b" / "qc.json")},
            {"kind": "human_review", "status": "pending", "details": "run level", "path": str(execution / "human_review.json")},
        ],
        uncertainty_register=[],
    )

    signals = {
        specimen["sample_id"]: {row["kind"] for row in specimen["quality_signals"]}
        for specimen in payload["specimens"]
    }
    assert signals["sample_a"] == {"qc_a", "human_review"}
    assert signals["sample_b"] == {"qc_b", "human_review"}


def test_base_sample_detector_assets_retain_cell_characterization_route(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    execution = outdir / "00_execution"
    execution.mkdir(parents=True)
    source = outdir / "01_input" / "sample" / "sample.converted_resolution.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    overlay = outdir / "03_stardist" / "sample" / "stardist_out" / "overlay.png"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"overlay")

    payload = build_specimen_atlas(
        execution_dir=execution,
        run_name="routes",
        success=True,
        analysis_intent="tissue_domain_discovery",
        uni2_sampling_mode="grid",
        cell_detection_mode="consensus",
        claim_ceiling="exploratory_description_only",
        project_records=[
            record(source, outdir, "input", "input_1"),
            record(overlay, outdir, "stardist", "stardist_1"),
        ],
        quality_signals=[],
        uncertainty_register=[],
    )

    specimen = payload["specimens"][0]
    context = next(layer for layer in specimen["layers"] if layer["id"] == "context")
    cells = next(layer for layer in specimen["layers"] if layer["id"] == "cells")
    assert not context["assets"]
    assert cells["assets"][0]["route"] == "cell identification and marker phenotyping"


def inspector_records(outdir, sample="sample A"):
    specifications = {
        "objects": ("03d_cell_consensus", "consensus_sample/objects.csv", b"id,x,y\n001,10,20\n"),
        "labels": ("03d_cell_consensus", "consensus_sample/labels.tif", b"native labels"),
        "shift": ("03_stardist", "prepared_crop/shift.json", json.dumps({"source_mpp": .5}).encode()),
        "image": ("03_stardist", "prepared_crop/" + sample + ".tif", b"native H&E"),
        "resolution": ("01_input", sample + ".converted_resolution.json", b'{"status":"pass"}'),
    }
    paths, records = {}, []
    for key, (folder, name, data) in specifications.items():
        path = outdir / folder / sample / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths[key] = path
        records.append(record(path, outdir, "input" if key == "resolution" else "cell_consensus" if key in ("objects", "labels") else "stardist", key))
    profile = outdir / "19_cell_profiles" / sample / "cell_profiles" / "cell_profiles_manifest.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"sample_id": sample, "cell_count": 1, "inputs": {
        "objects_sha256": hashlib.sha256(paths["objects"].read_bytes()).hexdigest(),
        "shift_sha256": hashlib.sha256(paths["shift"].read_bytes()).hexdigest(),
        "resolution_json_sha256": hashlib.sha256(paths["resolution"].read_bytes()).hexdigest()}}))
    records.append(record(profile, outdir, "cell_profiles", "profiles"))
    paths["profile"] = profile
    return records, paths


def test_cell_inspector_discovery_requires_provenance_and_emits_portable_manual_launch(tmp_path):
    outdir = tmp_path / "results"
    execution = outdir / "00_execution"
    execution.mkdir(parents=True)
    sample = "sample 'A' & <review>"
    records, paths = inspector_records(outdir, sample)
    discovered = discover_cell_inspector(records, execution, sample)
    assert discovered["status"] == "available"
    assert discovered["auto_launch"] is False and discovered["read_only"] is True
    assert discovered["raster_identity_status"] == "legacy_partial_or_unverified"
    assert str(tmp_path) not in json.dumps(discovered)
    assert discovered["datasets"]["labels"].startswith("../03d_cell_consensus/")
    assert discovered["datasets"]["profile_dir"].startswith("../19_cell_profiles/")
    assert "--atlas-manifest specimen_atlas.json" in discovered["launch_command"]
    payload = build_specimen_atlas(execution_dir=execution, run_name="Inspector", success=True,
        analysis_intent="exploratory", uni2_sampling_mode="grid", cell_detection_mode="consensus",
        claim_ceiling="exploratory_description_only", project_records=records, quality_signals=[], uncertainty_register=[])
    page = render_specimen_atlas(payload)
    assert "Interactive Cell Review" in page and "Portable launch manifest" in page
    assert "sample &#x27;A&#x27; &amp; &lt;review&gt;" in page
    assert "<script" not in page
    assert "Legacy / incomplete raster provenance" in page
    assert payload["specimens"][0]["cell_inspector"] == discovered
    paths["objects"].write_text("id,x,y\n999,10,20\n")
    assert discover_cell_inspector(records, execution, sample)["status"] == "unavailable"


def test_cell_inspector_missing_or_ambiguous_profiles_are_not_guessed(tmp_path):
    execution = tmp_path / "results" / "00_execution"
    execution.mkdir(parents=True)
    assert discover_cell_inspector([], execution, "sample A")["status"] == "unavailable"
    records, paths = inspector_records(execution.parent)
    assert discover_cell_inspector(records + [records[-1]], execution, "sample A")["status"] == "unavailable"
    no_resolution = [row for row in records if row["absolute_path"] != str(paths["resolution"])]
    missing = discover_cell_inspector(no_resolution, execution, "sample A")
    assert missing["status"] == "unavailable" and "resolution" in missing["reason"]


def test_cell_inspector_prefers_only_hash_bound_linked_derivative(tmp_path):
    import hashlib
    execution = tmp_path / 'results/00_execution'
    execution.mkdir(parents=True)
    records, paths = inspector_records(execution.parent)
    base = paths['profile']
    derived = execution.parent / '24_cell_tissue_links/sample A/cell_profiles/cell_profiles_manifest.json'
    derived.parent.mkdir(parents=True)
    manifest = json.loads(base.read_text())
    manifest['cell_hierarchy'] = {'source_profile_manifest_sha256': hashlib.sha256(base.read_bytes()).hexdigest()}
    derived.write_text(json.dumps(manifest))
    archive = derived.parent / 'hierarchy_source/cell_profiles_manifest.json'
    archive.parent.mkdir()
    archive.write_bytes(base.read_bytes())
    records += [record(derived, execution.parent, 'cell_tissue_links', 'linked'),
                record(archive, execution.parent, 'cell_tissue_links', 'source_provenance')]
    launch = discover_cell_inspector(records, execution, 'sample A')
    assert launch['status'] == 'available'
    assert launch['datasets']['profile_dir'].startswith('../24_cell_tissue_links/')
    assert launch['auto_launch'] is False
    manifest['cell_hierarchy']['source_profile_manifest_sha256'] = 'a' * 64
    derived.write_text(json.dumps(manifest))
    assert discover_cell_inspector(records, execution, 'sample A')['status'] == 'unavailable'


def test_region_inspector_discovery_requires_hash_bound_portable_bundle(tmp_path):
    outdir = tmp_path / "results"
    execution = outdir / "00_execution"
    execution.mkdir(parents=True)
    records, paths = inspector_records(outdir)
    manifest = json.loads(paths["profile"].read_text())
    for name in ("image", "labels"):
        manifest["inputs"][name+"_sha256"] = hashlib.sha256(paths[name].read_bytes()).hexdigest()
    root = outdir / "22_tissue_hierarchy" / "sample A" / "hierarchy"
    root.mkdir(parents=True)
    files = {"region_mask.ome.tif": b"region labels", "hierarchy_status.ome.tif": b"status", "parent_domains.ome.tif": b"parents",
             "grid_subdomains.csv": b"grid", "region_profiles/region_profiles.csv": b"regions", "region_profiles/feature_rows.csv": b"identities",
             "region_profiles/local.npy": b"features"}
    for name, data in files.items():
        path = root / name; path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        records.append(record(path, outdir, "tissue_hierarchy", name))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["inputs"]["domain_mask_sha256"] = digest(root / "parent_domains.ome.tif")
    paths["profile"].write_text(json.dumps(manifest))
    region_manifest_path = root / "region_profiles/region_profiles_manifest.json"
    region_manifest = {"observation_unit": "tissue_region", "hierarchy_id": "version1", "region_count": 3,
        "feature_blocks": {"local": {"path": "local.npy"}},
        "files": {name: digest(root / "region_profiles" / name) for name in ("local.npy", "region_profiles.csv", "feature_rows.csv")}}
    region_manifest_path.write_text(json.dumps(region_manifest))
    records.append(record(region_manifest_path, outdir, "tissue_hierarchy", "region_manifest"))
    summary_path = root / "hierarchy_summary.json"
    summary = {"sample_id": "sample A", "hierarchy_id": "version1", "inputs": {
        "image": {"sha256": manifest["inputs"]["image_sha256"]},
        "parent_mask": {"sha256": manifest["inputs"]["domain_mask_sha256"]},
        "shift_json": {"sha256": manifest["inputs"]["shift_sha256"]},
        "resolution_json": {"sha256": manifest["inputs"]["resolution_json_sha256"]}},
        "outputs": {name: digest(root / name) for name in (*files, "region_profiles/region_profiles_manifest.json")}}
    summary_path.write_text(json.dumps(summary))
    records.append(record(summary_path, outdir, "tissue_hierarchy", "hierarchy_summary"))
    launch = discover_cell_inspector(records, execution, "sample A")
    assert launch["datasets"]["hierarchy_dir"] == "../22_tissue_hierarchy/sample A/hierarchy"
    assert launch["region_inspector"]["status"] == "available_runtime_validation_required"
    assert launch["region_inspector"]["region_count"] == 3
    assert str(tmp_path) not in json.dumps(launch)
    (root / "region_mask.ome.tif").write_bytes(b"different region raster")
    failed = discover_cell_inspector(records, execution, "sample A")
    assert failed["status"] == "available"  # cell-only review remains available
    assert "hierarchy_dir" not in failed["datasets"]
    assert failed["region_inspector"]["status"] == "unavailable"
    (root / "region_mask.ome.tif").write_bytes(files["region_mask.ome.tif"])
    summary.pop("outputs")
    summary_path.write_text(json.dumps(summary))
    assert "hierarchy_dir" not in discover_cell_inspector(records, execution, "sample A")["datasets"]
    for malformed in (["wrong type"], {"image": "not an input record"}):
        summary["inputs"] = malformed
        summary_path.write_text(json.dumps(summary))
        launch = discover_cell_inspector(records, execution, "sample A")
        assert launch["status"] == "available" and launch["region_inspector"]["status"] == "unavailable"


def test_portable_inspector_manifest_survives_results_move(tmp_path):
    import shutil
    original = tmp_path / "original"
    execution = original / "00_execution"
    execution.mkdir(parents=True)
    records, _ = inspector_records(original)
    launch = discover_cell_inspector(records, execution, "sample A")
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    for path in launch["datasets"].values():
        assert (moved / "00_execution" / path).exists()


def test_inspector_discovery_rejects_wrong_native_image_or_label_hash(tmp_path):
    execution = tmp_path / "results" / "00_execution"
    execution.mkdir(parents=True)
    records, paths = inspector_records(execution.parent)
    manifest = json.loads(paths["profile"].read_text())
    for name in ("image", "labels"):
        manifest["inputs"][name + "_sha256"] = hashlib.sha256(paths[name].read_bytes()).hexdigest()
    paths["profile"].write_text(json.dumps(manifest))
    launch = discover_cell_inspector(records, execution, "sample A")
    assert launch["status"] == "available" and launch["raster_identity_status"] == "verified_sha256"
    assert all(item["status"] == "verified_sha256" for item in launch["raster_provenance"].values())
    for name, reason in (("image", "H&E-image SHA256"), ("labels", "canonical-label SHA256")):
        original = paths[name].read_bytes()
        paths[name].write_bytes(original + b"changed source, same filename")
        failed = discover_cell_inspector(records, execution, "sample A")
        assert failed["status"] == "unavailable" and reason in failed["reason"]
        paths[name].write_bytes(original)


def test_discovery_partial_and_invalid_raster_hashes_are_not_fully_verified(tmp_path):
    execution = tmp_path / "results" / "00_execution"
    execution.mkdir(parents=True)
    records, paths = inspector_records(execution.parent)
    manifest = json.loads(paths["profile"].read_text())
    manifest["inputs"]["labels_sha256"] = hashlib.sha256(paths["labels"].read_bytes()).hexdigest()
    paths["profile"].write_text(json.dumps(manifest))
    launch = discover_cell_inspector(records, execution, "sample A")
    assert launch["status"] == "available" and launch["raster_identity_status"] == "legacy_partial_or_unverified"
    assert launch["raster_provenance"]["labels"]["status"] == "verified_sha256"
    assert launch["raster_provenance"]["image"]["status"] == "legacy_missing_sha256"
    manifest["inputs"]["image_sha256"] = None
    paths["profile"].write_text(json.dumps(manifest))
    launch = discover_cell_inspector(records, execution, "sample A")
    assert launch["status"] == "unavailable" and "not a valid source hash" in launch["reason"]
