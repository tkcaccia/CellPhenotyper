import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from write_pipeline_execution_reports import (  # noqa: E402
    STAGE_DEFS,
    UNCERTAINTY_SPECS,
    build_uncertainty_register,
    collect_quality_signals,
    list_files,
    preserve_trace,
    render_landing_page,
    stage_summary,
    status_class,
    summarize_trace,
)


def write_trace(path: Path, process: str, realtime: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["process", "status", "realtime", "peak_rss"])
        writer.writerow([process, "COMPLETED", realtime, "2 GB"])


def test_targeted_run_does_not_replace_full_pipeline_trace(tmp_path: Path) -> None:
    execution_dir = tmp_path / "00_execution"
    execution_dir.mkdir()
    current_trace = execution_dir / "trace.tsv"

    write_trace(current_trace, "FULL_PROCESS", "10m")
    trace_source, metadata = preserve_trace(
        execution_dir, "full_run", True, "convert", "titan"
    )
    assert trace_source.name == "full_pipeline_trace.tsv"
    assert metadata["run_name"] == "full_run"

    write_trace(current_trace, "TARGETED_PROCESS", "5s")
    trace_source, metadata = preserve_trace(
        execution_dir, "targeted_run", True, "grandqc", "grandqc"
    )
    assert metadata["run_name"] == "full_run"
    assert summarize_trace(trace_source)[0]["process"] == "FULL_PROCESS"
    assert (execution_dir / "run_traces" / "targeted_run_grandqc_to_grandqc.tsv").exists()

    stored = json.loads((execution_dir / "full_pipeline_run.json").read_text())
    assert stored["start_point"] == "convert"
    assert len(list((execution_dir / "run_traces").glob("*.tsv"))) == 2


def test_project_paths_remain_in_published_tree_for_relative_links(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    target = tmp_path / "work" / "task" / "consensus_sample"
    target.mkdir(parents=True)
    (target / "objects.csv").write_text("label,x,y\n1,1,1\n")
    published_parent = outdir / "03d_cell_consensus" / "sample"
    published_parent.mkdir(parents=True)
    published_dir = published_parent / "consensus_sample"
    published_dir.symlink_to(target, target_is_directory=True)

    records = list_files(outdir / "03d_cell_consensus")
    assert len(records) == 1
    record = records[0]
    assert record["absolute_path"] == str(published_dir / "objects.csv")
    assert record["resolved_target_path"] == str(target / "objects.csv")
    assert record["absolute_path"].startswith(str(outdir))

    consensus = next(row for row in stage_summary(outdir) if row["id"] == "cell_consensus")
    assert consensus["key_files"] == [str(published_dir / "objects.csv")]


def test_report_contract_lists_new_consensus_and_medsam_qc_outputs() -> None:
    execution = next(row for row in STAGE_DEFS if row["id"] == "execution")
    consensus = next(row for row in STAGE_DEFS if row["id"] == "cell_consensus")
    gigatime = next(row for row in STAGE_DEFS if row["id"] == "gigatime")
    kodama = next(row for row in STAGE_DEFS if row["id"] == "kodama")
    clustering = next(row for row in STAGE_DEFS if row["id"] == "clustering")
    medsam = next(row for row in STAGE_DEFS if row["id"] == "medsam_refine_tissue")
    roi = next(row for row in STAGE_DEFS if row["id"] == "roi")
    assert "validation_readiness.json" in execution["expected"]
    assert "uncertainty_register.json" in execution["expected"]
    assert "uncertainty_register.tsv" in execution["expected"]
    assert "specimen_atlas.html" in execution["expected"]
    assert "specimen_atlas.json" in execution["expected"]
    assert "storage_preflight.json" in execution["expected"]
    assert "model_inventory.json" in execution["expected"]
    assert "model_inventory.tsv" in execution["expected"]
    assert "detector_agreement_benchmark.csv" in consensus["expected"]
    assert "detector_agreement_benchmark.json" in consensus["expected"]
    assert "gigatime_seam_qc.json" in gigatime["expected"]
    assert "gigatime_marker_score_qc.json" in gigatime["expected"]
    assert "_uni2_route_comparison.csv" in kodama["expected"]
    assert "_cluster_stability.csv" in clustering["expected"]
    assert "_cluster_resolution_candidates.csv" in clustering["expected"]
    assert "cluster_interpretation_summary.json" in clustering["expected"]
    assert "_medsam_tissue_support.png" in medsam["expected"]
    assert "_medsam_grandqc_empty_exclusion.png" in medsam["expected"]
    assert ".roi_qc.json" in roi["expected"]
    assert "index.html" in execution["expected"]


def test_reference_bundle_members_are_all_visible_key_files(tmp_path):
    names = {'reference_assignments.csv', 'reference_assignments.atlas.json', 'reference_assignments.mapping.json'}
    for stage_id in ('reference_mapping', 'region_reference_mapping'):
        stage = next(row for row in STAGE_DEFS if row['id'] == stage_id)
        assert set(stage['expected']) == names
        directory = tmp_path / stage['folder'] / 'sample'
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_text('inventory-only fixture')
    for stage in stage_summary(tmp_path):
        if stage['id'] in ('reference_mapping', 'region_reference_mapping'):
            assert {Path(path).name for path in stage['key_files']} == names


def test_cohort_bundle_members_and_uncertainty_are_visible(tmp_path):
    names={'cohort_niche_assignments.parquet','cohort_niche_model.json',
           'cohort_niche_summary.json','cohort_niches_completion.json'}
    definition=next(row for row in STAGE_DEFS if row['id']=='cohort_niches')
    assert set(definition['expected'])==names
    directory=tmp_path/definition['folder']/'cohort_niches'; directory.mkdir(parents=True)
    for name in names: (directory/name).write_text('inventory-only fixture')
    stage=next(row for row in stage_summary(tmp_path) if row['id']=='cohort_niches')
    assert {Path(path).name for path in stage['key_files']}==names
    assert 'not_a_probability' in UNCERTAINTY_SPECS['cohort_niches']


def test_refinement_provenance_is_reported_without_claiming_revalidation(tmp_path):
    refinement = tmp_path / "14_medsam_refine_tissue" / "sample" / "sample_standard_grown_mask_refined.ome.tif.provenance.json"
    vector = tmp_path / "15_cluster_geojson" / "sample" / "sample.geojson.provenance.json"
    refinement.parent.mkdir(parents=True)
    vector.parent.mkdir(parents=True)
    refinement.write_text(json.dumps({"output_binding_status": "complete", "input_uncertainty_provided": True}))
    vector.write_text(json.dumps({"schema_version": "cellphenotyper.vectorized_refinement_provenance.v1",
                                  "producer_output_binding": "exact_output_sha256_verified",
                                  "source_label_zero_included": True}))
    signals = {row["kind"]: row for row in collect_quality_signals(tmp_path)}
    for kind in ("refinement_provenance", "vectorized_refinement_provenance"):
        assert signals[kind]["status"] == "technical_provenance_recorded"
        assert "source hashes not reverified" in signals[kind]["details"]
    register = {row["stage_id"]: row for row in build_uncertainty_register(stage_summary(tmp_path), list(signals.values()))}
    assert register["cluster_geojson"]["evidence_paths"] == [str(vector)]
    assert register["cluster_geojson"]["calibration_status"] == "not_a_probability"
    assert "not each smoothed polygon" in register["cluster_geojson"]["interpretation_limit"]
    assert register["medsam_refine_tissue"]["evidence_paths"] == [str(refinement)]
    # Pending producers and unbound historical vectors remain reviewable, not pass.
    refinement.write_text(json.dumps({"output_binding_status": "pending_final_label_export"}))
    vector.write_text(json.dumps({"producer_output_binding": "legacy_unbound_output_files"}))
    signals = {row["kind"]: row for row in collect_quality_signals(tmp_path)}
    assert signals["refinement_provenance"]["status"] == "review"
    assert signals["vectorized_refinement_provenance"]["status"] == "review"
    # A stub cannot be promoted by completion-like fixture metadata.
    refinement.write_text(json.dumps({"stub": True, "output_binding_status": "complete"}))
    signals = {row["kind"]: row for row in collect_quality_signals(tmp_path)}
    assert signals["refinement_provenance"]["status"] == "review"


def test_vector_provenance_reports_follow_published_directory_links(tmp_path):
    results = tmp_path / "results"
    parent = results / "15_cluster_geojson"
    parent.mkdir(parents=True)
    target = tmp_path / "work" / "vector_task"
    target.mkdir(parents=True)
    (target / "sample.geojson.provenance.json").write_text(json.dumps({
        "schema_version": "cellphenotyper.vectorized_refinement_provenance.v1",
        "producer_output_binding": "exact_output_sha256_verified", "source_label_zero_included": True}))
    (parent / "sample").symlink_to(target, target_is_directory=True)
    (target / "cycle").symlink_to(target, target_is_directory=True)
    rows = [row for row in collect_quality_signals(results) if row["kind"] == "vectorized_refinement_provenance"]
    assert len(rows) == 1
    assert rows[0]["path"] == str(parent / "sample" / "sample.geojson.provenance.json")


def test_landing_page_prioritizes_review_signals_and_uses_relative_links(
    tmp_path: Path,
) -> None:
    execution = tmp_path / "00_execution"
    execution.mkdir()
    (execution / "validation_readiness.json").write_text(
        json.dumps({"claim_ceiling": "exploratory_description_only"})
    )
    preview = tmp_path / "02_grandqc" / "sample_grandqc_artifact_overlay.jpg"
    preview.parent.mkdir()
    preview.write_bytes(b"preview")
    record = {
        "output_id": "grandqc_1234",
        "stage_folder": "02_grandqc",
        "relative_path": preview.name,
        "absolute_path": str(preview),
    }
    page = render_landing_page(
        execution_dir=execution,
        run_name="review run",
        success=True,
        start_point="convert",
        end_point="cluster_geojson",
        analysis_intent="tissue_domain_discovery",
        uni2_sampling_mode="grid",
        cell_detection_mode="consensus",
        stages=[
            {
                "id": "grandqc",
                "folder": "02_grandqc",
                "title": "GrandQC",
                "present": True,
                "file_count": 1,
                "size_bytes": 7,
                "size_human": "7 B",
            }
        ],
        quality_signals=[
            {
                "kind": "input_image_qc",
                "status": "pass",
                "details": "mpp=0.25",
                "path": str(tmp_path / "01_input" / "qc.json"),
            },
            {
                "kind": "gigatime_seam_qc",
                "status": "fail",
                "details": "failed boundaries=2",
                "path": str(tmp_path / "05_gigatime" / "seam.json"),
            },
        ],
        trace_rows=[
            {
                "process": "UNI2",
                "tasks": 1,
                "realtime_human": "1h 0m 0s",
                "peak_human": "8.0 GB",
                "failed": 0,
            }
        ],
        project_records=[record],
    )
    assert "exploratory_description_only" in page
    assert page.index("Gigatime Seam Qc") < page.index("Input Image Qc")
    assert "../02_grandqc/sample_grandqc_artifact_overlay.jpg" in page
    assert "file://" not in page
    assert "Technical completion is not evidence of biological accuracy" in page
    assert "Absence of an uncertainty estimate is reported explicitly" in page
    assert status_class("independent_review_required") == "review"
    assert status_class("rejected") == "fail"


def test_uncertainty_register_covers_every_stage_and_exposes_missing_estimates(
    tmp_path: Path,
) -> None:
    assert {stage["id"] for stage in STAGE_DEFS} == set(UNCERTAINTY_SPECS)
    stages = [
        {
            **stage,
            "present": stage["id"] in {"gigatime", "clustering", "pathofmpred"},
            "key_files": [str(tmp_path / stage["folder"] / "evidence.json")],
        }
        for stage in STAGE_DEFS
    ]
    cluster_evidence = tmp_path / "cluster_summary.csv"
    signals = [
        {
            "kind": "clustering_stability",
            "status": "review",
            "path": str(cluster_evidence),
            "details": "unstable=2",
        }
    ]

    rows = build_uncertainty_register(stages, signals)

    assert len(rows) == len(STAGE_DEFS)
    cluster = next(row for row in rows if row["stage_id"] == "clustering")
    assert cluster["uncertainty_implementation"] == "quantified_with_abstention"
    assert cluster["evidence_paths"] == [str(cluster_evidence)]
    gigatime = next(row for row in rows if row["stage_id"] == "gigatime")
    assert gigatime["uncertainty_implementation"] == "technical_qc_only"
    assert gigatime["calibration_status"] == "biological_scores_uncalibrated"
    pathofm = next(row for row in rows if row["stage_id"] == "pathofmpred")
    assert pathofm["uncertainty_implementation"] == "not_quantified"
    assert pathofm["calibration_status"] == "uncalibrated_score"


def test_human_review_record_is_a_quality_signal(tmp_path: Path) -> None:
    review_path = tmp_path / "00_execution" / "human_review.json"
    review_path.parent.mkdir(parents=True)
    review_path.write_text(
        json.dumps(
            {
                "reviewer": {
                    "role": "pathology reviewer",
                    "independent_of_pipeline_development": True,
                    "blinded_to_outcomes": True,
                },
                "stage_decisions": [
                    {"stage": "grandqc", "decision": "accepted_with_limitations"}
                ],
                "overall": {
                    "decision": "accepted_with_limitations",
                    "approved_claim_ceiling": "exploratory_description_only",
                },
            }
        ),
        encoding="utf-8",
    )

    signals = collect_quality_signals(tmp_path)

    assert len(signals) == 1
    assert signals[0]["kind"] == "human_review"
    assert signals[0]["status"] == "accepted_with_limitations"
    assert "independent=True" in signals[0]["details"]
    assert "stage_decisions=1" in signals[0]["details"]


def test_storage_preflight_is_reported_as_run_level_quality_signal(tmp_path: Path) -> None:
    storage_path = tmp_path / "00_execution" / "storage_preflight.json"
    storage_path.parent.mkdir(parents=True)
    storage_path.write_text(
        json.dumps(
            {
                "status": "warning",
                "policy_mode": "fail",
                "filesystems": [
                    {
                        "expected_increment_bytes": 10 * 1024**3,
                        "worst_case_increment_bytes": 20 * 1024**3,
                        "expected_headroom_bytes": 5 * 1024**3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    signals = collect_quality_signals(tmp_path)

    assert len(signals) == 1
    assert signals[0]["kind"] == "storage_capacity"
    assert signals[0]["status"] == "warning"
    assert "expected_increment=10.0 GB" in signals[0]["details"]
    assert "restart_worst_case=20.0 GB" in signals[0]["details"]


def test_model_inventory_is_reported_as_release_quality_signal(tmp_path: Path) -> None:
    inventory_path = tmp_path / "00_execution" / "model_inventory.json"
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(
        json.dumps(
            {
                "status": "fail",
                "release_ready": False,
                "used_model_count": 3,
                "release_blockers": [
                    {"component_id": "medsam", "missing_release_fields": ["license"]}
                ],
            }
        ),
        encoding="utf-8",
    )

    signals = collect_quality_signals(tmp_path)

    assert len(signals) == 1
    assert signals[0]["kind"] == "model_provenance"
    assert signals[0]["status"] == "fail"
    assert "used_models=3" in signals[0]["details"]
    assert "blocked_models=1" in signals[0]["details"]


def test_main_records_analysis_intent_and_route_contract() -> None:
    main = (ROOT / "main.nf").read_text(encoding="utf-8")
    config = (ROOT / "nextflow.config").read_text(encoding="utf-8")
    contract = (ROOT / "lib" / "ScientificContract.groovy").read_text(encoding="utf-8")
    evidence = (ROOT / "lib" / "StudyEvidence.groovy").read_text(encoding="utf-8")

    assert "analysis_intent                = 'exploratory'" in config
    assert "analysis_intent=cell_phenotyping requires --uni2_sampling_mode cells" in contract
    assert "analysis_intent=tissue_domain_discovery requires --uni2_sampling_mode grid or both" in contract
    assert "analysis_contract.json" in main
    assert "clinical_use_validated: false" in contract
    assert "uncertainty_policy" in contract
    assert "pipeline_primary_intended_use" in contract
    assert "no cross-taxonomy consensus phenotype is fabricated" in contract
    assert "new JsonSlurper().parse(analysisContractFile)" in main
    assert "'--end-point', reportEndPoint" in main
    assert "validation_readiness.json" in evidence
    assert "engineering_feasibility_and_exploratory_description_only" in evidence
    assert "reference_standard" in evidence
    assert "prespecification.locked_before_testing" in evidence
    assert "Use the term reference standard for the benchmark" in evidence


def test_quality_signal_collection_surfaces_seams_stability_and_route_limits(tmp_path: Path) -> None:
    input_qc = tmp_path / "01_input" / "sample" / "sample.converted_resolution.json"
    input_qc.parent.mkdir(parents=True)
    input_qc.write_text(
        json.dumps(
            {
                "status": "pass",
                "effective_mpp": 0.25,
                "file_sha256": "a" * 64,
                "mpp_conflicts": [],
                "image_layout": {
                    "shape": [100, 120, 3],
                    "dtype": "uint8",
                    "channel_count": 3,
                    "pyramid_level_count": 3,
                },
            }
        )
    )
    roi_qc = tmp_path / "06_roi" / "sample" / "sample.roi_qc.json"
    roi_qc.parent.mkdir(parents=True)
    roi_qc.write_text(
        json.dumps(
            {
                "status": "pass",
                "roi_source_kind": "provided",
                "roi_sha256": "b" * 64,
                "valid_in_bounds_feature_count": 2,
                "feature_count": 2,
                "hole_count": 1,
                "outside_image_area_px2": 0.0,
                "geometry_modified": False,
                "failures": [],
            }
        )
    )
    readiness = tmp_path / "00_execution" / "validation_readiness.json"
    readiness.parent.mkdir(parents=True)
    readiness.write_text(
        json.dumps(
            {
                "status": "not_declared",
                "study_phase": "not_declared",
                "gate_passed": False,
                "claim_ceiling": "engineering_feasibility_and_exploratory_description_only",
                "issues": ["No study manifest was supplied."],
            }
        )
    )
    consensus = tmp_path / "03d_cell_consensus" / "sample" / "consensus_summary.json"
    consensus.parent.mkdir(parents=True)
    consensus.write_text(
        json.dumps(
            {
                "fusion_acceptance_policy": "broad_pair",
                "consensus_count": 19,
                "detector_count_qc": {
                    "status": "expected_non_exhaustive_scope_imbalance",
                    "broad_scope_max_min_ratio": 1.1,
                    "all_detector_max_min_ratio": 6.0,
                },
            }
        )
    )
    seam = tmp_path / "05_gigatime" / "sample" / "gigatime_seam_qc.json"
    seam.parent.mkdir(parents=True)
    seam.write_text(
        json.dumps(
            {
                "status": "fail",
                "failed_channel_boundaries": 2,
                "channel_boundary_tests": 10,
            }
        )
    )
    marker_qc = tmp_path / "05_gigatime" / "sample" / "gigatime_marker_score_qc.json"
    marker_qc.write_text(
        json.dumps(
            {
                "status": "review",
                "channels": [{"channel": "DAPI"}, {"channel": "PD-1"}],
                "warnings": ["PD-1: sampled p05-p95 range below 0.005"],
                "value_semantics": {
                    "value_type": "uncalibrated_virtual_marker_score",
                    "calibrated_probability": False,
                },
            }
        )
    )
    cluster = tmp_path / "11_clustering" / "sample" / "sample_cluster_summary.csv"
    cluster.parent.mkdir(parents=True)
    with cluster.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stability_mean_adjusted_rand_index",
                "stability_min_adjusted_rand_index",
                "unstable_observation_count",
                "abstained_observation_count",
                "cluster_analysis_role",
                "forced_cluster_count_requested",
                "forced_cluster_count_applied",
                "target_clusters",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stability_mean_adjusted_rand_index": 0.8,
                "stability_min_adjusted_rand_index": 0.7,
                "unstable_observation_count": 3,
                "abstained_observation_count": 2,
                "cluster_analysis_role": "sensitivity_forced_cluster_count",
                "forced_cluster_count_requested": True,
                "forced_cluster_count_applied": True,
                "target_clusters": 2,
            }
        )
    route = tmp_path / "10_kodama" / "sample" / "sample_uni2_route_comparison.csv"
    route.parent.mkdir(parents=True)
    route.write_text("metric,value\ncommon_grid_cores,100\n")
    interpretation = (
        tmp_path
        / "11_clustering"
        / "sample"
        / "sample_standard_cluster_interpretation"
        / "cluster_interpretation_summary.json"
    )
    interpretation.parent.mkdir(parents=True)
    interpretation.write_text(
        json.dumps(
            {
                "status": "independent_review_required",
                "spatial_coherence": {
                    "status": "descriptive_only",
                    "excess_over_prevalence_chance": 0.25,
                },
                "marker_enrichment": {"status": "descriptive_only"},
            }
        )
    )

    signals = collect_quality_signals(tmp_path)

    assert [signal["kind"] for signal in signals] == [
        "input_image_qc",
        "roi_geometry_qc",
        "validation_readiness",
        "cell_instance_fusion",
        "gigatime_seam_qc",
        "gigatime_marker_score_qc",
        "clustering_stability",
        "forced_cluster_count",
        "cluster_interpretation",
        "uni2_route_comparison",
    ]
    assert [signal["status"] for signal in signals] == [
        "pass",
        "pass",
        "not_declared",
        "expected_non_exhaustive_scope_imbalance",
        "fail",
        "review",
        "review",
        "sensitivity_only",
        "independent_review_required",
        "descriptive_only",
    ]
