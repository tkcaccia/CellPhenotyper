from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main.nf").read_text(encoding="utf-8")
PIPELINE_INPUTS = (ROOT / "lib" / "PipelineInputs.groovy").read_text(encoding="utf-8")
PARAMETERS = (ROOT / "PARAMETERS.md").read_text(encoding="utf-8")
SPATIAL = (ROOT / "subworkflows" / "post_grow_spatial_outputs.nf").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
RELEASE_WORKFLOW = (
    ROOT / ".github" / "workflows" / "publish-runtime-release.yml"
).read_text(encoding="utf-8")


def test_medsam_is_an_independent_restart_stage() -> None:
    assert "medsam: 'medsam_refine'" in PIPELINE_INPUTS
    assert "medsam_refine_tissue: 'medsam_refine'" in PIPELINE_INPUTS
    assert PIPELINE_INPUTS.index("'grow_tissue', 'medsam_refine', 'cluster_geojson'") > 0
    assert "def run_medsam_refine = should_run_stage('medsam_refine')" in MAIN
    assert "if (runMedsamRefine && grownRefineMethod == 'medsam_border_refine')" in SPATIAL


def test_cluster_geojson_restart_consumes_published_medsam_output() -> None:
    expected = (
        '${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/'
        '${sample_id}_${clusterVariant}_grown_mask_refined.ome.tif'
    )
    assert expected in SPATIAL
    assert "} else if (runClusterGeoJSON && grownRefineMethod == 'medsam_border_refine') {" in SPATIAL
    assert (
        "} else if (run_grow_tissue || run_medsam_refine || "
        "(run_cluster_geojson && uni2_grid_mode)) {"
    ) in MAIN


def test_documented_stage_list_includes_medsam_restart() -> None:
    assert "`grow_tissue`, `medsam_refine`, `cluster_geojson`" in PARAMETERS


def test_github_workflows_run_the_pytest_suite() -> None:
    assert "python -m pytest -q" in CI
    assert "python -m pytest -q tests" in RELEASE_WORKFLOW
    assert "unittest discover" not in RELEASE_WORKFLOW


def test_clustering_restart_reuses_published_marker_quantification() -> None:
    assert "def published_gigatime_quant_dir_ch" in MAIN
    assert 'quantification_${sample_id}", checkIfExists: true' in MAIN
    assert "? published_gigatime_quant_dir_ch" in MAIN


def test_both_mode_cell_route_restarts_without_reopening_uni2() -> None:
    route = (ROOT / "subworkflows" / "run_auxiliary_cell_route.nf").read_text(
        encoding="utf-8"
    )
    spatial = (ROOT / "subworkflows" / "run_auxiliary_cell_spatial.nf").read_text(
        encoding="utf-8"
    )
    assert "} else if (runClustering) {" in route
    assert "/10_kodama/${auxiliaryId}/kodama_output" in route
    assert "} else if (runClusterMask) {" in spatial
    assert "/11_clustering/${sample_id}/" in spatial
    assert "} else if (runGrowTissue || runMedsamRefine) {" in spatial
    assert "/12_cluster_mask/${sample_id}/" in spatial
    assert "/13_grown_tissue/${sample_id}/" in spatial
    assert "/14_medsam_refine_tissue/${sample_id}/" in spatial
