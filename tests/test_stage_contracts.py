from pathlib import Path


ROOT = Path(__file__).parents[1]
MAIN = (ROOT / "main.nf").read_text(encoding="utf-8")


def test_requested_stages_fail_when_their_output_channel_is_empty() -> None:
    assert "def requireStageOutput" in MAIN
    assert "emitted no outputs" in MAIN
    for stage in (
        "convert", "grandqc", "stardist", "cell_consensus", "tma", "cell_assignment",
        "cytoplasm", "gigatime", "marker_quantification", "uni2", "kodama",
        "clustering", "cluster_mask", "grow_tissue", "medsam_refine", "cluster_geojson",
    ):
        assert f"requireStageOutput('{stage}'" in MAIN


def test_gigatime_and_marker_exact_restarts_load_stardist_prerequisites() -> None:
    assert "def need_analysis_inputs = stage_index[end_point] >= stage_index['stardist']" in MAIN


def test_cell_free_grid_route_decouples_crop_from_stardist() -> None:
    assert "def cell_detection_enabled = cell_detection_mode != 'none'" in MAIN
    assert "def run_stardist = should_run_stage('stardist') && cell_detection_enabled" in MAIN
    assert "def run_analysis_crop = should_run_stage('stardist') && need_analysis_inputs" in MAIN
    assert "if (run_analysis_crop)" in MAIN
    assert "BUILD_CLUSTER_MASK_OUTPUT(crop_roi_ch" in MAIN
    assert "def image_for_growth_ch = crop_roi_ch" in MAIN


def test_grandqc_is_mandatory_for_every_analysis_stage() -> None:
    assert "GrandQC is a mandatory upstream stage" in MAIN
    assert "PREPARE_ANALYSIS_CROP(crop_input_ch)" in MAIN
    assert ".join(grandqc_clean_tissue_mask_ch)" in MAIN
    assert "grandqc_clean_tissue_intersection_roi" in (
        ROOT / "bin" / "prepare_analysis_crop.py"
    ).read_text(encoding="utf-8")
