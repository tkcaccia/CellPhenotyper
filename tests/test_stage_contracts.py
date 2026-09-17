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
    expected = (
        "run_gigatime || run_marker_quantification || run_grid_tiles || run_uni2 || "
        "run_cluster_mask"
    )
    assert expected in MAIN


def test_grandqc_is_mandatory_for_every_analysis_stage() -> None:
    assert "GrandQC is a mandatory upstream stage" in MAIN
    assert "PREPARE_ANALYSIS_CROP(crop_input_ch)" in MAIN
    assert ".join(grandqc_clean_tissue_mask_ch)" in MAIN
    assert "grandqc_clean_tissue_intersection_roi" in (
        ROOT / "bin" / "prepare_analysis_crop.py"
    ).read_text(encoding="utf-8")
