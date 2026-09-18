import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from build_uni2_spatial_grid import MaskReader, build_records  # noqa: E402


def test_grid_builder_filters_tissue_and_keeps_adjacent_cores(tmp_path: Path) -> None:
    tissue = np.zeros((6, 8), dtype=np.uint8)
    tissue[:, :4] = 1
    mask_path = tmp_path / "tissue.tif"
    tifffile.imwrite(mask_path, tissue)

    reader = MaskReader(mask_path)
    try:
        records, geometry, occupancy, selected = build_records(
            reader, context_size=4, stride=2, min_tissue_fraction=0.5
        )
    finally:
        reader.close()

    assert geometry["grid_rows"] == 3
    assert geometry["grid_cols"] == 4
    assert len(records) == 6
    assert selected.sum() == 6
    assert occupancy.shape == (3, 4)
    frame = pd.DataFrame(records)
    for _, row in frame.groupby("grid_row"):
        ordered = row.sort_values("grid_col")
        if len(ordered) > 1:
            np.testing.assert_array_equal(
                ordered["core_x0"].to_numpy()[1:], ordered["core_x1"].to_numpy()[:-1]
            )


def test_grid_builder_maps_full_resolution_cores_to_downsampled_tissue_mask(tmp_path: Path) -> None:
    tissue = np.zeros((3, 4), dtype=np.uint8)
    tissue[:, :2] = 1
    mask_path = tmp_path / "tissue_downsampled.tif"
    tifffile.imwrite(mask_path, tissue)

    reader = MaskReader(mask_path)
    try:
        records, geometry, occupancy, _ = build_records(
            reader,
            image_shape=(6, 8),
            context_size=4,
            stride=2,
            min_tissue_fraction=0.5,
        )
    finally:
        reader.close()

    assert len(records) == 6
    assert occupancy.shape == (3, 4)
    assert geometry["tissue_mask_scale_x"] == 0.5
    assert geometry["tissue_mask_scale_y"] == 0.5
    assert max(record["x"] for record in records) < 4


def test_grid_cluster_rasterizer_reconstructs_core_labels(tmp_path: Path) -> None:
    background = np.full((4, 4, 3), 180, dtype=np.uint8)
    background_path = tmp_path / "background.tif"
    tifffile.imwrite(background_path, background, photometric="rgb")

    objects = pd.DataFrame(
        [
            {"label": 1, "grid_row": 0, "core_x0": 0, "core_y0": 0, "core_x1": 2, "core_y1": 2},
            {"label": 2, "grid_row": 0, "core_x0": 2, "core_y0": 0, "core_x1": 4, "core_y1": 2},
            {"label": 3, "grid_row": 1, "core_x0": 0, "core_y0": 2, "core_x1": 2, "core_y1": 4},
            {"label": 4, "grid_row": 1, "core_x0": 2, "core_y0": 2, "core_x1": 4, "core_y1": 4},
        ]
    )
    objects_path = tmp_path / "grid.csv"
    objects.to_csv(objects_path, index=False)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "image_height_px": 4,
                "image_width_px": 4,
                "source_mpp_x": 0.25,
                "source_mpp_y": 0.25,
            }
        )
    )
    clusters_path = tmp_path / "clusters.csv"
    pd.DataFrame(
        {
            "label": [1, 2, 3, 4],
            "cluster": [1, 2, 3, 4],
            "interpretable_cluster": [1, 2, None, 4],
            "interpretation_status": [
                "accepted",
                "accepted",
                "abstained_seed_instability",
                "accepted",
            ],
        }
    ).to_csv(clusters_path, index=False)
    output_path = tmp_path / "cluster_mask.tif"
    uncertainty_path = tmp_path / "cluster_uncertainty_mask.tif"
    preview_path = tmp_path / "preview.png"
    uncertainty_preview_path = tmp_path / "uncertainty_preview.png"
    summary_path = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin" / "grid_clusters_to_mask.py"),
            "--grid-objects",
            str(objects_path),
            "--grid-metadata",
            str(metadata_path),
            "--map",
            str(clusters_path),
            "--out",
            str(output_path),
            "--uncertainty-out",
            str(uncertainty_path),
            "--summary",
            str(summary_path),
            "--preview",
            str(preview_path),
            "--uncertainty-preview",
            str(uncertainty_preview_path),
            "--preview-background",
            str(background_path),
        ],
        check=True,
    )

    observed = tifffile.imread(output_path)
    expected = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 4, 4], [0, 0, 4, 4]], dtype=np.uint16
    )
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        tifffile.imread(uncertainty_path),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 0, 0], [2, 2, 0, 0]], dtype=np.uint8),
    )
    assert preview_path.stat().st_size > 0
    assert uncertainty_preview_path.stat().st_size > 0
    summary = json.loads(summary_path.read_text())
    assert summary["grid_observations"] == 4
    assert summary["abstained_observations"] == 1
    assert summary["uncertainty_status_counts"]["abstained_seed_instability"] == 1


def test_pipeline_routes_grid_observations_through_uni2_and_kodama() -> None:
    main = (ROOT / "main.nf").read_text(encoding="utf-8")
    stages = (ROOT / "lib" / "PipelineInputs.groovy").read_text(encoding="utf-8")
    shared = (ROOT / "modules" / "extract_uni2_embeddings_shared.nf").read_text(encoding="utf-8")
    extraction = (ROOT / "subworkflows" / "extract_primary_uni2.nf").read_text(encoding="utf-8")
    assert "'grid_tiles', 'uni2', 'kodama'" in stages
    assert "def analysis_objects_ch = uni2_grid_mode ? grid_objects_ch : objects_assigned_ch" in main
    assert ".join(analysis_objects_ch)" in main
    assert "EXTRACT_PRIMARY_UNI2(" in main
    assert "tuple(sample_id, image_tif, tissue_mask_tif, 'grid', grid_objects_csv, resolution_json)" in extraction
    assert '--objects-csv \\"${observations_file}\\"' in shared


def test_both_mode_compares_routes_on_matched_grid_units() -> None:
    main = (ROOT / "main.nf").read_text(encoding="utf-8")
    route = (ROOT / "subworkflows" / "run_auxiliary_cell_route.nf").read_text(
        encoding="utf-8"
    )
    module = (ROOT / "modules" / "compare_uni2_routes.nf").read_text(encoding="utf-8")
    script = (ROOT / "bin" / "compare_uni2_routes.R").read_text(encoding="utf-8")

    assert "RUN_AUXILIARY_CELL_ROUTE(" in main
    assert "COMPARE_UNI2_ROUTES(comparisonInputCh)" in route
    assert "auxiliary_id.replaceFirst(/__cells$/" in route
    assert "uni2_route_comparison_${sample_id}" in module
    assert "procrustes_align" in script
    assert "neighbor_overlap" in script
    assert "virtual_marker_cross_validated_prediction" in script
    assert "does not select a preferred route automatically" in script


def test_grid_route_bypasses_cell_mask_growth_but_keeps_medsam() -> None:
    main = (ROOT / "main.nf").read_text(encoding="utf-8")
    auxiliary = (ROOT / "subworkflows" / "run_auxiliary_cell_spatial.nf").read_text(
        encoding="utf-8"
    )
    spatial = (ROOT / "subworkflows" / "post_grow_spatial_outputs.nf").read_text(
        encoding="utf-8"
    )

    assert "def run_grow_tissue = grow_tissue_requested && !uni2_grid_mode" in main
    assert (
        "} else if (uni2_grid_mode && (run_medsam_refine || run_cluster_geojson)) {"
        in main
    )
    assert "spatial_baseline_mask_ch = cluster_mask_ch" in main
    assert "GROW_TO_TISSUE(grow_input_ch)" in main
    assert "GROW_CELL_AUXILIARY_TO_TISSUE(growInputCh)" in auxiliary
    assert "def refinedMaskCh = spatial_baseline_mask_ch" in spatial
    assert "REFINE_GROWN_TISSUE_MEDSAM(refineInputCh, runtime_plan)" in spatial


def test_both_mode_runs_complete_namespaced_cell_route_and_can_restart() -> None:
    main = (ROOT / "main.nf").read_text(encoding="utf-8")
    route = (ROOT / "subworkflows" / "run_auxiliary_cell_route.nf").read_text(
        encoding="utf-8"
    )
    spatial = (ROOT / "subworkflows" / "run_auxiliary_cell_spatial.nf").read_text(
        encoding="utf-8"
    )

    assert "def run_auxiliary_cell_grow_tissue" in main
    assert "run_grow_tissue: run_auxiliary_cell_grow_tissue" in main
    assert '${sample_id}__cells' in route
    assert "/10_kodama/${auxiliaryId}/kodama_output" in route
    assert "/08_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif" in route
    for output_folder in (
        "11_clustering", "12_cluster_mask", "13_grown_tissue",
        "14_medsam_refine_tissue",
    ):
        assert output_folder in spatial
    assert "CELL_AUXILIARY_MASK_TO_GEOJSON(vectorInputCh)" in spatial
    assert ".join(finalProvenanceCh, by: [0, 1, 2], failOnMismatch: true, failOnDuplicate: true)" in spatial


def test_metro_routes_only_cell_masks_through_growth() -> None:
    metro = (ROOT / "docs" / "pipeline_metro" / "cellphenotyper_metro.mmd").read_text(
        encoding="utf-8"
    )

    assert "cell_cluster_mask -->|cell_morphology| grow_to_tissue" in metro
    assert "grid_cluster_mask -->|tissue_grid| medsam" in metro
    assert "grid_cluster_mask -->|tissue_grid| grow_to_tissue" not in metro


def test_uni2_gpu_selector_allows_unset_cuda_visible_devices() -> None:
    for module_name in ("extract_uni2_embeddings.nf", "extract_uni2_embeddings_shared.nf"):
        module = (ROOT / "modules" / module_name).read_text(encoding="utf-8")
        assert r"\${CUDA_VISIBLE_DEVICES:-}" in module
        assert r"\${GPU_SELECTOR%%,*}" in module
        assert r"\${CUDA_VISIBLE_DEVICES%%,*}" not in module
