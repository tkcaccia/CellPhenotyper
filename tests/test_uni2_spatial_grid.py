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
            reader, context_size=4, stride=4, min_tissue_fraction=0.5
        )
    finally:
        reader.close()

    assert geometry["grid_rows"] == 2
    assert geometry["grid_cols"] == 2
    assert len(records) == 2
    assert selected.sum() == 2
    assert occupancy.shape == (2, 2)
    assert all(r["tile_x0"] == r["core_x0"] for r in records)
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
            stride=4,
            min_tissue_fraction=0.5,
        )
    finally:
        reader.close()

    assert len(records) == 2
    assert occupancy.shape == (2, 2)
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
    # The semantic abstention overlay must not erase raw clusters before Bora.
    expected = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint16
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


def test_grid_route_uses_only_nonoverlapping_virchow2_tiles() -> None:
    main = (ROOT / "main.nf").read_text(encoding="utf-8")
    encoder = (ROOT / "modules" / "extract_virchow2_grid.nf").read_text(encoding="utf-8")
    assert "EXTRACT_VIRCHOW2_GRID(input_ch)" in main
    assert "uni2_sampling_mode != 'grid'" in main
    assert "non-overlapping Virchow2 grid observations" in main
    assert "--grid-objects" in encoder
    assert "--grid-metadata" in encoder
    assert "--checkpoint" in encoder


def test_grid_route_hands_original_image_coarse_mask_and_support_to_bora() -> None:
    main = (ROOT / "main.nf").read_text(encoding="utf-8")
    bridge = (ROOT / "modules" / "refine_virchow2_bora.nf").read_text(encoding="utf-8")
    assert "def run_grow_tissue = grow_tissue_requested && !uni2_grid_mode" in main
    assert "REFINE_VIRCHOW2_BORA(bora_input_ch)" in main
    for item in (".join(image_for_growth_variant_ch)", ".join(tissue_mask_variant_ch)",
                 ".join(grid_metadata_variant_ch)", ".join(bora_assets_ch)"):
        assert item in main
    for item in ("path(image_tif)", "path(coarse_mask_tif)", "path(tissue_mask_tif)",
                 "path(classifier_model)", "path(checkpoint_dir)", "--bora-source"):
        assert item in bridge


def test_metro_routes_only_cell_masks_through_growth() -> None:
    metro = (ROOT / "docs" / "pipeline_metro" / "cellphenotyper_metro.mmd").read_text(
        encoding="utf-8"
    )

    assert "cell_cluster_mask -->|cell_morphology| grow_to_tissue" in metro
    assert "grid_cluster_mask -->|tissue_grid| medsam" in metro
    assert "grid_cluster_mask -->|tissue_grid| grow_to_tissue" not in metro
