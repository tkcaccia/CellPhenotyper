import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from analyze_cell_neighborhoods import (  # noqa: E402
    SupportMask,
    RasterSupportMask,
    FEATURE_REPRESENTATION_VERSION,
    _niche_matrix,
    bounded_grid_shape,
    analyze_cells,
    build_radius_graph,
    discover_niches,
    neighborhood_features,
    segment_in_support,
    validate_cells,
)


def cells(xy, samples=None, phenotypes=None):
    frame = pd.DataFrame(xy, columns=["x_um", "y_um"])
    frame["sample_id"] = samples if samples is not None else "sample"
    frame["cell_id"] = [f"c{i:04d}" for i in range(len(frame))]
    if phenotypes is not None:
        frame["phenotype"] = phenotypes
    return validate_cells(frame)


def test_graph_rejects_hole_even_when_same_connected_component():
    mask = np.ones((8, 10), bool)
    mask[3, 4:6] = False
    support = SupportMask(mask)
    frame = cells([(2.5, 3.5), (7.5, 3.5), (2.5, 2.5)])
    graph, valid = build_radius_graph(frame, {"sample": support}, 8)
    assert valid.all()
    assert support.component_at(np.array([2.5, 3.5])) == support.component_at(np.array([7.5, 3.5]))
    assert 1 not in graph[0].indices
    assert 2 in graph[0].indices
    assert (graph - graph.T).nnz == 0
    assert not np.any(graph.diagonal())


def test_segment_rejects_diagonal_corner_background_and_gridline():
    mask = np.ones((5, 5), bool)
    mask[1, 2] = False
    support = SupportMask(mask)
    assert not segment_in_support(np.array([1.5, 1.5]), np.array([2.5, 2.5]), support)
    assert not segment_in_support(np.array([2.5, 2.5]), np.array([1.5, 1.5]), support)
    assert not segment_in_support(np.array([1.5, 2.0]), np.array([3.5, 2.0]), support)
    assert segment_in_support(np.array([1.5, 3.5]), np.array([3.5, 3.5]), support)


def test_graph_keeps_specimens_separate_at_identical_coordinates():
    frame = cells([(1.5, 1.5), (2.5, 1.5), (1.5, 1.5), (2.5, 1.5)], ["A", "A", "B", "B"])
    supports = {sample: SupportMask(np.ones((5, 5), bool)) for sample in ["A", "B"]}
    graph, valid = build_radius_graph(frame, supports, 2)
    assert valid.all() and graph.nnz == 4
    assert set(graph[0].indices) == {1}
    assert set(graph[2].indices) == {3}
    with pytest.raises(ValueError, match="exactly one"):
        build_radius_graph(frame, {"A": supports["A"]}, 2)


def test_graph_physical_scale_origin_and_anisotropy():
    frame = cells([(13, 21), (17, 21), (13, 25)])
    support = SupportMask(np.ones((10, 10)), mpp=(2, 1), origin_um=(10, 20))
    g1, valid = build_radius_graph(frame, {"sample": support}, 4.1)
    transformed = frame.copy()
    transformed[["x_um", "y_um"]] *= 3
    support2 = SupportMask(support.mask, mpp=(6, 3), origin_um=(30, 60))
    g2, _ = build_radius_graph(transformed, {"sample": support2}, 12.3)
    np.testing.assert_array_equal(g1.indices, g2.indices)
    np.testing.assert_array_equal(g1.indptr, g2.indptr)
    np.testing.assert_allclose(g2.data, g1.data * 3)
    assert valid.all()
    area1 = support.tissue_area_in_disk(np.array([13, 21]), 4)
    area2 = support2.tissue_area_in_disk(np.array([39, 63]), 12)
    assert area2 == 9 * area1


def test_density_compartments_mixing_and_axial_alignment():
    frame = cells([(4.5, 4.5), (5.5, 4.5), (4.5, 5.5)], phenotypes=["A", "A", None])
    frame["orientation_rad"] = [0, np.pi, np.pi / 2]
    supports = {"sample": SupportMask(np.ones((10, 10)))}
    graph, valid = build_radius_graph(frame, supports, 2)
    profiles, groups = neighborhood_features(frame, {2.: graph}, supports, valid)
    assert profiles.loc[0, "r2um_neighbor_count"] == 2
    assert profiles.loc[0, "r2um_tissue_area_um2"] == 13
    assert profiles.loc[0, "r2um_neighbor_density_per_mm2"] == pytest.approx(2e6 / 13)
    assert profiles.loc[0, "r2um_phenotype_fraction:A"] == .5
    assert profiles.loc[0, "r2um_phenotype_fraction:__unknown__"] == .5
    assert profiles.loc[0, "r2um_phenotype_entropy_nats"] == pytest.approx(np.log(2))
    assert profiles.loc[0, "r2um_axial_coherence"] == pytest.approx(0, abs=1e-10)
    assert profiles.loc[2, "r2um_axial_coherence"] == pytest.approx(1)
    assert profiles.loc[2, "r2um_alignment_to_focal_axis"] == pytest.approx(-1)
    assert set(groups) == {"density", "composition", "orientation"}


def test_density_denominator_excludes_background():
    mask = np.ones((10, 10))
    mask[:, :5] = 0
    support = SupportMask(mask)
    frame = cells([(5.5, 4.5), (6.5, 4.5)])
    graph, valid = build_radius_graph(frame, {"sample": support}, 2)
    profiles, _ = neighborhood_features(frame, {2.: graph}, {"sample": support}, valid)
    assert profiles.loc[0, "r2um_tissue_area_um2"] == 9
    assert profiles.loc[0, "r2um_neighbor_density_per_mm2"] == pytest.approx(1e6 / 9)


def test_isolated_and_unsupported_cells_retained_with_explicit_missingness():
    frame = cells([(2.5, 2.5), (8.5, 8.5), (-10, 0)])
    profiles, niches, graphs, summary = analyze_cells(frame, {"sample": SupportMask(np.ones((10, 10)))}, radii_um=(1,))
    assert len(profiles) == 3 and summary["niche_discovery"]["selected_k"] == 0
    assert graphs[1].nnz == 0
    assert profiles.r1um_neighbor_count.eq(0).all()
    assert profiles["r1um_phenotype_fraction:__unknown__"].isna().all()
    assert niches.niche_id.isna().all()
    assert niches.niche_status.tolist() == ["isolated_no_neighborhood", "isolated_no_neighborhood", "outside_tissue_support"]


def test_zero_distance_cells_are_neighbors():
    frame = cells([(2.5, 2.5), (2.5, 2.5)])
    support = {"sample": SupportMask(np.ones((6, 6)))}
    profiles, _, graphs, _ = analyze_cells(frame, support, radii_um=(1,))
    assert graphs[1].nnz == 2
    assert graphs[1].data.tolist() == [0, 0]
    assert profiles.r1um_neighbor_count.tolist() == [1, 1]


def test_domain_distance_and_feature_blocks_are_joined_by_identifiers():
    frame = cells([(2.5, 2.5), (3.5, 2.5), (7.5, 2.5)])
    support = {"sample": SupportMask(np.ones((10, 10)), mpp=(1, 1))}
    domains = np.ones((10, 10), int)
    domains[:, 5:] = 2
    block = pd.DataFrame({"sample_id": ["sample", "sample"], "cell_id": ["c0001", "c0000"], "marker": [6, 2]})
    profiles, _, _, _ = analyze_cells(frame, support, radii_um=(2,), feature_blocks={"markers": block}, domain_masks={"sample": domains})
    assert profiles.loc[0, "r2um_markers_mean:marker"] == 6
    assert profiles.loc[1, "r2um_markers_mean:marker"] == 2
    assert np.isnan(profiles.loc[2, "r2um_markers_mean:marker"])
    np.testing.assert_array_equal(profiles["own_markers:marker"], [2, 6, np.nan])
    assert profiles.own_markers_observed_fraction.tolist() == [1, 1, 0]
    np.testing.assert_allclose(profiles.domain_boundary_distance_um, [2, 1, 2])


def test_own_cell_features_distinguish_focal_cells_with_identical_neighbors_and_have_separate_weights():
    frame = cells([(9., 10.), (11., 10.), (10., 10.)], phenotypes=["same"] * 3)
    block = frame[["sample_id", "cell_id"]].assign(value=[0., 100., 5.])
    before = block.copy(deep=True)
    profiles, _, graphs, summary = analyze_cells(frame, {"sample": SupportMask(np.ones((40, 40)))},
        radii_um=(1.1,), feature_blocks={"morphology": block}, repeats=2)
    assert profiles["r1.1um_morphology_mean:value"].tolist() == [5., 5., 50.]
    assert profiles["own_morphology:value"].tolist() == [0., 100., 5.]
    assert summary["feature_groups"]["morphology"] == ["r1.1um_morphology_mean:value"]
    assert summary["feature_groups"]["own:morphology"] == ["own_morphology:value"]
    eligible = np.ones(len(profiles), bool)
    matrix, scaling = _niche_matrix(profiles, summary["feature_groups"], {}, eligible)
    weighted, weighted_scaling = _niche_matrix(profiles, summary["feature_groups"], {"own:morphology": 4}, eligible)
    assert not np.array_equal(matrix[0], matrix[1]), "Own features must enter the actual niche representation"
    offset = 0
    for name, definition in scaling.items():
        columns = slice(offset, offset + definition["dimensions"])
        np.testing.assert_array_equal(weighted[:, columns], matrix[:, columns] * (2 if name == "own:morphology" else 1))
        offset += definition["dimensions"]
    assert weighted_scaling["own:morphology"]["weight"] == 4
    assert weighted_scaling["morphology"]["weight"] == 1
    assert graphs[1.1].nnz == 4 and set(graphs[1.1][0].indices) == {2}
    pd.testing.assert_frame_equal(block, before)


def test_missing_indicators_survive_constant_observed_values_without_changing_raw_nans():
    profiles = pd.DataFrame({"sample_id": ["s"] * 4, "cell_id": list("abcd"),
        "in_tissue_support": True, "r1um_neighbor_count": 1, "constant_when_observed": [5., np.nan, 5., np.nan],
        "entirely_absent": np.nan, "constant_complete": 3.})
    original = profiles.copy(deep=True)
    groups = {"markers": ["constant_when_observed", "entirely_absent", "constant_complete"]}
    matrix, scaling = _niche_matrix(profiles, groups, {}, np.ones(4, bool))
    np.testing.assert_array_equal(matrix, [[0.], [1.], [0.], [1.]])
    assert scaling["markers"]["columns"] == []
    assert scaling["markers"]["means"] == scaling["markers"]["std"] == []
    assert scaling["markers"]["missing_indicator_columns"] == ["constant_when_observed"]
    assert scaling["markers"]["matrix_columns"] == [{"kind": "missing_indicator", "source_column": "constant_when_observed"}]
    niches, metadata = discover_niches(profiles, groups, fixed_k=2, repeats=2)
    assert niches.niche_id.nunique() == 2
    assert metadata["feature_representation_version"] == FEATURE_REPRESENTATION_VERSION == "2.0.0"
    assert metadata["confidence_is_calibrated_probability"] is False
    assert "not evidence of biological differences" in metadata["missingness_interpretation"]
    pd.testing.assert_frame_equal(profiles, original)


def test_missingness_only_outside_eligible_cells_does_not_create_training_dimensions():
    profiles = pd.DataFrame({"f": [5., 5., np.nan]})
    matrix, scaling = _niche_matrix(profiles, {"markers": ["f"]}, {}, np.array([True, True, False]))
    assert matrix.shape == (3, 0) and scaling == {}


def test_missing_modality_in_one_specimen_preserves_cells_coverage_and_graph_isolation():
    frame = cells([(10., 10.), (11., 10.)] * 2, ["A", "A", "B", "B"], ["same"] * 4)
    block = frame.loc[frame.sample_id.eq("A"), ["sample_id", "cell_id"]].assign(value=5.)
    supports = {name: SupportMask(np.ones((30, 30))) for name in ("A", "B")}
    profiles, niches, graphs, summary = analyze_cells(frame, supports, radii_um=(1.1,), feature_blocks={"marker": block}, repeats=2)
    pd.testing.assert_frame_equal(profiles[["sample_id", "cell_id"]], frame[["sample_id", "cell_id"]])
    assert len(niches) == 4
    np.testing.assert_array_equal(profiles["own_marker:value"], [5, 5, np.nan, np.nan])
    np.testing.assert_array_equal(profiles["r1.1um_marker_mean:value"], [5, 5, np.nan, np.nan])
    assert profiles.own_marker_observed_fraction.tolist() == [1, 1, 0, 0]
    assert profiles["r1.1um_marker_observed_fraction"].tolist() == [1, 1, 0, 0]
    assert summary["niche_discovery"]["scaling"]["marker"]["missing_indicator_columns"] == ["r1.1um_marker_mean:value"]
    assert summary["niche_discovery"]["scaling"]["own:marker"]["missing_indicator_columns"] == ["own_marker:value"]
    assert graphs[1.1].nnz == 4
    assert graphs[1.1][:2, 2:].nnz == graphs[1.1][2:, :2].nnz == 0


def test_group_definition_metadata_is_exact_and_can_refit_pooled_profiles():
    frame = cells([(9., 10.), (11., 10.), (10., 10.)] * 2, ["A"] * 3 + ["B"] * 3, ["same"] * 6)
    block = frame[["sample_id", "cell_id"]].assign(**{"feature with space": [0., 100., 5.] * 2})
    support = {name: SupportMask(np.ones((40, 40))) for name in ("A", "B")}
    profiles, niches, graphs, summary = analyze_cells(frame, support, radii_um=(1.1, 1.5),
        feature_blocks={"morphology": block.sample(frac=1, random_state=9)}, fixed_k=2, repeats=2)
    definitions = summary["feature_group_definitions"]
    assert summary["feature_representation_version"] == "2.0.0"
    assert set(definitions) == set(summary["feature_groups"])
    for name, definition in definitions.items():
        assert definition["columns"] == summary["feature_groups"][name]
        assert set(definition["columns"]) <= set(profiles)
    assert definitions["own:morphology"] == {"role": "own_cell_feature", "aggregation": "identity_no_imputation",
        "source_feature_block": "morphology", "source_features": ["feature with space"],
        "columns": ["own_morphology:feature%20with%20space"], "radii_um": [], "focal_cell_included": True,
        "coverage_columns": ["own_morphology_observed_fraction"]}
    assert definitions["morphology"]["radii_um"] == [1.1, 1.5]
    assert definitions["morphology"]["source_feature_block"] == "morphology"
    assert definitions["morphology"]["focal_cell_included"] is False
    assert definitions["composition"]["role"] == "neighbor_phenotype_composition"
    assert definitions["composition"]["absent_category"] == "zero_only_for_nonempty_neighborhood; otherwise_missing"
    refit, metadata = discover_niches(profiles, summary["feature_groups"], fixed_k=2, repeats=2)
    pd.testing.assert_frame_equal(niches, refit)
    assert metadata == summary["niche_discovery"]
    assert niches.niche_id.iloc[:3].tolist() == niches.niche_id.iloc[3:].tolist()
    for graph in graphs.values():
        assert graph[:3, 3:].nnz == graph[3:, :3].nnz == 0


def test_own_features_do_not_assign_isolated_or_unsupported_cells():
    frame = cells([(2., 2.), (8., 8.), (-1., -1.)])
    block = frame[["sample_id", "cell_id"]].assign(value=[1., np.nan, 3.])
    profiles, niches, _, summary = analyze_cells(frame, {"sample": SupportMask(np.ones((10, 10)))},
        radii_um=(1,), feature_blocks={"cellvit": block}, repeats=2)
    np.testing.assert_array_equal(profiles["own_cellvit:value"], [1., np.nan, 3.])
    assert len(niches) == 3 and niches.niche_id.isna().all()
    assert summary["niche_discovery"]["selected_k"] == 0
    assert summary["niche_discovery"]["feature_representation_version"] == "2.0.0"
    assert summary["feature_group_definitions"]["own:cellvit"]["columns"] == ["own_cellvit:value"]


def test_duplicate_keys_and_unmatched_feature_rows_fail():
    frame = cells([(1.5, 1.5), (2.5, 1.5)])
    frame.loc[1, "cell_id"] = frame.loc[0, "cell_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_cells(frame)
    frame.loc[1, "cell_id"] = "other"
    block = pd.DataFrame({"sample_id": ["absent"], "cell_id": ["c0000"], "f": [1]})
    with pytest.raises(ValueError, match="absent from canonical"):
        analyze_cells(frame, {"sample": SupportMask(np.ones((5, 5)))}, radii_um=(2,), feature_blocks={"markers": block})


def test_niches_recover_context_groups_and_are_permutation_equivariant():
    xy, phenotypes = [], []
    for offset, phenotype in [(0, "A"), (40, "B")]:
        for y in range(5):
            for x in range(5):
                xy.append((offset + 5.5 + x, 5.5 + y))
                phenotypes.append(phenotype)
    frame = cells(xy, phenotypes=phenotypes)
    support = {"sample": SupportMask(np.ones((20, 60)))}
    first = analyze_cells(frame, support, radii_um=(10, 15), seed=7)
    second = analyze_cells(frame.sample(frac=1, random_state=22), support, radii_um=(10, 15), seed=7)
    assert first[3]["niche_discovery"]["selected_k"] == 2
    assert adjusted_rand_score(phenotypes, first[1].niche_id.astype(int)) == 1
    assert first[1].niche_stability.eq(1).all()
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[3] == second[3]


def test_fixed_k_validation_and_noncalibrated_evidence():
    frame = pd.DataFrame({"sample_id": ["s"] * 12, "cell_id": list(map(str, range(12))), "in_tissue_support": True, "r1um_neighbor_count": 2, "f": [-10.] * 6 + [10.] * 6})
    niches, summary = discover_niches(frame, {"test": ["f"]}, fixed_k=2, repeats=2)
    assert summary["selected_k"] == 2
    assert summary["confidence_is_calibrated_probability"] is False
    assert niches.niche_status.eq("assigned_fixed_k").all()
    with pytest.raises(ValueError, match="distinct feature"):
        discover_niches(frame, {"test": ["f"]}, fixed_k=3, repeats=2)
    with pytest.raises(ValueError, match="Unknown feature weight"):
        discover_niches(frame, {"test": ["f"]}, feature_weights={"missing": 3})


def test_cli_writes_sparse_graph_provenance_and_multi_specimen_manifest(tmp_path):
    frame = cells([(2.5, 2.5), (3.5, 2.5), (2.5, 2.5), (3.5, 2.5)], ["A", "A", "B", "B"])
    frame.to_csv(tmp_path / "cells.csv", index=False)
    frame[["sample_id", "cell_id"]].assign(marker=[2., 4., 2., 4.]).to_csv(tmp_path / "features.csv", index=False)
    tifffile.imwrite(tmp_path / "mask.tif", np.ones((6, 6), np.uint8))
    (tmp_path / "supports.json").write_text(json.dumps({s: {"path": "mask.tif", "mpp": [1, 1]} for s in ["A", "B"]}))
    subprocess.run([sys.executable, str(ROOT / "bin/analyze_cell_neighborhoods.py"), "--cells", str(tmp_path / "cells.csv"),
        "--support-manifest", str(tmp_path / "supports.json"), "--radii-um", "2",
        "--feature-block", f"markers={tmp_path / 'features.csv'}", "--feature-weight", "own:markers=4",
        "--fixed-k", "2", "--stability-repeats", "2", "--outdir", str(tmp_path / "output")], check=True)
    output = tmp_path / "output"
    graph = sparse.load_npz(output / "neighborhood_graph_2um.npz")
    assert graph.nnz == 4 and graph[:2, 2:].nnz == 0
    summary = json.loads((output / "neighborhood_summary.json").read_text())
    assert summary["cell_count"] == 4
    assert len(summary["inputs"]["cells"]["sha256"]) == 64
    assert summary["software"]["script_sha256"]
    assert summary["feature_representation_version"] == "2.0.0"
    assert summary["niche_discovery"]["feature_representation_version"] == "2.0.0"
    assert summary["niche_discovery"]["scaling"]["own:markers"]["weight"] == 4
    assert summary["niche_discovery"]["scaling"]["markers"]["weight"] == 1
    assert summary["feature_group_definitions"]["own:markers"]["source_feature_block"] == "markers"
    actual = pd.read_csv(output / "neighborhood_profiles.csv")
    assert actual["own_markers:marker"].tolist() == [2., 4., 2., 4.]
    assert actual["r2um_markers_mean:marker"].tolist() == [4., 2., 4., 2.]
    assert len(pd.read_csv(output / "cellular_niches.csv")) == 4


def test_cli_refuses_implicit_multi_specimen_mask(tmp_path):
    frame = cells([(1.5, 1.5), (2.5, 2.5)], ["A", "B"])
    frame.to_csv(tmp_path / "cells.csv", index=False)
    tifffile.imwrite(tmp_path / "mask.tif", np.ones((5, 5), np.uint8))
    result = subprocess.run([sys.executable, str(ROOT / "bin/analyze_cell_neighborhoods.py"), "--cells", str(tmp_path / "cells.csv"), "--support-mask", str(tmp_path / "mask.tif"), "--support-mpp", "1", "--outdir", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "exactly one specimen" in result.stderr


def test_bounded_raster_graph_and_density_match_native_array_not_sampled_gap(tmp_path):
    native = np.ones((32, 64), np.uint8)
    native[:, 32] = 0  # erased by centre sampling at x=2,6,...,62
    native[4:7, 7:11] = 0
    path = tmp_path / "native.tif"
    tifffile.imwrite(path, native, tile=(16, 16), compression="deflate")
    mpp, origin = np.array([.25, .75]), np.array([13.25, 24.5])
    pixels = np.array([[30.5, 18.5], [33.5, 18.5], [28.5, 18.5], [32.5, 18.5], [-1., -1.]])
    frame = cells(pixels * mpp + origin)
    expected = SupportMask(native, tuple(mpp), tuple(origin))
    graph, valid = build_radius_graph(frame, {"sample": expected}, 2.)
    with RasterSupportMask(path, tuple(mpp), tuple(origin), tile_size=8, cache_tiles=2) as support:
        coarse = support.grid_tissue((8, 16))
        assert coarse[:, 8].all(), "The coarse representation must actually erase the test gap"
        actual, supported = build_radius_graph(frame, {"sample": support}, 2.)
        np.testing.assert_array_equal(actual.indptr, graph.indptr)
        np.testing.assert_array_equal(actual.indices, graph.indices)
        np.testing.assert_array_equal(actual.data, graph.data)
        np.testing.assert_array_equal(supported, valid)
        assert supported.tolist() == [True, True, True, False, False]
        assert 1 not in actual[0].indices and 2 in actual[0].indices
        for point in frame[["x_um", "y_um"]].to_numpy():
            for radius in (.5, 2., 50.):
                assert support.tissue_area_in_disk(point, radius) == expected.tissue_area_in_disk(point, radius)
        assert support.max_window_pixels_seen <= 8**2
        assert len(support._cache) <= 2
    assert not support._cache and support.reader.reader._tf is None


def test_bounded_raster_supercover_and_disconnected_specimens(tmp_path):
    image = np.ones((16, 32), np.uint8)
    image[:, 16] = 0
    image[1, 2] = 0
    path = tmp_path / "separate.tif"
    tifffile.imwrite(path, image, tile=(16, 16), compression="deflate")
    frame = cells([(14.5, 8.5), (17.5, 8.5), (13.5, 8.5)] * 2, ["A"] * 3 + ["B"] * 3)
    with RasterSupportMask(path, tile_size=4, cache_tiles=1) as a, RasterSupportMask(path, tile_size=4, cache_tiles=1) as b:
        graph, supported = build_radius_graph(frame, {"A": a, "B": b}, 8.)
        assert supported.all() and graph.nnz == 4
        assert set(graph[0].indices) == {2} and set(graph[3].indices) == {5}
        assert graph[:3, 3:].nnz == graph[3:, :3].nnz == 0
        assert not segment_in_support(np.array([1.5, 1.5]), np.array([2.5, 2.5]), a)
        assert not segment_in_support(np.array([2.5, 2.5]), np.array([1.5, 1.5]), a)
        assert not segment_in_support(np.array([1.5, 2.]), np.array([3.5, 2.]), a)
        assert segment_in_support(np.array([1.5, 3.5]), np.array([3.5, 3.5]), a)


def test_raster_view_geometry_values_and_analysis_grid_limits_fail_closed(tmp_path):
    path = tmp_path / "support.tif"
    image = np.ones((16, 16), np.float32)
    image[3, 3] = np.nan
    tifffile.imwrite(path, image, tile=(16, 16), compression="deflate")
    with RasterSupportMask(path, tile_size=4) as support:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            support.component_at(np.array([3.5, 3.5]))
    for view in ((.5, 0, 4, 4), (0, 0, 20, 4), (-1, 0, 4, 4)):
        with pytest.raises(ValueError, match="pixel-edge|within"):
            RasterSupportMask(path, window_xyxy=view)
    for limit in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            bounded_grid_shape((100, 100), limit)
    for shape in ((1, 1000000), (1000000, 1), (999, 1100)):
        reduced = bounded_grid_shape(shape, 13)
        assert min(reduced) > 0 and np.prod(reduced) <= 13
