"""Decision/CSR equivalence to the pre-scalar NumPy supercover implementation."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import analyze_cell_neighborhoods as neighborhoods


def legacy_segment_in_support(start, end, support):
    """Frozen production NumPy implementation before the 2026-09-05 scalar change."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if not np.all(np.isfinite([a, b])):
        return False
    cell = np.floor(a).astype(int)
    last = np.floor(b).astype(int)
    h, w = support.shape

    def inside(p):
        return support.is_tissue(int(p[0]), int(p[1]))

    if not inside(cell) or not inside(last):
        return False
    delta = b - a
    steps = np.sign(delta).astype(int)
    box_lo, box_hi = np.minimum(cell, last), np.maximum(cell, last)
    for axis in range(2):
        if delta[axis] == 0 and abs(a[axis] - round(a[axis])) < 1e-12:
            box_lo[axis] -= 1
    if np.all(box_lo >= 0) and box_hi[0] < w and box_hi[1] < h:
        if support.all_tissue_in_box(int(box_lo[0]), int(box_lo[1]), int(box_hi[0])+1, int(box_hi[1])+1):
            return True
    t_delta = np.full(2, np.inf)
    t_max = np.full(2, np.inf)
    moving = delta != 0
    t_delta[moving] = 1.0 / np.abs(delta[moving])
    next_edge = cell + (steps > 0)
    t_max[moving] = (next_edge[moving] - a[moving]) / delta[moving]
    boundary_axes = [i for i in range(2) if not moving[i] and abs(a[i] - round(a[i])) < 1e-12]

    def covered(p):
        if not inside(p):
            return False
        for axis in boundary_axes:
            other = p.copy()
            other[axis] -= 1
            if not inside(other):
                return False
        return True

    if not covered(cell):
        return False
    while not np.array_equal(cell, last):
        if np.isclose(t_max[0], t_max[1], rtol=0, atol=1e-12):
            side_x, side_y = cell.copy(), cell.copy()
            side_x[0] += steps[0]
            side_y[1] += steps[1]
            if not covered(side_x) or not covered(side_y):
                return False
            cell += steps
            t_max += t_delta
        else:
            axis = int(np.argmin(t_max))
            cell[axis] += steps[axis]
            t_max[axis] += t_delta[axis]
        if not covered(cell):
            return False
    return True


def lines_for_equivalence():
    rng = np.random.default_rng(5319)
    lines = list(rng.uniform((-2, -2), (82, 98), size=(5000, 2, 2)))
    # Floating, exact integer, half-integer, near-gridline, zero-length, and
    # negative-direction lines; each case is also checked in reverse.
    for _ in range(800):
        a, b = rng.integers((0, 0), (80, 96), size=(2, 2)).astype(float)
        lines.extend(((a, b), (a+.5, b+.5), (a, a)))
        axis = int(rng.integers(2))
        for epsilon in (0., -5e-13, 5e-13, -2e-12, 2e-12):
            start, end = a.copy(), b.copy()
            start[axis] += epsilon
            end[axis] = start[axis]
            lines.append((start, end))
    lines.extend(((a, b) for a, b in (
        ((0., 0.), (0., 0.)), ((79., 95.), (79., 95.)),
        ((0., 20.5), (79., 20.5)), ((20.5, 0.), (20.5, 95.)),
        ((1.5, 1.5), (78.5, 78.5)), ((78.5, 1.5), (1.5, 78.5)),
        ((np.nan, 2.), (3., 4.)), ((2., 3.), (np.inf, 4.)),
        ((-np.inf, 2.), (3., 4.)), ((2., 3.), (3., np.nan)),
    )))
    return lines


@pytest.mark.parametrize("backend", ["array", "raster"])
def test_scalar_decisions_equal_legacy_on_random_boundaries_corners_and_reverse(tmp_path, backend):
    rng = np.random.default_rng(79)
    mask = rng.random((96, 80)) > .035
    mask[:, 40] = False
    mask[15:80, 40] = True
    mask[45:49, 20:25] = False
    mask[0, 0] = mask[95, 79] = True
    with ExitStack() as stack:
        if backend == "array":
            support = neighborhoods.SupportMask(mask)
        else:
            path = tmp_path / "support.tif"
            tifffile.imwrite(path, mask.astype(np.uint8), tile=(16, 16), compression="deflate")
            support = stack.enter_context(neighborhoods.RasterSupportMask(path, tile_size=16, cache_tiles=3))
        for start, end in lines_for_equivalence():
            for a, b in ((start, end), (end, start)):
                assert neighborhoods.segment_in_support(a, b, support) == legacy_segment_in_support(a, b, support), (a, b)


@pytest.mark.parametrize("backend", ["array", "raster"])
def test_scalar_retains_absolute_corner_and_gridline_tolerances(tmp_path, backend):
    mask = np.ones((5, 5), np.uint8)
    mask[1, 2] = 0
    cases = [
        ((1.5, 1.5), (2.5, 2.5+1e-13), False),
        ((1.5, 1.5), (2.5, 2.5+1e-10), True),
        ((1.5, 2.+5e-13), (3.5, 2.+5e-13), False),
        ((1.5, 2.+2e-12), (3.5, 2.+2e-12), True),
    ]
    with ExitStack() as stack:
        if backend == "array":
            support = neighborhoods.SupportMask(mask)
        else:
            path = tmp_path / "tolerance.tif"
            tifffile.imwrite(path, mask, tile=(16, 16), compression="deflate")
            support = stack.enter_context(neighborhoods.RasterSupportMask(path, tile_size=2, cache_tiles=2))
        for start, end, expected in cases:
            for a, b in ((start, end), (end, start)):
                assert legacy_segment_in_support(a, b, support) == expected
                assert neighborhoods.segment_in_support(a, b, support) == expected


@pytest.mark.parametrize("backend", ["array", "raster"])
def test_scalar_preserves_complete_csr_and_row_identity(tmp_path, monkeypatch, backend):
    rng = np.random.default_rng(3253)
    mask = np.ones((96, 80), np.uint8)
    mask[15:80, 40] = 0
    mask[40:44, 5:20] = 0
    points = rng.uniform((1, 1), (78, 94), size=(160, 2))
    points[:6] = [[2., 3.], [2., 3.], [20.5, 40.], [20.5, 42.], [40., 20.], [40., 20.]]
    points[2], points[3] = points[0], points[1]  # Coincident cells within each specimen.
    cells = pd.DataFrame(points, columns=["x_um", "y_um"])
    cells["sample_id"] = np.where(np.arange(len(cells)) % 2, "sample_B", "sample_A")
    cells["cell_id"] = [f"cell_{i}" for i in range(len(cells))]
    cells = neighborhoods.validate_cells(cells)
    before = cells.copy(deep=True)
    with ExitStack() as stack:
        if backend == "array":
            supports = {sample: neighborhoods.SupportMask(mask) for sample in ("sample_A", "sample_B")}
        else:
            path = tmp_path / "support.tif"
            tifffile.imwrite(path, mask, tile=(16, 16), compression="deflate")
            supports = {sample: stack.enter_context(neighborhoods.RasterSupportMask(path, tile_size=16, cache_tiles=3))
                        for sample in ("sample_A", "sample_B")}
        actual, supported = neighborhoods.build_radius_graph(cells, supports, 24.)
        with monkeypatch.context() as patch:
            patch.setattr(neighborhoods, "segment_in_support", legacy_segment_in_support)
            expected, old_supported = neighborhoods.build_radius_graph(cells, supports, 24.)
        np.testing.assert_array_equal(supported, old_supported)
        for name in ("indices", "indptr", "data"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        assert np.count_nonzero(actual.data == 0.) >= 4
        pd.testing.assert_frame_equal(cells, before)


@pytest.mark.parametrize("backend", ["array", "raster"])
def test_scalar_microbenchmark_has_exact_decision_parity(tmp_path, backend):
    """Report isolated long-line timings only; no flaky speed threshold assertion."""
    mask = np.ones((256, 256), np.uint8)
    mask[80, 128] = 0  # Forces rectangle shortcut failure, but misses every line.
    lines = [(np.array([2.5, 12.5+offset]), np.array([250.5, 235.5+offset]))
             for offset in np.linspace(-8, 8, 64)]
    if backend == "raster":
        path = tmp_path / "benchmark_support.tif"
        tifffile.imwrite(path, mask, tile=(32, 32), compression="deflate")
    times = {"legacy": [], "scalar": []}
    decisions = []
    for repeat in range(3):
        variants = (("legacy", legacy_segment_in_support), ("scalar", neighborhoods.segment_in_support))
        if repeat % 2:
            variants = variants[::-1]
        for name, function in variants:
            with ExitStack() as stack:
                support = neighborhoods.SupportMask(mask) if backend == "array" else stack.enter_context(
                    neighborhoods.RasterSupportMask(path, tile_size=32, cache_tiles=16))
                start = time.perf_counter()
                result = [function(a, b, support) for a, b in lines]
                times[name].append(time.perf_counter() - start)
                decisions.append(result)
    assert all(result == decisions[0] for result in decisions)
    assert all(decisions[0])
    print(json.dumps({"benchmark": "isolated_long_line_supercover_warm_filesystem",
        "backend": backend, "line_count": len(lines), "repeats": 3,
        "exact_decision_parity": True, "seconds": times,
        "median_speed_ratio": float(np.median(times["legacy"]) / np.median(times["scalar"])),
        "scope": "synthetic fallback traversal only; not whole graph or pipeline runtime"}))
