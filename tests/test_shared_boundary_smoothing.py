import importlib.util
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union


SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "shared_boundary_smoothing.py"
SPEC = importlib.util.spec_from_file_location("shared_boundary_smoothing", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def zigzag_coverage():
    seam = [(5, 0), (4, 2), (6, 4), (4, 6), (6, 8), (5, 10)]
    left = Polygon([(0, 0), *seam, (0, 10), (0, 0)])
    right = Polygon([(10, 0), (10, 10), (5, 10), *reversed(seam[:-1]), (10, 0)])
    return [(1, left), (2, right)]


def test_shared_smoothing_preserves_coverage_and_reduces_seam_turns():
    source = zigzag_coverage()
    before = unary_union([geometry for _, geometry in source])
    result, metadata = MODULE.smooth_shared_boundary_coverage(
        source, smoothing_coefficient=0.5, smoothing_passes=1,
    )
    after = unary_union([geometry for _, geometry in result])

    assert metadata["applied"] is True
    assert metadata["outer_tissue_boundary_unchanged"] is True
    assert metadata["maximum_vertex_displacement_px"] > 0
    assert before.symmetric_difference(after).area < 1e-9
    assert result[0][1].intersection(result[1][1]).area < 1e-9
    original_shared = MODULE.linemerge(source[0][1].boundary.intersection(source[1][1].boundary))
    smoothed_shared = MODULE.linemerge(result[0][1].boundary.intersection(result[1][1].boundary))
    original_y = [point[1] for point in original_shared.coords]
    smoothed_x = [point[0] for point in smoothed_shared.coords]
    assert len(original_y) == len(smoothed_x)
    assert max(smoothed_x) - min(smoothed_x) < 2


def test_zero_strength_is_exact_noop():
    source = zigzag_coverage()
    result, metadata = MODULE.smooth_shared_boundary_coverage(
        source, smoothing_coefficient=0.0, smoothing_passes=1,
    )
    assert metadata["applied"] is False
    assert all(left.equals_exact(right, 0) for (_, left), (_, right) in zip(source, result))


def test_selective_smoothing_leaves_shallow_turns_unchanged():
    shallow = MODULE.LineString([(0, 0), (5, 0.1), (10, 0)])
    unchanged, displacement = MODULE.smooth_line(
        shallow,
        0.5,
        minimum_turn_degrees=10,
        minimum_adjacent_length=1,
    )
    assert displacement == 0
    assert unchanged.equals_exact(shallow, 0)

    sharp = MODULE.LineString([(0, 0), (5, 5), (10, 0)])
    changed, displacement = MODULE.smooth_line(
        sharp,
        0.5,
        minimum_turn_degrees=45,
        minimum_adjacent_length=1,
    )
    assert displacement > 0
    assert not changed.equals_exact(sharp, 0)
