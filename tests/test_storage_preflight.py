import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import storage_preflight as storage  # noqa: E402


def test_nextflow_wrapper_treats_yaml_boolean_false_as_off() -> None:
    source = (ROOT / "lib" / "StoragePreflight.groovy").read_text(encoding="utf-8")
    assert "rawMode instanceof Boolean && !rawMode ? 'off'" in source


def test_roi_bbox_fraction_is_clipped_and_padded(tmp_path: Path) -> None:
    roi = tmp_path / "sample.geojson"
    roi.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-10, 10], [50, 10], [50, 60], [-10, 60], [-10, 10]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # Clipped box is 50 x 50 of a 100 x 100 image, then 15% crop allowance.
    assert storage.roi_bbox_fraction(roi, 100, 100) == 0.2875


def test_route_factors_remove_obsolete_growth_and_scale_both_routes() -> None:
    factors = storage.adjusted_stage_factors(
        storage.active_stages("uni2", "cluster_geojson"),
        cell_detection_mode="consensus",
        uni2_sampling_mode="both",
        gigatime_enabled=True,
        marker_quantification_enabled=True,
        gigatime_channel_count=5,
        uni2_save_tiles=False,
    )
    assert "grow_tissue" not in factors
    assert factors["uni2"][0] == storage.STAGE_FACTORS["uni2"][0] * 2
    assert factors["cluster_geojson"][1] == storage.STAGE_FACTORS["cluster_geojson"][1] * 2


def test_gigatime_channels_and_saved_tiles_raise_capacity_estimate() -> None:
    base = storage.adjusted_stage_factors(
        ["gigatime", "uni2"],
        cell_detection_mode="consensus",
        uni2_sampling_mode="cells",
        gigatime_enabled=True,
        marker_quantification_enabled=True,
        gigatime_channel_count=5,
        uni2_save_tiles=False,
    )
    larger = storage.adjusted_stage_factors(
        ["gigatime", "uni2"],
        cell_detection_mode="consensus",
        uni2_sampling_mode="cells",
        gigatime_enabled=True,
        marker_quantification_enabled=True,
        gigatime_channel_count=10,
        uni2_save_tiles=True,
    )
    assert larger["gigatime"][0] == base["gigatime"][0] * 2
    assert larger["uni2"][0] > base["uni2"][0]


def test_capacity_has_distinct_expected_and_restart_states() -> None:
    gib = storage.GIB
    assert storage.assess_capacity(100 * gib, 30 * gib, 60 * gib, 20 * gib) == "pass"
    assert storage.assess_capacity(70 * gib, 30 * gib, 60 * gib, 20 * gib) == "warning"
    assert storage.assess_capacity(45 * gib, 30 * gib, 60 * gib, 20 * gib) == "fail"


def test_report_groups_output_work_and_cache_on_one_filesystem(
    tmp_path: Path, monkeypatch,
) -> None:
    image = tmp_path / "sample.tif"
    image.write_bytes(b"x" * 1024)
    outdir = tmp_path / "results"
    workdir = tmp_path / "work"
    cache = tmp_path / "cache"
    outdir.mkdir()
    workdir.mkdir()
    monkeypatch.setattr(storage, "probe_dimensions", lambda _: (1000, 500, "test"))
    monkeypatch.setattr(
        storage,
        "filesystem_info",
        lambda path: {
            "device": "same",
            "anchor": str(tmp_path),
            "available_bytes": 500 * storage.GIB,
            "capacity_bytes": 1000 * storage.GIB,
        },
    )
    args = storage.parser().parse_args(
        [
            "--input", str(image),
            "--outdir", str(outdir),
            "--workdir", str(workdir),
            "--output-json", str(outdir / "00_execution" / "storage_preflight.json"),
            "--start-point", "convert",
            "--end-point", "uni2",
            "--cache", f"hf={cache}",
            "--min-free-gib", "1",
        ]
    )
    report = storage.build_report(args)
    assert report["status"] == "pass"
    assert report["inputs"][0]["active_rgb_bytes"] == 1_500_000
    assert len(report["filesystems"]) == 1
    kinds = {row["kind"] for row in report["filesystems"][0]["paths"]}
    assert kinds == {"output", "work", "cache"}
    assert report["cache_inventory"][0]["label"] == "hf"
    assert report["filesystems"][0]["worst_case_increment_bytes"] > report["filesystems"][0]["expected_increment_bytes"]


def test_rellink_reduces_output_duplication(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "sample.tif"
    image.write_bytes(b"x")
    outdir = tmp_path / "results"
    workdir = tmp_path / "work"
    outdir.mkdir()
    workdir.mkdir()
    monkeypatch.setattr(storage, "probe_dimensions", lambda _: (10_000, 10_000, "test"))

    def build(mode: str) -> dict:
        args = storage.parser().parse_args(
            [
                "--input", str(image), "--outdir", str(outdir), "--workdir", str(workdir),
                "--output-json", str(outdir / "preflight.json"), "--start-point", "convert",
                "--end-point", "cluster_geojson", "--publish-dir-mode", mode, "--min-free-gib", "0",
            ]
        )
        return storage.build_report(args)

    copied = next(row for row in build("copy")["demands"] if row["kind"] == "output")
    linked = next(row for row in build("rellink")["demands"] if row["kind"] == "output")
    assert copied["expected_bytes"] > linked["expected_bytes"]
