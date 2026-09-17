"""Small real morphology producer/consumer contract tests; no learned model."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import profile_cell_morphology as morphology
import cell_morphology_io as io
import build_cell_profiles as profiles


def inputs(root, *, dtype="uint8", mpp=.25):
    root.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((16, 16))
    image = np.repeat(np.where((xx+yy) % 2, 32, 224).astype(np.uint8)[..., None], 3, axis=2)
    if dtype == "uint16":
        image = image.astype(np.uint16) * 257
    elif dtype == "float01":
        image = image.astype(np.float64) / 255
    tifffile.imwrite(root / "image.tif", image, photometric="rgb")
    labels = np.zeros((16, 16), np.uint32)
    labels[1:6, 1:6], labels[8:12, 8:12], labels[14, 14] = 7, 42, 99
    tifffile.imwrite(root / "labels.tif", labels)
    pd.DataFrame({"label": [42, 7, 99], "x": [9.5, 3., 14.5], "y": [9.5, 3., 14.5],
        "xmin": [8, 1, 14], "ymin": [8, 1, 14], "xmax": [12, 6, 15], "ymax": [12, 6, 15]}).to_csv(root / "objects.csv", index=False)
    (root / "shift.json").write_text(json.dumps({"source_mpp": mpp,
        "offset_crop_to_original": {"dx": 10, "dy": 20}, "crop_size": {"width": 16, "height": 16}}))
    (root / "resolution.json").write_text(json.dumps({"status": "pass", "mpp_x": mpp, "mpp_y": mpp,
        "width_px": 100, "height_px": 100}))
    args = profiles.parser().parse_args(["--objects", str(root / "objects.csv"), "--sample-id", root.name,
        "--image", str(root / "image.tif"), "--labels", str(root / "labels.tif"), "--shift", str(root / "shift.json"),
        "--resolution-json", str(root / "resolution.json"), "--outdir", str(root / "profiles")])
    return SimpleNamespace(root=root, args=args, white_level=1. if dtype == "float01" else None,
                           morphology=root / "morphology", image=image)


def produce(root, *, dtype="uint8", mpp=.25, lag=.5, white=None):
    fixture = inputs(root, dtype=dtype, mpp=mpp)
    summary = morphology.profile_cells(fixture.args.labels, fixture.args.image, fixture.args.objects,
        fixture.args.shift, fixture.morphology, resolution_json=fixture.args.resolution_json,
        white_level=fixture.white_level if white is None else white, texture_lag_um=lag, tile_size=8)
    fixture.summary = summary
    fixture.receipt = json.loads((fixture.morphology / io.COMPLETION).read_text())
    fixture.args.morphology = str(fixture.morphology / io.TABLE)
    return fixture


def load(fixture, table=None):
    return io.load_morphology_table(table or fixture.args.morphology,
        expected_inputs={name: value["sha256"] for name, value in fixture.receipt["sources"].items()},
        expected_geometry=fixture.receipt["geometry"], expected_ids=["42", "7", "99"])


def test_source_bound_producer_preserves_population_nan_and_actual_producer(tmp_path):
    fixture = produce(tmp_path)
    frame, binding, _ = load(fixture)
    assert frame.label.tolist() == ["42", "7", "99"]
    assert frame.texture_status.tolist() == ["ok", "ok", "no_pairs_at_requested_physical_lag"]
    assert np.isnan(frame.od_glcm_contrast.iloc[2])
    assert np.isfinite(frame.rgb_od_mean_mean).all()
    assert binding["reference_compatible"] is True
    assert set(fixture.receipt["sources"]) == {"image", "labels", "objects", "shift", "resolution_json"}
    cells, manifest = profiles.build_profiles(fixture.args)
    assert cells.cell_id.tolist() == ["42", "7", "99"]
    for group in ("morphology", "nuclear_texture"):
        record = manifest["feature_blocks"][group]
        assert record["reference_compatible"] is True
        assert record["feature_definition"] == fixture.receipt["feature_definitions"][group]
        assert record["feature_definition"]["producer"]["files"]["profile_cell_morphology.py"]["sha256"] == io.sha256(Path(morphology.__file__))
    matrix = np.load(Path(fixture.args.outdir) / manifest["feature_blocks"]["nuclear_texture"]["path"])
    assert np.isnan(matrix[2]).any() and np.isfinite(matrix[2]).any()
    assert manifest["feature_blocks"]["nuclear_texture"]["missing_cells"] == 1


def test_proportional_encodings_have_equal_values_and_reference_definitions(tmp_path):
    fixtures = [produce(tmp_path / name, dtype=name) for name in ("uint8", "uint16", "float01")]
    frames = [load(fixture)[0] for fixture in fixtures]
    for other, frame in zip(fixtures[1:], frames[1:]):
        np.testing.assert_allclose(frame[io.TEXTURE_FEATURES].to_numpy(float), frames[0][io.TEXTURE_FEATURES].to_numpy(float), atol=1e-12, equal_nan=True)
        assert other.receipt["feature_definitions"] == fixtures[0].receipt["feature_definitions"]
        assert other.receipt["settings"]["intensity"]["white_level"] != fixtures[0].receipt["settings"]["intensity"]["white_level"]


@pytest.mark.parametrize("variation", ["lag", "mpp", "explicit_white"])
def test_non_equivalent_texture_settings_change_reference_definition(tmp_path, variation):
    reference = produce(tmp_path / "reference")
    kwargs = {"lag": 1.} if variation == "lag" else ({"mpp": .5} if variation == "mpp" else {"white": 256.})
    query = produce(tmp_path / "query", **kwargs)
    for fixture in (reference, query):
        profiles.build_profiles(fixture.args)
    from cell_reference_atlas import load_profile
    frozen = load_profile(reference.args.outdir, ["nuclear_texture"])
    with pytest.raises(ValueError, match="Incompatible feature schema"):
        load_profile(query.args.outdir, ["nuclear_texture"], expected=frozen.schemas)


@pytest.mark.parametrize("source", ["image", "labels", "objects", "shift", "resolution_json"])
def test_foreign_same_id_sources_are_rejected(tmp_path, source):
    fixture = produce(tmp_path)
    path = Path(getattr(fixture.args, source))
    # Same semantics/shape/IDs, different bytes still cannot claim exact source.
    if source in ("image", "labels"):
        with path.open("ab") as handle:
            handle.write(b"different source identity")
    else:
        with path.open("a") as handle:
            handle.write("\n")
    with pytest.raises(ValueError, match="source input hash mismatch"):
        profiles.build_profiles(fixture.args)
    assert not (Path(fixture.args.outdir) / "cell_profiles_manifest.json").exists()


@pytest.mark.parametrize("artifact", [io.TABLE, io.SUMMARY])
def test_payload_hash_change_is_rejected(tmp_path, artifact):
    fixture = produce(tmp_path)
    with (fixture.morphology / artifact).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError):
        load(fixture)


def test_portable_receipt_does_not_open_recorded_upstream_paths_or_current_producer(tmp_path, monkeypatch):
    fixture = produce(tmp_path / "source")
    portable = tmp_path / "portable"
    portable.mkdir()
    for name in (io.TABLE, io.SUMMARY, io.COMPLETION):
        shutil.copy2(fixture.morphology / name, portable / name)
    original = io.identity
    def only_bundle(path):
        assert Path(path).parent == portable
        return original(path)
    monkeypatch.setattr(io, "identity", only_bundle)
    monkeypatch.setattr(io, "producer_identity", lambda: (_ for _ in ()).throw(AssertionError("Do not substitute current producer")))
    _, binding, _ = load(fixture, portable / io.TABLE)
    assert binding["feature_definitions"] == fixture.receipt["feature_definitions"]


def test_legacy_table_is_visible_but_not_reference_compatible(tmp_path):
    fixture = inputs(tmp_path)
    legacy = fixture.root / "legacy.csv"
    pd.DataFrame({"label": [7], "area_um2": [1.], "rgb_od_mean_mean": [.8]}).to_csv(legacy, index=False)
    fixture.args.morphology = str(legacy)
    cells, manifest = profiles.build_profiles(fixture.args)
    assert cells.cell_id.tolist() == ["42", "7", "99"]
    assert cells.morphology__available.tolist() == [False, True, False]
    for group in ("morphology", "nuclear_texture"):
        assert manifest["feature_blocks"][group]["reference_compatible"] is False
        assert "implementation_sha256" not in manifest["feature_blocks"][group]["feature_definition"]


def test_interrupted_new_table_cannot_downgrade_to_legacy(tmp_path, monkeypatch):
    fixture = inputs(tmp_path)
    monkeypatch.setattr(morphology, "complete_morphology", lambda *args, **kwargs: None)
    morphology.profile_cells(fixture.args.labels, fixture.args.image, fixture.args.objects, fixture.args.shift,
        fixture.morphology, resolution_json=fixture.args.resolution_json)
    with pytest.raises(ValueError, match="lacks its completion"):
        io.load_morphology_table(fixture.morphology / io.TABLE, expected_inputs={}, expected_geometry={}, expected_ids=[])


def test_source_mutation_during_measurement_never_completes(tmp_path, monkeypatch):
    fixture = inputs(tmp_path)
    original = morphology.appearance_features
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with Path(fixture.args.image).open("ab") as handle:
            handle.write(b"changed")
        return result
    monkeypatch.setattr(morphology, "appearance_features", mutate)
    with pytest.raises(ValueError, match="source or producer changed"):
        morphology.profile_cells(fixture.args.labels, fixture.args.image, fixture.args.objects, fixture.args.shift,
            fixture.morphology, resolution_json=fixture.args.resolution_json)
    assert not (fixture.morphology / io.COMPLETION).exists()


def test_output_is_never_overwritten(tmp_path):
    fixture = produce(tmp_path)
    before = {path: io.sha256(path) for path in fixture.morphology.iterdir()}
    with pytest.raises(FileExistsError):
        morphology.profile_cells(fixture.args.labels, fixture.args.image, fixture.args.objects,
            fixture.args.shift, fixture.morphology, resolution_json=fixture.args.resolution_json)
    assert all(io.sha256(path) == digest for path, digest in before.items())


@pytest.mark.parametrize("name", [io.TABLE, io.SUMMARY, io.COMPLETION])
def test_escape_is_rejected_before_payload_read(tmp_path, monkeypatch, name):
    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("not a morphology artifact")
    (root / name).symlink_to(outside)
    monkeypatch.setattr(io, "read_table", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Read before containment")))
    with pytest.raises(ValueError, match="escapes"):
        io.load_morphology_table(root / io.TABLE, expected_inputs={}, expected_geometry={}, expected_ids=[])


@pytest.mark.parametrize("artifact", ["table", "image"])
def test_consumer_detects_late_mutation_before_profile_completion(tmp_path, monkeypatch, artifact):
    fixture = produce(tmp_path)
    original = pd.DataFrame.to_parquet
    def mutate(frame, *args, **kwargs):
        result = original(frame, *args, **kwargs)
        path = fixture.args.morphology if artifact == "table" else fixture.args.image
        with Path(path).open("ab") as handle:
            handle.write(b"changed during later profile output")
        return result
    monkeypatch.setattr(pd.DataFrame, "to_parquet", mutate)
    with pytest.raises(ValueError, match="source or producer changed"):
        profiles.build_profiles(fixture.args)
    assert not (Path(fixture.args.outdir) / "cell_profiles_manifest.json").exists()


@pytest.mark.parametrize("mutation", ["settings", "effective_offsets", "producer", "count", "order"])
def test_semantic_receipt_corruption_is_rejected(tmp_path, mutation):
    fixture = produce(tmp_path)
    path = fixture.morphology / io.COMPLETION
    value = json.loads(path.read_text())
    if mutation == "settings":
        value["settings"]["intensity"]["levels"] = 16
    elif mutation == "effective_offsets":
        value["settings"]["texture"]["effective_offsets_um"][0][0] = .75
        import hashlib
        value["settings_sha256"] = hashlib.sha256(io.json_bytes(value["settings"])).hexdigest()
    elif mutation == "producer":
        value["producer"]["files"] = {}
    elif mutation == "count":
        value["cell_count"] = 1
    else:
        with pytest.raises(ValueError, match="population/order"):
            io.load_morphology_table(fixture.args.morphology,
                expected_inputs={name: item["sha256"] for name, item in fixture.receipt["sources"].items()},
                expected_geometry=fixture.receipt["geometry"], expected_ids=["7", "42", "99"])
        return
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load(fixture)


def test_without_passed_report_new_measurements_remain_non_reference_compatible(tmp_path):
    fixture = inputs(tmp_path)
    morphology.profile_cells(fixture.args.labels, fixture.args.image, fixture.args.objects,
                             fixture.args.shift, fixture.morphology)
    fixture.args.resolution_json = None
    fixture.args.morphology = str(fixture.morphology / io.TABLE)
    _, manifest = profiles.build_profiles(fixture.args)
    assert manifest["feature_blocks"]["morphology"]["reference_compatible"] is False
    assert manifest["feature_blocks"]["nuclear_texture"]["reference_compatible"] is False
