"""CellViT source-to-canonical attachment tests; no learned model is executed."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import build_cell_consensus as consensus
import build_cell_profiles as profiles
from cell_profile_io import sha256_file


def profile_inputs(directory):
    directory.mkdir(exist_ok=True)
    image = directory / "image.tif"
    tifffile.imwrite(image, np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3), photometric="rgb")
    labels = np.zeros((8, 12), np.uint32)
    labels[1:4, 1:4] = 1
    labels[3:6, 7:10] = 2
    labels[0:2, 10:12] = 3
    tifffile.imwrite(directory / "labels.tif", labels)
    objects = pd.DataFrame({"label": [2, 1, 3], "x": [8., 2., 10.], "y": [4., 2., 1.],
        "xmin": [7, 1, 10], "ymin": [3, 1, 0], "xmax": [10, 4, 12], "ymax": [6, 4, 2],
        "cellvitpp_id": ["007", "", "NA"]})
    objects.to_csv(directory / "objects.csv", index=False)
    (directory / "shift.json").write_text(json.dumps({"source_mpp": .5,
        "offset_crop_to_original": {"dx": 100, "dy": 200}, "crop_size": {"width": 12, "height": 8}}))
    (directory / "resolution.json").write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5,
                                                         "width_px": 200, "height_px": 300}))
    return profiles.parser().parse_args(["--objects", str(directory / "objects.csv"),
        "--image", str(image), "--labels", str(directory / "labels.tif"),
        "--shift", str(directory / "shift.json"), "--resolution-json", str(directory / "resolution.json"),
        "--sample-id", "sample_A", "--outdir", str(directory / "profiles")])


def legacy_bundle(directory):
    directory.mkdir()
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(directory / "cellvit_embeddings.npy", vectors)
    pd.DataFrame({"embedding_row": [0, 1, 2], "cellvitpp_id": ["007", "NA", "unmatched"],
        "x_px": [9., 10., 5.], "y_px": [4., 1., 5.], "source_graph_row": [0, 1, 2]}).to_csv(
        directory / "cellvit_embedding_ids.csv", index=False)
    (directory / "cellvit_embeddings_metadata.json").write_text(json.dumps({
        "schema_version": 1, "status": "exported", "representation": "synthetic fixture vectors",
        "provenance": {"model_provenance": {"resolved_revision": "package:cellvit==1.0.9",
            "checkpoints": [{"logical_name": "HIPT", "sha256": "a" * 64}]}},
        "dtype": "float32", "embedding_dimension": 4, "retained_cell_count": 3}))
    return vectors


def bound_bundle(args, directory, *, amp=False, population_variant=None, runtime_variant="fixture-v1",
                 channel_conversion="RGB unchanged"):
    """Make explicit synthetic payloads with the real completion producer.

    The declared runtime/checkpoint identities are test fixtures, not a claim
    that CellViT or a learned checkpoint ran. Producer execution itself has its
    own tests; no consumer validation routine is mocked here.
    """
    from cellvit_embedding_io import capture_inputs, complete_embedding_bundle
    import cellvit_embeddings
    directory.mkdir()
    raw_dir = directory / "raw"
    raw_dir.mkdir()
    vectors = np.arange(16, dtype=np.float32).reshape(4, 4)
    payload = {"cells": [{"id": name, "centroid": xy} for name, xy in
                          (("007", [9., 4.]), ("NA", [10., 1.]), ("unmatched", [5., 5.]))],
               "synthetic_contract_fixture": True, "population_variant": population_variant}
    (directory / "cellvit_cells.json").write_text(json.dumps(payload))
    raw_payload = {"cells": [*payload["cells"], {"id": "discarded", "centroid": [11., 7.]}]}
    raw_json = raw_dir / "cells.json"
    raw_json.write_text(json.dumps(raw_payload))
    (raw_dir / "cells.pt").write_bytes(b"synthetic graph-array fixture; no PyTorch or learned model")
    positions = np.asarray([cell["centroid"] for cell in raw_payload["cells"]], dtype=float)
    execution, source_paths = capture_inputs(args.image, args.shift, args.resolution_json)
    fixture_hash = lambda text: hashlib.sha256(("synthetic-not-learned:" + text).encode()).hexdigest()
    execution.update({"model": "HIPT", "taxonomy": "pannuke", "amp": amp,
        "used_model": False, "synthetic_contract_fixture": True,
        "checkpoint": {"status": "verified", "sha256": fixture_hash("checkpoint")},
        "prepared_image": {"sha256": fixture_hash("prepared image")},
        "preprocessing": {"native_size_px": [12, 8], "bands": 3, "tile_size_px": [512, 512],
            "compression": "jpeg", "jpeg_quality": 92, "pyramid": True, "bigtiff": True,
            "channel_conversion": channel_conversion},
        "runtime_identity": {"status": "verified_configured_python_entrypoint", "package_version": "1.0.9",
            "portable_identity": {"package_source_sha256": fixture_hash(runtime_variant),
                "executable_sha256": fixture_hash("executable"), "interpreter_sha256": fixture_hash("interpreter"),
                "package_version": "1.0.9", "entry_point": "synthetic_fixture_only"}}})
    with patch.object(cellvit_embeddings, "load_graph_arrays", return_value=(vectors, positions)):
        cellvit_embeddings.export_embeddings(raw_json, directory / "cellvit_cells.json", directory,
            {"synthetic_contract_fixture": True, "used_model": False}, execution=execution)
    receipt = complete_embedding_bundle(directory, execution, source_paths=source_paths)
    return vectors[:3], receipt


def bind_canonical_objects(args, source):
    from cellvit_embedding_io import population
    path = Path(args.objects)
    table = pd.read_csv(path, dtype={"cellvitpp_id": str}, keep_default_na=False)
    lookup = dict(population(json.loads((source / "cellvit_cells.json").read_text())))
    table["cellvitpp_x_px"] = [lookup[name][0] if name else "" for name in table.cellvitpp_id]
    table["cellvitpp_y_px"] = [lookup[name][1] if name else "" for name in table.cellvitpp_id]
    digest = sha256_file(source / "cellvit_cells.json")
    table["cellvitpp_source_sha256"] = [digest if name else "" for name in table.cellvitpp_id]
    table.to_csv(path, index=False)
    args.cellvit_features = str(source)
    return table


def bound_profile_fixture(tmp_path, **kwargs):
    args = profile_inputs(tmp_path)
    source = tmp_path / "bound_cellvit"
    vectors, receipt = bound_bundle(args, source, **kwargs)
    table = bind_canonical_objects(args, source)
    return args, source, vectors, receipt, table


def test_bound_profile_preserves_full_population_and_accepts_legitimate_fused_centroid(tmp_path):
    args, source, vectors, receipt, original = bound_profile_fixture(tmp_path)
    frozen = {path: sha256_file(path) for path in source.iterdir() if path.is_file()}
    cells, manifest = profiles.build_profiles(args)
    assert cells.cell_id.tolist() == ["2", "1", "3"]
    assert cells.cellvitpp_id.tolist() == ["007", "", "NA"]
    np.testing.assert_array_equal(cells[["x_crop_px", "y_crop_px"]], original[["x", "y"]])
    assert cells.x_crop_px.iloc[0] == 8 and cells.cellvitpp_x_px.iloc[0] == 9
    assert cells[["cellvitpp_x_px", "cellvitpp_y_px"]].dtypes.eq(np.dtype("float64")).all()
    assert cells.loc[1, ["cellvitpp_x_px", "cellvitpp_y_px"]].isna().all()
    stored = pd.read_parquet(Path(args.outdir) / "cell_profiles.parquet")
    np.testing.assert_array_equal(stored[["cellvitpp_x_px", "cellvitpp_y_px"]],
                                  cells[["cellvitpp_x_px", "cellvitpp_y_px"]])
    assert cells.cellvit_available.tolist() == [True, False, True]
    assert cells.cellvit_status.tolist() == ["available_verified_reference_definition", "missing_no_cellvit_source",
                                            "available_verified_reference_definition"]
    record = manifest["feature_blocks"]["cellvit"]
    assert record["reference_compatible"] is True
    assert record["feature_definition"] == receipt["feature_definition"]
    assert record["feature_definition"]["source_mpp"] == .5 and record["feature_definition"]["amp"] is False
    assert record["canonical_correspondence"]["canonical_fused_centroid_used_for_matching"] is False
    assert record["source_binding"]["retained_population_sha256"] == sha256_file(source / "cellvit_cells.json")
    np.testing.assert_array_equal(np.load(Path(args.outdir) / record["path"]),
                                  np.stack([vectors[0], np.full(4, np.nan), vectors[1]]))
    assert all(sha256_file(path) == digest for path, digest in frozen.items())
    from cell_reference_atlas import load_profile
    assert load_profile(args.outdir, ["cellvit"]).blocks["cellvit"].shape == (3, 4)


@pytest.mark.parametrize("input_name", ["image", "shift", "resolution_json"])
def test_same_ids_from_foreign_image_shift_or_calibration_fail(tmp_path, input_name):
    args, _, _, _, _ = bound_profile_fixture(tmp_path)
    source = Path(getattr(args, input_name))
    other = source.with_name("foreign_" + source.name)
    if input_name == "image":
        values = tifffile.imread(source)
        values[0, 0, 0] ^= 1
        tifffile.imwrite(other, values, photometric="rgb")
    else:
        data = json.loads(source.read_text())
        if input_name == "resolution_json":
            data.update(mpp_x=.6, mpp_y=.6)
        else:
            data["offset_crop_to_original"]["dx"] += 1
        other.write_text(json.dumps(data))
    setattr(args, input_name, str(other))
    with pytest.raises(ValueError, match="source input hash mismatch|geometry/calibration mismatch"):
        profiles.build_profiles(args)
    assert not (Path(args.outdir) / "cell_profiles_manifest.json").exists()


def test_valid_other_population_with_identical_ids_and_coordinates_is_rejected(tmp_path):
    args, _, _, _, _ = bound_profile_fixture(tmp_path)
    other = tmp_path / "other_population"
    bound_bundle(args, other, population_variant="different detector population")
    args.cellvit_features = str(other)
    with pytest.raises(ValueError, match="source population differs"):
        profiles.build_profiles(args)
    assert not (Path(args.outdir) / "cell_profiles_manifest.json").exists()


@pytest.mark.parametrize("mutation,match", [
    ("detector_centroid", "detector centroids differ"), ("source_hash", "source population differs"),
    ("missing_id", "absent from its declared"), ("duplicate_id", "multiple canonical"),
    ("partial_fields", "Incomplete canonical"), ("unmapped_coordinates", "without a source ID")])
def test_canonical_correspondence_cannot_be_guessed_or_silently_repaired(tmp_path, mutation, match):
    args, _, _, _, table = bound_profile_fixture(tmp_path)
    if mutation == "detector_centroid":
        table.loc[0, "cellvitpp_x_px"] += .1
    elif mutation == "source_hash":
        table.loc[0, "cellvitpp_source_sha256"] = "a" * 64
    elif mutation == "missing_id":
        table.loc[0, "cellvitpp_id"] = "missing"
    elif mutation == "duplicate_id":
        table.loc[2, "cellvitpp_id"] = "007"
    elif mutation == "partial_fields":
        table = table.drop(columns="cellvitpp_y_px")
    else:
        table.loc[1, "cellvitpp_x_px"] = 2.
    table.to_csv(args.objects, index=False)
    with pytest.raises(ValueError, match=match):
        profiles.build_profiles(args)
    assert not (Path(args.outdir) / "cell_profiles_manifest.json").exists()


def test_new_receipt_with_legacy_canonical_objects_does_not_invent_correspondence(tmp_path):
    args = profile_inputs(tmp_path)
    source = tmp_path / "bound_cellvit"
    bound_bundle(args, source)
    args.cellvit_features = str(source)
    cells, manifest = profiles.build_profiles(args)
    assert cells.cellvit_available.tolist() == [True, False, True]
    record = manifest["feature_blocks"]["cellvit"]
    assert record["source_binding"]["status"] == "verified_exact_inputs_and_population"
    assert record["reference_compatible"] is False
    assert record["canonical_correspondence"]["status"] == "legacy_unverified_canonical_correspondence"


@pytest.mark.parametrize("timing", ["after_validation", "during_attachment"])
def test_embedding_payload_mutation_cannot_establish_a_new_attachment_baseline(tmp_path, monkeypatch, timing):
    import cellvit_embedding_io
    args, source, _, _, _ = bound_profile_fixture(tmp_path)
    payload = source / "cellvit_embeddings.npy"
    def mutate():
        changed = np.load(payload, mmap_mode="r+")
        changed[0, 0] += 1
        changed.flush()
    if timing == "after_validation":
        original = cellvit_embedding_io.load_cellvit_embedding_bundle
        def changed_after_validation(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(cellvit_embedding_io, "load_cellvit_embedding_bundle", changed_after_validation)
        message = "differ from their validated completion"
    else:
        original = np.lib.format.open_memmap
        def changed_during_attachment(*args, **kwargs):
            result = original(*args, **kwargs)
            if Path(args[0]).name == "cellvit.npy" and kwargs.get("mode") == "w+":
                mutate()
            return result
        monkeypatch.setattr(np.lib.format, "open_memmap", changed_during_attachment)
        message = "changed during canonical profile attachment"
    with pytest.raises(ValueError, match=message):
        profiles.build_profiles(args)
    assert not (Path(args.outdir) / "cell_profiles_manifest.json").exists()


@pytest.mark.parametrize("change", ["none", "amp", "mpp", "runtime", "preprocessing"])
def test_actual_definition_compatibility_includes_physical_preprocessing_amp_and_runtime(tmp_path, change):
    from cell_reference_atlas import load_profile
    left_args, _, _, _, _ = bound_profile_fixture(tmp_path / "left")
    profiles.build_profiles(left_args)
    reference = load_profile(left_args.outdir, ["cellvit"])
    right_args = profile_inputs(tmp_path / "right")
    if change == "mpp":
        for name in ("shift", "resolution_json"):
            path = Path(getattr(right_args, name))
            value = json.loads(path.read_text())
            value.update(source_mpp=.6) if name == "shift" else value.update(mpp_x=.6, mpp_y=.6)
            path.write_text(json.dumps(value))
    source = tmp_path / "right/bound_cellvit"
    bound_bundle(right_args, source, amp=change == "amp", runtime_variant="different-fixture" if change == "runtime" else "fixture-v1",
                 channel_conversion="sRGB conversion" if change == "preprocessing" else "RGB unchanged")
    bind_canonical_objects(right_args, source)
    _, manifest = profiles.build_profiles(right_args)
    assert manifest["feature_blocks"]["cellvit"]["reference_compatible"] is True
    if change == "none":
        assert load_profile(right_args.outdir, ["cellvit"], reference.schemas).schemas == reference.schemas
    else:
        with pytest.raises(ValueError, match="Incompatible feature schema/definition"):
            load_profile(right_args.outdir, ["cellvit"], reference.schemas)


def test_legacy_vectors_are_visible_but_not_reference_compatible(tmp_path):
    args = profile_inputs(tmp_path)
    source = tmp_path / "legacy_cellvit"
    vectors = legacy_bundle(source)
    args.cellvit_features = str(source)
    cells, manifest = profiles.build_profiles(args)
    assert cells.cell_id.tolist() == ["2", "1", "3"]
    assert cells.cellvitpp_id.tolist() == ["007", "", "NA"]
    assert cells.cellvit_available.tolist() == [True, False, True]
    assert cells.cellvit_status.tolist() == ["available_unverified_for_reference", "missing_no_cellvit_source",
                                            "available_unverified_for_reference"]
    record = manifest["feature_blocks"]["cellvit"]
    assert record["reference_compatible"] is False
    assert record["source_binding"]["status"] == "legacy_unverified"
    assert record["canonical_correspondence"]["status"] == "legacy_unverified_canonical_correspondence"
    assert record["missing_cells"] == 1 and record["source_cells_not_canonical"] == 1
    expected = np.stack([vectors[0], np.full(4, np.nan), vectors[1]])
    np.testing.assert_array_equal(np.load(Path(args.outdir) / record["path"]), expected)
    from cell_reference_atlas import load_profile
    with pytest.raises(ValueError, match="not reference-compatible"):
        load_profile(args.outdir, ["cellvit"])


def consensus_run(tmp_path, monkeypatch, *, mutation=None):
    """Real matching/CSV lineage; only unavailable native TIFF/preview IO is shimmed."""
    source = tmp_path / "cellvit_cells.json"
    source.write_text(json.dumps({"cells": [{"id": "007", "centroid": [5., 3.],
        "contour": [[4, 2], [6, 2], [6, 4], [4, 4]], "type_id": 2, "type_name": "inflammatory"}]}))
    stardist = tmp_path / "stardist.csv"
    stardist.write_text("label,x,y\n01,3,3\n")
    hovernet = tmp_path / "hovernet.json"
    hovernet.write_text('{"cells":[]}')
    image, shift = tmp_path / "image.tif", tmp_path / "shift.json"
    tifffile.imwrite(image, np.full((16, 16, 3), 128, np.uint8), photometric="rgb")
    shift.write_text(json.dumps({"source_mpp": .5, "offset_crop_to_original": {"dx": 0, "dy": 0},
                                 "crop_size": {"width": 16, "height": 16}}))
    outdir = tmp_path / "consensus"
    monkeypatch.setitem(sys.modules, "pyvips", SimpleNamespace(Image=SimpleNamespace(
        new_from_file=lambda *a, **k: SimpleNamespace(width=16, height=16))))
    def write_mask(records, width, height, path, *args):
        labels = np.zeros((height, width), np.uint32)
        for record in records:
            x, y = int(record["x"]), int(record["y"])
            labels[y, x] = record["label"]
            record["mask_seed_x"], record["mask_seed_y"] = x, y
        tifffile.imwrite(path, labels)
        return {"reserved_seed_count": len(records), "relocated_seed_count": 0}
    monkeypatch.setattr(consensus, "write_mask", write_mask)
    monkeypatch.setattr(consensus, "validate_mask_label_coverage", lambda path, n: {"present_label_count": n})
    def preview(*args):
        if mutation == "after_fusion":
            source.write_bytes(source.read_bytes() + b" ")
    monkeypatch.setattr(consensus, "write_preview", preview)
    if mutation == "during_parse":
        original = consensus.load_cells
        def changed(path, detector, **kwargs):
            result = original(path, detector, **kwargs)
            if detector == "cellvitpp":
                source.write_bytes(source.read_bytes() + b" ")
            return result
        monkeypatch.setattr(consensus, "load_cells", changed)
    if mutation == "during_fusion":
        original_edges = consensus.candidate_edges
        def changed_edges(*args, **kwargs):
            result = original_edges(*args, **kwargs)
            source.write_bytes(source.read_bytes() + b" ")
            return result
        monkeypatch.setattr(consensus, "candidate_edges", changed_edges)
    monkeypatch.setattr(sys, "argv", ["build_cell_consensus.py", "--stardist-objects", str(stardist),
        "--hovernet-cells", str(hovernet), "--cellvit-cells", str(source), "--image", str(image),
        "--shift", str(shift), "--outdir", str(outdir)])
    digest = sha256_file(source)
    consensus.main()
    return outdir, digest


def test_consensus_exports_detector_centroid_without_replacing_fused_centroid(tmp_path, monkeypatch):
    outdir, digest = consensus_run(tmp_path, monkeypatch)
    cells = pd.read_csv(outdir / "objects.csv", dtype={"cellvitpp_id": str, "stardist_id": str})
    assert cells.label.tolist() == [1]
    assert cells.cellvitpp_id.tolist() == ["007"] and cells.stardist_id.tolist() == ["01"]
    assert cells[["x", "y"]].to_numpy().tolist() == [[4., 3.]]
    assert cells[["cellvitpp_x_px", "cellvitpp_y_px"]].to_numpy().tolist() == [[5., 3.]]
    assert cells.cellvitpp_source_sha256.eq(digest).all()
    summary = json.loads((outdir / "consensus_summary.json").read_text())
    assert summary["cellvit_source_binding"]["sha256"] == digest


@pytest.mark.parametrize("mutation", ["during_parse", "during_fusion", "after_fusion"])
def test_consensus_rejects_source_mutation_before_completion(tmp_path, monkeypatch, mutation):
    with pytest.raises(RuntimeError, match="source population changed"):
        consensus_run(tmp_path, monkeypatch, mutation=mutation)
    assert not (tmp_path / "consensus/consensus_summary.json").exists()
