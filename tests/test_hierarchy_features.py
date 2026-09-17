import json
import ast
import hashlib
import re
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from cell_profile_io import RasterReader, sha256_file
from discover_tissue_hierarchy import load_embedding_block, main as discover, validate_feature_definitions
from prepare_hierarchy_features import inspect_snapshot, parse_args, prepare, read_resized_field, resolve_fields
from model_provenance import package_version


def extractor_functions(*names, **extra):
    """Exercise standalone production functions without optional torchvision."""
    path = ROOT / "bin/extract_uni2_embeddings.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {"Path": Path, "json": json, "np": np, "hashlib": hashlib, "re": re,
                 "sha256_file": sha256_file, "package_version": package_version, "__file__": str(path), **extra}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def fixture(tmp_path):
    height = width = 96
    yy, xx = np.mgrid[:height, :width]
    image = np.stack((xx * 2, yy * 2, np.full_like(xx, 100)), axis=-1).astype(np.uint8)
    image[40:56, 40:56] = (30, 30, 220)
    tifffile.imwrite(tmp_path / "image.tif", image, tile=(16, 16), compression="deflate")
    records = []
    for y in range(6):
        for x in range(6):
            records.append({"label": y * 6+x+1, "x": x*16+8, "y": y*16+8, "grid_row": y, "grid_col": x,
                            "core_x0": x*16, "core_y0": y*16, "core_x1": (x+1)*16, "core_y1": (y+1)*16})
    grid = pd.DataFrame(records)
    grid.to_csv(tmp_path / "grid.csv", index=False)
    metadata = {"observation_type": "spatial_grid", "coordinate_space": "crop_roi_level0_pixels", "source_mpp_x": .5, "source_mpp_y": .5, "image_height_px": height, "image_width_px": width}
    (tmp_path / "grid.json").write_text(json.dumps(metadata))
    (tmp_path / "shift.json").write_text(json.dumps({"crop_size": {"height": height, "width": width}, "offset_crop_to_original": {"dx": 100, "dy": 200}}))
    (tmp_path / "resolution.json").write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5, "width_px": 300, "height_px": 400}))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"architecture": "vit_giant_patch14_224", "pretrained_cfg": {"mean": [.485, .456, .406], "std": [.229, .224, .225]}}))
    # A mock's byte-identified fixture, not a real UNI2 checkpoint.
    (snapshot / "model.safetensors").write_bytes(b"test-only-model-weights")
    args = parse_args(["--image", str(tmp_path / "image.tif"), "--grid-objects", str(tmp_path / "grid.csv"), "--grid-metadata", str(tmp_path / "grid.json"), "--shift-json", str(tmp_path / "shift.json"), "--resolution-json", str(tmp_path / "resolution.json"), "--model-snapshot", str(snapshot), "--sample-id", "specimen", "--local-field-um", "8", "--context-field-um", "24", "--batch", "5", "--outdir", str(tmp_path / "features")])
    return args, grid, image


def mock_encoder_factory(calls):
    def factory(args, model):
        calls.append(("load", model["model_revision"]))
        def encode(images):
            calls.append(("batch", images.copy()))
            return np.concatenate((images.mean(axis=(1, 2)), images.std(axis=(1, 2))), axis=1).astype(np.float32)
        return encode
    return factory


def test_independent_fields_binary_loader_and_safe_complete_reuse(tmp_path):
    args, grid, _ = fixture(tmp_path)
    calls = []
    summary = prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert sum(name == "load" for name, _ in calls) == 1
    assert all(len(values) <= 5 for name, values in calls if name == "batch")
    definitions = json.loads((args.outdir / "embedding_metadata.json").read_text())
    assert definitions["local"]["field_width_source_px"] == 16
    assert definitions["context"]["field_width_source_px"] == 48
    assert definitions["context"]["input_context_width_source_px"] == 48
    assert all(value["independent_transformer_context"] for value in definitions.values())
    assert definitions["context"]["effective_model_mpp_xy"] == pytest.approx([24/224, 24/224])
    local, names, local_meta = load_embedding_block(args.outdir / "local", grid, tmp_path / "unused.npy", definitions["local"], chunk_rows=3)
    context, _, context_meta = load_embedding_block(args.outdir / "context", grid, tmp_path / "unused2.npy", definitions["context"], chunk_rows=3)
    assert local.shape == context.shape == (36, 6)
    assert local_meta["zero_copy_readonly"] and not local.flags.writeable
    assert local_meta["observations_present"] == 36
    assert context_meta["observations_missing"] == 20
    valid = np.isfinite(context).all(axis=1)
    assert not np.allclose(local[valid], context[valid])
    assert not (tmp_path / "unused.npy").exists()
    calls_before = len(calls)
    cached = prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert cached["reused"] == {"local": True, "context": True}
    assert len(calls) == calls_before
    assert summary["no_expert_annotations_used"]
    provenance = summary["model_provenance"]
    assert provenance["source_repository"] == "https://huggingface.co/MahmoodLab/UNI2-h"
    assert provenance["requested_revision"] == "local_snapshot" and provenance["used_model"] is True
    assert provenance["resolved_revision"] == summary["model"]["model_revision"]
    assert {record["logical_name"] for record in provenance["checkpoints"]} == {"config.json", "model.safetensors"}
    assert all(sha256_file(record["cache_path"]) == record["sha256"] for record in provenance["checkpoints"])


def test_parent_hierarchy_cli_consumes_binary_fields_without_csv_feature_rewrite(tmp_path):
    args, grid, image = fixture(tmp_path)
    prepare(args, encoder_factory=mock_encoder_factory([]))
    parent = np.ones(image.shape[:2], np.uint16)
    parent[:, 48:] = 2
    parent[40:56, 40:44] = 0
    parent_path = tmp_path / "parents.tif"
    tifffile.imwrite(parent_path, parent)
    uncertainty_path, support_path = tmp_path / "uncertainty.tif", tmp_path / "support.tif"
    uncertainty = np.zeros(parent.shape, np.uint8)
    uncertainty[20:24, 20:24] = 254
    tifffile.imwrite(uncertainty_path, uncertainty)
    tifffile.imwrite(support_path, (parent > 0).astype(np.uint8))
    digest = sha256_file(parent_path)
    output = tmp_path / "hierarchy"
    command = ["--image", str(args.image), "--parent-mask", str(parent_path), "--parent-uncertainty", str(uncertainty_path), "--support-mask", str(support_path), "--grid-objects", str(args.grid_objects), "--grid-metadata", str(args.grid_metadata), "--shift-json", str(args.shift_json), "--resolution-json", str(args.resolution_json), "--local-embeddings", str(args.outdir / "local"), "--context-embeddings", str(args.outdir / "context"), "--embedding-metadata", str(args.outdir / "embedding_metadata.json"), "--sample-id", "specimen", "--min-observations", "4", "--repeats", "2", "--outdir", str(output)]
    # This portable test covers binary feature ingestion and source binding,
    # not native fitting. Actual native CLI/SpatialData suites cover KODAMA.
    command = ["--discovery-method", "legacy_kmeans", *command]
    discover(command)
    assert sha256_file(parent_path) == digest
    np.testing.assert_array_equal(tifffile.imread(output / "parent_domains.ome.tif"), parent)
    summary = json.loads((output / "hierarchy_summary.json").read_text())
    assert summary["embedding_provenance"]["context"]["observations_missing"] == 20
    assert summary["inputs"]["image"]["sha256"] == sha256_file(args.image)
    assert sha256_file(output / "parent_uncertainty.ome.tif") == sha256_file(uncertainty_path)
    assert np.all(tifffile.imread(output / "hierarchy_status.ome.tif")[uncertainty > 0] == 10)
    for relative, digest in summary["outputs"].items():
        assert sha256_file(output / relative) == digest
    # Same dimensions are insufficient: old fields must not be attached to a
    # different image or crop transform after reuse/resume.
    tifffile.imwrite(args.image, np.zeros_like(image))
    command[-1] = str(tmp_path / "wrong_image")
    with pytest.raises(ValueError, match="image checksum differs"):
        discover(command)


def test_cached_features_reject_changed_checkpoint_and_corruption(tmp_path):
    args, grid, _ = fixture(tmp_path)
    calls = []
    prepare(args, encoder_factory=mock_encoder_factory(calls))
    original = sha256_file(args.outdir / "local/features.npy")
    weights = args.model_snapshot / "model.safetensors"
    weights.write_bytes(b"changed-checkpoint")
    with pytest.raises(FileExistsError, match="different"):
        prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert sha256_file(args.outdir / "local/features.npy") == original
    weights.write_bytes(b"test-only-model-weights")
    matrix = np.load(args.outdir / "local/features.npy", mmap_mode="r+")
    matrix[0, 0] = -123
    matrix.flush()
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare(args, encoder_factory=mock_encoder_factory(calls))


def test_no_complete_fields_abstain_without_model_inference(tmp_path):
    args, _, _ = fixture(tmp_path)
    args.local_field_um, args.context_field_um = 64., 128.
    calls = []
    summary = prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert calls == []
    assert summary["model_provenance"]["used_model"] is False
    for name in ("local", "context"):
        manifest = json.loads((args.outdir / name / "embedding_manifest.json").read_text())
        assert manifest["observations_encoded"] == 0 and manifest["observations_missing"] == 36
        assert np.isnan(np.load(args.outdir / name / "features.npy")).all()


def test_binary_loader_aligns_ids_and_rejects_coordinates_and_definitions(tmp_path):
    args, grid, _ = fixture(tmp_path)
    prepare(args, encoder_factory=mock_encoder_factory([]))
    block = args.outdir / "local"
    metadata = json.loads((args.outdir / "embedding_metadata.json").read_text())["local"]
    expected = np.load(block / "features.npy").copy()
    rows = pd.read_csv(block / "feature_rows.csv")
    permutation = np.arange(len(rows))[::-1]
    rows = rows.iloc[permutation].reset_index(drop=True)
    rows.to_csv(block / "feature_rows.csv", index=False)
    np.save(block / "features.npy", expected[permutation])
    manifest = json.loads((block / "embedding_manifest.json").read_text())
    for key, filename in (("matrix", "features.npy"), ("rows", "feature_rows.csv")):
        manifest[key]["sha256"] = sha256_file(block / filename)
    (block / "embedding_manifest.json").write_text(json.dumps(manifest))
    values, _, provenance = load_embedding_block(block, grid, tmp_path / "aligned.npy", metadata, chunk_rows=2)
    np.testing.assert_array_equal(values, expected)
    assert not provenance["zero_copy_readonly"]
    with pytest.raises(ValueError, match="definition disagrees"):
        load_embedding_block(block, grid, tmp_path / "bad.npy", {**metadata, "pooling": "mean_patch"})
    rows.loc[0, "cx"] += 1
    rows.to_csv(block / "feature_rows.csv", index=False)
    manifest["rows"]["sha256"] = sha256_file(block / "feature_rows.csv")
    (block / "embedding_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coordinates disagree"):
        load_embedding_block(block, grid, tmp_path / "bad.npy", metadata)


def test_crop_fields_keep_physical_scale_and_do_not_center_crop(tmp_path):
    geometry = {"mpp_xy": [.5, .5]}
    one = resolve_fields(8, 24, geometry)
    two = resolve_fields(8, 24, {"mpp_xy": [.25, .25]})
    assert two["context"]["field_pixels"] == 2 * one["context"]["field_pixels"]
    assert two["context"]["effective_model_mpp_xy"] == one["context"]["effective_model_mpp_xy"]
    image = np.zeros((48, 48, 3), np.uint8)
    image[:, :8] = 255
    tifffile.imwrite(tmp_path / "edge.tif", image)
    with RasterReader(tmp_path / "edge.tif") as reader:
        resized = read_resized_field(reader, 0, 0, 48)
    assert resized[:, :20].mean() > 250  # entire outer field is present
    assert resized[:, 180:].mean() == 0
    with pytest.raises(ValueError, match="strictly wider"):
        resolve_fields(8, 8, geometry)
    with pytest.raises(ValueError, match="rounding"):
        resolve_fields(8, 8.01, geometry)
    with pytest.raises(ValueError, match="pixel limit"):
        resolve_fields(8, 24, geometry, max_window_pixels=100)


def test_calibration_mismatch_fails_before_encoder_and_creating_output(tmp_path):
    args, _, _ = fixture(tmp_path)
    report = json.loads(args.resolution_json.read_text())
    report["mpp_x"] = report["mpp_y"] = .55
    args.resolution_json.write_text(json.dumps(report))
    calls = []
    with pytest.raises(ValueError, match="differs from verified"):
        prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert not calls and not args.outdir.exists()


def test_snapshot_identity_is_content_bound_and_loader_is_strict_local(tmp_path):
    args, _, _ = fixture(tmp_path)
    one = inspect_snapshot(args.model_snapshot)
    (args.model_snapshot / "model.safetensors").write_bytes(b"different-bytes")
    two = inspect_snapshot(args.model_snapshot)
    assert one["model_revision"] != two["model_revision"]
    assert one["model_revision"].startswith("sha256:")
    source = (ROOT / "bin/extract_uni2_embeddings.py").read_text().split("def load_local_uni2_encoder", 1)[1].split("# Region readers", 1)[0]
    assert "pretrained=False" in source and "strict=True" in source and "weights_only=True" in source
    assert "hf_hub_download" not in source and "from_pretrained" not in source


def test_legacy_extractor_cache_binds_mpp_weights_inputs_and_parameters_without_token(tmp_path):
    args, _, _ = fixture(tmp_path)
    values = SimpleNamespace(image=str(args.image), mask=str(args.image), objects_csv=str(args.grid_objects), resolution_json=str(args.resolution_json),
                             outdir="first-location", paired_inner_square_outdir="", tiles_root="", hf_token="SECRET-MUST-NOT-APPEAR",
                             zero_outside_mask=False, target_mpp=.5, tile_size=224, inner_square_fixed_px=64,
                             encoder="uni2-h", pooling="cls", device="cpu", disable_grid_resume=False)
    function = extractor_functions("build_extraction_cache_contract")["build_extraction_cache_contract"]
    options = {"model_state_sha256": "a"*64, "backend": "timm_hf", "pooling": "cls", "calibration": {"source_mpp": .5, "target_mpp": .5}, "global_scaling": {"lo": [0, 0, 0], "hi": [255, 255, 255]}, "preprocessing": "RGB normalize"}
    contract, first = function(values, **options)
    assert "SECRET" not in json.dumps(contract) and "hf_token" not in contract["parameters"]
    values.outdir = "copied-location"
    assert function(values, **options)[1] == first
    for name, value in (("target_mpp", 1.0), ("tile_size", 448), ("inner_square_fixed_px", 96)):
        original = getattr(values, name)
        setattr(values, name, value)
        assert function(values, **options)[1] != first
        setattr(values, name, original)
    assert function(values, **{**options, "model_state_sha256": "b"*64})[1] != first
    assert function(values, **{**options, "calibration": {"source_mpp": .273774, "target_mpp": .5}})[1] != first
    args.resolution_json.write_text(json.dumps({"status": "pass", "mpp_x": .273774, "mpp_y": .273774}))
    assert function(values, **options)[1] != first
    with pytest.raises(ValueError, match="actual loaded encoder"):
        function(values, **{**options, "model_state_sha256": "main"})


def test_legacy_extractor_rejects_old_or_corrupted_grid_cache(tmp_path):
    function = extractor_functions("validate_extraction_grid_cache")["validate_extraction_grid_cache"]
    path = tmp_path / "uni2_embeddings_shard0000.csv.gz"
    path.write_bytes(b"fixture-shard")
    identity = "a"*64
    marker = {"cache_contract_sha256": identity, "shards": 1, "shard_files": [{"name": path.name, "sha256": sha256_file(path)}]}
    assert function(marker, identity, tmp_path)
    assert not function({"shards": 1, "observation_type": "grid"}, identity, tmp_path)
    assert not function(marker, "b"*64, tmp_path)
    path.write_bytes(b"modified-shard")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        function(marker, identity, tmp_path)


@pytest.mark.parametrize("architecture", ["vit_giant_patch14_224", "vit_huge_patch14_224"])
def test_pinned_loader_executes_cpu_tensor_path_and_rejects_incomplete_state(tmp_path, monkeypatch, architecture):
    torch = pytest.importorskip("torch")
    args, _, _ = fixture(tmp_path)
    config_path = args.model_snapshot / "config.json"
    config = json.loads(config_path.read_text())
    config["architecture"] = architecture
    config_path.write_text(json.dumps(config))
    assert inspect_snapshot(args.model_snapshot)["architecture"] == architecture
    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((1536, 3)))
        def forward(self, images):
            return images.mean(dim=(2, 3)) @ self.weight.T
    fixture_model = TinyEncoder()
    checkpoint = args.model_snapshot / "pytorch_model.bin"
    torch.save(fixture_model.state_dict(), checkpoint)
    calls = []
    timm = ModuleType("timm")
    def create_model(architecture, **kwargs):
        calls.append((architecture, kwargs))
        return TinyEncoder()
    timm.create_model = create_model
    layers = ModuleType("timm.layers")
    layers.SwiGLUPacked = torch.nn.Identity
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "timm.layers", layers)
    functions = extractor_functions("load_local_uni2_encoder", "fingerprint_encoder_state", torch=torch)
    encoder = functions["load_local_uni2_encoder"](args.model_snapshot, checkpoint.name, "cpu")
    embeddings = encoder(np.full((2, 224, 224, 3), 128, np.uint8))
    assert embeddings.shape == (2, 1536) and np.isfinite(embeddings).all()
    assert calls[0][1]["pretrained"] is False
    assert calls[0][0] == architecture
    assert calls[0][1]["depth"] == 24 and calls[0][1]["embed_dim"] == 1536 and calls[0][1]["reg_tokens"] == 8
    digest = functions["fingerprint_encoder_state"](fixture_model)
    assert digest == functions["fingerprint_encoder_state"](TinyEncoder())
    with torch.no_grad():
        fixture_model.weight[0, 0] += 1
    assert functions["fingerprint_encoder_state"](fixture_model) != digest
    torch.save({}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        functions["load_local_uni2_encoder"](args.model_snapshot, checkpoint.name, "cpu")
    config["architecture"] = "arbitrary_model"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="compatible"):
        inspect_snapshot(args.model_snapshot)
    with pytest.raises(ValueError, match="compatible"):
        functions["load_local_uni2_encoder"](args.model_snapshot, checkpoint.name, "cpu")
