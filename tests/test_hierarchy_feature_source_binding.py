"""Tiny source-race/reuse contracts; synthetic vectors, never learned inference."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import prepare_hierarchy_features as module
from cell_profile_io import sha256_file
from test_hierarchy_features import fixture, mock_encoder_factory
from test_process_code_dependencies import local_python_imports


def test_feature_producer_identity_covers_recursive_static_local_python_imports():
    code_root = ROOT / "bin"
    declared = {(code_root / name).resolve() for name in module.PRODUCER_CODE_FILES}
    assert len(declared) == len(module.PRODUCER_CODE_FILES)
    pending = [(code_root / "prepare_hierarchy_features.py").resolve()]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        assert path in declared, f"Unbound local Python dependency: {path.name}"
        pending.extend(local_python_imports(path))


def source_path(args, name):
    if name in ("checkpoint", "config"):
        return args.model_snapshot / ("model.safetensors" if name == "checkpoint" else "config.json")
    return Path(getattr(args, name))


def mutate_bytes(path):
    """Keep TIFF/JSON/CSV readable but alter its exact identity."""
    path.write_bytes(path.read_bytes() + b"\n ")


def assert_no_complete_field(root, name):
    assert not (root / name / "embedding_manifest.json").exists()
    request = root / name / "field_request.json"
    if request.exists():
        assert json.loads(request.read_text()).get("status") != "complete"
    assert not (root / "hierarchy_features_summary.json").exists()


@pytest.mark.parametrize("name", ["image", "grid_objects", "grid_metadata", "shift_json",
                                  "resolution_json", "checkpoint", "config"])
def test_input_or_model_change_during_encoding_never_completes_current_field(tmp_path, name):
    args, _, _ = fixture(tmp_path)
    target = source_path(args, name)
    original = sha256_file(target)
    calls = []
    def factory(arguments, model):
        def encode(images):
            if not calls:
                mutate_bytes(target)
            calls.append(len(images))
            return images.mean(axis=(1, 2)).astype(np.float32)
        return encode
    with pytest.raises((ValueError, RuntimeError), match="changed|identity|source|mismatch"):
        module.prepare(args, encoder_factory=factory)
    assert calls and sha256_file(target) != original
    assert_no_complete_field(args.outdir, "local")
    assert not (args.outdir / "context" / "embedding_manifest.json").exists()


def test_changed_source_in_second_field_preserves_prior_bytes_without_current_completion(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path)
    active, earlier, calls = [], {}, []
    original_produce = module.produce_field
    def produce(*values, **options):
        directory = Path(values[4] if len(values) > 4 else options["directory"])
        active[:] = [directory.name]
        if directory.name == "context":
            earlier.update({name: (args.outdir / "local" / name).read_bytes()
                            for name in ("embedding_manifest.json", "features.npy", "feature_rows.csv")})
        return original_produce(*values, **options)
    monkeypatch.setattr(module, "produce_field", produce)
    def factory(arguments, model):
        def encode(images):
            if active == ["context"] and "context" not in calls:
                mutate_bytes(args.image)
            calls.append(active[0])
            return images.mean(axis=(1, 2)).astype(np.float32)
        return encode
    with pytest.raises((ValueError, RuntimeError), match="changed|identity|source|mismatch"):
        module.prepare(args, encoder_factory=factory)
    assert set(calls) == {"local", "context"}
    assert earlier and all((args.outdir / "local" / name).read_bytes() == value for name, value in earlier.items())
    assert_no_complete_field(args.outdir, "context")
    # Preserving the prior artifacts is not recertifying them for a changed image.
    prior = json.loads(earlier["embedding_manifest.json"])
    assert prior["source_inputs"]["image"]["sha256"] != sha256_file(args.image)


def test_config_changed_after_its_read_is_not_hashed_as_if_original_config_was_consumed(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path)
    target = source_path(args, "config").resolve()
    original_read = Path.read_text
    changed, calls = [], []
    def read(path, *values, **options):
        contents = original_read(path, *values, **options)
        if path.resolve() == target and not changed:
            mutate_bytes(target)
            changed.append(True)
        return contents
    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises((ValueError, RuntimeError), match="changed|identity|source|mismatch"):
        module.prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert changed and calls == []
    assert_no_complete_field(args.outdir, "local")


def test_image_changed_after_first_pixel_read_is_not_adopted_as_original_input(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path)
    original_window = module.RasterReader.window
    changed, calls = [], []
    def window(reader, *values, **options):
        pixels = original_window(reader, *values, **options)
        if reader.path.resolve() == args.image.resolve() and not changed:
            mutate_bytes(args.image)
            changed.append(True)
        return pixels
    monkeypatch.setattr(module.RasterReader, "window", window)
    with pytest.raises((ValueError, RuntimeError), match="changed|identity|source|mismatch"):
        module.prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert changed
    assert_no_complete_field(args.outdir, "local")


@pytest.mark.parametrize("name", ["grid_metadata", "shift_json", "resolution_json"])
def test_geometry_changed_after_its_read_cannot_be_adopted_by_later_hashing(tmp_path, monkeypatch, name):
    args, _, _ = fixture(tmp_path)
    target = source_path(args, name).resolve()
    original_read = Path.read_text
    changed, calls = [], []
    def read(path, *values, **options):
        contents = original_read(path, *values, **options)
        if path.resolve() == target and not changed:
            mutate_bytes(target)
            changed.append(True)
        return contents
    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises((ValueError, RuntimeError), match="changed|identity|source|mismatch"):
        module.prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert changed and calls == []
    assert_no_complete_field(args.outdir, "local")


def test_grid_changed_after_dataframe_read_cannot_be_adopted_by_later_hashing(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path)
    original_read = module.pd.read_csv
    changed, calls = [], []
    def read(path, *values, **options):
        frame = original_read(path, *values, **options)
        if Path(path).resolve() == args.grid_objects.resolve() and not changed:
            mutate_bytes(args.grid_objects)
            changed.append(True)
        return frame
    monkeypatch.setattr(module.pd, "read_csv", read)
    with pytest.raises((ValueError, RuntimeError), match="changed|identity|source|mismatch"):
        module.prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert changed and calls == []
    assert_no_complete_field(args.outdir, "local")


def test_unchanged_completed_fields_reuse_exact_payloads_without_loading_model(tmp_path):
    args, _, _ = fixture(tmp_path)
    calls = []
    first = module.prepare(args, encoder_factory=mock_encoder_factory(calls))
    assert any(name == "load" for name, _ in calls)
    assert (args.outdir / "hierarchy_features_summary.json").exists()
    pinned = {f"{field}/{name}": (args.outdir / field / name).read_bytes()
              for field in ("local", "context")
              for name in ("embedding_manifest.json", "features.npy", "feature_rows.csv")}
    def forbidden_model_load(*values):
        pytest.fail("Unchanged complete fields must not load an encoder")
    second = module.prepare(args, encoder_factory=forbidden_model_load)
    assert first["request_key"] == second["request_key"]
    assert second["reused"] == {"local": True, "context": True}
    assert first["encoder_loaded_this_invocation"] is True
    assert second["encoder_loaded_this_invocation"] is False
    assert second["sources_unchanged_after_generation"] is True
    assert all(sha256_file(path) == digest for path, digest in second["source_identity"].items())
    assert all((args.outdir / relative).read_bytes() == value for relative, value in pinned.items())


def test_source_changed_after_fields_complete_prevents_final_summary(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path)
    original_write = module.write_json
    changed = []
    def write(path, value):
        original_write(path, value)
        if Path(path).name == "embedding_metadata.json":
            mutate_bytes(args.grid_objects)
            changed.append(True)
    monkeypatch.setattr(module, "write_json", write)
    with pytest.raises((ValueError, RuntimeError), match="changed|identity|source|mismatch"):
        module.prepare(args, encoder_factory=mock_encoder_factory([]))
    assert changed
    assert all((args.outdir / field / "embedding_manifest.json").exists() for field in ("local", "context"))
    assert not (args.outdir / "hierarchy_features_summary.json").exists()
