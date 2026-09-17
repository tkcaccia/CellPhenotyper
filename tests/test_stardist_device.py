"""Validate explicit TensorFlow device selection with fake APIs, never models."""
import ast
import builtins
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "bin/run_stardist_roi_segmentation.py"


class FakeTensorFlow:
    def __init__(self, *, cuda=False, gpus=("Metal:0",), initialized=False):
        self.events = []
        self.gpus = list(gpus)
        self.visible = list(gpus)
        self.initialized = initialized
        self.test = SimpleNamespace(is_built_with_cuda=lambda: cuda)
        self.config = SimpleNamespace(
            list_physical_devices=self.list_physical,
            set_visible_devices=self.set_visible,
            get_visible_devices=self.get_visible,
        )

    def list_physical(self, kind):
        assert kind == "GPU"
        self.events.append("list_physical")
        return list(self.gpus)

    def set_visible(self, devices, kind):
        assert kind == "GPU"
        self.events.append("set_visible")
        if self.initialized:
            raise RuntimeError("Physical devices cannot be modified after initialization")
        self.visible = list(devices)

    def get_visible(self, kind):
        assert kind == "GPU"
        return list(self.visible)


@pytest.fixture
def implementation():
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
        "configure_tensorflow_device", "initialize_stardist_runtime", "load_stardist_model_filtered"}]
    assert len(functions) == 3
    namespace = {"List": List, "io": io, "contextlib": contextlib,
                 "STARDIST_AVAILABLE": False, "STARDIST_IMPORT_ERROR": None,
                 "_STARDIST_RUNTIME_INITIALIZED": False, "StarDist2D": None,
                 "_local_stardist_candidates": lambda name: [], "log": lambda message: None}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def fake_imports(monkeypatch, tensorflow):
    real_import = builtins.__import__
    events = []

    class NoModel:
        @staticmethod
        def from_pretrained(name):
            events.append("model_init")
            return {"used_model": False, "name": name}

    def isolated_import(name, *args, **kwargs):
        if name == "tensorflow":
            if tensorflow is None:
                raise ModuleNotFoundError("No optional TensorFlow runtime")
            return tensorflow
        if name == "stardist.models":
            events.append("model_import")
            if tensorflow is not None:
                events.append(tuple(tensorflow.visible))
            return SimpleNamespace(StarDist2D=NoModel)
        if name == "stardist.plot":
            return SimpleNamespace(render_label=lambda *args, **kwargs: None)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    return events


def test_cpu_disables_metal_gpu_before_stardist_import_and_model_initialization(implementation, monkeypatch):
    tensorflow = FakeTensorFlow(cuda=False, gpus=("Metal:0",))
    events = fake_imports(monkeypatch, tensorflow)
    assert implementation["initialize_stardist_runtime"]("cpu") is True
    assert tensorflow.visible == []
    assert events[:2] == ["model_import", ()]
    result = implementation["load_stardist_model_filtered"]("model-free-fixture")
    assert result["used_model"] is False and events[-1] == "model_init"


def test_cuda_requires_cuda_build_not_merely_any_tensorflow_gpu(implementation, monkeypatch):
    events = fake_imports(monkeypatch, FakeTensorFlow(cuda=False, gpus=("Metal:0",)))
    with pytest.raises(RuntimeError, match="not built with CUDA"):
        implementation["initialize_stardist_runtime"]("cuda")
    assert events == []


def test_cuda_requires_a_visible_gpu_and_does_not_fallback_to_cpu(implementation):
    with pytest.raises(RuntimeError, match="no visible CUDA GPU"):
        implementation["configure_tensorflow_device"]("cuda", FakeTensorFlow(cuda=True, gpus=()))


def test_cuda_visibility_is_configured_before_model_initialization(implementation, monkeypatch):
    tensorflow = FakeTensorFlow(cuda=True, gpus=("CUDA:0",))
    events = fake_imports(monkeypatch, tensorflow)
    assert implementation["initialize_stardist_runtime"]("cuda") is True
    assert tensorflow.events == ["list_physical", "set_visible"]
    assert events[:2] == ["model_import", ("CUDA:0",)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_already_initialized_tensorflow_fails_without_running_a_model(implementation, monkeypatch, device):
    events = fake_imports(monkeypatch, FakeTensorFlow(cuda=True, gpus=("CUDA:0",), initialized=True))
    with pytest.raises(RuntimeError, match="after TensorFlow device initialization"):
        implementation["initialize_stardist_runtime"](device)
    assert events == []


def test_standalone_auto_keeps_existing_tensorflow_selection(implementation, monkeypatch):
    tensorflow = FakeTensorFlow(cuda=False, gpus=("Metal:0",))
    events = fake_imports(monkeypatch, tensorflow)
    assert implementation["initialize_stardist_runtime"]() is True
    assert tensorflow.events == [] and tensorflow.visible == ["Metal:0"]
    assert events[:2] == ["model_import", ("Metal:0",)]


def test_missing_optional_framework_preserves_precomputed_label_fallback(implementation, monkeypatch):
    events = fake_imports(monkeypatch, None)
    assert implementation["initialize_stardist_runtime"]("cpu") is False
    assert implementation["STARDIST_AVAILABLE"] is False
    assert isinstance(implementation["STARDIST_IMPORT_ERROR"], ImportError)
    assert events == []


def test_visible_device_readback_must_honor_explicit_cpu_selection(implementation):
    tensorflow = FakeTensorFlow()
    tensorflow.config.get_visible_devices = lambda kind: ["unexpected GPU"]
    with pytest.raises(RuntimeError, match="did not honor explicit StarDist cpu"):
        implementation["configure_tensorflow_device"]("cpu", tensorflow)


def test_cli_declares_device_and_initializes_runtime_before_segmentation():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    device_arg = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                      and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--device")
    assert {arg.arg: ast.literal_eval(arg.value) for arg in device_arg.keywords}["default"] == "auto"
    initialize = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "initialize_stardist_runtime")
    segmentation = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id in {"stardist_segment_big", "stardist_segment_roi"}]
    assert initialize.lineno < min(node.lineno for node in segmentation)
