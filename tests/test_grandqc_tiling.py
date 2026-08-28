import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "bin" / "run_grandqc_artifact_analysis.py"
SPEC = importlib.util.spec_from_file_location("run_grandqc_artifact_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _FakeProperties:
    def __init__(self, gib: float):
        self.total_memory = int(gib * (1024 ** 3))


class _FakeCuda:
    def __init__(self, gib: float):
        self.gib = gib

    def get_device_properties(self, _index: int):
        return _FakeProperties(self.gib)


class _FakeTorch:
    def __init__(self, gib: float):
        self.cuda = _FakeCuda(gib)


class GrandQCTileSelectionTest(unittest.TestCase):
    def test_auto_uses_large_context_on_supported_gpu(self):
        size, meta = MODULE.resolve_artifact_tile_size(0, "cuda", _FakeTorch(16.0))
        self.assertEqual(size, 1024)
        self.assertEqual(meta["mode"], "hardware_auto")

    def test_auto_falls_back_for_small_gpu_or_cpu(self):
        self.assertEqual(MODULE.resolve_artifact_tile_size(0, "cuda", _FakeTorch(4.0))[0], 512)
        self.assertEqual(MODULE.resolve_artifact_tile_size(0, "cpu", _FakeTorch(64.0))[0], 512)

    def test_explicit_tile_size_is_validated(self):
        self.assertEqual(MODULE.resolve_artifact_tile_size(768, "cpu", _FakeTorch(0.0))[0], 768)
        with self.assertRaises(ValueError):
            MODULE.resolve_artifact_tile_size(750, "cuda", _FakeTorch(16.0))


if __name__ == "__main__":
    unittest.main()
