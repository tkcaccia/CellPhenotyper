import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "bin" / "gigatime_hardware.py"
SPEC = importlib.util.spec_from_file_location("gigatime_hardware", MODULE_PATH)
hardware = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hardware
SPEC.loader.exec_module(hardware)


def settings(**overrides):
    values = {
        "enabled": True,
        "profile": "balanced",
        "requested_batch": 1,
        "requested_block": 1024,
        "requested_output_gib": 2.0,
        "max_auto_batch": 16,
        "max_auto_block": 3072,
        "max_auto_output_gib": 16.0,
        "task_memory_gib": 18.0,
        "min_free_system_gib": 8.0,
        "system_mem_available_gib": 23.0,
        "cuda_mem_free_gib": 14.5,
        "cuda_mem_total_gib": 15.5,
        "use_cuda": True,
    }
    values.update(overrides)
    return hardware.choose_gigatime_hardware_settings(**values)


class GigaTIMEHardwarePolicyTest(unittest.TestCase):
    def test_cuda_batch_uses_vram_tier_instead_of_host_ram_tier(self):
        result = settings()
        self.assertEqual(result["derived_caps"]["memory_batch"], 4)
        self.assertEqual(result["derived_caps"]["gpu_batch"], 8)
        self.assertEqual(result["effective"]["batch_size"], 8)
        self.assertEqual(result["effective"]["block_size"], 1024)

    def test_cpu_batch_remains_limited_by_host_memory(self):
        result = settings(use_cuda=False, cuda_mem_free_gib=None, cuda_mem_total_gib=None)
        self.assertEqual(result["effective"]["batch_size"], 4)

    def test_auto_caps_remain_hard_limits(self):
        result = settings(max_auto_batch=4, max_auto_block=768)
        self.assertEqual(result["effective"]["batch_size"], 4)
        self.assertEqual(result["effective"]["block_size"], 768)

    def test_unknown_cuda_memory_keeps_conservative_requested_batch(self):
        result = settings(cuda_mem_free_gib=None, cuda_mem_total_gib=None)
        self.assertEqual(result["effective"]["batch_size"], 1)

    def test_low_available_ram_fails_before_inference(self):
        with self.assertRaisesRegex(RuntimeError, "cannot run safely"):
            settings(system_mem_available_gib=10.0)


if __name__ == "__main__":
    unittest.main()
