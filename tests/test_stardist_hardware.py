import ast
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[1]


def load_block_candidate_function():
    source = (ROOT / "bin" / "run_stardist_roi_segmentation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_big_block_candidates"
    )
    namespace = {"List": List, "Optional": Optional}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<stardist-block-policy>", "exec"), namespace)
    return namespace["build_big_block_candidates"]


def test_gpu_memory_cap_does_not_reinsert_unsafe_requested_block() -> None:
    candidates = load_block_candidate_function()(4096, memory_budget_gb=10.0, gpu_memory_gib=15.47)
    assert candidates[0] == 2048
    assert 4096 not in candidates
    assert candidates == sorted(set(candidates), reverse=True)


def test_unconstrained_block_keeps_requested_size() -> None:
    candidates = load_block_candidate_function()(4096)
    assert candidates[0] == 4096
