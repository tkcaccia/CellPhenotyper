from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from run_pathsegmentor_semantics import axis_starts, blend_window, load_prompt_panel  # noqa: E402


def test_breast_prompt_panel_is_fixed_valid_and_hierarchical() -> None:
    panel = load_prompt_panel(ROOT / "resources/pathsegmentor_breast_prompts.json")
    assert panel["organ"] == "breast"
    assert len(panel["prompts"]) == 6
    assert len({record["id"] for record in panel["prompts"]}) == 6
    assert all("-level " in record["text"] for record in panel["prompts"])
    assert all(record["text"].endswith("in breast pathology") for record in panel["prompts"])


def test_prompt_panel_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"prompts": [
        {"id": "tissue_tumor", "text": "a"},
        {"id": "tissue_tumor", "text": "b"},
    ]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_prompt_panel(path)


@pytest.mark.parametrize(
    ("length", "window", "stride", "expected"),
    [(100, 200, 50, [0]), (100, 40, 30, [0, 30, 60]), (101, 40, 30, [0, 30, 60, 61])],
)
def test_axis_starts_covers_image_edge(length, window, stride, expected) -> None:
    assert axis_starts(length, window, stride) == expected


def test_blend_window_has_no_zero_weight_seams() -> None:
    window = blend_window(32, 48)
    assert window.shape == (32, 48)
    assert window.dtype == np.float32
    assert np.all(window >= 0.05)
    assert window.max() <= 1.0
