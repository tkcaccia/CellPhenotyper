"""Display enhancement must never become a marker or QC transformation."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from gigatime_display import tone_map_dapi_preview


def test_tone_map_is_global_monotone_and_mask_independent():
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    original = raw.copy()
    shown, config = tone_map_dapi_preview(raw)
    assert np.array_equal(raw, original)
    assert shown.shape == raw.shape
    assert np.all(np.diff(shown.reshape(-1).astype(int)) >= 0)
    assert shown[8, 0] < raw[8, 0]  # less washed-out midtones
    assert config["native_scores_and_qc_unchanged"]
    assert config["applied"]


def test_dapi_exporter_only_changes_full_image_display(tmp_path):
    pytest.importorskip("torch")
    from run_gigatime_on_crop import JpegTileExporter

    grid = np.tile(np.arange(32, 224, dtype=np.uint8), (192, 1))
    raw_probs = (grid.astype(np.float32) / 255.0)[None]
    before = raw_probs.copy()
    exporter = JpegTileExporter(
        outdir=tmp_path, full_shape_yx=(192, 192), tile_size=192,
        quality=95, preview_max_side=192, save_tiles=True,
        channel_indices=[0], channel_names=["DAPI"],
    )
    exporter.save_tile(0, 192, 0, 192, raw_probs)
    exporter.write_outputs({"test": True})
    manifest = json.loads((tmp_path / "gigatime_jpg_manifest.json").read_text())
    assert manifest["full_image_preview_display_transforms"]["DAPI"]["applied"]
    assert np.array_equal(raw_probs, before)
    tile = np.asarray(Image.open(tmp_path / "jpg_tiles/DAPI/y000000_x000000.jpg"))
    preview = np.asarray(Image.open(tmp_path / "jpg_image/DAPI.jpg"))
    assert preview.mean() < tile.mean()
