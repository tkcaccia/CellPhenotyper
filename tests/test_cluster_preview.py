from pathlib import Path
import sys

import numpy as np
import tifffile


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import labels_to_cluster_mask  # noqa: E402


def test_preview_uses_tiled_tiff_fallback_when_memmap_fails(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "background.tif"
    tifffile.imwrite(path, np.zeros((8, 10, 3), dtype=np.uint8), photometric="rgb")
    expected = np.full((4, 5, 3), [21, 42, 84], dtype=np.uint8)

    monkeypatch.setattr(
        labels_to_cluster_mask.tiff,
        "memmap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("not memory-mappable")),
    )
    monkeypatch.setattr(
        labels_to_cluster_mask,
        "read_tiled_tiff_preview",
        lambda *_args, **_kwargs: expected.copy(),
    )

    observed = labels_to_cluster_mask.read_preview_background(
        str(path),
        expected_shape=(8, 10),
        downsample_factor=2,
        allow_full_read=False,
    )

    np.testing.assert_array_equal(observed, expected)
