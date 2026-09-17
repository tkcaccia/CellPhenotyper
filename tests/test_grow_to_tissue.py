from pathlib import Path
import sys

import numpy as np
import tifffile


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import grow_to_tissue  # noqa: E402


def test_downsampled_grow_resamples_tissue_from_independent_grid(
    tmp_path: Path, monkeypatch
) -> None:
    seeds = np.zeros((32, 40), dtype=np.uint8)
    seeds[8:16, 8:16] = 1
    seeds[20:28, 24:32] = 2
    tissue = np.ones((9, 11), dtype=np.uint8)

    seed_path = tmp_path / "seeds.tif"
    tissue_path = tmp_path / "tissue.tif"
    output_path = tmp_path / "grown.tif"
    tifffile.imwrite(seed_path, seeds)
    tifffile.imwrite(tissue_path, tissue)

    monkeypatch.setattr(
        grow_to_tissue,
        "improve_tissue_mask",
        lambda tissue_bool, **_: tissue_bool,
    )
    grown, fraction = grow_to_tissue.grow_downsampled_to_fullres(
        seed_path=str(seed_path),
        tissue_path=str(tissue_path),
        out_flat_tif=str(output_path),
        factor=4,
        block_rows=8,
        min_seed_area=0,
        restrict_to_seeded_components_flag=False,
        fill_holes_area=0,
        close_radius=0,
        mpp_x=0.25,
        mpp_y=0.25,
    )

    assert grown.shape == seeds.shape
    assert set(np.unique(grown)) == {1, 2}
    assert fraction == 1.0
