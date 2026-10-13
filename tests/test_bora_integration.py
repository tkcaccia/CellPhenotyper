from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile


yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
REVISION = "00d9cc5d3d06cc4d55a5ec2e71676d7f7f52c92e"
PARAMS = yaml.safe_load((ROOT / "pipeline_paramers.yml").read_text(encoding="utf-8"))
SCHEMA = json.loads((ROOT / "nextflow_schema.json").read_text(encoding="utf-8"))
CONFIG = (ROOT / "nextflow.config").read_text(encoding="utf-8")
MODULE = (ROOT / "modules" / "refine_with_bora.nf").read_text(encoding="utf-8")
SPATIAL = (ROOT / "subworkflows" / "post_grow_spatial_outputs.nf").read_text(
    encoding="utf-8"
)
FINALIZER = (ROOT / "bin" / "finalize_bora_refinement.py").read_text(
    encoding="utf-8"
)
SUPPORT_CONTRACT = (ROOT / "bin" / "bora_support_contract.py").read_text(
    encoding="utf-8"
)
sys.path.insert(0, str(ROOT / "bin"))

from bora_support_contract import verify_support  # noqa: E402
from ome_tiff_metadata import restore_ome_physical_size  # noqa: E402


class _ArrayReader:
    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array)

    def spatial_shape(self) -> tuple[int, int]:
        return self.array.shape

    def read(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        return self.array[y0:y1, x0:x1]


def test_bora_is_the_pinned_production_default() -> None:
    assert PARAMS["grown_tissue_refine_method"] == "bora"
    assert PARAMS["bora_revision"] == REVISION
    assert PARAMS["bora_outer_dilation"] == 0
    assert PARAMS["bora_smooth_radius"] == 0
    assert PARAMS["bora_annealed_wand"] is True
    assert PARAMS["bora_watershed_resolution"] == "native"
    assert PARAMS["bora_geojson_profile"] == "accurate"
    assert f"bora_revision                  = '{REVISION}'" in CONFIG


def test_schema_exposes_the_same_bora_contract() -> None:
    properties = SCHEMA["definitions"]["representation_options"]["properties"]
    assert properties["grown_tissue_refine_method"]["default"] == "bora"
    assert properties["bora_revision"]["default"] == REVISION
    assert properties["bora_revision"]["pattern"] == "^[0-9a-f]{40}$"
    assert properties["bora_pyramid"]["const"] is True


def test_bora_replaces_refinement_and_vectorization_in_one_process() -> None:
    assert "python -m bora.cli refine" in MODULE
    assert "--geojson" in MODULE
    assert "--report" in MODULE
    assert "finalize_bora_refinement.py" in CONFIG
    assert "REFINE_WITH_BORA(refineInputCh, runtime_plan)" in SPATIAL
    assert "MASK_TO_GEOJSON" in SPATIAL  # explicit non-Bora fallback only
    assert "grownRefineMethod == 'bora'" in SPATIAL
    assert '--shift-json "${resolution_json}"' in MODULE
    assert "rebase_geojson_file" in FINALIZER


def test_adapter_fails_closed_on_support_leakage() -> None:
    assert "new_foreground_outside_baseline_pixels" in SUPPORT_CONTRACT
    assert "baseline_outside_scaled_tissue_pixels" in SUPPORT_CONTRACT
    assert "foreground_on_confirmed_artifact_outlier_pixels" in SUPPORT_CONTRACT
    assert "raise RuntimeError" in SUPPORT_CONTRACT
    assert '"refinement_engine": "Bora"' in FINALIZER
    assert "cellphenotyper.bora_geojson_provenance.v1" in FINALIZER


def test_scaled_tissue_disagreement_is_diagnostic_but_new_growth_fails() -> None:
    baseline = np.array([[1, 1, 0], [0, 2, 0]], dtype=np.uint16)
    final = baseline.copy()
    tissue = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
    uncertainty = np.zeros_like(baseline, dtype=np.uint8)
    counts = verify_support(
        _ArrayReader(final),
        _ArrayReader(baseline),
        _ArrayReader(tissue),
        _ArrayReader(uncertainty),
        block_rows=1,
    )
    assert counts["baseline_outside_scaled_tissue_pixels"] == 1
    assert counts["new_foreground_outside_baseline_pixels"] == 0

    grown = final.copy()
    grown[0, 2] = 1
    with pytest.raises(RuntimeError, match="outside the upstream cluster-label footprint"):
        verify_support(
            _ArrayReader(grown),
            _ArrayReader(baseline),
            _ArrayReader(tissue),
            _ArrayReader(uncertainty),
            block_rows=2,
        )


def test_bora_ome_metadata_repair_preserves_pixels(tmp_path: Path) -> None:
    path = tmp_path / "bora.ome.tif"
    labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)
    tifffile.imwrite(path, labels, ome=True, metadata={"axes": "YX"})
    before = tifffile.imread(path)
    result = restore_ome_physical_size(path, mpp_x=0.25, mpp_y=0.26)
    after = tifffile.imread(path)
    assert np.array_equal(before, after)
    assert result["physical_size_x_um"] == pytest.approx(0.25)
    assert result["physical_size_y_um"] == pytest.approx(0.26)
    assert result["repair"] == "ImageDescription-only; pixel tiles unchanged"


@pytest.mark.parametrize(
    "container_file",
    [
        "docker/Dockerfile.full.cpu",
        "docker/Dockerfile.full.gpu",
        "docker/Dockerfile.runtime-update.gpu",
        "docker/Dockerfile.source-refresh.gpu",
        "singularity/cellphenotyper_full_cpu.def",
        "singularity/cellphenotyper_full_gpu.def",
    ],
)
def test_release_runtimes_pin_bora_revision(container_file: str) -> None:
    text = (ROOT / container_file).read_text(encoding="utf-8")
    assert REVISION in text
    assert "tkcaccia/Bora" in text
