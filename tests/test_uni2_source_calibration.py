"""CPU-only regression for verified UNI2 source calibration on every NF route."""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from cell_profile_io import RasterReader


def source_geometry_functions():
    """Run the real extractor's pure geometry functions without loading torch."""
    names = {"_safe_float", "read_source_mpp", "infer_source_mpp", "read_resolution_json_mpp", "resolve_extraction_tile_size"}
    source = ast.parse((ROOT / "bin/extract_uni2_embeddings.py").read_text())
    definitions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    future = ast.parse("from __future__ import annotations").body[0]
    namespace = {"json": json, "Path": Path, "tifffile": tifffile, "re": re}
    exec(compile(ast.Module(body=[future] + definitions, type_ignores=[]), "extractor_geometry", "exec"), namespace)
    return namespace


def resolution_preflight(module_name):
    content = (ROOT / "modules" / module_name).read_text()
    match = re.search(r"python -c '([^']+)' [\"']\$\{resolution_(?:file|json)\}[\"']", content)
    assert match, "Every extraction route must validate the passed report before loading a model"
    return match.group(1)


@pytest.mark.parametrize("module_name", ["extract_uni2_embeddings.nf", "extract_uni2_embeddings_shared.nf", "build_uni2_spatial_grid.nf"])
def test_real_bad_tiff_metadata_is_overridden_before_patch_geometry(tmp_path, module_name):
    # Reproduce the real slide's 10 pixels/cm TIFF tags, but use a tiny native
    # image. These tags imply 1000 um/px; the verified override is ~0.274 um/px.
    source = tmp_path / "crop.tif"
    pixels = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
    tifffile.imwrite(source, pixels, photometric="rgb", resolution=(10, 10), resolutionunit="CENTIMETER", tile=(32, 32))
    mpp = 0.273774374855905
    report = tmp_path / "resolution.json"
    report.write_text(json.dumps({"status": "pass", "mpp_x": mpp, "mpp_y": mpp, "effective_mpp": mpp}))
    completed = subprocess.run([sys.executable, "-O", "-c", resolution_preflight(module_name), str(report)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    geometry = source_geometry_functions()
    assert geometry["infer_source_mpp"](str(source)) == 1000.
    assert geometry["resolve_extraction_tile_size"](256, 1000., .25)[0] == 1
    verified_mpp = geometry["read_resolution_json_mpp"](report)
    size, effective = geometry["resolve_extraction_tile_size"](256, verified_mpp, .25)
    assert verified_mpp == mpp and size == 234
    assert effective == pytest.approx(size * mpp / 256)
    with RasterReader(source) as image:
        patch = image.window(100, 100, 100 + size, 100 + size)
    np.testing.assert_array_equal(patch, pixels[100:334, 100:334])
    assert patch.shape == (234, 234, 3)
    # A 64-um intended field of view, with only native-pixel rounding error.
    assert abs(size * verified_mpp - 256 * .25) <= verified_mpp / 2


@pytest.mark.parametrize("module_name", ["extract_uni2_embeddings.nf", "extract_uni2_embeddings_shared.nf", "build_uni2_spatial_grid.nf"])
@pytest.mark.parametrize("change", [
    {"status": "fail"}, {"mpp_x": 1000, "mpp_y": 1000}, {"mpp_x": 0},
    {"mpp_y": .75}, {"mpp_x": None}, {"mpp_y": float("nan")},
    {"source_mpp": 1000}, {"source_mpp_x": .25}, {"effective_mpp": .75},
])
def test_module_preflight_rejects_missing_failed_or_conflicting_calibration(tmp_path, module_name, change):
    report = tmp_path / "resolution.json"
    report.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5, **change}))
    completed = subprocess.run([sys.executable, "-O", "-c", resolution_preflight(module_name), str(report)], capture_output=True, text=True)
    assert completed.returncode != 0


def test_all_primary_and_auxiliary_routes_carry_sample_keyed_verified_reports():
    primary = (ROOT / "subworkflows/extract_primary_uni2.nf").read_text()
    auxiliary = (ROOT / "subworkflows/run_auxiliary_cell_uni2.nf").read_text()
    route = (ROOT / "subworkflows/run_auxiliary_cell_route.nf").read_text()
    main = (ROOT / "main.nf").read_text()
    profiles = (ROOT / "subworkflows/run_cell_profile_atlas.nf").read_text()
    grid = (ROOT / 'subworkflows/prepare_uni2_spatial_grid.nf').read_text()
    assert 'PREPARE_UNI2_SPATIAL_GRID(image_input_ch, crop_roi_ch, tissue_mask_ch, art_ch, converted_resolution_report_ch,' in main
    assert 'resolution_report_ch' in grid and 'shift_json_ch' not in grid
    assert "empty_uni2_resolution" not in primary + auxiliary
    assert primary.count(".join(strictJoin, resolution_ch)") == 6
    assert ".join(strictJoin, resolution_ch)" in auxiliary
    assert "failOnDuplicate: true, failOnMismatch: true" in primary and "failOnDuplicate: true, failOnMismatch: true" in auxiliary
    assert "'cell', placeholder_observations_file, resolution_json" in primary
    assert "'grid', grid_objects_csv, resolution_json" in primary
    for mode in ("tile", "cyto", "inner_square", "nuclei"):
        assert "resolution_json, '" + mode + "'" in primary
    assert "'cell', empty_observations, resolution_json" in auxiliary
    assert "verified_resolution_ch," in route
    assert "grid_metadata_ch, converted_resolution_report_ch," in main
    assert "shift_json_ch, converted_resolution_report_ch, cyto_mask_ch," in main
    assert "primary_tile_ch, resolution_ch, flags.run_uni2, false" in profiles
    for module_name in ("extract_uni2_embeddings.nf", "extract_uni2_embeddings_shared.nf"):
        module = (ROOT / "modules" / module_name).read_text()
        assert 'path(resolution_file)' in module
        assert '--resolution-json "${resolution_file}"' in module
        assert module.index('requires a passed finite isotropic') < module.index('TOKEN_ENV_FILE=')
