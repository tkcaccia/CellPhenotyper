"""Source-bound CellViT profile/reference/SpatialData integration, no inference.

Run in the optional SpatialData runtime. The existing atlas runtime creates
tiny synthetic producer bundles, canonical profiles and a frozen atlas; this
runtime uses the real SpatialData writer and reader without importing models.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

sd = pytest.importorskip("spatialdata")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import export_spatialdata as exporter
from cell_profile_io import sha256_file
from reference_mapping_io import load_reference_mapping


PREPARE = r'''
import json
import sys
from pathlib import Path
root, directory = map(Path, sys.argv[1:])
sys.path[:0] = [str(root / "bin"), str(root / "tests")]
from test_cellvit_profile_binding import bound_profile_fixture
import build_cell_profiles as profiles
from cell_reference_atlas import build_atlas, map_profile
from reference_mapping_io import write_reference_mapping

sources = []
for sample in ("synthetic_reference_A", "synthetic_reference_B", "synthetic_query"):
    args, _, _, _, objects = bound_profile_fixture(directory / sample)
    args.sample_id = sample
    objects["reviewed_label"] = "synthetic_contract_group_not_biological_annotation"
    objects.to_csv(args.objects, index=False)
    profiles.build_profiles(args)
    sources.append(args)
atlas_dir = directory / "frozen_atlas"
atlas = build_atlas([args.outdir for args in sources[:2]], atlas_dir,
                   "synthetic-cellvit-contract-v1", ["cellvit"], label_column="reviewed_label")
assert atlas["reference_observation_count"] == 4
assert atlas["excluded_missing_features"] == 2
query = sources[2]
output = directory / "mapping" / "reference_assignments.csv"
mapping = write_reference_mapping(query.outdir, atlas_dir, output, map_profile)
assert mapping.reference_status.tolist() == ["assigned", "missing_features", "assigned"]
tissue = directory / "synthetic_crop_outline.geojson"
tissue.write_text(json.dumps({"type": "FeatureCollection", "features": [{
    "type": "Feature", "properties": {"value": 1, "source": "synthetic rectangular fixture"},
    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [12, 0], [12, 8], [0, 8], [0, 0]]]}}]}))
inputs = {key: str(getattr(query, key)) for key in ("image", "labels", "shift", "resolution_json")}
inputs.update(profile_dir=query.outdir, tissue_geojson=str(tissue), tissue_coordinates="crop_pixels",
              cell_reference_mapping=str(output.with_suffix(".mapping.json")), expected_sample_id=query.sample_id)
(directory / "inputs.json").write_text(json.dumps(inputs))
print(json.dumps({"used_model": False, "synthetic_graph_vectors": True, "canonical_cells": 3,
                  "reference_cells": 4, "mapping_status": mapping.reference_status.tolist()}))
'''


def test_bound_cellvit_profile_reference_and_real_spatialdata_roundtrip(tmp_path):
    atlas_python = os.environ.get("CELLPHENOTYPER_ATLAS_TEST_PYTHON") or str(ROOT / ".venv-spatial/bin/python")
    if not Path(atlas_python).is_file():
        pytest.skip("A separate atlas test Python is required for profile/reference construction")
    result = subprocess.run([atlas_python, "-c", PREPARE, str(ROOT), str(tmp_path)],
        capture_output=True, text=True, timeout=45,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    execution = json.loads(result.stdout.strip().splitlines()[-1])
    assert execution["used_model"] is False and execution["synthetic_graph_vectors"] is True
    inputs = json.loads((tmp_path / "inputs.json").read_text())
    profile = Path(inputs["profile_dir"])
    manifest = json.loads((profile / "cell_profiles_manifest.json").read_text())
    cells = pd.read_parquet(profile / "cell_profiles.parquet")
    block = manifest["feature_blocks"]["cellvit"]
    features = np.load(profile / block["path"], allow_pickle=False)
    receipt = Path(inputs["cell_reference_mapping"])
    expected_mapping, mapping_record, mapping_files = load_reference_mapping(receipt, profile, expected_unit="cell")
    source_hashes = {path: sha256_file(path) for path in tmp_path.rglob("*") if path.is_file()}

    output = tmp_path / "cellvit_reference.zarr"
    summary = exporter.export_spatialdata(**inputs, outdir=output, tile_size=16, workers=1)
    package = sd.SpatialData.read(output)
    table = package.tables["cells"]
    assert summary["versions"]["spatialdata"] == sd.__version__
    assert summary["canonical_cell_count"] == table.n_obs == 3
    assert table.obs_names.tolist() == cells.cell_uid.tolist()
    assert table.obs.cell_id.astype(str).tolist() == ["2", "1", "3"]
    assert table.obs.instance_id.tolist() == [2, 1, 3]
    assert table.obs.cellvitpp_id.astype(str).tolist() == ["007", "", "NA"]
    assert table.obs.cellvit_available.tolist() == [True, False, True]
    assert table.obs.cellvit_status.astype(str).tolist() == [
        "available_verified_reference_definition", "missing_no_cellvit_source",
        "available_verified_reference_definition"]
    assert set(table.obsm) == {"cellvit", "spatial"}
    assert table.obsm["cellvit"].dtype == np.float32
    np.testing.assert_array_equal(table.obsm["cellvit"], features)
    np.testing.assert_array_equal(features[[0, 2]], np.arange(8, dtype=np.float32).reshape(2, 4))
    assert np.isnan(table.obsm["cellvit"][1]).all()
    for column in ("x_crop_px", "y_crop_px", "x_um", "y_um", "cellvitpp_x_px", "cellvitpp_y_px"):
        np.testing.assert_array_equal(table.obs[column], cells[column])
    assert table.obs.x_crop_px.iloc[0] == 8 and table.obs.cellvitpp_x_px.iloc[0] == 9
    assert table.obs.cellvitpp_x_px.dtype == table.obs.cellvitpp_y_px.dtype == np.float64
    assert np.isnan(table.obs.cellvitpp_x_px.iloc[1]) and np.isnan(table.obs.cellvitpp_y_px.iloc[1])
    np.testing.assert_array_equal(table.obsm["spatial"], cells[["x_um", "y_um"]])
    np.testing.assert_array_equal(package.images["he_image"].data.compute(),
                                  np.moveaxis(tifffile.imread(inputs["image"]), -1, 0))
    np.testing.assert_array_equal(package.labels["canonical_cells"].data.compute(), tifffile.imread(inputs["labels"]))
    from spatialdata.transformations import get_transformation
    transform = get_transformation(package.labels["canonical_cells"], to_coordinate_system="original_um")
    np.testing.assert_array_equal(transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
                                  [[.5, 0, 50], [0, .5, 100], [0, 0, 1]])

    stored_manifest = json.loads(table.uns["cellphenotyper"]["profile_manifest_json"])
    assert stored_manifest == manifest
    stored_block = stored_manifest["feature_blocks"]["cellvit"]
    assert stored_block["reference_compatible"] is True
    assert stored_block["source_binding"]["status"] == "verified_exact_inputs_and_population"
    assert stored_block["canonical_correspondence"]["status"] == "verified_source_ids_detector_centroids_and_population"
    assert stored_block["canonical_correspondence"]["canonical_fused_centroid_used_for_matching"] is False
    assert stored_block["source_binding"]["receipt"]["execution"]["used_model"] is False
    assert stored_block["feature_definition"] == stored_block["source_binding"]["receipt"]["feature_definition"]
    assert stored_block["feature_definition"]["source_mpp"] == .5
    assert stored_block["feature_definition"]["amp"] is False
    assert stored_block["feature_definition"]["preprocessing"]["channel_conversion"] == "RGB unchanged"
    assert stored_block["feature_definition"]["runtime"]["entry_point"] == "synthetic_fixture_only"
    retained_sha = stored_block["source_binding"]["retained_population_sha256"]
    assert table.obs.cellvitpp_source_sha256.astype(str).tolist() == [retained_sha, "", retained_sha]
    assert stored_block["source_binding"]["receipt"]["execution"]["inputs"]["image"]["sha256"] == sha256_file(inputs["image"])
    assert package.attrs["cellphenotyper"]["canonical_raster_lineage"]["status"] == "verified_image_and_labels"

    assert table.obs.reference_status.astype(str).tolist() == ["assigned", "missing_features", "assigned"]
    for name in exporter.REFERENCE_FIELDS:
        expected = expected_mapping[name].to_numpy()
        actual = table.obs[name].to_numpy()
        if expected.dtype.kind in "fiu":
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual.tolist() == expected.tolist()
    portable = json.loads(table.uns["cellphenotyper"]["reference_mapping_json"])
    assert portable == summary["reference_mappings"]["cell"] == package.attrs["cellphenotyper"]["reference_mappings"]["cell"]
    assert portable["receipt"] == mapping_record
    assert portable["receipt_json"] == receipt.read_text()
    assert portable["atlas_manifest_json"] == mapping_files["atlas_manifest"].read_text()
    atlas = json.loads(portable["atlas_manifest_json"])
    assert atlas["feature_schemas"]["cellvit"]["feature_definition"] == block["feature_definition"]
    assert table.obs.reference_atlas_id.astype(str).eq(atlas["atlas_id"]).all()
    for key, path in mapping_files.items():
        assert portable["artifacts"][key] == {"filename": path.name, "sha256": sha256_file(path),
                                              "size_bytes": path.stat().st_size}
    assert all(sha256_file(path) == digest for path, digest in source_hashes.items())
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) < 10_000_000
