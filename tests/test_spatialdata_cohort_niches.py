"""Real cohort fitting in the CPU runtime, then model-free SpatialData export."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sd = pytest.importorskip("spatialdata")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import cohort_niche_io as cohort_io
import export_spatialdata as exporter
from cell_profile_io import sha256_file


# Fit only seven cells per specimen in a separate, existing CPU environment.
# The export environment itself has neither sklearn nor Torch installed.
BUILD_FIXTURE = r'''
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
root, destination = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "bin"))
from build_cell_profiles import build_profiles, parser
from assemble_spatial_cell_profiles import assemble
from fit_cohort_niches import fit_cohort
from cell_profile_io import sha256_file
profiles = []
for sample in ("A", "B"):
    folder = destination / sample
    folder.mkdir(parents=True)
    xy = np.array([[4,4],[5,4],[4,5],[32,4],[33,4],[32,5],[21,4]])
    ids = ["007", "2", "3", "4", "5", "6", "8"]
    labels = np.zeros((12,40), np.uint16)
    for name,(x,y) in zip(ids,xy): labels[y,x] = int(name)
    support = np.ones((12,40), np.uint8); support[:,20:24] = 0
    image = np.arange(12*40*3, dtype=np.uint16).reshape(12,40,3).astype(np.uint8)
    for name,value in (("image",image),("labels",labels),("support",support)):
        tifffile.imwrite(folder / (name+".tif"), value)
    pd.DataFrame({"label":ids, "x":xy[:,0], "y":xy[:,1],
        "xmin":xy[:,0], "ymin":xy[:,1], "xmax":xy[:,0]+1, "ymax":xy[:,1]+1}).to_csv(folder/"objects.csv",index=False)
    (folder/"shift.json").write_text(json.dumps({"crop_size":{"width":40,"height":12},
        "offset_crop_to_original":{"dx":100,"dy":200}}))
    (folder/"resolution.json").write_text(json.dumps({"status":"pass","mpp_x":.5,"mpp_y":.5}))
    base = folder/"base"
    args = ["--sample-id",sample,"--outdir",str(base)]
    for name,flag in (("objects.csv","--objects"),("image.tif","--image"),("labels.tif","--labels"),
        ("support.tif","--tissue-mask"),("shift.json","--shift"),("resolution.json","--resolution-json")):
        args += [flag,str(folder/name)]
    cells,manifest = build_profiles(parser().parse_args(args))
    # Explicit synthetic fixture, not a claimed learned-model representation.
    values = np.array([[0.],[1.],[0.],[100.],[101.],[100.],[np.nan]],dtype=np.float64)
    path = base/"feature_blocks/synthetic.npy"
    np.save(path,values,allow_pickle=False)
    manifest["feature_blocks"]["synthetic"] = {"path":"feature_blocks/synthetic.npy",
        "sha256":sha256_file(path),"shape":list(values.shape),"dtype":"float64",
        "feature_names":["fixture_feature"],"reference_compatible":True,
        "feature_definition":{"method":"synthetic_test_fixture_not_learned"}}
    # Preserve original reference and predicted-marker fields through attachment.
    cells["reference_assignment"] = ["original_group"]*6 + ["unmatched"]
    cells["reference_status"] = ["assigned"]*6 + ["missing_features"]
    cells["predicted__nucleus__CD3__mean"] = np.array([.1,.2,.3,.4,.5,.6,np.nan],dtype=np.float64)
    cells.to_csv(base/"cell_profiles.csv",index=False)
    cells.to_parquet(base/"cell_profiles.parquet",index=False)
    for name in ("cell_profiles.csv","cell_profiles.parquet"):
        manifest["files"][name] = sha256_file(base/name)
    (base/"cell_profiles_manifest.json").write_text(json.dumps(manifest))
    profile = folder/"cell_profiles"
    assemble(base,folder/"support.tif",folder/"shift.json",folder/"resolution.json",profile,
        radii_um=(3.,),feature_groups=("synthetic",),repeats=2)
    profiles.append(profile)
    (folder/"domains.geojson").write_text(json.dumps({"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{"value":1,"classification":{"name":"tissue"}},
         "geometry":{"type":"Polygon","coordinates":[[[0,0],[40,0],[40,12],[0,12],[0,0]]]}}]}))
fit_cohort(profiles,destination/"cohort",fixed_k=2,repeats=2)
'''


@pytest.fixture(scope="module")
def fitted_template(tmp_path_factory):
    destination = tmp_path_factory.mktemp("cohort_export_template")
    python = ROOT / ".venv-spatial/bin/python"
    if not python.is_file():
        pytest.skip("Existing CPU cohort fitting environment is unavailable")
    result = subprocess.run([str(python), "-c", BUILD_FIXTURE, str(ROOT), str(destination)],
        capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    return destination


@pytest.fixture
def cohort_case(fitted_template, tmp_path):
    return Path(shutil.copytree(fitted_template, tmp_path / "fixture"))


def inputs(case, sample="A"):
    folder = case / sample
    return dict(profile_dir=folder / "cell_profiles", image=folder / "image.tif",
        labels=folder / "labels.tif", shift=folder / "shift.json", resolution_json=folder / "resolution.json",
        tissue_geojson=folder / "domains.geojson", tissue_coordinates="crop_pixels", tile_size=16, workers=1)


def read_cohort(case, sample="A"):
    return cohort_io.load_cohort_bundle(case / "cohort", profile_dir=case / sample / "cell_profiles", sample_id=sample)


def assert_cohort(table, expected, record):
    assert table.obs_names.tolist() == expected.cell_uid.tolist()
    values = expected[cohort_io.COHORT_COLUMNS].copy()
    values.index = table.obs.index
    safe = exporter._safe_obs(values)
    for column in cohort_io.COHORT_COLUMNS:
        pd.testing.assert_series_equal(table.obs[column], safe[column], check_exact=True)
    assert json.loads(table.uns["cellphenotyper"]["cohort_niches_json"]) == record


def test_actual_cohort_attachment_is_exact_additive_and_portable(cohort_case, tmp_path):
    args = inputs(cohort_case)
    expected, record = read_cohort(cohort_case)
    assert expected.cell_id.tolist() == ["007", "2", "3", "4", "5", "6", "8"]
    assert expected.cohort_niche_id.dtype == pd.Int64Dtype()
    assert expected.cohort_niche_id.isna().tolist() == [False]*6 + [True]
    assert expected.cohort_niche_status.iloc[-1] == "outside_tissue_support"
    before = {p: sha256_file(p) for p in cohort_case.rglob("*") if p.is_file()}
    original = pd.read_parquet(args["profile_dir"] / "cell_profiles.parquet")
    output = tmp_path / "cohort.zarr"
    summary = exporter.export_spatialdata(**args, cohort_niches=cohort_case / "cohort", outdir=output)
    loaded = sd.SpatialData.read(output)
    table = loaded.tables["cells"]
    assert_cohort(table, expected, record)
    assert summary["cohort_niches"] == loaded.attrs["cellphenotyper"]["cohort_niches"] == record
    assert summary["reference_mappings"] == {}  # Existing source fields are not a new reference receipt.
    mapping = json.loads(table.uns["cellphenotyper"]["field_name_mapping_json"])
    safe_original = exporter._safe_obs(original.set_axis(table.obs.index))
    for name in original:
        pd.testing.assert_series_equal(table.obs[mapping["obs"][name]], safe_original[name],
                                       check_names=False, check_exact=True)
    assert set(table.obs) == {*mapping["obs"].values(), "spatial_region", "instance_id"}
    np.testing.assert_array_equal(table.obsm["synthetic"], np.load(args["profile_dir"] / "feature_blocks/synthetic.npy"))
    np.testing.assert_array_equal(table.obsp["neighbors_3um"].toarray(),
        sparse.load_npz(args["profile_dir"] / "neighborhood_graph_3um.npz").toarray())
    assert table.obs.instance_id.tolist() == [7,2,3,4,5,6,8]
    import tifffile
    np.testing.assert_array_equal(loaded.labels["canonical_cells"].data.compute(), tifffile.imread(args["labels"]))
    np.testing.assert_array_equal(loaded.images["he_image"].data.compute(), np.moveaxis(tifffile.imread(args["image"]), -1, 0))
    from spatialdata.transformations import get_transformation
    transform = get_transformation(loaded.labels["canonical_cells"], to_coordinate_system=exporter.COORDINATE_SYSTEM)
    np.testing.assert_array_equal(transform.to_affine_matrix(input_axes=("x","y"), output_axes=("x","y")),
        [[.5,0,50],[0,.5,100],[0,0,1]])
    assert all(sha256_file(path) == digest for path,digest in before.items())
    # Nothing in exported cohort provenance requires the fit/profile directories.
    cohort_case.rename(tmp_path / "unmounted_source_directories")
    assert_cohort(sd.SpatialData.read(output).tables["cells"], expected, record)


def test_default_export_keeps_original_niches_without_cohort_attachment(cohort_case, tmp_path):
    args = inputs(cohort_case)
    original = pd.read_parquet(args["profile_dir"] / "cell_profiles.parquet")
    output = tmp_path / "plain.zarr"
    summary = exporter.export_spatialdata(**args, outdir=output)
    table = sd.SpatialData.read(output).tables["cells"]
    assert summary["cohort_niches"] == {"status":"not_provided"}
    assert not set(cohort_io.COHORT_COLUMNS) & set(table.obs)
    assert "cohort_niches_json" not in table.uns["cellphenotyper"]
    np.testing.assert_array_equal(table.obs.niche_id, original.niche_id)


@pytest.mark.parametrize("source", ["cohort/cohort_niche_assignments.parquet", "A/cell_profiles/cell_profiles.parquet"])
def test_stale_bundle_or_current_profile_fails_before_write(cohort_case, tmp_path, source):
    path = cohort_case / source
    frame = pd.read_parquet(path)
    name = "cohort_niche_centroid_margin" if source.startswith("cohort/") else "predicted__nucleus__CD3__mean"
    frame.loc[0, name] = np.nextafter(frame.loc[0, name], np.inf)
    frame.to_parquet(path, index=False)
    output = tmp_path / "stale.zarr"
    with pytest.raises(ValueError, match="(?i)sha256|hash"):
        exporter.export_spatialdata(**inputs(cohort_case), cohort_niches=cohort_case / "cohort", outdir=output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["reverse", "foreign", "extra"])
def test_consumer_never_reorders_or_silently_accepts_extra_assignments(cohort_case, tmp_path, monkeypatch, change):
    original = cohort_io.load_cohort_bundle
    def changed(*args, **kwargs):
        frame,record = original(*args, **kwargs)
        if change == "reverse": frame = frame.iloc[::-1].copy()
        elif change == "foreign": frame.loc[0, "cell_uid"] = "foreign:uid"
        else: frame["unexpected_score"] = 1.
        return frame,record
    monkeypatch.setattr(cohort_io, "load_cohort_bundle", changed)
    output = tmp_path / "bad_identity.zarr"
    with pytest.raises(ValueError, match="Cohort niche|Cohort niches"):
        exporter.export_spatialdata(**inputs(cohort_case), cohort_niches=cohort_case / "cohort", outdir=output)
    assert not output.exists()


def test_casefolded_profile_collision_is_not_overwritten(cohort_case, tmp_path, monkeypatch):
    original = exporter.cell_table
    def collided(*args, **kwargs):
        table = original(*args, **kwargs)
        table.obs["COHORT_NICHE_ID"] = 99
        return table
    monkeypatch.setattr(exporter, "cell_table", collided)
    output = tmp_path / "collision.zarr"
    with pytest.raises(ValueError, match="collide"):
        exporter.export_spatialdata(**inputs(cohort_case), cohort_niches=cohort_case / "cohort", outdir=output)
    assert not output.exists()


@pytest.mark.parametrize("source", ["cohort/cohort_niches_completion.json", "A/cell_profiles/feature_blocks/synthetic.npy"])
def test_sources_changed_during_write_are_rejected(cohort_case, tmp_path, monkeypatch, source):
    original = sd.SpatialData.write
    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        with (cohort_case / source).open("ab") as stream: stream.write(b"changed during export")
        return result
    monkeypatch.setattr(sd.SpatialData, "write", changed)
    with pytest.raises(ValueError, match="SHA256"):
        exporter.export_spatialdata(**inputs(cohort_case), cohort_niches=cohort_case / "cohort", outdir=tmp_path / "changing.zarr")


@pytest.mark.parametrize("target", ["score", "nullable_id", "table_metadata", "package_metadata"])
def test_exact_readback_rejects_changed_values_or_provenance(cohort_case, tmp_path, monkeypatch, target):
    import zarr
    original = sd.SpatialData.write
    def changed(self, path, *args, **kwargs):
        result = original(self, path, *args, **kwargs)
        store = zarr.open_group(path, mode="r+")
        if target == "score":
            array = store["tables/cells/obs/cohort_niche_centroid_margin"]
            array[0] = np.nextafter(array[0], np.inf)
        elif target == "nullable_id":
            store["tables/cells/obs/cohort_niche_id/mask"][-1] = False
        elif target == "table_metadata":
            store["tables/cells/uns/cellphenotyper/cohort_niches_json"][()] = "{}"
        else:
            attrs = dict(store.attrs["cellphenotyper"])
            attrs["cohort_niches"] = {"status":"altered"}
            store.attrs["cellphenotyper"] = attrs
        return result
    monkeypatch.setattr(sd.SpatialData, "write", changed)
    with pytest.raises(RuntimeError, match="cohort niche"):
        exporter.export_spatialdata(**inputs(cohort_case), cohort_niches=cohort_case / "cohort", outdir=tmp_path / "changed.zarr")


def test_cli_attaches_requested_specimen_without_a_fitting_runtime(cohort_case, tmp_path):
    args = inputs(cohort_case, "B")
    command = [sys.executable, str(ROOT / "bin/export_spatialdata.py")]
    for name,value in args.items(): command += ["--" + name.replace("_", "-"), str(value)]
    output = tmp_path / "cli.zarr"
    command += ["--cohort-niches", str(cohort_case / "cohort"), "--outdir", str(output), "--sample-id", "B"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["cohort_niches"] is True and summary["cell_count"] == 7
    expected,record = read_cohort(cohort_case, "B")
    assert_cohort(sd.SpatialData.read(output).tables["cells"], expected, record)
