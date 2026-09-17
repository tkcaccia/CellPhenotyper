import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from cell_profile_io import sha256_file
from integrate_measured_assay import INPUT_SCHEMA, OUTPUT_SCHEMA, REGION_SCHEMA, integrate


def fixture(tmp_path, *, region=False, parquet=False):
    source = tmp_path / "canonical"
    source.mkdir()
    key = "region_uid" if region else "cell_uid"
    frame = pd.DataFrame({"sample_id": ["s1"] * 3, key: ["s1:seg:2", "s1:seg:1", "s1:seg:3"],
        "region_id" if region else "cell_id": ["002", "001", "003"],
        "x_um": [10., 20., 30.], "y_um": [5., 5., 5.],
        "predicted__nucleus__CD3__mean": [.1, .2, .3]})
    if region:
        frame["area_um2"] = 2500.
    table = source / ("regions.parquet" if region else "cell_profiles.parquet")
    frame.to_parquet(table, index=False)
    geometry = source / "regions.geojson"
    geometry.write_text('{"type":"FeatureCollection","features":[]}')
    registry = {"schema_version": REGION_SCHEMA if region else "1.0.0",
        "sample_id": "s1", "observation_unit": "spatial_bin" if region else "cell",
        "coordinate_system": "original_slide_micrometres", "region_count" if region else "cell_count": 3,
        "files": {table.name: sha256_file(table)}}
    if region:
        registry["region_definition"] = {"kind": "fixed_spatial_bins", "description": "Locked 50 micrometre bins",
            "artifact": {"path": str(geometry), "sha256": sha256_file(geometry)}}
    registry_manifest = source / ("regions_manifest.json" if region else "cell_profiles_manifest.json")
    registry_manifest.write_text(json.dumps(registry))
    assay = pd.DataFrame({"sample_id": ["s1", "s1"], key: [frame[key][2], frame[key][0]],
        "assay_observation_id": ["003", "NA"], "assay_x": [120., 40.], "assay_y": [20., 20.],
        "registered_x_um": [30.1, 10.1], "registered_y_um": [5., 5.],
        "CD3_abundance": [16777217.25, np.nan], "Ki67_abundance": [np.nan, 1.25]})
    measured = tmp_path / ("measured.parquet" if parquet else "measured.csv")
    if parquet:
        assay.to_parquet(measured, index=False)
    else:
        assay.to_csv(measured, index=False)
    transform = tmp_path / "registration.json"
    transform.write_text(json.dumps({"method": "affine", "matrix": [[.25, 0, 0], [0, .25, 0], [0, 0, 1]]}))
    raw = tmp_path / "independent_assay_raw.txt"
    raw.write_text("independent instrument data fixture")
    contract = {"schema_version": INPUT_SCHEMA, "assay_id": "mIF-1", "assay_type": "protein_imaging",
        "assay_platform": "cyclic-immunofluorescence", "assay_protocol": "locked-assay-protocol-v1",
        "panel_version": "panel1", "sample_id": "s1", "observation_unit": registry["observation_unit"],
        "reference_design": "serial_section_region_level" if region else "same_section_registered",
        "coordinates": {"source_frame": "assay_image_level0", "source_units": "pixel",
            "target_frame": "original_slide_micrometres", "target_units": "um"},
        "registration": {"status": "passed", "locked_before_prediction_review": True,
            "method": "independent_landmark_affine", "independent_landmarks": 5,
            "median_error_um": .2, "p95_error_um": .7, "acceptance_p95_um": 1.,
            "target_observations_sha256": sha256_file(table), "source_observations_sha256": sha256_file(measured),
            "transform_artifact": {"path": str(transform), "sha256": sha256_file(transform)}},
        "matching": {"method": "provided_region_ids" if region else "provided_one_to_one_ids",
            "independent_of_predicted_markers": True, "protocol": "locked matching before virtual-marker review",
            "eligible_observation_count": 3, "minimum_matched_fraction": .5, "matched_fraction": 2/3,
            "maximum_match_distance_um": 1.},
        "source_artifacts": [{"path": str(raw), "sha256": sha256_file(raw), "role": "raw_assay_measurements"}],
        "markers": [{"name": name, "column": f"{name}_abundance", "units": "instrument_arbitrary_units",
            "measurement_type": "protein_intensity", "normalization": "none; raw independent-assay export"}
                    for name in ("CD3", "Ki67")]}
    manifest = tmp_path / "assay.json"
    manifest.write_text(json.dumps(contract))
    kwargs = {"measured": measured, "assay_manifest": manifest, "outdir": tmp_path / "integrated"}
    if region:
        kwargs.update(regions=table, regions_manifest=registry_manifest)
    else:
        kwargs["cell_profiles"] = source
    return kwargs, frame, assay, contract


def save_contract(kwargs, contract):
    Path(kwargs["assay_manifest"]).write_text(json.dumps(contract))


def save_assay(kwargs, assay, contract):
    path = Path(kwargs["measured"])
    if path.suffix == ".parquet":
        assay.to_parquet(path, index=False)
    else:
        assay.to_csv(path, index=False)
    contract["registration"]["source_observations_sha256"] = sha256_file(path)
    save_contract(kwargs, contract)


@pytest.mark.parametrize("parquet", [False, True])
def test_separate_measured_modality_preserves_order_ids_precision_missingness_and_sources(tmp_path, parquet):
    kwargs, canonical, _, _ = fixture(tmp_path, parquet=parquet)
    before = {str(path): sha256_file(path) for path in Path(kwargs["cell_profiles"]).iterdir() if path.is_file()}
    result, manifest = integrate(**kwargs)
    assert manifest["schema_version"] == OUTPUT_SCHEMA
    assert result.cell_uid.tolist() == canonical.cell_uid.tolist()
    assert result.cell_id.tolist() == ["002", "001", "003"]
    assert "predicted__nucleus__CD3__mean" not in result
    assert result.measured_status.tolist() == ["matched", "unmatched", "matched"]
    assert result.assay_observation_id.iloc[0] == "NA"
    matrix = np.load(Path(kwargs["outdir"]) / "measured_values.npy", allow_pickle=False)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix, [[np.nan, 1.25], [np.nan, np.nan], [16777217.25, np.nan]], equal_nan=True)
    assert manifest["markers"][0]["missing_observations"] == 2
    assert not manifest["predicted_features_modified"]
    assert not manifest["biological_accuracy_validated"]
    for path, digest in before.items():
        assert sha256_file(path) == digest
    pd.testing.assert_frame_equal(result, pd.read_parquet(Path(kwargs["outdir"]) / "measured_observations.parquet"))
    for path, digest in manifest["files"].items():
        assert sha256_file(Path(kwargs["outdir"]) / path) == digest


@pytest.mark.parametrize("change,match", [
    (lambda c: c.update(reference_design="serial_section_region_level"), "Serial sections"),
    (lambda c: c.update(assay_platform="10x Visium HD"), "Visium"),
    (lambda c: c["registration"].update(status="failed"), "status=passed"),
    (lambda c: c["registration"].update(independent_landmarks=2), "independent_landmarks"),
    (lambda c: c["registration"].update(p95_error_um=2.), "acceptance gate"),
    (lambda c: c["registration"].update(locked_before_prediction_review=False), "prediction review"),
    (lambda c: c["matching"].update(method="highest_marker_correlation"), "no inferred correlation"),
    (lambda c: c["matching"].update(eligible_observation_count=2), "complete supplied registry"),
    (lambda c: c["matching"].update(minimum_matched_fraction=.9), "below the locked minimum"),
    (lambda c: c["matching"].update(matched_fraction=.9), "actual unique matches"),
    (lambda c: c["coordinates"].update(target_units="pixel"), "original-slide micrometres"),
    (lambda c: c["markers"][0].pop("units"), "units"),
    (lambda c: c["markers"].append(c["markers"][0].copy()), "Duplicate marker"),
    (lambda c: c.update(sample_id="different_sample"), "sample/observation"),
])
def test_fail_closed_contract_gates_before_any_output(tmp_path, change, match):
    kwargs, _, _, contract = fixture(tmp_path)
    change(contract)
    save_contract(kwargs, contract)
    with pytest.raises(ValueError, match=match):
        integrate(**kwargs)
    assert not Path(kwargs["outdir"]).exists()


@pytest.mark.parametrize("case,match", [("foreign", "foreign cell_uid"), ("duplicate", "Duplicate cell_uid"),
    ("source_duplicate", "Duplicate assay_observation_id"), ("coordinates", "maximum_match_distance"),
    ("infinite", "Nonfinite"), ("sample", "sample identity")])
def test_rejects_invalid_explicit_matches(tmp_path, case, match):
    kwargs, _, assay, contract = fixture(tmp_path)
    if case == "foreign":
        assay.loc[0, "cell_uid"] = "unknown:cell"
    elif case == "duplicate":
        assay.loc[0, "cell_uid"] = assay.loc[1, "cell_uid"]
    elif case == "source_duplicate":
        assay.loc[0, "assay_observation_id"] = assay.loc[1, "assay_observation_id"]
    elif case == "coordinates":
        assay.loc[0, "registered_x_um"] = 1000.
    elif case == "infinite":
        assay.loc[0, "CD3_abundance"] = np.inf
    else:
        assay.loc[0, "sample_id"] = "foreign"
    save_assay(kwargs, assay, contract)
    with pytest.raises(ValueError, match=match):
        integrate(**kwargs)
    assert not Path(kwargs["outdir"]).exists()


def test_all_marker_missing_match_remains_distinct_from_unmatched(tmp_path):
    kwargs, _, assay, contract = fixture(tmp_path)
    assay[["CD3_abundance", "Ki67_abundance"]] = np.nan
    save_assay(kwargs, assay, contract)
    result, manifest = integrate(**kwargs)
    assert result.measured_status.tolist() == ["matched_all_markers_missing", "unmatched", "matched_all_markers_missing"]
    assert result.measured_matched.tolist() == [True, False, True]
    assert manifest["matched_fraction"] == 2 / 3


def test_serial_sections_are_separate_regions_and_visium_not_cell_ground_truth(tmp_path):
    kwargs, canonical, _, contract = fixture(tmp_path, region=True)
    contract["assay_platform"] = "Visium"
    contract["assay_type"] = "spatial_transcriptomics"
    save_contract(kwargs, contract)
    result, manifest = integrate(**kwargs)
    assert manifest["observation_unit"] == "spatial_bin"
    assert manifest["identity_key"] == "region_uid"
    assert "cell_uid" not in result
    assert result.region_uid.tolist() == canonical.region_uid.tolist()
    assert manifest["region_definition"]["kind"] == "fixed_spatial_bins"


@pytest.mark.parametrize("target,match", [("source", "source measured table hash"),
    ("registry", "Registry table hash"), ("transform", "artifact hash mismatch"),
    ("raw", "artifact hash mismatch")])
def test_tampered_provenance_rejected(tmp_path, target, match):
    kwargs, _, _, contract = fixture(tmp_path)
    if target == "source":
        Path(kwargs["measured"]).write_text("tampered")
    elif target == "registry":
        (Path(kwargs["cell_profiles"]) / "cell_profiles.parquet").write_bytes(b"tampered")
    elif target == "transform":
        Path(contract["registration"]["transform_artifact"]["path"]).write_text("tampered")
    else:
        Path(contract["source_artifacts"][0]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match=match):
        integrate(**kwargs)


def test_existing_output_and_canonical_subdirectory_are_never_mutated(tmp_path):
    kwargs, _, _, _ = fixture(tmp_path)
    kwargs["outdir"] = Path(kwargs["cell_profiles"]) / "measured"
    with pytest.raises(ValueError, match="separate from the canonical"):
        integrate(**kwargs)
    kwargs["outdir"] = Path(kwargs["cell_profiles"])
    with pytest.raises(FileExistsError):
        integrate(**kwargs)


def test_size_guards_fail_before_writes(tmp_path):
    kwargs, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="max-observations"):
        integrate(**kwargs, max_observations=2)
    with pytest.raises(ValueError, match="max-matrix-values"):
        integrate(**kwargs, max_matrix_values=5)
    assert not Path(kwargs["outdir"]).exists()


def test_cli_actual_csv_parquet_and_matrix_export(tmp_path):
    kwargs, _, _, _ = fixture(tmp_path)
    command = [sys.executable, str(ROOT / "bin/integrate_measured_assay.py")]
    for key, value in kwargs.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["biological_accuracy_validated"] is False
    output = Path(kwargs["outdir"])
    assert pd.read_parquet(output / "measured_observations.parquet").shape[0] == 3
    assert np.load(output / "measured_values.npy").shape == (3, 2)


def test_csv_duplicate_measurement_header_cannot_be_silently_mangled(tmp_path):
    kwargs, _, _, contract = fixture(tmp_path)
    path = Path(kwargs["measured"])
    text = path.read_text().replace("Ki67_abundance", "CD3_abundance", 1)
    path.write_text(text)
    contract["registration"]["source_observations_sha256"] = sha256_file(path)
    save_contract(kwargs, contract)
    with pytest.raises(ValueError, match="Duplicate table columns"):
        integrate(**kwargs)
