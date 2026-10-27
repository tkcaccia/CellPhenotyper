"""Pipeline integration checks that never execute learned-model processes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]


def test_opt_in_profile_and_existing_defaults_validate() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    yaml = pytest.importorskip("yaml")
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    jsonschema.Draft7Validator.check_schema(schema)
    validator = jsonschema.Draft7Validator(schema)
    validator.validate(yaml.safe_load((ROOT / "pipeline_paramers.yml").read_text()))
    profile = yaml.safe_load((ROOT / "params/cell_identification.yml").read_text())
    validator.validate(profile)
    assert profile["cell_detection_mode"] == "cellvitpp"
    assert profile["cell_identification_enable"] is True
    assert profile["cell_segmentation_review_enable"] is False


@pytest.mark.parametrize("overrides", [
    {"cell_identification_enable": True, "compute_device": "cpu"},
    {"cell_segmentation_review_enable": True},
    {"end_point": "cell_identification", "cell_identification_enable": False},
    {"cellvit_taxonomies": "all,consep"},
    {"cellvit_taxonomies": "unknown"},
    {"cellvit_input_scope": "partial"},
    {"cellvit_classification_device": "cuda:0"},
    {"virchow2_feature_cache": "relative/cache"},
    {"cell_segmentation_review_max_pixels": 0},
])
def test_invalid_new_route_parameters_rejected(overrides: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(schema).validate(overrides)


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow unavailable")
def test_nextflow_parameter_guards_context_join_and_new_modules_parse(tmp_path: Path) -> None:
    """Actual Groovy/dataflow execution; the harness has no process invocations."""
    project = tmp_path / "settings_probe"
    project.mkdir()
    text = f"""nextflow.enable.dsl=2
import groovy.json.JsonOutput
include {{ resolveCellIdentificationSettings; annotateCellIdentificationContract }} from '{ROOT}/modules/cell_identification_settings'
include {{ RUN_CELL_IDENTIFICATION }} from '{ROOT}/modules/cell_identification'
include {{ RUN_STARDIST_ROI_SEGMENTATION }} from '{ROOT}/modules/run_stardist_roi_segmentation'
workflow {{
  def settings = resolveCellIdentificationSettings([cell_identification_enable:true,cellvit_input_scope:'roi',cellvit_taxonomies:'consep,ocelot'], 'gpu')
  assert settings.enabled && settings.input_scope == 'roi' && settings.taxonomies == ['consep','ocelot']
  def fullslide = resolveCellIdentificationSettings([cell_identification_enable:true], 'gpu')
  assert fullslide.input_scope == 'full_slide' && fullslide.taxonomies == ['all']
  def disabled = resolveCellIdentificationSettings([:], 'cpu')
  assert !disabled.enabled && !disabled.review
  def contract = annotateCellIdentificationContract([scientific_route:[:]], settings, 'cell_identification')
  assert contract.observation_unit == 'detected_nuclear_instance'
  println 'PARAMETER_GUARDS_AND_MODULE_PARSE_OK'
}}
"""
    (project / "main.nf").write_text(text)
    env = dict(os.environ, NXF_OFFLINE="true", NXF_ANSI_LOG="false")
    result = subprocess.run(["nextflow", "run", str(project / "main.nf")], cwd=project, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PARAMETER_GUARDS_AND_MODULE_PARSE_OK" in result.stdout

    # Nextflow propagates an included-function failure to the workflow outcome.
    # Check that real failure separately rather than swallowing it inside DSL2.
    (project / "invalid.nf").write_text(f"""nextflow.enable.dsl=2
include {{ resolveCellIdentificationSettings }} from '{ROOT}/modules/cell_identification_settings'
workflow {{ resolveCellIdentificationSettings([cell_identification_enable:true], 'cpu') }}
""")
    rejected = subprocess.run(["nextflow", "run", str(project / "invalid.nf")], cwd=project, env=env,
                              capture_output=True, text=True, timeout=60)
    assert rejected.returncode != 0
    assert "requires an explicitly resolved GPU runtime" in rejected.stdout + rejected.stderr


def test_full_slide_route_requires_persistent_full_image_and_complete_project() -> None:
    main = (ROOT / "main.nf").read_text()
    module = (ROOT / "modules/cell_identification.nf").read_text()
    settings = (ROOT / "modules/cell_identification_settings.nf").read_text()
    config = (ROOT / "nextflow.config").read_text()
    assert "params.cellvit_input_scope ?: 'full_slide'" in settings
    assert "cell_detection_settings = resolveCellDetectionSettings" in main
    assert "Full-slide CellViT and crop consensus cannot run together" in main
    primary = (ROOT / "modules/cell_segmentation_cellvit.nf").read_text()
    assert "RUN_PRIMARY_CELLVIT(crop_roi_ch" in main
    assert "if (fullslide)" in primary
    assert "RUN_CELL_IDENTIFICATION_FULLSLIDE(full_image_ch" in primary
    assert "EXPORT_CELLVIT_PRIMARY_SEGMENTATION(primary_input_ch)" in primary
    assert "validateFullslideCellIdentificationInputs(params, sample_rows.size())" in main
    assert "cellvit_fullslide_shift" in (ROOT / "modules/cell_identification_settings.nf").read_text()
    assert "The crop shift cannot be reused" in (ROOT / "modules/cell_identification_settings.nf").read_text()
    assert "RUN_CELLVIT_FULLSLIDE(fullslide_input_ch" in module
    assert "--action run" in module
    assert "cellvit_fullslide_complete.json" in module
    assert "CELLVIT_PROJECT_COMPLETE.json" in module
    assert "experiment.cellphenotyper.json" in module
    assert "--bora-geojson" in module
    assert "--expected-bora-sha256" in module
    assert "CELLVIT_SELECTED_PROJECT" in module
    assert "cellvit_fullslide_combined_output" in config
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    assert "cellvit_fullslide_combined_output" in schema["definitions"]["cell_options"]["properties"]
    assert "--max-new-shards" not in module
    assert config.count("withName: RUN_CELLVIT_FULLSLIDE") >= 2


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow unavailable")
def test_full_slide_inputs_fail_closed_and_roi_naming_helpers_are_unchanged(tmp_path: Path) -> None:
    probe = tmp_path / "main.nf"
    probe.write_text(f"""nextflow.enable.dsl=2
include {{ resolveCellIdentificationSettings; validateFullslideCellIdentificationInputs }} from '{ROOT}/modules/cell_identification_settings'
include {{ detectImageSuffix; imageSuffixPriority; deriveSampleId; extractCziRegionLabel; extractCziRegionOrder; buildSampleId }} from '{ROOT}/modules/image_input_helpers'
workflow {{
  assert resolveCellIdentificationSettings([cell_identification_enable:true], 'gpu').input_scope == 'full_slide'
  assert resolveCellIdentificationSettings([cell_identification_enable:true,cellvit_input_scope:'roi'], 'gpu').input_scope == 'roi'
  assert detectImageSuffix('slide.ome.tif') == '.ome.tif'
  assert imageSuffixPriority('.ome.tif') > imageSuffixPriority('.tif')
  assert deriveSampleId(new File('/tmp/slide.ome.tif')) == 'slide'
  assert extractCziRegionLabel('slide - ScanRegion2.geojson') == 'ScanRegion2'
  assert extractCziRegionOrder('slide - ScanRegion2.geojson') == 2
  assert buildSampleId(new File('/tmp/slide.czi'), 'ScanRegion2') == 'slide__ScanRegion2'
  assert validateFullslideCellIdentificationInputs([
    cellvit_fullslide_image:'/tmp/image.tif', cellvit_fullslide_shift:'/tmp/full.shift.json',
    cellvit_fullslide_clean_mask:'/tmp/mask.tif', cellvit_fullslide_bundle:'/tmp/bundle',
    cellvit_classifier_cache:'/tmp/models', cellvit_fullslide_gpu_uuid:'GPU-1',
    cellvit_fullslide_runtime_sif:'/tmp/runtime.sif', cellvit_fullslide_runtime_sif_sha256:'a'*64,
    cellvit_fullslide_runtime_python:'/opt/cellvit/python',
    cellvit_fullslide_runtime_executable:'/opt/cellvit/cellvit-inference'], 1)
  println 'FULLSLIDE_SCOPE_AND_INPUT_GUARDS_OK'
}}
""")
    result = subprocess.run(["nextflow", "run", str(probe)], cwd=tmp_path,
                            env=dict(os.environ, NXF_OFFLINE="true", NXF_ANSI_LOG="false"),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FULLSLIDE_SCOPE_AND_INPUT_GUARDS_OK" in result.stdout

    rejected = tmp_path / "missing.nf"
    rejected.write_text(f"""nextflow.enable.dsl=2
include {{ validateFullslideCellIdentificationInputs }} from '{ROOT}/modules/cell_identification_settings'
workflow {{ validateFullslideCellIdentificationInputs([cellvit_fullslide_image:'/tmp/image.tif'], 1) }}
""")
    failed = subprocess.run(["nextflow", "run", str(rejected)], cwd=tmp_path,
                            env=dict(os.environ, NXF_OFFLINE="true", NXF_ANSI_LOG="false"),
                            capture_output=True, text=True, timeout=60)
    assert failed.returncode != 0
    assert "cellvit_fullslide_shift" in failed.stdout + failed.stderr

    incomplete_bora = tmp_path / "incomplete_bora.nf"
    incomplete_bora.write_text(f"""nextflow.enable.dsl=2
include {{ validateFullslideCellIdentificationInputs }} from '{ROOT}/modules/cell_identification_settings'
workflow {{ validateFullslideCellIdentificationInputs([
  cellvit_fullslide_image:'/tmp/image.tif', cellvit_fullslide_shift:'/tmp/full.shift.json',
  cellvit_fullslide_clean_mask:'/tmp/mask.tif', cellvit_fullslide_bundle:'/tmp/bundle',
  cellvit_classifier_cache:'/tmp/models', cellvit_fullslide_gpu_uuid:'GPU-1',
  cellvit_fullslide_runtime_sif:'/tmp/runtime.sif', cellvit_fullslide_runtime_sif_sha256:'a'*64,
  cellvit_fullslide_runtime_python:'/opt/cellvit/python',
  cellvit_fullslide_runtime_executable:'/opt/cellvit/cellvit-inference',
  cellvit_fullslide_bora_geojson:'/tmp/bora.geojson'], 1) }}
""")
    rejected_bora = subprocess.run(["nextflow", "run", str(incomplete_bora)], cwd=tmp_path,
                                   env=dict(os.environ, NXF_OFFLINE="true", NXF_ANSI_LOG="false"),
                                   capture_output=True, text=True, timeout=60)
    assert rejected_bora.returncode != 0
    assert "Bora merge requires" in rejected_bora.stdout + rejected_bora.stderr


def test_review_module_has_no_automatic_geometry_promotion_or_gpu_route() -> None:
    module = (ROOT / "modules/review_cell_segmentation.nf").read_text()
    flag = "cell_segmentation_review_auto_adjust"
    assert re.search(
        rf"def autoAdjustFlag\s*=\s*params\.{flag}\s*\?\s*'--auto-adjust'\s*:\s*''",
        module,
    )
    assert module.count("'--auto-adjust'") == 1
    assert "${autoAdjustFlag}" in module
    config = (ROOT / "nextflow.config").read_text()
    assert re.search(rf"^\s*{flag}\s*=\s*false(?:\s|$)", config, re.M)
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    setting = schema["definitions"]["cell_options"]["properties"][flag]
    assert setting["type"] == "boolean" and setting["default"] is False
    assert "label 'compute_medium'" in module
    assert "gpu_capable" not in module
    assert "--gpu" not in module
    assert 'original_labels.tif"), emit: original_labels' in module
    assert 'tuned_candidate_labels.tif"), emit: candidate_labels' in module
    assert "--stardist-metadata" in module
