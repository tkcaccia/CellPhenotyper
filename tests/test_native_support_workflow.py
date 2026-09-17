"""Run the actual profile module with model-free command recorders.

These tests verify invocation order, parameter gating and immutable input wiring,
not the image algorithm or biological gap detection.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/build_spatial_cell_profiles.nf"


def test_native_support_default_and_helper_fingerprint():
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    option = schema["definitions"]["cell_atlas_options"]["properties"]["cell_neighborhood_support_mode"]
    assert option["default"] == "provided"
    assert option["enum"] == ["provided", "brightfield_native"]
    assert re.search(r"cell_neighborhood_support_mode\s*=\s*'provided'", (ROOT / "nextflow.config").read_text())
    scripts = MODULE.read_text().split("def scripts =", 1)[1].split("def codeFingerprint", 1)[0]
    assert "'build_native_tissue_support.py'" in scripts


def run_profile_module(tmp_path, mode):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    for directory in ("modules", "lib", "bin"):
        (tmp_path / directory).mkdir()
    shutil.copy2(MODULE, tmp_path / "modules/build_spatial_cell_profiles.nf")
    for name in ("TaskRuntime.groovy", "HostRuntime.groovy", "HardwarePolicy.groovy", "PipelineHelpers.groovy", "ProcessCode.groovy"):
        shutil.copy2(ROOT / "lib" / name, tmp_path / "lib" / name)
    # The real module hashes its script inventory. Scripts are never executed by
    # this recorder; their contents only represent model-free fixture identities.
    for name in re.findall(r"'([a-z0-9_]+\.py)'", MODULE.read_text()):
        (tmp_path / "bin" / name).write_text("# model-free command fixture\n")
    recorder = tmp_path / "record_profile_commands"
    recorder.write_text(f"#!{sys.executable}\n" + '''import json, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-c':
    print('Model-free profile command recorder')
    raise SystemExit(0)
script = Path(args[0]).name
record_path = Path('command_calls.json')
calls = json.loads(record_path.read_text()) if record_path.exists() else []
calls.append({'script': script, 'args': args[1:], 'used_image_algorithm': False})
record_path.write_text(json.dumps(calls))
outdir = Path(args[args.index('--outdir') + 1])
outdir.mkdir()
if script == 'profile_cell_morphology.py':
    (outdir/'cell_morphology.csv').write_text('label,area_um2\\n')
elif script == 'build_native_tissue_support.py':
    (outdir/'support.tif').write_text('model-free mask fixture')
    (outdir/'support_manifest.json').write_text(json.dumps({'model_free': True}))
elif script == 'assemble_spatial_cell_profiles.py':
    (outdir/'recorded_commands.json').write_text(json.dumps(calls))
    (outdir/'cell_profiles_manifest.json').write_text(json.dumps({'model_free': True}))
''')
    recorder.chmod(0o755)
    for name in ("image.tif", "labels.tif", "objects.csv", "shift.json", "resolution.json", "support.tif"):
        (tmp_path / name).write_text("model-free input fixture\n")
    params = {key: 0 for key in re.findall(r"params\.([A-Za-z_][A-Za-z_0-9]*)", MODULE.read_text())}
    params.update({"cell_atlas_python": str(recorder), "outdir_base": str(tmp_path / "output"),
                   "publish_dir_mode": "copy", "cell_neighborhood_support_mode": mode,
                   "cell_neighborhood_radii_um": "25,50,100", "cell_neighborhood_feature_groups": "",
                   "cell_niche_seed": 17, "cell_niche_fixed_k": 0, "cell_niche_max_k": 8})
    (tmp_path / "params.json").write_text(json.dumps(params))
    (tmp_path / "nextflow.config").write_text("process.executor = 'local'\nexecutor.queueSize = 1\n")
    (tmp_path / "main.nf").write_text('''nextflow.enable.dsl=2
include { BUILD_SPATIAL_CELL_PROFILES } from './modules/build_spatial_cell_profiles'
workflow {
  def runtime_plan = [schema_version:1, compute_device:'cpu', profile:'conservative',
    cpu_budget:1, memory_budget_gb:2, stages:[cell_profiles:[cpus:1, memory_gb:2]], settings:[:]]
  def features = [markers:false, uni2:false, cellvit:false, domains:false, compartments:false, uncertainty:false]
  def inputs = Channel.of(tuple('sample', file('image.tif'), file('labels.tif'), file('objects.csv'),
    file('shift.json'), file('resolution.json'), file('support.tif'), [], [], [], [], [], [], [], features))
  BUILD_SPATIAL_CELL_PROFILES(inputs, runtime_plan)
}
''')
    result = subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "main.nf"),
        "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false", "-work-dir", str(tmp_path / "work")],
        cwd=tmp_path, env={**os.environ, "NXF_OFFLINE": "true"}, capture_output=True, text=True, timeout=60)
    output = tmp_path / "output/19_cell_profiles/sample/cell_profiles/recorded_commands.json"
    calls = json.loads(output.read_text()) if output.exists() else []
    return result, calls


@pytest.mark.parametrize("mode", ["provided", "brightfield_native"])
def test_actual_profile_module_native_support_command_and_immutable_inputs(tmp_path, mode):
    result, calls = run_profile_module(tmp_path, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    scripts = [call["script"] for call in calls]
    expected = ["profile_cell_morphology.py", "build_cell_profiles.py"]
    if mode == "brightfield_native":
        expected.append("build_native_tissue_support.py")
    assert scripts == expected + ["assemble_spatial_cell_profiles.py"]
    assert all(not call["used_image_algorithm"] for call in calls)
    base, assembly = calls[1]["args"], calls[-1]["args"]
    assert base[base.index("--tissue-mask") + 1] == "support.tif"
    assert assembly[assembly.index("--support-mask") + 1] == "support.tif"
    assert "--native-support-mask" not in base
    if mode == "provided":
        assert "--native-support-mask" not in assembly
    else:
        native = calls[2]["args"]
        assert native == ["--image", "image.tif", "--support-mask", "support.tif", "--shift", "shift.json",
                          "--resolution-json", "resolution.json", "--outdir", "native_support"]
        assert assembly[assembly.index("--native-support-mask") + 1] == "native_support/support.tif"
        assert assembly[assembly.index("--native-support-coordinates") + 1] == "crop_pixels"
        assert assembly[assembly.index("--native-support-manifest") + 1] == "native_support/support_manifest.json"
        assert not any("geojson" in value.lower() for value in native)


@pytest.mark.parametrize("mode", ["native", "", None, 1])
def test_actual_profile_module_rejects_invalid_support_mode(tmp_path, mode):
    result, calls = run_profile_module(tmp_path, mode)
    assert result.returncode != 0
    assert "cell_neighborhood_support_mode must be provided or brightfield_native" in result.stdout + result.stderr
    assert calls == []


def test_two_specimens_real_native_support_and_profile_assembly_through_nextflow(tmp_path):
    """Real tiny image computation, not encoder inference or biological validation."""
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    import numpy as np
    import pandas as pd
    import tifffile
    from scipy import sparse

    sys.path.insert(0, str(ROOT / "bin"))
    from build_native_tissue_support import verify_support_bundle
    from cell_profile_io import sha256_file

    rows, originals = [], {}
    expected_ids = ["7", "2", "11", "3", "5"]
    # The gap-centred fifth canonical cell intentionally remains in the labels
    # and profile: native support must not use that cell mask to invent tissue.
    locations = [(7, 72, 32), (2, 24, 16), (11, 64, 16), (3, 32, 32), (5, 48, 24)]
    for sample, white, origin, upstream_exclusion in (
        ("native_A", (255, 255, 255), (100, 200), False),
        ("native_B", (232, 233, 249), (300, 400), True),
    ):
        root = tmp_path / sample
        root.mkdir()
        rgb = np.empty((64, 96, 3), np.uint8)
        rgb[:] = (205, 155, 175)
        rgb[:, 40:56] = white
        labels = np.zeros((64, 96), np.uint16)
        objects = []
        for label, x, y in locations:
            labels[y-1:y+2, x-1:x+2] = label
            if label != 5:
                rgb[y-1:y+2, x-1:x+2] = (80, 50, 120)
            objects.append({"label": label, "x": x, "y": y, "xmin": x-1, "ymin": y-1,
                            "xmax": x+2, "ymax": y+2})
        coarse = np.ones((4, 6), np.uint8)
        if upstream_exclusion:
            coarse[:, 0] = 0
            rgb[:, :16] = white  # Enough independent upstream-excluded samples for colour adaptation.
        tifffile.imwrite(root / "image.tif", rgb, photometric="rgb", tile=(16, 16), compression="deflate")
        tifffile.imwrite(root / "labels.tif", labels, tile=(16, 16), compression="deflate")
        tifffile.imwrite(root / "support.tif", coarse)
        pd.DataFrame(objects).to_csv(root / "objects.csv", index=False)
        (root / "shift.json").write_text(json.dumps({"crop_size": {"width": 96, "height": 64},
            "offset_crop_to_original": {"dx": origin[0], "dy": origin[1]}}))
        (root / "resolution.json").write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
        row = {key: str(root / name) for key, name in (("image", "image.tif"), ("labels", "labels.tif"),
            ("objects", "objects.csv"), ("support", "support.tif"), ("shift", "shift.json"), ("resolution", "resolution.json"))}
        row["sample_id"] = sample
        rows.append(row)
        originals.update({Path(path): sha256_file(path) for key, path in row.items() if key != "sample_id"})
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(rows))
    result = subprocess.run([nextflow, "-log", str(tmp_path / "actual.nextflow.log"), "run", str(ROOT / "cell_profiles.nf"),
        "-ansi-log", "false", "-work-dir", str(tmp_path / "actual_work"),
        "--cell_profile_samples", str(samples), "--outdir_base", str(tmp_path / "actual_output"),
        "--cell_atlas_python", sys.executable, "--cell_profiles_spatialdata", "false",
        "--cell_profiles_uni2_enable", "false", "--cell_profiles_markers_enable", "false",
        "--cell_profiles_cpus", "1", "--cell_profiles_memory_gb", "2",
        "--cell_neighborhood_support_mode", "brightfield_native", "--cell_neighborhood_radii_um", "30",
        "--cell_niche_fixed_k", "1"],
        cwd=tmp_path, env={**os.environ, "NXF_OFFLINE": "true", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert originals == {path: sha256_file(path) for path in originals}
    base_profiles = {}
    for path in (tmp_path / "actual_work").glob("*/*/base_profiles/cell_profiles_manifest.json"):
        base_profiles[json.loads(path.read_text())["sample_id"]] = path.parent
    assert set(base_profiles) == {"native_A", "native_B"}
    all_uids = []
    for row in rows:
        sample = row["sample_id"]
        profile = tmp_path / "actual_output/19_cell_profiles" / sample / "cell_profiles"
        cells = pd.read_parquet(profile / "cell_profiles.parquet")
        base = pd.read_parquet(base_profiles[sample] / "cell_profiles.parquet")
        pd.testing.assert_frame_equal(cells[base.columns], base)
        assert cells.cell_id.tolist() == expected_ids
        assert cells.sample_id.eq(sample).all() and cells.cell_uid.is_unique
        assert cells.in_tissue_support.tolist() == [True, True, True, True, False]
        assert cells.niche_status.iloc[-1] == "outside_tissue_support"
        all_uids.extend(cells.cell_uid)
        manifest = json.loads((profile / "cell_profiles_manifest.json").read_text())
        assert set(manifest["feature_blocks"]).issubset({"morphology", "nuclear_texture"})
        support = manifest["neighborhoods"]["graph_support"]
        assert support["path"] == "neighborhood_support/support.tif"
        assert support["canonical_measurement_support_unchanged"] is True
        bundle = profile / "neighborhood_support/support_manifest.json"
        verified = verify_support_bundle(bundle, image_sha256=sha256_file(row["image"]),
            support_mask=row["support"], shift=row["shift"], resolution_json=row["resolution"])
        assert verified["biological_validation"] == "not_established"
        assert verified["pixel_counts"]["2"] > 0
        assert verified["white_reference"]["status"] == (
            "nominal_white_fallback" if sample == "native_A" else "estimated_from_upstream_exclusions")
        for filename in ("support.tif", "reasons.tif", "support_manifest.json"):
            key = f"neighborhood_support/{filename}"
            assert manifest["files"][key] == sha256_file(profile / key)
        graph = sparse.load_npz(profile / "neighborhood_graph_30um.npz")
        assert graph.shape == (5, 5) and graph.nnz == 4
        assert set(graph[0].indices) == {2} and set(graph[1].indices) == {3}
        assert graph[4].nnz == 0
        for left, right in zip(*graph.nonzero()):
            assert (cells.x_crop_px.iloc[left] < 40) == (cells.x_crop_px.iloc[right] < 40)
    assert len(all_uids) == len(set(all_uids)) == 10
