"""Fresh-process Python/R comparison contracts; synthetic features, no models.

The real portable graph exporter serializes a declared synthetic corrected KNN
fixture. This exercises the comparison, not KODAMA training or tissue accuracy.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin/compare_tissue_spatial_clustering.py"
EXPORTER = ROOT / "bin/kodama_graph_export.R"
SEEDS = [11, 29]
VARIANTS = ["baseline", "local_feature", "boundary_feature"]
IDS = ["001", "NA", "cell C", "04", "left_5", "left_6",
       "007", "right_8", "09", "right_10", "11", "right_12"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(directory):
    return {str(path.relative_to(directory)): sha(path) for path in directory.rglob("*") if path.is_file()}


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@pytest.fixture(scope="module")
def rscript():
    executable = shutil.which("Rscript")
    if executable is None:
        pytest.skip("Existing Rscript unavailable; tests never install it")
    result = subprocess.run([executable, "-e",
        'p<-c("Matrix","igraph","digest","jsonlite");if(!all(vapply(p,requireNamespace,logical(1),quietly=TRUE)))quit(status=77)'],
        capture_output=True, text=True, timeout=30, env={**os.environ, "LC_ALL": "C"})
    if result.returncode == 77:
        pytest.skip("Existing R sparse graph dependencies unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.fixture(scope="module")
def immutable_fixture(tmp_path_factory, rscript):
    directory = tmp_path_factory.mktemp("spatial_comparison_inputs")
    native = directory / "native"
    native.mkdir()
    # Source order groups the two native communities. Grid CSV order is
    # deliberately different, and IDs are never interpreted numerically.
    coordinates = [(col, row, 6 + col*12, 6 + row*12)
                   for columns in ((0, 1), (2, 3)) for row in range(3) for col in columns]
    frame = pd.DataFrame({"label": IDS, "x": [item[2] for item in coordinates],
        "y": [item[3] for item in coordinates], "grid_row": [item[1] for item in coordinates],
        "grid_col": [item[0] for item in coordinates]})
    frame.to_csv(directory / "native_order.csv", index=False, float_format="%.17g")
    frame.iloc[[8, 1, 11, 4, 0, 10, 3, 6, 9, 2, 7, 5]].to_csv(directory / "grid.csv", index=False, float_format="%.17g")
    code = r'''
      a<-commandArgs(TRUE);source(a[1]);directory<-a[2]
      cells<-read.csv(a[3],colClasses=c(label="character"),na.strings=character(),check.names=FALSE)
      ids<-cells$label;n<-length(ids);stopifnot(n==12L,"NA"%in%ids,"001"%in%ids)
      xy<-as.matrix(cells[c("x","y")]);storage.mode(xy)<-"double";rownames(xy)<-ids
      set.seed(417);pca<-matrix(rnorm(n*50)*.01,nrow=n,dimnames=list(ids,paste0("PC",1:50)))
      pca[7:12,]<-pca[7:12,]+5
      common_ids<-ids;vis<-cbind(seq_len(n)*17,rev(seq_len(n))*19);rownames(vis)<-ids
      pca_file<-file.path(directory,"pca_full_50.RData");save(pca,xy,common_ids,file=pca_file)
      index<-matrix(NA_integer_,n,5);distance<-matrix(.2,n,5)
      for(i in seq_len(n))index[i,]<-if(i<=6)setdiff(1:6,i) else setdiff(7:12,i)
      distance[1,1]<-0
      fit<-list(knn_is_kodama_corrected=TRUE,res=matrix(0L,1,n),parameters=list(classifier="knn",ncomp=50L))
      receipt<-export_portable_kodama_graph(fit,ids,directory,pca_file,
        package_version="synthetic-contract-fixture",package_revision="no-native-training-or-model-inference",
        materialize=function(fit)list(indices=index,distances=distance))
      representation_metadata<-list(pca_file=basename(pca_file),pca_file_md5=receipt$pca_file_md5,
        pca_file_sha256=receipt$pca_file_sha256,kodama_graph_available=TRUE,
        kodama_graph_file="kodama_graph.rds",kodama_graph_manifest_file="kodama_graph.json",
        kodama_graph_sha256=receipt$file_sha256,requested_kodama_ncomp=50L,
        requested_pca_components=50L,source_scope="synthetic fixture; no KODAMA fitting or learned model execution")
      save(vis,xy,common_ids,representation_metadata,file=file.path(directory,"kodama_full_50.RData"))
    '''
    result = subprocess.run([rscript, "-e", code, str(EXPORTER), str(native), str(directory / "native_order.csv")],
        capture_output=True, text=True, timeout=60, env={**os.environ, "LC_ALL": "C"})
    assert result.returncode == 0, result.stdout + result.stderr
    yy, xx = np.indices((52, 64))
    image = np.stack([235 - 2*xx, 210 - yy, 180 - xx//2 - yy//2], axis=-1).astype(np.uint8)
    tifffile.imwrite(directory / "resolution_image.tif", image, photometric="rgb", tile=(16, 16), compression="deflate")
    crop = image[7:43, 8:56]
    tifffile.imwrite(directory / "crop.tif", crop, photometric="rgb", tile=(16, 16), compression="deflate")
    support = np.ones((9, 12), dtype=np.uint8)
    support[:, 6] = 0  # A represented gap separates the two spatial halves.
    support[1, 1] = 0  # First centre unsupported; its native vertex must survive.
    tifffile.imwrite(directory / "support.tif", support, tile=(16, 16), compression="deflate")
    json_write(directory / "resolution.json", {"status": "pass", "file_sha256": sha(directory / "resolution_image.tif"),
        "mpp_x": .5, "mpp_y": .75, "width_px": 64, "height_px": 52})
    json_write(directory / "crop_summary.json", {"crop_bbox_xyxy": {"x0": 8, "y0": 7, "x1": 56, "y1": 43},
        "crop_size": {"width": 48, "height": 36}, "full_size": {"width": 64, "height": 52},
        "offset_crop_to_original": {"dx": 8, "dy": 7}})
    json_write(directory / "support_summary.json", {"analysis_crop_shape_yx": [36, 48],
        "output_shape_yx": [9, 12], "scale_mask_per_crop_x": .25, "scale_mask_per_crop_y": .25,
        "output_origin_original_pixels_xy": [8, 7], "output_pixel_size_original_pixels_xy": [4, 4],
        "sampling": "crop_aligned_pixel_centres_into_original_grandqc_mask"})
    json_write(directory / "grid_metadata.json", {"observation_type": "spatial_grid",
        "coordinate_space": "crop_roi_level0_pixels", "image_width_px": 48, "image_height_px": 36,
        "tissue_mask_width_px": 12, "tissue_mask_height_px": 9,
        "grid_stride_source_px": 12, "grid_rows": 3, "grid_cols": 4,
        "grid_origin_x": 0, "grid_origin_y": 0, "retained_units": 12,
        "source_mpp_x": .25, "source_mpp_y": .25})
    return directory


@pytest.fixture
def inputs(immutable_fixture, tmp_path):
    directory = tmp_path / "inputs"
    shutil.copytree(immutable_fixture, directory)
    return directory


def invoke(directory, outdir, *extra, environment=None):
    command = [sys.executable, str(CLI), "--source", str(directory / "native"),
        "--grid", str(directory / "grid.csv"), "--grid-metadata", str(directory / "grid_metadata.json"),
        "--image", str(directory / "crop.tif"), "--support", str(directory / "support.tif"),
        "--support-summary", str(directory / "support_summary.json"), "--crop-summary", str(directory / "crop_summary.json"),
        "--resolution", str(directory / "resolution.json"), "--resolution-image", str(directory / "resolution_image.tif"),
        "--outdir", str(outdir), "--dimensions", "50", "--threads", "1", "--target-clusters", "2",
        "--seeds", *map(str, SEEDS), "--radius-um", "9.1", "--descriptor-radius-um", "1",
        "--path-step-um", "1.5", "--spatial-weight", ".1", *extra]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C",
        "PYTHONPYCACHEPREFIX": str(outdir.parent / (outdir.name + "_isolated_bytecode")),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.update(environment or {})
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=env)
    return result


def diagnostics(result, outdir):
    logs = "\n".join(path.name + ":\n" + path.read_text() for path in outdir.glob("*.log")) if outdir.exists() else ""
    return result.stdout + result.stderr + logs


def test_full_cli_three_variants_two_seeds_keep_exact_native_population(inputs, tmp_path):
    before = tree_hashes(inputs)
    output = tmp_path / "comparison"
    result = invoke(inputs, output)
    assert result.returncode == 0, diagnostics(result, output)
    verification = json.loads((output / "verification.json").read_text())
    assert verification["status"] == "pass" and verification["observation_count"] == 12
    assert verification["sources_unchanged"] and verification["frozen_code_unchanged"]
    assert verification["source_sha256_before"] == verification["source_sha256_after"]
    assert verification["exact_output_population_and_coordinates"]
    assert verification["python_runtime"]["source_bytecode_isolated"]
    assert not verification["production_defaults_changed"] and not verification["models_rerun"]
    assert not verification["expert_annotation_used"]
    geometry = verification["geometry"]
    assert geometry["all_crop_pixels_exact"] and geometry["pixels_checked"] == 36*48
    assert geometry["support_explicit_origin_present"]
    assert verification["lattice"] == {"exact_index_to_coordinate_match": True, "stride_source_pixels": 12,
                                        "rows": 3, "columns": 4}
    assert geometry["authoritative_mpp_xy"] == [.5, .75]
    assert geometry["historical_grid_mpp_xy"] == [.25, .25] and not geometry["historical_mpp_agrees"]
    assert verification["adjacency"]["support_mpp_xy"] == [2., 3.]
    assert verification["adjacency"]["unsupported_count"] == 1
    assert verification["adjacency"]["edge_count"] == 12
    assert all(record["returncode"] == 0 for name, record in verification["runs"].items() if name in ("inspect", "fit"))
    expected = pd.read_csv(inputs / "native_order.csv", dtype={"label": str}, keep_default_na=False, float_precision="round_trip")
    observed = pd.read_csv(output / "observations.csv", dtype={"label": str}, keep_default_na=False, float_precision="round_trip")
    pd.testing.assert_frame_equal(observed, expected)
    assert observed.label.tolist() == IDS
    vertices = pd.read_csv(output / "vertices.csv", dtype={"label": str}, keep_default_na=False)
    assert vertices.label.tolist() == IDS and vertices.in_tissue_support.tolist() == [False] + [True]*11
    assignments = pd.read_csv(output / "assignments.csv", dtype={"label": str}, keep_default_na=False, float_precision="round_trip")
    assert len(assignments) == 12*3*len(SEEDS)
    assert set(assignments.variant) == set(VARIANTS) and set(assignments.seed) == set(SEEDS)
    for variant in VARIANTS:
        for seed in SEEDS:
            selected = assignments[(assignments.variant == variant) & (assignments.seed == seed)].reset_index(drop=True)
            assert selected.label.tolist() == IDS and selected.cluster.nunique() == 2
            np.testing.assert_array_equal(selected[["x", "y"]], expected[["x", "y"]])
            assert selected.cluster.iloc[:6].nunique() == selected.cluster.iloc[6:].nunique() == 1
            assert selected.cluster.iloc[0] != selected.cluster.iloc[6]
            assert selected.in_tissue_support.tolist() == [False] + [True]*11
    contract = json.loads((output / "input_contract.json").read_text())
    assert contract["requested_kodama_ncomp"] == 50 and contract["dimensions"] == 50
    assert contract["exact_population_and_coordinates"] and contract["source_grid_reordered_by_literal_id"]
    assert "synthetic" in contract["feature_provenance"]["legacy_source_scope"]
    assert set(contract["feature_provenance"]["missing_fields"]) == {
        "embedding_input_provenance", "rawdata_input_sha256", "pca_preprocessing"}
    assert contract["feature_provenance"]["embedding_input_provenance"] is None
    comparison = verification["clustering"]
    assert comparison["zero_weight_control_exact"] and comparison["target_clusters"] == 2
    assert len(comparison["runs"]) == 6 and len(comparison["comparisons"]) == 7
    for name, expected_hash in verification["output_sha256"].items():
        assert sha(output / name) == expected_hash
    for mode in ("inspect", "fit"):
        binding = json.loads((output / f"{mode}_input_verification.json").read_text())
        requested = json.loads((output / f"{mode}_input_hashes.json").read_text())
        assert binding["inputs"] == requested
        assert binding["outputs"]
        for path, expected_hash in {**binding["inputs"], **binding["outputs"]}.items():
            assert sha(path) == expected_hash
    assert tree_hashes(inputs) == before
    assert not list(output.rglob("*.pyc")) and not list(output.glob("source-cache-*"))


@pytest.mark.parametrize("fault,expected_error", [
    ("report_hash", "Physical-resolution report does not bind"),
    ("crop_pixels", "Crop pixels differ"),
    ("support_shape", "Actual support shape contradicts"),
    ("support_scale", "Support scale does not cover"),
    ("support_origin", "Support output_origin_original_pixels_xy contradicts"),
    ("crop_offset", "Crop size, origin and bounding box disagree"),
    ("grid_frame", "exact crop coordinate frame"),
    ("grid_coordinates", "Grid coordinates differ from saved PCA coordinates"),
    ("grid_foreign_id", "Grid population differs from saved features"),
    ("grid_index_permutation", "Grid indices do not identify the saved observation coordinates"),
])
def test_source_and_geometry_conflicts_fail_without_success_receipt(inputs, tmp_path, fault, expected_error):
    if fault == "report_hash":
        path = inputs / "resolution.json"
        record = json.loads(path.read_text()); record["file_sha256"] = "0"*64
        json_write(path, record)
    elif fault == "crop_pixels":
        path = inputs / "crop.tif"
        pixels = tifffile.imread(path)
        pixels[3, 5, 1] ^= np.uint8(1)
        tifffile.imwrite(path, pixels, photometric="rgb", tile=(16, 16), compression="deflate")
    elif fault == "support_shape":
        tifffile.imwrite(inputs / "support.tif", np.ones((8, 11), dtype=np.uint8), tile=(16, 16), compression="deflate")
    elif fault in ("support_scale", "support_origin"):
        path = inputs / "support_summary.json"
        record = json.loads(path.read_text())
        if fault == "support_scale":
            record["scale_mask_per_crop_x"] = .5
        else:
            record["output_origin_original_pixels_xy"] = [0, 0]
        json_write(path, record)
    elif fault == "crop_offset":
        path = inputs / "crop_summary.json"
        record = json.loads(path.read_text()); record["offset_crop_to_original"]["dx"] = 0
        json_write(path, record)
    elif fault == "grid_frame":
        path = inputs / "grid_metadata.json"
        record = json.loads(path.read_text()); record["coordinate_space"] = "original_level0_pixels"
        json_write(path, record)
    else:
        path = inputs / "grid.csv"
        frame = pd.read_csv(path, dtype={"label": str}, keep_default_na=False)
        if fault == "grid_coordinates":
            frame["x"] = frame["x"].astype(float)
            frame.loc[0, "x"] += .125
        elif fault == "grid_foreign_id":
            frame.loc[0, "label"] = "foreign"
        else:
            frame.loc[[0, 1], ["grid_row", "grid_col"]] = frame.loc[[1, 0], ["grid_row", "grid_col"]].to_numpy()
        frame.to_csv(path, index=False, float_format="%.17g")
    before = tree_hashes(inputs)
    output = tmp_path / "rejected"
    result = invoke(inputs, output)
    assert result.returncode != 0, diagnostics(result, output)
    assert expected_error in diagnostics(result, output)
    receipt = output / "verification.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        assert record["status"] == "failed" and record["sources_unchanged"]
    assert not (output / "assignments.csv").exists()
    assert tree_hashes(inputs) == before


def test_preexisting_output_is_not_modified(inputs, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user_results.txt").write_text("Existing results must be retained.\n")
    before, before_inputs = tree_hashes(output), tree_hashes(inputs)
    result = invoke(inputs, output)
    assert result.returncode != 0 and "Output must be a new directory" in result.stderr
    assert tree_hashes(output) == before and tree_hashes(inputs) == before_inputs


def test_adjacency_mutation_between_python_binding_and_r_consumption_is_rejected(inputs, tmp_path, rscript):
    wrapper_dir = tmp_path / "wrapper_bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "Rscript"
    # This test-only executable changes one valid generated edge value after
    # Python has hashed fit inputs, then delegates to the actual installed R.
    # It never changes any source image, native graph, PCA or production code.
    wrapper.write_text(f"#!{sys.executable}\n" + f"REAL_R = {rscript!r}\n" + r'''
import csv
import os
from pathlib import Path
import sys
if len(sys.argv) > 2 and sys.argv[2] == "fit":
    outdir = Path(sys.argv[5])
    path = outdir / "adjacency.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    assert rows and (outdir / "fit_input_hashes.json").is_file()
    assert rows[0]["boundary_weight"] != "0.123456789"
    rows[0]["boundary_weight"] = "0.123456789"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "test_mutation_applied.txt").write_text("valid boundary weight changed after binding\n")
os.execv(REAL_R, [REAL_R, *sys.argv[1:]])
''')
    wrapper.chmod(0o755)
    before = tree_hashes(inputs)
    output = tmp_path / "mutation_rejected"
    result = invoke(inputs, output, environment={"PATH": str(wrapper_dir) + os.pathsep + os.environ.get("PATH", "")})
    assert result.returncode != 0, diagnostics(result, output)
    assert (output / "test_mutation_applied.txt").is_file()
    assert "R input changed during consumption" in diagnostics(result, output)
    receipt = json.loads((output / "verification.json").read_text())
    assert receipt["status"] == "failed" and receipt["sources_unchanged"]
    assert receipt["runs"]["inspect"]["returncode"] == 0
    assert receipt["runs"]["fit"]["returncode"] != 0
    assert not (output / "assignments.csv").exists()
    assert tree_hashes(inputs) == before
