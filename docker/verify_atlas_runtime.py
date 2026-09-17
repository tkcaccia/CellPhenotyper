#!/usr/bin/env python3
"""Build/runtime probe using actual imports, native algorithms and SpatialData I/O.

This is engineering compatibility evidence, not model inference, biological
validation, a transitive dependency lock or a GPU compatibility certification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import struct
import subprocess
import sys
import tempfile


def requirements_check(path, active=None):
    from packaging.requirements import Requirement

    path = Path(path).resolve()
    active = set() if active is None else set(active)
    if path in active:
        raise ValueError("Cyclic atlas requirements inclusion")
    active.add(path)
    records = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            records.update(requirements_check(path.parent / line[3:].strip(), active))
            continue
        requirement = Requirement(line)
        version = importlib.metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f"Runtime dependency mismatch: {requirement.name}={version}, required {requirement.specifier}")
        records[requirement.name] = version
    return records


def requirements_hashes(path):
    """Bind every recursively included source file, not only the top-level pins."""
    base = Path(path).resolve().parent
    records = {}

    def visit(candidate):
        candidate = candidate.resolve()
        if not candidate.is_relative_to(base):
            raise ValueError("Atlas requirements include a file outside their source directory")
        name = str(candidate.relative_to(base))
        if name in records:
            return
        records[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        for line in candidate.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("-r "):
                visit(candidate.parent / line[3:].strip())

    visit(Path(path))
    return records


def atlas_probe(source_root):
    import numpy as np
    import pandas as pd
    import tifffile
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from skimage.measure import regionprops_table
    import spatialdata as sd
    from cell_profile_io import sha256_file
    from export_spatialdata import export_spatialdata

    for name in ("torch", "tensorflow", "timm"):
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f"Atlas runtime is not ML-isolated: unexpected {name} package")
    for name in ("profile_cell_morphology", "build_cell_profiles", "assemble_spatial_cell_profiles",
                 "analyze_cell_neighborhoods", "discover_tissue_hierarchy", "link_cell_tissue_hierarchy",
                 "cell_reference_atlas", "integrate_measured_assay", "cell_inspector", "specimen_atlas"):
        importlib.import_module(name)
    points = np.array([[0., 0.], [0., .1], [10., 10.], [10., 10.1]])
    projected = PCA(n_components=2, svd_solver="full").fit_transform(points)
    groups = KMeans(n_clusters=2, n_init=2, random_state=17).fit_predict(projected)
    if not (groups[0] == groups[1] and groups[2] == groups[3] and groups[0] != groups[2]):
        raise RuntimeError("Native PCA/KMeans functional probe failed")
    with tempfile.TemporaryDirectory(prefix="cellphenotyper_atlas_probe_") as temporary:
        root = Path(temporary)
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        labels = np.zeros((16, 16), np.uint32)
        labels[2:5, 2:5], labels[10:14, 10:14] = 1, 7
        tifffile.imwrite(root / "image.tif", image, tile=(16, 16), compression="deflate", photometric="rgb")
        tifffile.imwrite(root / "labels.tif", labels, tile=(16, 16), compression="deflate")
        morphology = regionprops_table(labels, properties=("label", "area", "eccentricity", "solidity"))
        if not np.array_equal(morphology["label"], [1, 7]) or not np.array_equal(morphology["area"], [9, 16]):
            raise RuntimeError("Native region morphology functional probe failed")
        (root / "shift.json").write_text(json.dumps({"source_mpp": .5,
            "crop_size": {"width": 16, "height": 16}, "offset_crop_to_original": {"dx": 100, "dy": 200}}))
        (root / "tissue.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {"value": 1}, "geometry": {"type": "Polygon",
            "coordinates": [[[0, 0], [16, 0], [16, 16], [0, 16], [0, 0]]]}}]}))
        profile = root / "profiles"
        profile.mkdir()
        cells = pd.DataFrame({"sample_id": ["runtime-probe"] * 2, "cell_id": ["1", "7"],
            "cell_uid": ["runtime-probe:1", "runtime-probe:7"], "x_um": [51.5, 56.], "y_um": [101.5, 106.],
            "predicted__nucleus__CD3__mean": [16777217.25, np.nan]})
        cells.to_csv(profile / "cell_profiles.csv", index=False)
        cells.to_parquet(profile / "cell_profiles.parquet", index=False)
        cells[["sample_id", "cell_id", "cell_uid"]].to_csv(profile / "feature_rows.csv", index=False)
        features = np.array([[1., 2.], [np.nan, np.nan]], np.float32)
        np.save(profile / "features.npy", features)
        manifest = {"schema_version": "1.0.0", "observation_unit": "cell", "sample_id": "runtime-probe",
            "cell_count": 2, "source_mpp": .5, "crop_origin_um": [50., 100.], "crop_size_px": [16, 16],
            "inputs": {"image_sha256": sha256_file(root / "image.tif"), "labels_sha256": sha256_file(root / "labels.tif")},
            "feature_blocks": {"probe": {"path": "features.npy", "sha256": sha256_file(profile / "features.npy"), "shape": [2, 2]}},
            "files": {name: sha256_file(profile / name) for name in ("cell_profiles.csv", "cell_profiles.parquet", "feature_rows.csv")}}
        (profile / "cell_profiles_manifest.json").write_text(json.dumps(manifest))
        export_spatialdata(profile, root / "image.tif", root / "labels.tif", root / "shift.json", root / "atlas.zarr",
            tissue_geojson=root / "tissue.geojson", tissue_coordinates="crop_pixels", tile_size=16, workers=1)
        loaded = sd.SpatialData.read(root / "atlas.zarr")
        if (not np.array_equal(loaded.labels["canonical_cells"].data.compute(), labels)
                or not np.array_equal(loaded.tables["cells"].obsm["probe"], features, equal_nan=True)
                or loaded.tables["cells"].obs_names.tolist() != cells.cell_uid.tolist()
                or not np.array_equal(loaded.tables["cells"].obs.predicted__nucleus__CD3__mean.to_numpy(),
                                      cells.predicted__nucleus__CD3__mean.to_numpy(), equal_nan=True)):
            raise RuntimeError("Actual canonical-profile/SpatialData exact round trip failed")
    return {"native_pca_kmeans": "passed", "native_region_morphology": "passed", "pipeline_source_imports": "passed",
            "canonical_profile_spatialdata_roundtrip": "passed", "models_loaded": False}


def model_probe(source_root):
    import numpy as np
    import torch
    import timm
    import prepare_hierarchy_features
    from extract_uni2_embeddings import load_local_uni2_encoder

    original = np.array([1., 2.], dtype=np.float32)
    if not np.array_equal(torch.from_numpy(original).numpy(), original):
        raise RuntimeError("Existing ML runtime NumPy/PyTorch bridge failed")
    return {"hierarchy_feature_module_import": "passed", "uni2_encoder_module_import": "passed", "numpy_torch_bridge": "passed",
            "torch": torch.__version__, "timm": timm.__version__, "cuda_build": torch.version.cuda,
            "models_loaded": False, "gpu_execution": "not_tested", "weights": "not_accessed"}


def _native_executable(value):
    value = str(value)
    path = Path(value)
    if (not value or value != value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK)):
        raise ValueError("Native hierarchy Rscript must be an existing absolute executable; no fallback")
    return path.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hierarchy_native_probe(source_root, rscript, library=None, *, strict_isolation=False):
    """Exercise the actual production native runner, not a graph or CLI stub.

    Only 30 synthetic feature rows are fitted. Exact RData/graph readback and
    consumption hashes are checked by the producer, and all returned partitions
    and payload hashes are independently checked here before publishing proof.
    """
    rscript = _native_executable(rscript)
    source_root = Path(source_root).resolve()
    code = [source_root / "bin" / name for name in
            ("run_hierarchy_kodama.R", "kodama_graph_export.R", "kodama_graph_clustering.R")]
    code_before = {str(path): _sha256(path) for path in code}
    executable_before = _sha256(rscript)
    if library is not None:
        library = Path(library).resolve()
        if not library.is_dir():
            raise ValueError("Explicit native R library must exist")
    env = dict(os.environ)
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
                 "R_HOME", "R_LIBS", "R_LIBS_USER", "R_LIBS_SITE"):
        env.pop(name, None)
    env.update(R_ENVIRON_USER=os.devnull, R_PROFILE_USER=os.devnull,
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    native_prefix = rscript.parent.parent
    if (native_prefix / "lib").is_dir():
        env["LD_LIBRARY_PATH"] = str(native_prefix / "lib")
    ids = ["001", "NA", "quoted,cell"] + [f"synthetic_{i:02d}" for i in range(3, 30)]
    rng = random.Random(1705)
    local = [[rng.gauss(-3 if i < 15 else 3, 1) for _ in range(4)] for i in range(30)]
    context = [[rng.gauss(-2 if i < 15 else 2, 1) for _ in range(4)] for i in range(30)]
    values = {"local": local, "context": context, "combined": [a + b for a, b in zip(local, context)]}
    with tempfile.TemporaryDirectory(prefix="cellphenotyper_native_hierarchy_probe_") as temporary:
        root = Path(temporary).resolve()
        with (root / "rows.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["label", "x", "y"])
            writer.writerows((label, i * .5 + 12, (29-i) * .75 - 3) for i, label in enumerate(ids))
        representations = {}
        for name, matrix in values.items():
            path = root / f"{name}.f64"
            with path.open("wb") as stream:
                for row in matrix:
                    stream.write(struct.pack("<" + "d" * len(row), *row))
            representations[name] = {"path": path.name, "sha256": _sha256(path),
                                     "shape": [30, len(matrix[0])], "constant": False}
        parameters = {"ncomp": 50, "M": 3, "Tcycle": 2, "landmarks": 20, "cores": 1,
                      "neighbors": 8, "seed": 17, "cluster_seeds": [17, 18], "resolutions": [.5, 1.]}
        manifest = {"format": "cellphenotyper_hierarchy_kodama_input", "schema_version": "1.0.0",
            "rows": {"path": "rows.csv", "sha256": _sha256(root / "rows.csv")},
            "representations": representations, "parameters": parameters,
            "scope": {"source": "synthetic Gaussian axes; no learned models", "observations": 30},
            "provenance": {"purpose": "native container engineering capability probe"}}
        manifest_path = root / "input_manifest.json"
        manifest_path.write_text(json.dumps(manifest, allow_nan=False))
        input_before = {str(path.resolve()): _sha256(path) for path in root.iterdir()}
        output = root / "native"
        command = [str(rscript), "--vanilla", str(code[0]), str(manifest_path), str(output), _sha256(manifest_path)]
        if library is not None:
            command.append(str(library))
        result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=120)
        if result.returncode:
            raise RuntimeError("Native hierarchy functional probe failed:\n" + result.stdout + result.stderr)
        summary_path = output / "runner_summary.json"
        summary_hash = _sha256(summary_path)
        summary = json.loads(summary_path.read_text())
        if (summary.get("status") != "pass" or summary.get("partition_run_count") != 12
                or summary.get("partition_rows") != 360 or summary.get("feature_protocol") != "raw_data_native_retained_graph"
                or summary.get("sources_before") != input_before or summary.get("sources_after") != input_before
                or summary.get("producer_before") != code_before or summary.get("producer_after") != code_before):
            raise RuntimeError("Native hierarchy functional probe receipt differs")
        for before, after in (("runtime_before", "runtime_after"), ("native_package_before_load", "native_package_after")):
            if not summary.get(before) or summary[before] != summary.get(after):
                raise RuntimeError("Native hierarchy runtime bytes changed during probe")
        for relative, expected in summary["output_sha256"].items():
            path = (output / relative).resolve()
            if not path.is_relative_to(output) or _sha256(path) != expected:
                raise RuntimeError("Native hierarchy functional output hash differs")
        partitions = {}
        with (output / "partitions.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["representation"], float(row["resolution"]), int(row["seed"]))
                partitions.setdefault(key, []).append(row["label"])
                degree, cluster = int(row["degree"]), int(row["cluster"])
                if (degree == 0) != (cluster == 0) or degree < 0 or cluster < 0:
                    raise RuntimeError("Native hierarchy isolate assignment differs")
                for column in ("affinity_margin", "own_affinity_fraction"):
                    if degree == 0:
                        if row[column] != "NA":
                            raise RuntimeError("An isolated observation has fabricated evidence")
                    elif not math.isfinite(float(row[column])):
                        raise RuntimeError("Supported graph evidence is nonfinite")
        expected_runs = {(name, resolution, seed) for name in values for resolution in parameters["resolutions"]
                         for seed in parameters["cluster_seeds"]}
        if set(partitions) != expected_runs or any(labels != ids for labels in partitions.values()):
            raise RuntimeError("Native hierarchy probe lost or reordered observations")
        for name in values:
            record = summary["representations"][name]
            effective = record["effective_native_parameters"]
            if (record["status"] != "fit" or effective["ncomp"] != 50 or effective["classifier"] != "knn"
                    or effective["backend"] != "cpu" or effective["return.graph"] != "handle"
                    or effective["visual.init"] is not False or record["spatial_graph_builds"] != 0):
                raise RuntimeError("Native hierarchy fit protocol differs")
        if strict_isolation:
            for name, package in summary["runtime"]["packages"].items():
                if not Path(package["path"]).resolve().is_relative_to(native_prefix):
                    raise RuntimeError(f"Native R package outside isolated native prefix: {name}: {package['path']}")
        if (any(_sha256(path) != expected for path, expected in input_before.items())
                or any(_sha256(path) != expected for path, expected in code_before.items())
                or _sha256(rscript) != executable_before or _sha256(summary_path) != summary_hash):
            raise RuntimeError("Native hierarchy probe inputs, code or receipt changed")
        return {"native_raw_data_kodama_handle": "passed", "portable_graph_exact_readback": "passed",
                "native_graph_leiden": "passed", "literal_ids_all_rows": "passed", "models_loaded": False,
                "rscript": str(rscript), "rscript_sha256": executable_before, "native_library": str(library) if library else None,
                "native_package_isolation_checked": strict_isolation,
                "runner_summary_sha256": summary_hash, "runner_receipt": summary,
                "temporary_probe_artifacts": "removed after verification; their exact hashes are retained in the receipt",
                "scope": "30 synthetic rows; ncomp=50 distinct from 4/8 input features; M=3,Tcycle=2; no full-tissue/learned/GPU validation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--component", choices=("atlas", "model", "hierarchy-native"), default="atlas")
    parser.add_argument("--native-rscript", default=os.environ.get("CELLPHENOTYPER_HIERARCHY_RSCRIPT", "/opt/micromamba/envs/kodama-r/bin/Rscript"))
    parser.add_argument("--native-r-library", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict-isolation", action="store_true", help="Require all critical atlas package origins inside the selected Python prefix")
    args = parser.parse_args(argv)
    if args.report is not None and args.report.exists():
        raise FileExistsError("Runtime reports require a new path")
    if args.component == "atlas" and sys.version_info[:2] != (3, 12):
        raise RuntimeError("Packaged atlas runtime requires the tested Python 3.12 minor series")
    source = args.source_root.resolve()
    sys.path.insert(0, str(source / "bin"))
    pins = requirements_check(args.requirements) if args.requirements else {}
    if args.component == "hierarchy-native":
        result = hierarchy_native_probe(source, args.native_rscript, args.native_r_library,
                                        strict_isolation=args.strict_isolation)
    else:
        result = atlas_probe(source) if args.component == "atlas" else model_probe(source)
    origins = {}
    for name in (("numpy", "pandas", "scipy", "sklearn", "skimage", "pyarrow", "tifffile", "spatialdata", "zarr", "dask", "geopandas", "shapely", "PIL")
                 if args.component == "atlas" else ("numpy", "torch", "timm") if args.component == "model" else ()):
        module = importlib.import_module(name)
        origin = str(Path(module.__file__).resolve())
        origins[name] = {"path": origin, "version": getattr(module, "__version__", "not_exposed")}
        if args.strict_isolation and not Path(origin).is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"Package outside isolated Python prefix: {name}: {origin}")
    report = {"schema": "cellphenotyper.atlas_runtime_probe.v1", "component": args.component,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(),
        "prefix": sys.prefix, "requirements_checked": pins, "strict_isolation_checked": args.strict_isolation,
        "imported_package_origins": origins,
        "result": result, "biological_validation": False, "container_build_verified_by_this_probe": False,
        "limitation": "Functional CPU engineering probe. Model weights, real learned inference, GPU/driver compatibility and independent biological accuracy are not tested."}
    if args.requirements:
        report["requirements_sha256"] = hashlib.sha256(args.requirements.read_bytes()).hexdigest()
        report["requirements_files_sha256"] = requirements_hashes(args.requirements)
    packages, duplicates = {}, {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        if name in packages and packages[name] != dist.version:
            duplicates.setdefault(name, [packages[name]]).append(dist.version)
        else:
            packages.setdefault(name, dist.version)
    report["installed_packages"], report["duplicate_distribution_versions"] = packages, duplicates
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(payload)
    print(payload)
    return report


if __name__ == "__main__":
    main()
