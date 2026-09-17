#!/usr/bin/env python3
"""Annotation-free spatial-constraint experiment on an immutable saved grid.

Creates a new derivative directory only. The original graph, encoder features,
profiles and production defaults are not changed. No learned models are run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


CODE_FILES = ("compare_tissue_spatial_clustering.py", "compare_tissue_spatial_clustering.R",
              "kodama_spatial_regularization.R", "kodama_graph_clustering.R", "kodama_graph_export.R",
              "build_tissue_spatial_adjacency.py", "analyze_cell_neighborhoods.py", "cell_profile_io.py",
              "profile_cell_morphology.py", "cell_morphology_io.py", "uni2_grid.py")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid JSON number {x}")))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _integers(value, names):
    result = []
    for name in names:
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or int(number) != number:
            raise ValueError(f"{name} must be an integer")
        result.append(int(number))
    return result


def verify_geometry(args, raster_reader):
    """Bind new physical geometry to report bytes and exact source crop pixels."""
    import numpy as np
    report, crop, grid, support_meta = map(read_json,
        (args.resolution, args.crop_summary, args.grid_metadata, args.support_summary))
    if report.get("status") != "pass" or report.get("file_sha256") != digest(args.resolution_image):
        raise ValueError("Physical-resolution report does not bind the supplied source image")
    if any(isinstance(report[k], bool) or not isinstance(report[k], (float, int)) for k in ("mpp_x", "mpp_y")):
        raise ValueError("Physical pixel size must be numeric, not boolean")
    mpp = tuple(float(report[k]) for k in ("mpp_x", "mpp_y"))
    if not all(math.isfinite(v) and .01 <= v <= 10 for v in mpp):
        raise ValueError("Invalid physical pixel size")
    box = _integers(crop["crop_bbox_xyxy"], ("x0", "y0", "x1", "y1"))
    width, height = _integers(crop["crop_size"], ("width", "height"))
    full_width, full_height = _integers(crop["full_size"], ("width", "height"))
    ox, oy = _integers(crop["offset_crop_to_original"], ("dx", "dy"))
    if box != [ox, oy, ox + width, oy + height] or min(width, height) <= 0 or min(ox, oy) < 0 or box[2] > full_width or box[3] > full_height:
        raise ValueError("Crop size, origin and bounding box disagree")
    if (full_width, full_height) != tuple(_integers(report, ("width_px", "height_px"))):
        raise ValueError("Crop source shape differs from passed resolution report")
    if grid.get("observation_type") != "spatial_grid" or grid.get("coordinate_space") != "crop_roi_level0_pixels" or (grid.get("image_width_px"), grid.get("image_height_px")) != (width, height):
        raise ValueError("Grid metadata does not describe the exact crop coordinate frame")
    if support_meta.get("analysis_crop_shape_yx") != [height, width]:
        raise ValueError("Support geometry is not bound to the crop shape")
    with raster_reader(args.image) as image, raster_reader(args.resolution_image) as original, raster_reader(args.support) as support:
        if (image.width, image.height) != (width, height) or (original.width, original.height) != (full_width, full_height):
            raise ValueError("Actual image shapes contradict crop/source metadata")
        if len(image.reader.shape) != 3 or image.reader.shape[2] != 3 or len(original.reader.shape) != 3 or original.reader.shape[2] != 3:
            raise ValueError("Crop and resolution image must be RGB")
        if len(support.reader.shape) != 2 or support_meta.get("output_shape_yx") != [support.height, support.width] or (grid.get("tissue_mask_width_px"), grid.get("tissue_mask_height_px")) != (support.width, support.height):
            raise ValueError("Actual support shape contradicts grid/support metadata")
        for key, expected in (("scale_mask_per_crop_x", support.width / width), ("scale_mask_per_crop_y", support.height / height)):
            if not math.isclose(float(support_meta[key]), expected, rel_tol=1e-12, abs_tol=0):
                raise ValueError("Support scale does not cover exactly the crop extent")
        for key, expected in (("tissue_mask_scale_x", support.width / width), ("tissue_mask_scale_y", support.height / height)):
            if key in grid and not math.isclose(float(grid[key]), expected, rel_tol=1e-12, abs_tol=0):
                raise ValueError("Grid support scale contradicts crop extent")
        support_fields = {
            "output_origin_original_pixels_xy": [ox, oy],
            "output_pixel_size_original_pixels_xy": [width / support.width, height / support.height],
            "source_pixel_size_original_pixels_xy": [full_width / support_meta["source_mask_shape_yx"][1],
                                                     full_height / support_meta["source_mask_shape_yx"][0]]
            if "source_mask_shape_yx" in support_meta else None,
        }
        for key, expected in support_fields.items():
            if key in support_meta and (expected is None or not np.allclose(support_meta[key], expected, rtol=1e-12, atol=0)):
                raise ValueError(f"Support {key} contradicts crop/source geometry")
        if "sampling" in support_meta and support_meta["sampling"] != "crop_aligned_pixel_centres_into_original_grandqc_mask":
            raise ValueError("Unsupported support sampling convention")
        # All pixels, bounded windows: shape and file hashes alone do not prove
        # that a re-encoded crop belongs to the image in the resolution report.
        checked = 0
        for y in range(0, height, 512):
            for x in range(0, width, 512):
                x1, y1 = min(width, x + 512), min(height, y + 512)
                a = image.window(x, y, x1, y1)
                b = original.window(x + ox, y + oy, x1 + ox, y1 + oy)
                if a.dtype != b.dtype or not np.array_equal(a, b):
                    raise ValueError(f"Crop pixels differ from report-bound source at window {x},{y}")
                checked += a.shape[0] * a.shape[1]
        support_shape = [support.height, support.width]
    old_mpp = [grid.get("source_mpp_x"), grid.get("source_mpp_y")]
    return mpp, {"source_report_status": "pass", "all_crop_pixels_exact": True, "pixels_checked": checked,
        "image_shape_yx": [height, width], "support_shape_yx": support_shape,
        "authoritative_mpp_xy": list(mpp), "historical_grid_mpp_xy": old_mpp,
        "historical_mpp_agrees": all(isinstance(x, (float, int)) and math.isclose(x, y, rel_tol=1e-9) for x, y in zip(old_mpp, mpp)),
        "geometry_policy": "Report-bound MPP applies only to this new spatial derivative; historical encoder features/calibration remain unchanged and unverified",
        "support_provenance": "Supplied mask bytes are frozen for this experiment; historical image/mask lineage is not independently verified",
        "support_explicit_origin_present": "output_origin_original_pixels_xy" in support_meta,
        "support_scope": "Exact traversal of supplied support pixels; coarse masks cannot reveal unrepresented thin gaps"}


def verify_lattice(observations, metadata, centered_axis_lattice):
    """A grid index must identify its recorded physical centre, not just a row."""
    import numpy as np
    width, height, stride, rows, cols, ox, oy, retained = _integers(metadata, (
        "image_width_px", "image_height_px", "grid_stride_source_px", "grid_rows",
        "grid_cols", "grid_origin_x", "grid_origin_y", "retained_units"))
    xc, xs, _ = centered_axis_lattice(width, stride)
    yc, ys, _ = centered_axis_lattice(height, stride)
    if (rows, cols, ox, oy, retained) != (len(yc), len(xc), int(xs[0]), int(ys[0]), len(observations)):
        raise ValueError("Grid lattice metadata contradicts crop geometry/population")
    indices = observations[["grid_row", "grid_col"]].to_numpy(dtype=float)
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all() or (indices < 0).any():
        raise ValueError("Grid indices must be nonnegative integers")
    rr, cc = indices.astype(np.int64).T
    if (rr >= rows).any() or (cc >= cols).any() or observations.duplicated(["grid_row", "grid_col"]).any():
        raise ValueError("Grid indices exceed lattice or duplicate a position")
    if not np.array_equal(observations[["x", "y"]].to_numpy(), np.column_stack((xc[cc], yc[rr]))):
        raise ValueError("Grid indices do not identify the saved observation coordinates")
    return {"exact_index_to_coordinate_match": True, "stride_source_pixels": stride, "rows": rows, "columns": cols}


def compare(args):
    for name in ("source", "grid", "grid_metadata", "image", "support", "support_summary", "crop_summary", "resolution", "resolution_image", "outdir"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.outdir.exists() or args.source in args.outdir.parents:
        raise ValueError("Output must be a new directory outside the immutable source bundle")
    if args.dimensions < 1 or args.threads < 1 or args.target_clusters != 2 or len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require positive dimensions/threads, distinct repeated seeds, explicit sensitivity K=2")
    for name in ("radius_um", "descriptor_radius_um", "path_step_um", "boundary_sigma", "spatial_weight", "resolution_value"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    executable = shutil.which("Rscript")
    if not executable:
        raise RuntimeError("Rscript unavailable; this experiment never installs runtimes")
    manifest = read_json(args.source / "kodama_graph.json")
    pca_name = manifest["pca_file"]
    if not isinstance(pca_name, str) or Path(pca_name).name != pca_name or pca_name in (".", ".."):
        raise ValueError("Unsafe PCA sidecar name")
    source_files = [args.source / x for x in ("kodama_graph.rds", "kodama_graph.json", pca_name, f"kodama_full_{args.dimensions}.RData")]
    source_files += [getattr(args, name) for name in ("grid", "grid_metadata", "image", "support", "support_summary", "crop_summary", "resolution", "resolution_image")]
    code_dir = Path(__file__).resolve().parent
    source_files += [code_dir / name for name in CODE_FILES]
    hashes = {str(path): digest(path) for path in source_files}
    args.outdir.mkdir(parents=True)
    frozen = args.outdir / "code"
    frozen.mkdir()
    for name in CODE_FILES:
        shutil.copyfile(code_dir / name, frozen / name)
        if digest(frozen / name) != hashes[str(code_dir / name)]:
            raise ValueError("Executed code changed while making immutable snapshot")
    settings = {"seeds": args.seeds, "spatial_weight": args.spatial_weight,
                "resolution": args.resolution_value, "target_clusters": args.target_clusters}
    write_json(args.outdir / "settings.json", settings)
    generated_hashes = {str(args.outdir / "settings.json"): digest(args.outdir / "settings.json")}
    evidence = {"status": "running", "schema_version": "1.0.0", "source_sha256_before": hashes,
        "expert_annotation_used": False, "models_rerun": False, "observation_unit": "tissue_grid",
        "claim": "Controlled developmental spatial-constraint comparison, not independently validated tissue accuracy",
        "production_defaults_changed": False, "commands": [], "runs": {},
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    receipt = args.outdir / "verification.json"
    write_json(receipt, evidence)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), VECLIB_MAXIMUM_THREADS=str(args.threads))
    def run_r(mode, observations):
        # Bind every R-consumed file to the exact bytes selected by Python.
        inputs = {str(path): hashes[str(path)] for path in source_files[:4]}
        inputs.update({str(frozen / name): hashes[str(code_dir / name)] for name in CODE_FILES if name.endswith(".R")})
        consumed = [args.outdir / "settings.json", observations]
        if mode == "fit":
            consumed += [args.outdir / name for name in ("adjacency.csv", "vertices.csv")]
        for path in consumed:
            inputs[str(path)] = generated_hashes.get(str(path), hashes.get(str(path)))
        if any(expected is None or digest(path) != expected for path, expected in inputs.items()):
            raise ValueError("R input changed before consumption")
        binding = args.outdir / f"{mode}_input_hashes.json"
        write_json(binding, inputs)
        generated_hashes[str(binding)] = digest(binding)
        command = [executable, str(frozen / "compare_tissue_spatial_clustering.R"), mode,
            str(args.source), str(observations), str(args.outdir), str(args.dimensions), str(args.outdir / "settings.json"), str(binding)]
        evidence["commands"].append(command)
        started = time.monotonic()
        with (args.outdir / f"{mode}.log").open("w") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        evidence["runs"][mode] = {"returncode": result.returncode, "wall_seconds": time.monotonic() - started}
        write_json(receipt, evidence)
        if result.returncode:
            raise RuntimeError(f"R {mode} failed; inspect {args.outdir / (mode + '.log')}")
        verification_path = args.outdir / f"{mode}_input_verification.json"
        verification_hash = digest(verification_path)
        verified = read_json(verification_path)
        if digest(verification_path) != verification_hash:
            raise ValueError("R verification receipt changed during consumption")
        generated_hashes[str(verification_path)] = verification_hash
        if verified.get("inputs") != inputs or any(digest(path) != expected for path, expected in inputs.items()):
            raise ValueError("R consumed inputs differ from the producer-bound files")
        evidence.setdefault("r_input_bindings", {})[mode] = {"verified_before_and_after": True, "sha256": inputs}
        produced = ("observations.csv", "input_contract.json") if mode == "inspect" else ("assignments.csv", "clustering_comparison.json")
        output_hashes = verified.get("outputs", {})
        if not all(str(args.outdir / name) in output_hashes for name in produced):
            raise ValueError("R omitted an output identity receipt")
        for path, expected in output_hashes.items():
            if Path(path).parent != args.outdir or digest(path) != expected:
                raise ValueError("R-produced artifact changed before Python consumption")
            generated_hashes[path] = expected
    try:
        # A fresh empty prefix prevents reading historical timestamp bytecode.
        # The CLI is intended for a fresh process; don't reuse imported helpers.
        if any(Path(name).stem in sys.modules for name in CODE_FILES if name.endswith(".py") and name != Path(__file__).name):
            raise RuntimeError("Run comparison in a fresh Python process")
        with tempfile.TemporaryDirectory(prefix="source-cache-", dir=args.outdir) as prefix:
            sys.dont_write_bytecode = True
            sys.pycache_prefix = prefix
            sys.path.insert(0, str(frozen))
            import numpy as np
            import pandas as pd
            from cell_profile_io import RasterReader
            from build_tissue_spatial_adjacency import build_adjacency
            from uni2_grid import centered_axis_lattice
            started = time.monotonic()
            mpp, geometry = verify_geometry(args, RasterReader)
            evidence["geometry"] = geometry
            evidence["runs"]["geometry"] = {"wall_seconds": time.monotonic() - started}
            print(f"Verified all {geometry['pixels_checked']} crop pixels and physical geometry", flush=True)
            run_r("inspect", args.grid)
            observations = pd.read_csv(args.outdir / "observations.csv", dtype={"label": str}, keep_default_na=False, float_precision="round_trip")
            evidence["lattice"] = verify_lattice(observations, read_json(args.grid_metadata), centered_axis_lattice)
            started = time.monotonic()
            edges, vertices, adjacency_meta = build_adjacency(observations, args.image, args.support, image_mpp=mpp,
                radius_um=args.radius_um, descriptor_radius_um=args.descriptor_radius_um,
                path_step_um=args.path_step_um, boundary_sigma=args.boundary_sigma)
            if vertices.label.tolist() != observations.label.tolist():
                raise ValueError("Adjacency builder changed observation IDs or order")
            edges.to_csv(args.outdir / "adjacency.csv", index=False, float_format="%.17g")
            vertices.to_csv(args.outdir / "vertices.csv", index=False, float_format="%.17g")
            write_json(args.outdir / "adjacency.json", adjacency_meta)
            for name in ("adjacency.csv", "vertices.csv", "adjacency.json"):
                generated_hashes[str(args.outdir / name)] = digest(args.outdir / name)
            evidence["runs"]["adjacency"] = {"wall_seconds": time.monotonic() - started}
            print(f"Built {len(edges)} local edges, retaining all {len(vertices)} observations", flush=True)
            run_r("fit", args.outdir / "observations.csv")
            assignments = pd.read_csv(args.outdir / "assignments.csv", dtype={"label": str}, keep_default_na=False, float_precision="round_trip")
            for variant in ("baseline", "local_feature", "boundary_feature"):
                for seed in args.seeds:
                    selected = assignments[(assignments.variant == variant) & (assignments.seed == seed)]
                    if selected.label.tolist() != observations.label.tolist() or selected.cluster.nunique() != args.target_clusters:
                        raise ValueError("Comparison omitted/reordered observations or failed requested sensitivity K")
                    if not np.array_equal(selected[["x", "y"]].to_numpy(), observations[["x", "y"]].to_numpy()):
                        raise ValueError("Clustering changed source coordinates")
            if len(assignments) != len(observations) * 3 * len(args.seeds):
                raise ValueError("Unexpected additional comparison rows")
            evidence["python_runtime"] = {"version": sys.version, "source_bytecode_isolated": True,
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "tifffile")}}
            evidence["clustering"] = read_json(args.outdir / "clustering_comparison.json")
            evidence["adjacency"] = adjacency_meta
            evidence.update(status="pass", observation_count=len(observations), exact_output_population_and_coordinates=True)
    except Exception as error:
        evidence.update(status="failed", error=str(error))
        raise
    finally:
        after = {path: digest(path) for path in hashes}
        evidence["source_sha256_after"] = after
        evidence["sources_unchanged"] = after == hashes
        evidence["frozen_code_unchanged"] = all(digest(frozen / name) == hashes[str(code_dir / name)] for name in CODE_FILES)
        evidence["generated_artifacts_sha256"] = generated_hashes
        evidence["generated_artifacts_unchanged"] = all(Path(path).is_file() and digest(path) == expected for path, expected in generated_hashes.items())
        if not evidence["sources_unchanged"] or not evidence["frozen_code_unchanged"] or not evidence["generated_artifacts_unchanged"]:
            evidence.update(status="failed", error="Sources, executed snapshot or generated artifacts changed during comparison")
        evidence["output_sha256"] = {str(path.relative_to(args.outdir)): digest(path) for path in args.outdir.iterdir() if path.is_file() and path != receipt}
        write_json(receipt, evidence)
    if evidence["status"] != "pass":
        raise RuntimeError(evidence["error"])
    print(f"Comparison verified: {evidence['observation_count']} observations, three variants, {len(args.seeds)} seeds", flush=True)
    return evidence


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "grid", "grid-metadata", "image", "support", "support-summary", "crop-summary", "resolution", "resolution-image", "outdir"):
        value.add_argument("--" + name, type=Path, required=True)
    value.add_argument("--dimensions", type=int, default=50)
    value.add_argument("--threads", type=int, default=1)
    value.add_argument("--target-clusters", type=int, default=2)
    value.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    value.add_argument("--radius-um", type=float, default=30.)
    value.add_argument("--descriptor-radius-um", type=float, default=2.)
    value.add_argument("--path-step-um", type=float, default=2.)
    value.add_argument("--boundary-sigma", type=float, default=.15)
    value.add_argument("--spatial-weight", type=float, default=.1)
    value.add_argument("--resolution-value", type=float, default=.3)
    return value


if __name__ == "__main__":
    compare(parser().parse_args())
