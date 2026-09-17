#!/usr/bin/env python3
"""Add calibrated neighbourhoods/niches to a canonical cell profile directory.

Inputs are immutable. Graph endpoints, visibility and density always use the
selected support raster's exact pixels through bounded windows. Only optional
domain-boundary distances may use a smaller analysis grid. This cannot recover
gaps that an upstream low-resolution tissue mask never represented.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from analyze_cell_neighborhoods import (KEYS, SupportMask, RasterSupportMask, DomainGrid,
                                       bounded_grid_shape, sample_domain_grid, analyze_cells)
from cell_profile_io import RasterReader, calibration, sha256_file


def assemble(profile_dir, support_mask, shift, resolution_json, outdir, *, domain_mask=None,
             radii_um=(25., 50., 100.), feature_groups=('available',), fixed_k=None, max_k=8,
             seed=17, repeats=5, max_support_pixels=50_000_000,
             feature_weights=None, max_working_mb=1024,
             native_support_mask=None, native_support_coordinates=None,
             native_support_manifest=None,
             support_tile_size=256, support_cache_tiles=16,
             feature_storage="table", row_batch_size=4096, column_batch_size=64, fit_limit=20000):
    source, outdir = Path(profile_dir), Path(outdir)
    if outdir.exists():
        raise FileExistsError("Spatial profiles require a new output directory")
    if feature_storage not in ("table", "arrays"):
        raise ValueError("feature_storage must be table or arrays")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (row_batch_size, column_batch_size)):
        raise ValueError("Neighbourhood row/column batch sizes must be positive integers")
    if isinstance(fit_limit, bool) or not isinstance(fit_limit, int) or fit_limit < 10:
        raise ValueError("Neighbourhood fit_limit must be an integer >=10")
    source_manifest_sha = sha256_file(source / "cell_profiles_manifest.json")
    manifest = json.loads((source / "cell_profiles_manifest.json").read_text())
    cells = pd.read_parquet(source / "cell_profiles.parquet")
    if manifest.get("spatial_graphs") or any(name.startswith("niche_") for name in cells.columns):
        raise ValueError("Reassembly requires the original base profiles, not already assembled neighbourhood/niche profiles")
    if sha256_file(source / "cell_profiles.parquet") != manifest["files"]["cell_profiles.parquet"]:
        raise ValueError("Cell profile content does not match its manifest")
    row_ids = pd.read_csv(source / "feature_rows.csv", dtype=str, keep_default_na=False)
    if not row_ids.equals(cells[["sample_id", "cell_id", "cell_uid"]].astype(str)):
        raise ValueError("Feature rows do not align with canonical cell IDs")
    if len(cells) != manifest["cell_count"] or cells.duplicated(KEYS).any():
        raise ValueError("Canonical cell count/identity mismatch")
    sample = str(manifest["sample_id"])
    if set(cells.sample_id.astype(str)) != {sample}:
        raise ValueError("One specimen per profile directory is required")
    cal = calibration(shift, resolution_json)
    # Relative tolerance at a large slide origin can hide whole-pixel shifts.
    # Permit only sub-millionth-pixel numerical roundoff in the physical origin.
    if not np.isclose(cal["mpp"], manifest["source_mpp"], rtol=1e-6, atol=0) or not np.allclose(
            cal["origin_px"] * cal["mpp"], manifest["crop_origin_um"], rtol=0, atol=cal["mpp"] * 1e-6):
        raise ValueError("Cell-profile and support coordinate frames disagree")
    if manifest.get("crop_size_px") != [cal["width"], cal["height"]]:
        raise ValueError("Cell-profile and support coordinate frames disagree on crop dimensions")
    if bool(native_support_mask) != bool(native_support_coordinates):
        raise ValueError("Native support requires both a mask and explicit native_support_coordinates")
    if native_support_coordinates not in (None, "crop_pixels", "original_pixels"):
        raise ValueError("native_support_coordinates must be crop_pixels or original_pixels")
    native_provenance = None
    native_manifest_sha256 = sha256_file(native_support_manifest) if native_support_manifest else None
    if native_support_manifest:
        if not native_support_mask or native_support_coordinates != "crop_pixels":
            raise ValueError("Image-derived support provenance requires an explicit native crop support mask")
        from build_native_tissue_support import verify_support_bundle
        native_provenance = verify_support_bundle(native_support_manifest,
            image_sha256=manifest["inputs"].get("image_sha256"), support_mask=support_mask,
            shift=shift, resolution_json=resolution_json)
        if Path(native_support_mask).resolve() != (Path(native_support_manifest).parent / "support.tif").resolve():
            raise ValueError("Selected native mask does not match the image-derived support bundle")
    with RasterReader(support_mask) as reader:
        if len(reader.reader.shape) != 2:
            raise ValueError("Support mask must be two dimensional")
        shape = (reader.height, reader.width)
    pixel_size = (cal["width"] * cal["mpp"] / shape[1], cal["height"] * cal["mpp"] / shape[0])
    analysis_shape = bounded_grid_shape(shape, max_support_pixels)
    selected_support, selected_window = support_mask, None
    if native_support_mask:
        selected_support, pixel_size = native_support_mask, (cal["mpp"], cal["mpp"])
        with RasterReader(native_support_mask) as reader:
            if len(reader.reader.shape) != 2:
                raise ValueError("Native support mask must be two dimensional")
            if native_support_coordinates == "crop_pixels":
                if (reader.height, reader.width) != (cal["height"], cal["width"]):
                    raise ValueError("Native crop support must match the exact calibrated crop dimensions")
            else:
                if not np.equal(cal["origin_px"], np.floor(cal["origin_px"])).all():
                    raise ValueError("Original-pixel native support requires integer crop offsets")
                full = json.loads(Path(shift).read_text()).get("full_size", {})
                if (reader.width, reader.height) != (full.get("width"), full.get("height")):
                    raise ValueError("Original-pixel native support requires matching shift full_size dimensions")
                x0, y0 = map(int, cal["origin_px"])
                selected_window = (x0, y0, x0 + cal["width"], y0 + cal["height"])
    if tuple(feature_groups) == ('available',):
        feature_groups = tuple(sorted(manifest.get('feature_blocks', {})))
    elif 'available' in feature_groups:
        raise ValueError("Use available alone, or explicitly name neighbourhood feature groups")
    if not math.isfinite(max_working_mb) or max_working_mb <= 0:
        raise ValueError("Neighbourhood max_working_mb must be positive and finite")
    dimensions = sum(int(manifest.get('feature_blocks', {}).get(name, {}).get('shape', [0, 0])[1]) for name in feature_groups)
    estimated_working_mb = len(cells) * max(1, dimensions * (1 + len(radii_um)) + 16 * len(radii_um)) * 96 / 1024**2
    if feature_storage == "table" and estimated_working_mb > max_working_mb:
        raise ValueError(f"Neighbourhood dense working estimate {estimated_working_mb:.1f} MiB exceeds {max_working_mb:g} MiB; declare more memory or explicitly select fewer feature groups")
    blocks, array_blocks = {}, {}
    for name in feature_groups:
        if name in blocks or name in array_blocks:
            raise ValueError(f"Repeated neighbourhood feature group {name}")
        record = manifest["feature_blocks"].get(name)
        if not record:
            raise ValueError(f"Requested neighbourhood feature block is unavailable: {name}")
        path = (source / record["path"]).resolve()
        if not path.is_relative_to(source.resolve()) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Feature block identity failure: {name}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != tuple(record["shape"]) or values.shape[0] != len(cells):
            raise ValueError(f"Feature block row count mismatch: {name}")
        if feature_storage == "arrays":
            array_blocks[name] = (record, values)
        else:
            frame = pd.DataFrame(values, columns=record.get("feature_names") or [f"feature_{i}" for i in range(values.shape[1])])
            blocks[name] = pd.concat([cells[KEYS].reset_index(drop=True), frame], axis=1)
    selected_support_sha256 = sha256_file(selected_support)
    with RasterSupportMask(selected_support, pixel_size, tuple(manifest["crop_origin_um"]),
            window_xyxy=selected_window, tile_size=support_tile_size,
            cache_tiles=support_cache_tiles) as support:
        supports = {sample: support}
        domains = {}
        if domain_mask:
            # This sampled grid is only a domain-distance approximation. It has
            # no authority over graph visibility, endpoint support or density.
            analysis_mpp = (cal["width"] * cal["mpp"] / analysis_shape[1],
                            cal["height"] * cal["mpp"] / analysis_shape[0])
            if native_provenance:
                # The image-rule derivative changes graph/density support only;
                # keep the original bounded domain-distance definition intact.
                original_mpp = (cal["width"] * cal["mpp"] / shape[1], cal["height"] * cal["mpp"] / shape[0])
                with RasterSupportMask(support_mask, original_mpp, support.origin_um,
                        tile_size=support_tile_size, cache_tiles=support_cache_tiles) as original_support:
                    domain_support = SupportMask(original_support.grid_tissue(analysis_shape), analysis_mpp, support.origin_um)
            else:
                domain_support = SupportMask(support.grid_tissue(analysis_shape), analysis_mpp, support.origin_um)
            domains[sample] = DomainGrid(sample_domain_grid(domain_mask, analysis_shape), domain_support)
        profiles, niches, graphs, summary = analyze_cells(cells, supports, radii_um=radii_um,
            domain_masks=domains, feature_blocks=blocks, fixed_k=fixed_k, seed=seed,
            max_k=max_k, repeats=repeats, feature_weights=feature_weights,
            defer_niches=feature_storage == "arrays", fit_limit=fit_limit)
        if sha256_file(selected_support) != selected_support_sha256:
            raise ValueError("Spatial support changed during neighbourhood computation")
        support_record = {"path": str(Path(selected_support).resolve()), "sha256": sha256_file(selected_support),
            "explicit_native_support": bool(native_support_mask),
            "source_coordinates": native_support_coordinates or "crop_extent_scaled_support_grid",
            "source_grid_matches_native_crop": support.shape == (cal["height"], cal["width"]) and tuple(support.mpp) == (cal["mpp"], cal["mpp"]),
            "view_window_xyxy": list(support.window_xyxy), "shape_yx": list(support.shape),
            "mpp_xy": list(support.mpp), "origin_um_xy": list(support.origin_um),
            "tile_size": support.tile_size, "cache_tiles": support.cache_tiles,
            "graph_and_density_resampling": "none",
            "limitation": "Exact for supplied raster values, not a guarantee of biological tissue detection or gaps absent from the source mask."}
    sorted_index = pd.MultiIndex.from_frame(profiles[KEYS])
    original_index = pd.MultiIndex.from_frame(cells[KEYS])
    order = sorted_index.get_indexer(original_index)
    if (order < 0).any() or len(np.unique(order)) != len(cells):
        raise ValueError("Neighbourhood computation changed canonical identity")
    profiles = profiles.iloc[order].reset_index(drop=True)
    store_groups = {}
    if feature_storage == "arrays":
        from assemble_neighborhood_arrays import build_arrays
        from niche_clustering import discover_niches_streaming
        outdir.mkdir(parents=True)
        profiles, store_groups, read_values = build_arrays(outdir, profiles, graphs, order,
            array_blocks, summary, row_batch_size=row_batch_size,
            column_batch_size=column_batch_size, max_working_mb=max_working_mb)
        sorted_profiles = profiles.iloc[np.argsort(order)].reset_index(drop=True)
        # A new child directory keeps the engine's no-overwrite contract while
        # the owning temporary context removes only this run's fitting scratch.
        with tempfile.TemporaryDirectory(prefix=".niche_fit_", dir=outdir) as scratch:
            niches, summary["niche_discovery"] = discover_niches_streaming(sorted_profiles,
                summary["feature_groups"], read_values, Path(scratch) / "work",
                fixed_k=fixed_k, max_k=max_k, seed=seed, repeats=repeats,
                feature_weights=feature_weights, max_working_mb=max_working_mb,
                row_batch_size=row_batch_size, fit_limit=fit_limit)
    profiles = profiles.merge(niches, on=KEYS, how="left", validate="one_to_one", sort=False)
    if not profiles.cell_uid.equals(cells.cell_uid):
        raise ValueError("Neighbourhood join changed canonical row order")
    outdir.mkdir(parents=True, exist_ok=feature_storage == "arrays")
    if native_provenance:
        bundle = outdir / "neighborhood_support"
        bundle.mkdir()
        for filename in ("support.tif", "reasons.tif", "support_manifest.json"):
            shutil.copy2(Path(native_support_manifest).parent / filename, bundle / filename)
        if sha256_file(bundle / "support_manifest.json") != native_manifest_sha256:
            raise ValueError("Native support manifest changed during neighbourhood computation")
        verify_support_bundle(bundle / "support_manifest.json",
            image_sha256=manifest["inputs"].get("image_sha256"), support_mask=support_mask,
            shift=shift, resolution_json=resolution_json)
        if sha256_file(bundle / "support.tif") != support_record["sha256"]:
            raise ValueError("Native support changed during neighbourhood computation")
        support_record.update({"path": "neighborhood_support/support.tif", "path_basis": "profile_directory",
            "producer_manifest": "neighborhood_support/support_manifest.json",
            "producer_manifest_sha256": sha256_file(bundle / "support_manifest.json"),
            "image_derived_method": native_provenance["method"],
            "biological_validation": native_provenance["biological_validation"],
            "canonical_measurement_support_unchanged": True})
    shutil.copytree(source / "feature_blocks", outdir / "feature_blocks")
    shutil.copy2(source / "feature_rows.csv", outdir / "feature_rows.csv")
    profiles.to_csv(outdir / "cell_profiles.csv", index=False)
    profiles.to_parquet(outdir / "cell_profiles.parquet", index=False)
    if store_groups:
        from neighborhood_feature_io import finalize_store
        manifest["neighborhood_feature_store"] = finalize_store(outdir, store_groups)
    graph_records = {}
    for radius, graph in graphs.items():
        filename = f"neighborhood_graph_{radius:g}um.npz"
        sparse.save_npz(outdir / filename, graph[order][:, order])
        graph_records[f"{radius:g}"] = {"path": filename, "sha256": sha256_file(outdir / filename)}
    # The standalone analyser uses a sorted graph_cells.csv axis. Assembly
    # restores the canonical profile order and exports feature_rows.csv instead.
    # Do not carry the standalone filename/order claim into this bundle.
    summary["graph_contract"] = (
        "CSR values are Euclidean distances in um, including stored zero-distance edges; "
        "both axes follow canonical feature_rows.csv order; no self edges; per-specimen "
        "radius with conservative pixel-supercover tissue traversal; within-row CSR "
        "storage order is not a cell-identity ordering guarantee")
    summary["graph_axes"] = {"path": "feature_rows.csv", "sha256": sha256_file(outdir / "feature_rows.csv"),
        "count": len(cells), "keys": ["sample_id", "cell_id", "cell_uid"], "same_order_on_both_axes": True}
    summary["domain_resampling"] = "nearest pixel centre on bounded analysis grid; used only for approximate boundary distances" if domain_mask else "unavailable"
    summary["graph_support"] = support_record
    summary["selected_cell_feature_groups"] = list(feature_groups)
    summary["feature_storage"] = feature_storage
    summary["dense_working_memory"] = {"estimated_mb": estimated_working_mb, "budget_mb": max_working_mb,
        "materialized": feature_storage == "table",
        "scope": "Dense feature/scaling copies (counterfactual table estimate in arrays mode); excludes sparse graph, raster cache and interpreter overhead"}
    (outdir / "neighborhood_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    manifest["parent_profile_manifest_sha256"] = source_manifest_sha
    manifest["spatial_graphs"] = graph_records
    manifest["neighborhoods"] = {"summary": "neighborhood_summary.json", "radii_um": list(radii_um), "niche_count": summary["niche_discovery"]["selected_k"], "tissue_domain_count_is_independent": True}
    manifest["inputs"]["support_mask_sha256"] = sha256_file(support_mask)
    if native_support_mask:
        manifest["inputs"]["native_support_mask_sha256"] = support_record["sha256"]
    manifest["neighborhoods"]["graph_support"] = support_record
    manifest["files"] = {name: sha256_file(outdir / name) for name in ("cell_profiles.csv", "cell_profiles.parquet", "feature_rows.csv", "neighborhood_summary.json")}
    if native_provenance:
        manifest["files"].update({f"neighborhood_support/{name}": sha256_file(outdir / "neighborhood_support" / name)
            for name in ("support.tif", "reasons.tif", "support_manifest.json")})
    # Completion is written only after all actually consumed base bytes still
    # agree. Copied arrays retain exactly the same content and canonical axes.
    if sha256_file(source / "cell_profiles_manifest.json") != source_manifest_sha:
        raise ValueError("Base profile manifest changed during neighbourhood assembly")
    for name in ("cell_profiles.parquet", "feature_rows.csv"):
        if sha256_file(source / name) != json.loads((source / "cell_profiles_manifest.json").read_text())["files"][name]:
            raise ValueError("Base profile rows changed during neighbourhood assembly")
    for record in manifest.get("feature_blocks", {}).values():
        if sha256_file(outdir / record["path"]) != record["sha256"]:
            raise ValueError("Copied feature block changed during neighbourhood assembly")
    (outdir / "cell_profiles_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return profiles, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("profile-dir", "support-mask", "shift", "resolution-json", "outdir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--domain-mask")
    p.add_argument("--native-support-mask", help="Optional authoritative native-pixel tissue mask; never inferred from domains or nuclei")
    p.add_argument("--native-support-coordinates", choices=("crop_pixels", "original_pixels"), help="Required explicit coordinate frame for --native-support-mask")
    p.add_argument("--native-support-manifest", help="Optional bound image-derived support receipt; copies a verified portable bundle into the profile")
    p.add_argument("--support-tile-size", type=int, default=256)
    p.add_argument("--support-cache-tiles", type=int, default=16)
    p.add_argument("--radii-um", default="25,50,100")
    p.add_argument("--feature-groups", default="available", help="available selects all present blocks; empty string uses neighbourhood statistics only")
    p.add_argument("--feature-weights", default="{}", help="JSON mapping of group to positive weight; own:block and block are separate")
    p.add_argument("--max-working-mb", type=float, default=1024)
    p.add_argument("--feature-storage", choices=("table", "arrays"), default="table", help="arrays stores own/neighbor vectors outside the scalar table and uses bounded fitting")
    p.add_argument("--row-batch-size", type=int, default=4096)
    p.add_argument("--column-batch-size", type=int, default=64)
    p.add_argument("--fit-limit", type=int, default=20000, help="Maximum seeded training cells; scaling and assignment still use every eligible cell")
    p.add_argument("--fixed-k", type=int)
    p.add_argument("--max-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-support-pixels", type=int, default=50_000_000, help="Limit optional domain-distance analysis grid, never downsample graph/density support")
    args = vars(p.parse_args())
    args["radii_um"] = tuple(float(v) for v in args["radii_um"].split(","))
    args["feature_groups"] = tuple(v.strip() for v in args["feature_groups"].split(",") if v.strip())
    args["feature_weights"] = json.loads(args["feature_weights"])
    if not isinstance(args["feature_weights"], dict):
        p.error("--feature-weights must be a JSON object")
    cells, manifest = assemble(**args)
    print(json.dumps({"cells": len(cells), "niches": manifest["neighborhoods"]["niche_count"]}))


if __name__ == "__main__":
    main()
