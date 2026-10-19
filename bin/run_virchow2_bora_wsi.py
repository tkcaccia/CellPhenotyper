#!/usr/bin/env python3
"""CellPhenotyper -> current Bora, using bounded calibrated WSI windows.

Input labels come only from CellPhenotyper's published coarse tile mask.
The expert reference is not an argument. Virchow2 sees the original RGB at
0.5 um/px, with disjoint 224-pixel encoder tiles. Bora supplies RGB9+19,
Bora-AF geometry, Neighbor-context fastPLS identity, and compact GeoJSON.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from PIL import Image
import tifffile

from ome_tiff_metadata import read_mpp_json
from virchow2_runtime import CHECKPOINT_SHA256, REVISION, load_model, sha256, spatial_tokens_nonoverlap


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_mask_reader(path: Path):
    import pyvips

    image = pyvips.Image.new_from_file(str(path), access="random")
    if image.bands != 1 or image.format not in ("uchar", "ushort", "uint"):
        raise ValueError(f"Expected a single-channel integer mask: {path}")
    return image


class CalibratedSlide:
    """Windowed CellPhenotyper crop with independently checked MPP sidecar."""

    def __init__(self, image_path: Path, resolution_json: Path, grid: dict):
        import pyvips
        from bora.wsi import NativeWindow, ome_mpp

        self.NativeWindow = NativeWindow
        self.image = pyvips.Image.new_from_file(str(image_path), access="random")
        if self.image.bands != 3 or self.image.format != "uchar":
            raise ValueError("Expected a uint8 RGB tissue image")
        self.shape_xy = (self.image.width, self.image.height)
        self.native_mpp_xy = np.asarray(read_mpp_json(resolution_json, fallback=0.0), dtype=float)
        declared = np.asarray([grid["source_mpp_x"], grid["source_mpp_y"]], dtype=float)
        if (not np.all(np.isfinite(self.native_mpp_xy)) or
                not np.allclose(self.native_mpp_xy, declared, rtol=.002, atol=.0005)):
            raise ValueError("Crop resolution sidecar disagrees with the Virchow2 grid")
        try:
            (width, height), ome_spacing = ome_mpp(image_path)
        except ValueError:
            pass  # The prepared crop may be plain TIFF; the mandatory sidecar is authoritative.
        else:
            if (width, height) != self.shape_xy or not np.allclose(
                    ome_spacing, self.native_mpp_xy, rtol=.002, atol=.0005):
                raise ValueError("OME physical sizes disagree with the crop resolution sidecar")

    def windows(self, core_pixels: int, halo_pixels: int):
        width, height = self.shape_xy
        minimum_xy = np.ceil(224 * .5 / self.native_mpp_xy).astype(int)
        if width < minimum_xy[0] or height < minimum_xy[1]:
            raise ValueError("The tissue crop is physically smaller than one Virchow2/Bora field")
        for y in range(0, height, core_pixels):
            for x in range(0, width, core_pixels):
                cw, ch = min(core_pixels, width - x), min(core_pixels, height - y)
                x0, y0 = max(0, x - halo_pixels), max(0, y - halo_pixels)
                x1, y1 = min(width, x + cw + halo_pixels), min(height, y + ch + halo_pixels)
                # Narrow WSI edge cores still need Bora's minimum 224-pixel
                # working field. Extend the read halo without moving the core.
                x0 = max(0, min(x0, width - int(minimum_xy[0])))
                y0 = max(0, min(y0, height - int(minimum_xy[1])))
                x1 = min(width, max(x1, x0 + int(minimum_xy[0])))
                y1 = min(height, max(y1, y0 + int(minimum_xy[1])))
                yield self.NativeWindow(x0, y0, x1 - x0, y1 - y0, x, y, cw, ch)

    def read(self, window):
        region = self.image.crop(window.x, window.y, window.width, window.height)
        native = np.ndarray((window.height, window.width, 3), dtype=np.uint8,
                            buffer=region.write_to_memory()).copy()
        working_xy = np.maximum(1, np.rint(np.asarray([window.width, window.height]) *
                                         self.native_mpp_xy / .5).astype(int))
        working = cv2.resize(native, tuple(working_xy), interpolation=cv2.INTER_AREA)
        working_mpp = self.native_mpp_xy * np.asarray(
            [window.width, window.height]) / working_xy
        if np.any(np.abs(working_mpp - .5) > .01):
            raise ValueError("WSI window could not be resampled to Virchow2's physical scale")
        return native, working, working_mpp


def mask_window(reader, window, source_shape_xy, *, binary: bool) -> np.ndarray:
    """Map exact native window bounds onto a potentially coarser mask raster."""
    sx = reader.width / source_shape_xy[0]
    sy = reader.height / source_shape_xy[1]
    x0 = max(0, min(reader.width - 1, int(np.floor(window.x * sx))))
    y0 = max(0, min(reader.height - 1, int(np.floor(window.y * sy))))
    x1 = max(x0 + 1, min(reader.width, int(np.ceil((window.x + window.width) * sx))))
    y1 = max(y0 + 1, min(reader.height, int(np.ceil((window.y + window.height) * sy))))
    region = reader.crop(x0, y0, x1 - x0, y1 - y0)
    dtype = {"uchar": np.uint8, "ushort": np.uint16, "uint": np.uint32}[reader.format]
    values = np.ndarray((y1 - y0, x1 - x0), dtype=dtype, buffer=region.write_to_memory())
    # Absolute pixel-center mapping avoids small shifts at each WSI window.
    xs = np.minimum(reader.width - 1,
                    np.floor((np.arange(window.x, window.x + window.width) + .5) * sx).astype(int)) - x0
    ys = np.minimum(reader.height - 1,
                    np.floor((np.arange(window.y, window.y + window.height) + .5) * sy).astype(int)) - y0
    result = np.asarray(values[np.ix_(ys, xs)])
    return result != 0 if binary else result


def working_coarse(native: np.ndarray, working_shape_yx: tuple[int, int]) -> np.ndarray:
    if native.ndim != 2 or native.dtype.kind not in "iu":
        raise ValueError("A 2D integer coarse tile mask is required")
    if np.any(native > 255):
        raise ValueError("Bora supports at most 255 anonymous coarse IDs")
    return cv2.resize(native.astype(np.uint8), working_shape_yx[::-1],
                      interpolation=cv2.INTER_NEAREST)


def remap_unobserved_partial_edges(coarse: np.ndarray, observed_ids: np.ndarray,
                                   window, source_shape_xy: tuple[int, int],
                                   grid_stride_source_px: int) -> tuple[np.ndarray, dict]:
    """Assign only unsupported *partial outer tiles* to adjacent observed groups.

    Bora's fixed RGB conversion takes 90-pixel, nearly pure observations. A
    last grid column only a few source pixels wide can acquire its own KODAMA
    label but has no eligible observation. Such padded edge tiles are assigned
    to their nearest supported group; an unsupported interior group is an error.
    """
    from scipy import ndimage as ndi

    present = np.unique(coarse[coarse > 0])
    missing = np.setdiff1d(present, np.asarray(observed_ids, dtype=present.dtype))
    if not len(missing):
        return coarse, dict(merged_partial_edge_groups=[], merged_pixels=0)
    width, height = source_shape_xy
    stride = int(grid_stride_source_px)
    if stride < 1:
        raise ValueError("Positive source grid stride required")
    allowed = np.zeros(coarse.shape, dtype=bool)
    if width % stride and window.x + window.width == width:
        edge_x = (width // stride) * stride
        start_x = int(np.floor((edge_x - window.x) * coarse.shape[1] / window.width)) - 2
        allowed[:, max(0, start_x):] = True
    if height % stride and window.y + window.height == height:
        edge_y = (height // stride) * stride
        start_y = int(np.floor((edge_y - window.y) * coarse.shape[0] / window.height)) - 2
        allowed[max(0, start_y):, :] = True
    unsupported = np.isin(coarse, missing)
    if np.any(unsupported & ~allowed):
        raise ValueError(f"Coarse groups without eligible Virchow2 observations occur away "
                         f"from partial outer tiles: {missing.tolist()}")
    supported = (coarse > 0) & ~unsupported
    if not supported.any():
        raise ValueError("No supported coarse group exists for the partial edge")
    nearest = ndi.distance_transform_edt(~supported, return_distances=False,
                                         return_indices=True)
    result = coarse.copy()
    result[unsupported] = coarse[tuple(axis[unsupported] for axis in nearest)]
    if np.any(result[unsupported] == 0):
        raise ValueError("Partial edge could not be assigned to an observed group")
    return result, dict(merged_partial_edge_groups=missing.astype(int).tolist(),
                        merged_pixels=int(unsupported.sum()))


def make_palette(count: int) -> np.ndarray:
    from colorsys import hsv_to_rgb

    return np.asarray([
        [round(255 * channel) for channel in hsv_to_rgb((index * .61803398875) % 1, .72, .91)]
        for index in range(count)
    ], dtype=np.uint8)


def cleanup_wsi(output: Path, shape: tuple[int, int], *, core: int = 2048) -> dict:
    from bora.mask_cleanup import fill_small_regions

    raw = np.memmap(output / "classified.raw.dat", mode="r", dtype=np.uint8, shape=shape)
    cleaned = np.memmap(output / "classified.cleaned.dat", mode="w+", dtype=np.uint8, shape=shape)
    changed = 0
    for y in range(0, shape[0], core):
        for x in range(0, shape[1], core):
            y1, x1 = min(shape[0], y + core), min(shape[1], x + core)
            ay0, ax0 = max(0, y - 32), max(0, x - 32)
            ay1, ax1 = min(shape[0], y1 + 32), min(shape[1], x1 + 32)
            local = np.asarray(raw[ay0:ay1, ax0:ax1])
            if not local.any():
                cleaned[y:y1, x:x1] = 0
                continue
            result, _ = fill_small_regions(local, 20)
            part = result[y - ay0:y1 - ay0, x - ax0:x1 - ax0]
            changed += int(np.count_nonzero(part != raw[y:y1, x:x1]))
            cleaned[y:y1, x:x1] = part
    cleaned.flush()
    report = dict(native_shape_yx=list(shape), minimum_area_native_pixels=20,
                  strictly_less_than_threshold=True, halo_pixels=32,
                  changed_pixels=changed)
    atomic_json(output / "CLEANED_COMPLETE.json", report)
    return report


def run(args) -> dict:
    # Bora includes a compiled, vendored fastPLS core. Import the installed
    # package instead of putting an unbuilt source checkout ahead of it.
    try:
        from bora._vendor.fastpls import PLS  # noqa: F401
        from bora.classifier import NeighborContextClassifier
        from bora.pipeline import Case, prepare
        from bora.wsi_export import export as export_vectors, fit_window
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Install the current Bora wheel, including its native fastPLS extension, "
            "in this process's Python environment before running CellPhenotyper"
        ) from exc

    if args.bora_source:
        source_version = args.bora_source / "pyproject.toml"
        if not source_version.is_file() or 'version = "0.2.1"' not in source_version.read_text():
            raise ValueError("The staged Bora source is not the expected 0.2.1 release")

    import torch
    from threadpoolctl import threadpool_limits

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.core_pixels < 256 or args.halo_pixels < 0 or args.deadline_seconds < 180:
        raise ValueError("Use a core >=256 pixels, nonnegative halo and >=180-second deadline")
    grid = json.loads(args.grid_metadata.read_text(encoding="utf-8"))
    slide = CalibratedSlide(args.image, args.resolution_json, grid)
    if (grid.get("tile_layout") != "nonoverlapping" or
            int(grid.get("context_overlap_source_px", -1)) != 0 or
            int(grid["model_tile_size_px"]) != 224 or
            abs(float(grid["target_mpp"]) - .5) > 1e-6 or
            [int(grid["image_width_px"]), int(grid["image_height_px"])] != list(slide.shape_xy) or
            not np.allclose(slide.native_mpp_xy,
                            [float(grid["source_mpp_x"]), float(grid["source_mpp_y"])], atol=.001)):
        raise ValueError("CellPhenotyper grid is not the calibrated, disjoint Virchow2 input")
    coarse_reader = load_mask_reader(args.coarse)
    tissue_reader = load_mask_reader(args.tissue)
    if (coarse_reader.width, coarse_reader.height) != slide.shape_xy:
        raise ValueError("Coarse tile mask must be at native WSI pixel resolution")
    model = NeighborContextClassifier.load(args.classifier)
    torch.set_num_threads(2)
    encoder = load_model(args.checkpoint, device=args.device)
    # Bound model memory by physical field of view, independent of whether the
    # native WSI was scanned at 0.25, 0.5, or 1.0 µm/pixel.
    core_native = max(1, int(np.floor(args.core_pixels * .5 / max(slide.native_mpp_xy))))
    halo_native = max(0, int(np.ceil(args.halo_pixels * .5 / min(slide.native_mpp_xy))))
    windows = list(slide.windows(core_native, halo_native))
    selected = ([int(args.window_index)] if args.window_index is not None
                else list(range(len(windows))))
    if any(index < 0 or index >= len(windows) for index in selected):
        raise ValueError("WSI window index outside the slide")
    args.output.mkdir(parents=True)
    started = monotonic()
    manifest = dict(image=str(args.image.resolve()), image_sha256=sha256(args.image),
                    coarse_sha256=sha256(args.coarse), tissue_sha256=sha256(args.tissue),
                    grid_metadata_sha256=sha256(args.grid_metadata),
                    resolution_json_sha256=sha256(args.resolution_json),
                    classifier_sha256=sha256(args.classifier),
                    checkpoint_sha256=CHECKPOINT_SHA256, encoder_revision=REVISION,
                    source_shape_xy=list(slide.shape_xy),
                    source_mpp_xy=slide.native_mpp_xy.tolist(),
                    encoder_tile_px=224, encoder_stride_px=224, encoder_tile_overlap_px=0,
                    inner_square_used=False,
                    requested_core_working_pixels=args.core_pixels,
                    requested_halo_working_pixels=args.halo_pixels,
                    core_native_pixels=core_native, halo_native_pixels=halo_native,
                    total_windows=len(windows),
                    selected_windows=selected if args.window_index is not None else "all",
                    expert_annotation_read=False)
    atomic_json(args.output / "RUN_MANIFEST.json", manifest)
    full = args.window_index is None
    native_shape = (slide.shape_xy[1], slide.shape_xy[0])
    raw = (np.memmap(args.output / "classified.raw.dat", mode="w+", dtype=np.uint8,
                     shape=native_shape) if full else None)
    if raw is not None:
        raw[:] = 0
        raw.flush()
    completed = 0
    with threadpool_limits(limits=2):
        for index in selected:
            if monotonic() - started > args.deadline_seconds - 120:
                atomic_json(args.output / "STOPPED_AT_CHECKPOINT.json",
                            dict(next_index=index, completed=completed, reason="deadline margin"))
                raise TimeoutError("Bora WSI stage stopped before its deadline; no downstream job may start")
            window = windows[index]
            stamp = monotonic()
            native, working, mpp = slide.read(window)
            support_native = mask_window(tissue_reader, window, slide.shape_xy, binary=True)
            coarse_native = mask_window(coarse_reader, window, slide.shape_xy, binary=False)
            if not np.any(support_native):
                native_result = np.zeros(support_native.shape, np.uint8)
                details = dict(background=True, encoder_ran=False)
            else:
                coarse = working_coarse(coarse_native, working.shape[:2])
                if not np.any(coarse):
                    raise ValueError(f"Tissue-bearing WSI window {index} has no coarse tile group")
                support = cv2.resize(support_native.astype(np.uint8), working.shape[1::-1],
                                     interpolation=cv2.INTER_NEAREST) > 0
                tokens, base_meta = spatial_tokens_nonoverlap(
                    encoder, working, device=args.device, batch=args.batch)
                from bora.conversion import observations
                _, observed_ids = observations(tokens, coarse)
                coarse, edge_audit = remap_unobserved_partial_edges(
                    coarse, observed_ids, window, slide.shape_xy,
                    int(grid["grid_stride_source_px"]))
                variants = {}
                for name, factor in (("scale_1", 2), ("scale_2", 4)):
                    resized = np.asarray(Image.fromarray(working).resize(
                        (max(1, round(working.shape[1] / factor)),
                         max(1, round(working.shape[0] / factor))),
                        Image.Resampling.LANCZOS))
                    features, _ = spatial_tokens_nonoverlap(
                        encoder, resized, device=args.device, batch=args.batch)
                    variants[name] = dict(values=features, shape_yx=list(resized.shape[:2]))
                case = Case(working, coarse, support, tokens, mpp,
                            slide.native_mpp_xy, support_native,
                            palette=make_palette(int(coarse.max())), variants=variants,
                            provenance=dict(**manifest, window_index=index, spatial_tokens=base_meta))
                prepared = prepare(case, wsi_window=True)
                identities = model.predict(prepared["descriptors"])
                lookup = np.r_[np.uint8(0), identities]
                native_result = lookup[prepared["regions"]]
                native_result[~support_native] = 0
                details = dict(background=False, encoder_ran=True,
                               group_ids=prepared["group_ids"].tolist(),
                               group_classes=identities.tolist(),
                               conversion=prepared["details"]["conversion"],
                               appearance=prepared["details"]["appearance"],
                               working_shape_yx=list(working.shape[:2]),
                               working_mpp_xy=mpp.tolist(), spatial_tokens=base_meta,
                               partial_edge_assignment=edge_audit)
            core_y, core_x = window.core_slices
            core = native_result[core_y, core_x]
            if full:
                raw[window.core_y:window.core_y + window.core_height,
                    window.core_x:window.core_x + window.core_width] = core
                raw.flush()
            else:
                tifffile.imwrite(args.output / f"window-{index:03d}.tif", core,
                                 compression="deflate")
            atomic_json(args.output / f"window-{index:03d}.json",
                        dict(index=index, core_xywh=[window.core_x, window.core_y,
                                                     window.core_width, window.core_height],
                             native_window_xywh=[window.x, window.y, window.width, window.height],
                             seconds=monotonic() - stamp, **details))
            completed += 1
            print(json.dumps(dict(window=index, total=len(windows), seconds=round(monotonic()-stamp, 2),
                                  **{key: details[key] for key in ("background", "encoder_ran")})),
                  flush=True)
            del native, working, support_native, coarse_native, native_result
    if not full:
        index = selected[0]
        window = windows[index]
        pilot_raw = tifffile.imread(args.output / f"window-{index:03d}.tif")
        features, rendered, vector_audit = fit_window(
            pilot_raw, window.core_x, window.core_y,
            dict(native_mpp_xy=slide.native_mpp_xy.tolist(),
                 class_names={str(key): value for key, value in model.names_.items()}))
        tifffile.imwrite(args.output / "mask.ome.tif", rendered, ome=True,
                         compression="deflate", photometric="minisblack",
                         metadata=dict(axes="YX", PhysicalSizeX=float(slide.native_mpp_xy[0]),
                                       PhysicalSizeY=float(slide.native_mpp_xy[1]),
                                       PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm"))
        atomic_json(args.output / "mask.geojson",
                    dict(type="FeatureCollection", coordinate_space="native_image_pixel_edges",
                         native_shape_yx=list(native_shape),
                         native_mpp_xy=slide.native_mpp_xy.tolist(),
                         pilot_core_xywh=[window.core_x, window.core_y,
                                          window.core_width, window.core_height],
                         features=features))
        report = dict(**manifest, windows_completed=completed, pilot_only=True,
                      vector_audit=vector_audit, wall_seconds=monotonic() - started)
        atomic_json(args.output / "PILOT_COMPLETE.json", report)
        return report
    raw.flush()
    report = dict(**manifest, native_shape_yx=list(native_shape),
                  native_mpp_xy=slide.native_mpp_xy.tolist(),
                  class_names={str(key): value for key, value in model.names_.items()},
                  classifier="Neighbor context, native fastPLS",
                  windows_completed=completed, pilot_only=False,
                  model_load_refit_seconds=model.fit_seconds_)
    atomic_json(args.output / "CLASSIFIED_COMPLETE.json", report)
    del encoder
    cleanup_wsi(args.output, native_shape)
    # The CPU vector workers must never inherit the CUDA runtime by fork.
    multiprocessing.set_start_method("spawn", force=True)
    vector = export_vectors(args.output, core=core_native, fit_core=min(2048, core_native),
                            workers=args.vector_workers)
    final = np.memmap(args.output / "classified.final.dat", mode="r", dtype=np.uint8,
                      shape=native_shape)
    tifffile.imwrite(args.output / "mask.ome.tif", final, bigtiff=True, tile=(512, 512),
                     compression="deflate", photometric="minisblack", ome=True,
                     metadata=dict(axes="YX", PhysicalSizeX=float(slide.native_mpp_xy[0]),
                                   PhysicalSizeY=float(slide.native_mpp_xy[1]),
                                   PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm"))
    report["vector"] = vector
    report["wall_seconds"] = monotonic() - started
    atomic_json(args.output / "PIPELINE_COMPLETE.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ("image", "coarse", "tissue", "resolution-json", "grid-metadata",
                "classifier", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--bora-source", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--core-pixels", type=int, default=2048)
    parser.add_argument("--halo-pixels", type=int, default=384)
    parser.add_argument("--deadline-seconds", type=int, default=21600)
    parser.add_argument("--vector-workers", type=int, default=2)
    parser.add_argument("--window-index", type=int)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(dict(output=str(args.output), windows_completed=result["windows_completed"],
                          pilot_only=result["pilot_only"], wall_seconds=result["wall_seconds"])),
          flush=True)


if __name__ == "__main__":
    main()
