#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def _normalize_compression(compression: str) -> str:
    value = (compression or "").strip().lower()
    if value in {"jpeg", "jpg", "tiff_jpeg"}:
        return "jpeg"
    if value in {"deflate", "zlib", "adobe_deflate"}:
        return "deflate"
    if value in {"lzw", "none", "uncompressed"}:
        return value
    return value or "jpeg"


def _parse_input_region(raw_value: str) -> tuple[str, int | None]:
    value = (raw_value or "").strip()
    if not value:
        return "", None
    match = re.search(r"(?i)(scanregion)(\d+)", value)
    if match:
        idx = int(match.group(2))
        return f"ScanRegion{idx}", idx
    if value.isdigit():
        idx = int(value)
        return f"ScanRegion{idx}", idx
    return value, None


def _select_scene_index(scene_names: tuple[str, ...], region_label: str, region_index: int | None) -> int:
    if not scene_names:
        return 0
    if region_label:
        wanted = region_label.lower()
        for idx, scene_name in enumerate(scene_names):
            if scene_name.lower() == wanted:
                return idx
        for idx, scene_name in enumerate(scene_names):
            if wanted in scene_name.lower():
                return idx
    if region_index is not None:
        if region_index < 0 or region_index >= len(scene_names):
            raise ValueError(
                f"Requested CZI region index {region_index} is out of range for scenes: {', '.join(scene_names)}"
            )
        return region_index
    if len(scene_names) > 1:
        raise ValueError(
            "CZI contains multiple regions/scenes. Provide --input-region or a region-specific GeoJSON "
            "such as '<image>.czi - ScanRegion0.geojson'."
        )
    return 0


def _squeeze_axis(arr, dims, dim_name):
    axis = dims.index(dim_name)
    arr = arr.take(indices=0, axis=axis)
    dims.pop(axis)
    return arr, dims


def _normalize_czi_array(data, dims_order: str):
    import numpy as np

    arr = np.asarray(data)
    dims = list(dims_order)
    if len(dims) != arr.ndim:
        raise ValueError(f"Unexpected CZI dims/order mismatch: dims='{dims_order}' shape={arr.shape}")

    for dim_name in list(dims):
        if dim_name in {"Y", "X", "C", "S"}:
            continue
        arr, dims = _squeeze_axis(arr, dims, dim_name)

    if "C" in dims and "S" in dims:
        arr, dims = _squeeze_axis(arr, dims, "C")

    channel_dim = "S" if "S" in dims else ("C" if "C" in dims else "")
    if channel_dim:
        axis = dims.index(channel_dim)
        arr = np.moveaxis(arr, axis, -1)
        dims.pop(axis)
        dims.append(channel_dim)

    if "Y" not in dims or "X" not in dims:
        raise ValueError(f"CZI scene does not contain Y/X axes after normalization: dims={dims}")

    ordered_axes = [dims.index("Y"), dims.index("X")]
    if channel_dim:
        ordered_axes.append(dims.index(channel_dim))
    arr = np.transpose(arr, ordered_axes)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] == 2:
            arr = np.concatenate([arr, arr[:, :, 1:2]], axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]

    return arr


def _normalize_czi_mosaic_array(data):
    import numpy as np

    arr = np.asarray(data)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] == 2:
            arr = np.concatenate([arr, arr[:, :, 1:2]], axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]

    if arr.ndim not in {2, 3}:
        raise ValueError(f"Unexpected CZI mosaic output shape after normalization: {arr.shape}")

    return arr


def _write_array_as_tiff(array, dst: Path, compression: str, tile: int) -> str:
    from tifffile import imwrite

    compression = _normalize_compression(compression)
    photometric = None
    if getattr(array, "ndim", 0) == 3 and array.shape[2] >= 3:
        photometric = "rgb"

    write_kwargs = dict(
        compression=compression,
        photometric=photometric,
        bigtiff=getattr(array, "nbytes", 0) >= (2 * 1024 ** 3),
    )
    if tile and getattr(array, "ndim", 0) >= 2:
        write_kwargs["tile"] = (tile, tile)

    try:
        imwrite(str(dst), array, **write_kwargs)
    except Exception:
        write_kwargs.pop("tile", None)
        imwrite(str(dst), array, **write_kwargs)
    return "tifffile"


def _convert_czi(src: Path, dst: Path, compression: str, tile: int, input_region: str) -> str:
    try:
        from aicspylibczi import CziFile
    except Exception as exc:
        raise RuntimeError(
            "CZI support requires the 'aicspylibczi' package baked into the container."
        ) from exc

    region_label, region_index = _parse_input_region(input_region)
    czi = CziFile(str(src))
    is_mosaic = bool(czi.is_mosaic())

    if is_mosaic:
        scene_boxes = czi.get_all_mosaic_scene_bounding_boxes() or czi.get_all_scene_bounding_boxes()
    else:
        scene_boxes = czi.get_all_scene_bounding_boxes()

    scene_indices = tuple(sorted(scene_boxes)) if scene_boxes else (0,)
    scene_names = tuple(f"ScanRegion{idx}" for idx in scene_indices)
    selected_pos = _select_scene_index(scene_names, region_label, region_index)
    selected_scene = scene_indices[selected_pos]

    if is_mosaic:
        if selected_scene not in scene_boxes:
            raise ValueError(f"Unable to resolve mosaic bounding box for CZI scene {selected_scene}")
        bbox = scene_boxes[selected_scene]
        region = (bbox.x, bbox.y, bbox.w, bbox.h)
        data = czi.read_mosaic(region=region, scale_factor=1.0, C=0)
        array = _normalize_czi_mosaic_array(data)
        backend_name = "aicspylibczi-mosaic"
    else:
        data, shape_info = czi.read_image(S=selected_scene)
        dims_order = "".join(dim_name for dim_name, _ in shape_info)
        array = _normalize_czi_array(data, dims_order)
        backend_name = "aicspylibczi-scene"

    backend = _write_array_as_tiff(array, dst, compression, tile)
    selected_name = scene_names[selected_pos] if selected_pos < len(scene_names) else str(selected_scene)
    return f"{backend}+{backend_name}(scene={selected_name})"


def _convert_with_pyvips(src: Path, dst: Path, compression: str, tile: int, quality: int, pyramid: bool) -> str:
    import pyvips

    compression = _normalize_compression(compression)
    image = pyvips.Image.new_from_file(str(src), access="sequential")
    if image.bands > 4:
        image = image[:3]
    elif image.bands == 4:
        image = image[:3]

    try:
        if image.bands >= 3 and image.interpretation not in ("srgb", "rgb"):
            image = image.colourspace("srgb")
    except Exception:
        pass

    kwargs = dict(
        compression=compression,
        tile=True,
        tile_width=tile,
        tile_height=tile,
        pyramid=bool(pyramid),
        bigtiff=True,
        properties=False,
    )
    if compression == "jpeg":
        kwargs["Q"] = quality
    elif compression in {"deflate", "lzw"}:
        kwargs["predictor"] = True

    image.tiffsave(str(dst), **kwargs)
    return "pyvips"


def _convert_with_pillow(src: Path, dst: Path, compression: str, quality: int) -> str:
    import numpy as np
    from PIL import Image, ImageSequence
    from tifffile import imwrite

    compression = _normalize_compression(compression)
    image = Image.open(src)
    frame = next(ImageSequence.Iterator(image), image)
    if frame.mode not in ("RGB", "RGBA", "L", "I;16", "I"):
        frame = frame.convert("RGB")

    if compression == "jpeg":
        save_kwargs = dict(format="TIFF", compression="tiff_jpeg", quality=quality)
        if frame.mode == "RGBA":
            frame = frame.convert("RGB")
        frame.save(str(dst), **save_kwargs)
        return "pillow"

    array = np.array(frame)
    photometric = None
    if array.ndim == 3 and array.shape[2] >= 3:
        if array.shape[2] > 3:
            array = array[..., :3]
        photometric = "rgb"

    imwrite(
        str(dst),
        array,
        compression=compression,
        photometric=photometric,
        bigtiff=array.nbytes >= (2 * 1024 ** 3),
    )
    return "pillow"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert generic image inputs to TIFF with configurable compression.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output TIFF path.")
    parser.add_argument("--input-region", default="", help="Optional input region selector (used for CZI scene extraction).")
    parser.add_argument("--compression", default="jpeg", help="TIFF compression (default: jpeg).")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality for lossy TIFF compression (default: 90).")
    parser.add_argument("--tile", type=int, default=512, help="Tile size for streamed WSI writes (default: 512).")
    parser.add_argument("--pyramid", action="store_true", help="Write a pyramidal TIFF.")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".czi":
        backend = _convert_czi(src, dst, args.compression, args.tile, args.input_region)
        print(
            f"[INFO] Converted {src.name} -> {dst.name} with {backend} "
            f"(compression={args.compression}, input_region={args.input_region or 'auto'})"
        )
        return

    vips_error = None
    try:
        backend = _convert_with_pyvips(src, dst, args.compression, args.tile, args.quality, args.pyramid)
        print(
            f"[INFO] Converted {src.name} -> {dst.name} with {backend} "
            f"(compression={args.compression}, pyramid={args.pyramid})"
        )
        return
    except Exception as exc:
        vips_error = exc

    backend = _convert_with_pillow(src, dst, args.compression, args.quality)
    print(
        f"[WARN] pyvips conversion failed for {src.name}: {vips_error}. "
        f"Fell back to {backend} (compression={args.compression})."
    )


if __name__ == "__main__":
    main()
