#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape


def _normalize_compression(compression: str) -> str:
    value = (compression or "").strip().lower()
    if value in {"jpeg", "jpg", "tiff_jpeg"}:
        return "jpeg"
    if value in {"deflate", "zlib", "adobe_deflate"}:
        return "deflate"
    if value in {"lzw", "none", "uncompressed"}:
        return value
    return value or "jpeg"


def _normalize_channel_order(channel_order: str) -> str:
    order = ("RGB" if channel_order is None else channel_order).strip().upper()
    if len(order) != 3 or set(order) != {"R", "G", "B"}:
        raise ValueError(
            "--channel-order must contain R, G, and B exactly once "
            "(one of RGB, RBG, GRB, GBR, BRG, or BGR)."
        )
    return order


def _rgb_indices_from_source_order(channel_order: str) -> tuple[int, int, int]:
    """Return source-plane indices needed to write canonical R, G, B output.

    ``channel_order`` describes the meaning of source planes 0, 1, and 2.
    For example, BGR maps source plane 2 to output red, plane 1 to green,
    and plane 0 to blue.
    """
    order = _normalize_channel_order(channel_order)
    return tuple(order.index(channel) for channel in "RGB")


def _is_rgb_compatible_photometric(value: str) -> bool:
    """Return whether TIFF storage decodes to canonical three-channel RGB."""
    return str(value or "").strip().upper() in {"RGB", "YCBCR"}


def _ome_dtype(dtype_name: str) -> str:
    mapping = {
        "uint8": "uint8",
        "int8": "int8",
        "uint16": "uint16",
        "int16": "int16",
        "uint32": "uint32",
        "int32": "int32",
        "float32": "float",
        "float64": "double",
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported planar RGB pixel type: {dtype_name}") from exc


def _unit_to_micrometres(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    normalized = (unit or "um").strip().lower().replace("μ", "µ")
    if normalized in {"µm", "um", "micrometer", "micrometre"}:
        return value
    if normalized in {"nm", "nanometer", "nanometre"}:
        return value / 1000.0
    if normalized in {"mm", "millimeter", "millimetre"}:
        return value * 1000.0
    return value


def _ome_physical_sizes_micrometres(ome_xml: str | None) -> tuple[float | None, float | None]:
    if not ome_xml:
        return None, None
    try:
        root = ET.fromstring(ome_xml)
        pixels = root.find(".//{*}Pixels")
        if pixels is None:
            return None, None
        x = float(pixels.attrib["PhysicalSizeX"]) if pixels.attrib.get("PhysicalSizeX") else None
        y = float(pixels.attrib["PhysicalSizeY"]) if pixels.attrib.get("PhysicalSizeY") else x
        return (
            _unit_to_micrometres(x, pixels.attrib.get("PhysicalSizeXUnit")),
            _unit_to_micrometres(y, pixels.attrib.get("PhysicalSizeYUnit")),
        )
    except (ET.ParseError, TypeError, ValueError):
        return None, None


def _build_rgb_ome_xml(
    image_name: str,
    width: int,
    height: int,
    dtype_name: str,
    mpp_x: float | None,
    mpp_y: float | None,
) -> str:
    physical = ""
    if mpp_x is not None and mpp_y is not None:
        physical = (
            f' PhysicalSizeX="{mpp_x:.12g}" PhysicalSizeXUnit="µm"'
            f' PhysicalSizeY="{mpp_y:.12g}" PhysicalSizeYUnit="µm"'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0" Name="{escape(image_name, {chr(34): "&quot;"})}">'
        f'<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="{_ome_dtype(dtype_name)}"'
        f' SizeX="{width}" SizeY="{height}" SizeC="3" SizeZ="1" SizeT="1"'
        f' Interleaved="true"{physical}>'
        '<Channel ID="Channel:0:0" Name="RGB" SamplesPerPixel="3"><LightPath/></Channel>'
        '<TiffData IFD="0" PlaneCount="1"/>'
        '</Pixels></Image></OME>'
    )


def _ensure_explicit_rgb_ome_tiff(
    path: Path,
    channel_order: str,
    compression: str,
    quality: int,
    tile: int,
) -> str:
    """Stream a three-plane grayscale OME-TIFF into explicit RGB storage.

    Bio-Formats commonly preserves brightfield VSI colour as three CYX
    MINISBLACK planes.  This is scientifically valid OME data but ambiguous to
    generic pathology readers.  The rewrite joins those planes into canonical
    interleaved RGB while preserving physical calibration and a SubIFD pyramid.
    """
    import tifffile

    order = _normalize_channel_order(channel_order)
    with tifffile.TiffFile(str(path)) as tif:
        if not tif.series:
            raise RuntimeError(f"Converted TIFF contains no image series: {path}")
        series = tif.series[0]
        axes = str(getattr(series, "axes", "") or "")
        shape = tuple(int(value) for value in series.shape)
        page = series.pages[0]
        photometric = str(getattr(getattr(page, "photometric", None), "name", getattr(page, "photometric", ""))).upper()
        samples_per_pixel = int(getattr(page, "samplesperpixel", 1) or 1)
        channel_axis = "C" if "C" in axes else ("S" if "S" in axes else "")
        channel_count = shape[axes.index(channel_axis)] if channel_axis else samples_per_pixel
        if _is_rgb_compatible_photometric(photometric) and channel_count == 3:
            return "already-rgb-compatible"
        if channel_count != 3 or len(series.pages) < 3:
            raise RuntimeError(
                "Planar brightfield image cannot be converted to RGB automatically: "
                f"axes={axes!r} shape={shape} photometric={photometric!r} "
                f"channels={channel_count} pages={len(series.pages)}."
            )
        width = shape[axes.index("X")]
        height = shape[axes.index("Y")]
        dtype_name = str(series.dtype)
        mpp_x, mpp_y = _ome_physical_sizes_micrometres(tif.ome_metadata)

    if mpp_x is None or mpp_y is None:
        raise RuntimeError(
            "Cannot normalize planar brightfield channels to RGB without physical pixel-size metadata."
        )

    try:
        import pyvips
    except Exception as exc:
        raise RuntimeError("Planar RGB normalization requires pyvips in the runtime image.") from exc

    source_planes = [
        pyvips.Image.new_from_file(str(path), access="sequential", page=idx)
        for idx in range(3)
    ]
    rgb_indices = _rgb_indices_from_source_order(order)
    rgb_planes = [source_planes[idx] for idx in rgb_indices]
    image = rgb_planes[0].bandjoin(rgb_planes[1:]).copy(
        interpretation="srgb",
        xres=1000.0 / mpp_x,
        yres=1000.0 / mpp_y,
    )
    ome_xml = _build_rgb_ome_xml(path.name, width, height, dtype_name, mpp_x, mpp_y)
    image.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml)

    normalized_compression = _normalize_compression(compression)
    if normalized_compression == "uncompressed":
        normalized_compression = "none"
    save_kwargs = dict(
        tile=True,
        tile_width=tile,
        tile_height=tile,
        pyramid=True,
        subifd=True,
        bigtiff=True,
        compression=normalized_compression,
        properties=False,
        resunit="cm",
    )
    if normalized_compression == "jpeg":
        save_kwargs["Q"] = quality
    elif normalized_compression in {"deflate", "lzw"}:
        save_kwargs["predictor"] = "horizontal"

    temporary = path.with_name(f".{path.name}.rgb-normalizing.tif")
    try:
        image.tiffsave(str(temporary), **save_kwargs)
        with tifffile.TiffFile(str(temporary)) as tif:
            series = tif.series[0]
            page = series.pages[0]
            observed_photometric = str(
                getattr(getattr(page, "photometric", None), "name", getattr(page, "photometric", ""))
            ).upper()
            # TIFF/JPEG stores three-channel colour as YCbCr even when the
            # source image and OME channel contract are canonical RGB.  All
            # standard TIFF readers decode those samples back to RGB, so both
            # tags are valid RGB-compatible storage.  Lossless TIFF codecs
            # retain the literal RGB photometric tag.
            if (
                not _is_rgb_compatible_photometric(observed_photometric)
                or int(getattr(page, "samplesperpixel", 1) or 1) != 3
                or not tif.ome_metadata
                or len(getattr(series, "levels", ()) or ()) < 2
            ):
                raise RuntimeError(
                    "RGB normalization did not produce a three-sample RGB-compatible "
                    "pyramidal OME-TIFF: "
                    f"photometric={observed_photometric!r} "
                    f"samples={getattr(page, 'samplesperpixel', None)} "
                    f"ome={bool(tif.ome_metadata)} levels={len(getattr(series, 'levels', ()) or ())}."
                )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return f"planar-{order}-to-interleaved-RGB"


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


def _convert_vsi(
    src: Path,
    dst: Path,
    compression: str,
    quality: int,
    series_index: int,
    channel_order: str,
    tile: int,
) -> str:
    """Convert an Olympus VSI and its sibling _<stem>_ data directory.

    VSI headers are not self-contained. Bio-Formats resolves the companion
    directory relative to the header, so fail early with a useful message when
    a transfer or staging operation omitted it.
    """
    companion = src.parent / f"_{src.stem}_"
    if not companion.is_dir():
        raise FileNotFoundError(
            f"Olympus VSI companion directory is missing: {companion}. "
            f"Keep {src.name} beside _{src.stem}_ when copying or staging the sample."
        )

    bioformats2raw = shutil.which("bioformats2raw")
    raw2ometiff = shutil.which("raw2ometiff")
    if not bioformats2raw or not raw2ometiff:
        raise RuntimeError(
            "VSI conversion requires bioformats2raw and raw2ometiff in the runtime image."
        )

    compression_key = _normalize_compression(compression)
    ome_compression = {
        "jpeg": "JPEG",
        "lzw": "LZW",
        "none": "UNCOMPRESSED",
        "uncompressed": "UNCOMPRESSED",
    }.get(compression_key, "LZW")

    with tempfile.TemporaryDirectory(prefix=f"{src.stem}.bioformats.", dir=str(dst.parent)) as tmp:
        zarr_dir = Path(tmp) / "pyramid.zarr"
        subprocess.run(
            [
                bioformats2raw,
                "--overwrite",
                "--use-existing-resolutions",
                "-s",
                str(series_index),
                "--max-workers",
                "4",
                str(src),
                str(zarr_dir),
            ],
            check=True,
        )
        command = [
            raw2ometiff,
            "--compression",
            ome_compression,
            "--max_workers",
            "4",
        ]
        if ome_compression == "JPEG":
            command.extend(["--quality", str(max(0.0, min(1.0, quality / 100.0)))])
        command.extend([str(zarr_dir), str(dst)])
        subprocess.run(command, check=True)

    rgb_backend = _ensure_explicit_rgb_ome_tiff(
        dst,
        channel_order=channel_order,
        compression=compression,
        quality=quality,
        tile=tile,
    )
    return f"bioformats2raw+raw2ometiff(series={series_index})+{rgb_backend}"


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
    parser.add_argument("--output", help="Output TIFF path.")
    parser.add_argument(
        "--normalize-existing",
        action="store_true",
        help=(
            "Normalize --input in place when it contains three planar grayscale "
            "brightfield channels; explicit RGB/YCbCr inputs are unchanged."
        ),
    )
    parser.add_argument("--input-region", default="", help="Optional input region selector (used for CZI scene extraction).")
    parser.add_argument("--compression", default="jpeg", help="TIFF compression (default: jpeg).")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality for lossy TIFF compression (default: 90).")
    parser.add_argument("--tile", type=int, default=512, help="Tile size for streamed WSI writes (default: 512).")
    parser.add_argument("--pyramid", action="store_true", help="Write a pyramidal TIFF.")
    parser.add_argument(
        "--vsi-series-index",
        type=int,
        default=1,
        help="Bio-Formats series containing the primary Olympus VSI WSI (default: 1).",
    )
    parser.add_argument(
        "--channel-order",
        default="RGB",
        type=_normalize_channel_order,
        help=(
            "Meaning of source planes before canonical RGB output (default: RGB). "
            "Allowed values: RGB, RBG, GRB, GBR, BRG, BGR."
        ),
    )
    args = parser.parse_args()

    src = Path(args.input)
    if args.normalize_existing:
        backend = _ensure_explicit_rgb_ome_tiff(
            src,
            channel_order=args.channel_order,
            compression=args.compression,
            quality=args.quality,
            tile=args.tile,
        )
        print(
            f"[INFO] RGB normalization for {src.name}: {backend} "
            f"(channel_order={args.channel_order})"
        )
        return
    if not args.output:
        parser.error("--output is required unless --normalize-existing is used")
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".czi":
        backend = _convert_czi(src, dst, args.compression, args.tile, args.input_region)
        print(
            f"[INFO] Converted {src.name} -> {dst.name} with {backend} "
            f"(compression={args.compression}, input_region={args.input_region or 'auto'})"
        )
        return

    if src.suffix.lower() == ".vsi":
        backend = _convert_vsi(
            src,
            dst,
            args.compression,
            args.quality,
            args.vsi_series_index,
            args.channel_order,
            args.tile,
        )
        print(
            f"[INFO] Converted {src.name} -> {dst.name} with {backend} "
            f"(compression={args.compression}, pyramid=True, channel_order={args.channel_order})"
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
