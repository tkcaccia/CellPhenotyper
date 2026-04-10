#!/usr/bin/env python3
import argparse
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
    parser.add_argument("--compression", default="jpeg", help="TIFF compression (default: jpeg).")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality for lossy TIFF compression (default: 90).")
    parser.add_argument("--tile", type=int, default=512, help="Tile size for streamed WSI writes (default: 512).")
    parser.add_argument("--pyramid", action="store_true", help="Write a pyramidal TIFF.")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

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
