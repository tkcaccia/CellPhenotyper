#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import tifffile


def infer_hw(image_path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(str(image_path)) as tf:
        if not tf.series:
            raise RuntimeError(f"No TIFF series found in {image_path}")
        series = tf.series[0]
        level0 = series.levels[0] if getattr(series, "levels", None) else series
        shape = tuple(level0.shape)
        axes = (getattr(level0, "axes", None) or getattr(series, "axes", None) or "").replace("S", "C")

    if len(shape) == 2:
        return int(shape[0]), int(shape[1])

    if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])

    if len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            return int(shape[0]), int(shape[1])
        if shape[0] in (1, 3, 4):
            return int(shape[1]), int(shape[2])

    if len(shape) >= 2:
        return int(shape[-2]), int(shape[-1])

    raise RuntimeError(f"Could not infer image size from shape={shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a full-image rectangular ROI GeoJSON.")
    parser.add_argument("--image", required=True, help="Input OME-TIFF image path.")
    parser.add_argument("--out", required=True, help="Output GeoJSON path.")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = infer_hw(image_path)
    polygon = [
        [0.0, 0.0],
        [float(w), 0.0],
        [float(w), float(h)],
        [0.0, float(h)],
        [0.0, 0.0],
    ]

    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "full_image_roi",
                    "source_image": image_path.name,
                    "width_px": int(w),
                    "height_px": int(h),
                },
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
            }
        ],
    }
    out_path.write_text(json.dumps(geo), encoding="utf-8")


if __name__ == "__main__":
    main()
