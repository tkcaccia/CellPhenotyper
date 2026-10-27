#!/usr/bin/env python3
"""One pinned Virchow2 embedding per disjoint, physically calibrated tissue tile."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from virchow2_runtime import CHECKPOINT_SHA256, PATCH, REVISION, load_model, pooled_tiles, sha256


def validate_grid(frame: list[dict], metadata: dict) -> None:
    """Reject legacy overlapping/inner-square observations before model loading."""
    footprint = int(metadata["extraction_tile_size_source_px"])
    stride = int(metadata["grid_stride_source_px"])
    if (metadata.get("tile_layout") != "nonoverlapping" or
            metadata.get("tile_origin") != "top_left" or
            int(metadata.get("context_overlap_source_px", -1)) != 0 or
            footprint != stride or
            int(metadata["model_tile_size_px"]) != PATCH or
            abs(float(metadata["target_mpp"]) - .5) > 1e-6):
        raise ValueError("Expected top-left non-overlapping 224-pixel Virchow2 tiles at 0.5 um/px")
    if not frame:
        raise ValueError("Grid has no tissue tiles")
    used = set()
    for item in frame:
        row, col = int(item["grid_row"]), int(item["grid_col"])
        x0, y0 = int(item["tile_x0"]), int(item["tile_y0"])
        if ((row, col) in used or x0 != col * stride or y0 != row * stride or
                int(item["tile_x1"]) != x0 + footprint or
                int(item["tile_y1"]) != y0 + footprint or
                int(item["core_x0"]) != x0 or int(item["core_y0"]) != y0):
            raise ValueError("Grid tile overlaps, shifts, duplicates, or disagrees with its core")
        used.add((row, col))
    if len(used) != int(metadata["retained_units"]):
        raise ValueError("Grid metadata count differs from observation CSV")


def read_tile(image, x0: int, y0: int, size: int) -> np.ndarray:
    """Read the exact non-overlapping source footprint; whiten only its edge pad."""
    width = max(0, min(size, image.width - x0))
    height = max(0, min(size, image.height - y0))
    if width == 0 or height == 0 or x0 < 0 or y0 < 0:
        raise ValueError("Grid tile lies outside the source image")
    region = image.crop(x0, y0, width, height)
    pixels = np.ndarray((height, width, 3), dtype=np.uint8, buffer=region.write_to_memory()).copy()
    patch = np.full((size, size, 3), 255, dtype=np.uint8)
    patch[:height, :width] = pixels
    return cv2.resize(patch, (PATCH, PATCH), interpolation=cv2.INTER_AREA if size >= PATCH else cv2.INTER_CUBIC)


def run(image_path: Path, objects_path: Path, metadata_path: Path, checkpoint: Path,
        output: Path, *, batch: int = 4, device: str = "cuda") -> dict:
    import pyvips

    if batch < 1:
        raise ValueError("Positive batch size required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with objects_path.open(newline="", encoding="utf-8") as stream:
        frame = list(csv.DictReader(stream))
    validate_grid(frame, metadata)
    image = pyvips.Image.new_from_file(str(image_path), access="random")
    if ((image.width, image.height, image.bands, image.format) !=
            (int(metadata["image_width_px"]), int(metadata["image_height_px"]), 3, "uchar")):
        raise ValueError("Grid and uint8 RGB image disagree")
    footprint = int(metadata["extraction_tile_size_source_px"])
    mpp_x, mpp_y = float(metadata["source_mpp_x"]), float(metadata["source_mpp_y"])
    if max(abs(footprint * mpp_x / PATCH - .5), abs(footprint * mpp_y / PATCH - .5)) > .01:
        raise ValueError("Source tile footprint is not calibrated to Virchow2 0.5 um/px")
    model = load_model(checkpoint, device=device)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "virchow2_embeddings_shard0000.csv.gz"
    fields = ["cell_id", "observation_type", "cx", "cy", "area_px"] + [
        f"feat_{index + 1}" for index in range(CHANNEL_COUNT)
    ]
    with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for start in range(0, len(frame), batch):
            rows = frame[start:start + batch]
            patches = np.stack([read_tile(image, int(row["tile_x0"]), int(row["tile_y0"]), footprint)
                                for row in rows])
            values = pooled_tiles(model, patches, device=device)
            if values.shape != (len(rows), CHANNEL_COUNT) or not np.isfinite(values).all():
                raise ValueError("Invalid Virchow2 tile embeddings")
            for row, features in zip(rows, values):
                writer.writerow([int(row["label"]), "grid", int(row["x"]), int(row["y"]),
                                 int(row["area_px"]), *features.tolist()])
    report = dict(encoder="Virchow2", encoder_revision=REVISION,
                  checkpoint_sha256=CHECKPOINT_SHA256, grid_sha256=sha256(objects_path),
                  grid_metadata_sha256=sha256(metadata_path), image=str(image_path),
                  observations=len(frame), features_per_tile=CHANNEL_COUNT,
                  target_mpp_xy=[.5, .5], tile_size_model_px=PATCH,
                  tile_size_source_px=footprint, tile_stride_source_px=footprint,
                  tile_overlap_source_px=0, inner_square_used=False,
                  embeddings_sha256=sha256(csv_path))
    (output / "VIRCHOW2_COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


CHANNEL_COUNT = 2560


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("image", "grid-objects", "grid-metadata", "checkpoint", "outdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(json.dumps(run(args.image, args.grid_objects, args.grid_metadata,
                         args.checkpoint, args.outdir, batch=args.batch, device=args.device)),
          flush=True)


if __name__ == "__main__":
    main()
