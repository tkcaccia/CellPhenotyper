#!/usr/bin/env python3
"""Create three SYNTHETIC image/feature fixtures. Never loads an encoder.

Values deliberately follow a simple analytic pattern. They are not Virchow2
inference, biological evidence or a model performance benchmark.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import struct
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from export_experiment import FILENAME, export_experiment


def write_png(path, width, height, tissue):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    pixels = bytearray()
    for y in range(height):
        pixels.append(0)
        for x in range(width):
            texture = ((x // 7 + y // 11) % 9) - 4
            colour = (185, 95, 130) if x < width // 2 else (225, 185, 195)
            pixels.extend(max(0, min(255, value + texture + tissue * 2)) for value in colour)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def write_features(path, width, height, unit, tissue):
    dimension, spacing = (2560, 224) if unit == "tile" else (1280, 14)
    labels = []
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observation_id", "x", "y", *[f"feat_{i}" for i in range(1, dimension + 1)]])
        identifier = 0
        for y in range(spacing // 2, height, spacing):
            for x in range(spacing // 2, width, spacing):
                identifier += 1
                sign = -1 if x < width / 2 else 1
                # Keep all channels explicit with finite nonconstant values.
                groups = [f"{sign * (1 + i / 20) + x / width * (i % 3) + y / height * (i % 5) / 10 + tissue / 100:.5f}"
                          for i in range(16)]
                writer.writerow([str(identifier), x, y, *[groups[i % 16] for i in range(dimension)]])
                labels.append(dict(observation_id=str(identifier), observation_unit=unit,
                                   class_name="epithelium" if sign < 0 else "stroma"))
    return dimension, spacing, labels


def create_demo(directory):
    directory = Path(directory).absolute()
    if directory.exists():
        raise ValueError(f"Destination already exists: {directory}; choose a new directory to preserve existing work")
    directory.mkdir(parents=True)
    spec = dict(name="SYNTHETIC three-tissue integration demo — no Virchow2 inference",
                settings=dict(training_tissue_ids=["synthetic-1", "synthetic-2"], query_tissue_ids=["synthetic-3"]),
                provenance=dict(synthetic=True, encoder_executed=False,
                                purpose="Software integration only; values are analytic fixtures, not learned embeddings"), tissues=[])
    truth = []
    for index in range(1, 4):
        tid = f"synthetic-{index}"
        folder = directory / tid
        folder.mkdir()
        image = folder / "image.png"
        write_png(image, 448, 448, index)
        annotations = []
        for label, colour, x0, x1 in (("epithelium", "#b95f82", 0, 224), ("stroma", "#e1b9c3", 224, 448)):
            annotations.append(dict(type="Feature", id=f"roi-{label}",
                                    properties=dict(name=f"SYNTHETIC {label}", classification=dict(name=label, color=colour),
                                                    tissue_id=tid, sample_id=tid, synthetic=True),
                                    geometry=dict(type="Polygon", coordinates=[[[x0, 0], [x1, 0], [x1, 448], [x0, 448], [x0, 0]]])))
        geojson = folder / "annotations.geojson"
        geojson.write_text(json.dumps(dict(type="FeatureCollection", coordinate_space="image_level0_pixels",
                                           features=annotations), indent=2) + "\n")
        tissue = dict(tissue_id=tid, name=f"SYNTHETIC tissue {index}", specimen_id=f"synthetic-specimen-{index}",
                      image=dict(path=str(image), mpp_x=.5, mpp_y=.5),
                      provenance=dict(synthetic=True, split="train" if index < 3 else "query"),
                      artifacts=[dict(role="tissue_annotations", path=str(geojson))], feature_sets=[])
        for kind, unit in (("virchow2-pooled", "tile"), ("virchow2-spatial", "token")):
            path = folder / f"{unit}_features.csv.gz"
            dimension, spacing, labels = write_features(path, 448, 448, unit, index)
            truth.extend(dict(tissue_id=tid, split="train" if index < 3 else "query", **label) for label in labels)
            tissue["feature_sets"].append(dict(feature_set_id=kind, name=f"SYNTHETIC {unit} features",
                                                kind=kind, observation_unit=unit, path=str(path), format="csv.gz",
                                                feature_columns_prefix="feat_", id_column="observation_id",
                                                x_column="x", y_column="y", coordinate_frame="image",
                                                encoder=dict(name="Virchow2", revision="0" * 40, checkpoint_sha256="0" * 64),
                                                preprocessing=dict(synthetic=True, feature_definition="CLS+mean_patch" if unit == "tile" else "spatial_patch_tokens",
                                                                   data_generation="analytic_fixture", channels=dimension, encoder_executed=False),
                                                grid=dict(tile_width_px=spacing, tile_height_px=spacing,
                                                          spacing_x_px=spacing, spacing_y_px=spacing,
                                                          origin_x_px=0, origin_y_px=0)))
        spec["tissues"].append(tissue)
    with (directory / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["tissue_id", "split", "observation_id", "observation_unit", "class_name"])
        writer.writeheader()
        writer.writerows(truth)
    output = directory / FILENAME
    payload = export_experiment(spec, output)
    (directory / "README.txt").write_text("SYNTHETIC SOFTWARE FIXTURE. No encoder or GPU ran.\nOpen experiment.cellphenotyper.json in wsiTools Desktop.\nTrain on tissues 1 and 2; query tissue 3. Local IDs intentionally repeat.\nlabels.csv is separate reference data, never predictor input.\nThese values cannot support biological or Virchow2 performance claims.\n")
    return output, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New destination directory")
    args = parser.parse_args()
    try:
        output, payload = create_demo(args.directory)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Cannot create synthetic demo: {exc}\n")
    print(f"SYNTHETIC demo ready: {output}\n{len(payload['tissues'])} tissues, pooled and spatial cached features; no inference was run.")


if __name__ == "__main__":
    main()
