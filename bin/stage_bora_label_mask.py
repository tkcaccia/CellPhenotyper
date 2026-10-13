#!/usr/bin/env python3
"""Stage a compressed categorical TIFF as a temporary contiguous Bora mask.

Published CellPhenotyper masks remain compressed pyramidal OME-TIFFs.  Bora's
streaming backend currently requires ``tifffile.memmap`` input, so this adapter
decompresses level 0 blockwise into task-local scratch without materializing the
whole slide in RAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from tiff_preview import _resolve_zarr_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-rows", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if source_path == output_path:
        raise ValueError("Bora staging input and output must be different paths")
    output_path.unlink(missing_ok=True)

    import zarr

    store = None
    destination = None
    try:
        with tifffile.TiffFile(source_path) as tif:
            level = tif.series[0].levels[0]
            axes = str(level.axes)
            if axes != "YX" or len(level.shape) != 2:
                raise ValueError(
                    f"Bora label mask must have YX axes, got axes={axes!r}, "
                    f"shape={level.shape}"
                )
            height, width = map(int, level.shape)
            dtype = np.dtype(level.dtype)
            if dtype.kind not in "uib":
                raise ValueError(f"Bora label mask must be integer, got {dtype}")
            store = level.aszarr()
            source = _resolve_zarr_array(zarr.open(store, mode="r"))
            if tuple(map(int, source.shape)) != (height, width):
                raise RuntimeError(
                    f"TIFF/Zarr shape mismatch: {source.shape} != {(height, width)}"
                )
            destination = tifffile.memmap(
                output_path,
                shape=(height, width),
                dtype=dtype,
                bigtiff=True,
                metadata={"axes": "YX"},
            )
            block_rows = max(1, int(args.block_rows))
            for y0 in range(0, height, block_rows):
                y1 = min(height, y0 + block_rows)
                destination[y0:y1, :] = np.asarray(source[y0:y1, :])
            destination.flush()
            del destination
            destination = None

        with tifffile.TiffFile(output_path) as staged:
            page = staged.pages[0]
            if not page.is_memmappable:
                raise RuntimeError(f"Staged Bora mask is not memory-mappable: {output_path}")
            observed_shape = tuple(map(int, staged.series[0].shape))
            if observed_shape != (height, width):
                raise RuntimeError(
                    f"Staged Bora mask shape mismatch: {observed_shape} != {(height, width)}"
                )
        print(
            f"[INFO] Staged compressed Bora input as task-local contiguous TIFF: "
            f"{output_path} shape={(height, width)} dtype={dtype}"
        )
    except Exception:
        if destination is not None:
            del destination
        output_path.unlink(missing_ok=True)
        raise
    finally:
        if store is not None and hasattr(store, "close"):
            store.close()


if __name__ == "__main__":
    main()
