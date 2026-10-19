#!/usr/bin/env python3
"""Check virtual DAPI against nuclei on the native GigaTIME inference grid.

This is a technical plausibility check, not validation against measured DAPI.
The nuclear segmentation must come from the input H&E, independently of GigaTIME.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

from quantify_gigatime_intensity import LazyImageReader, LazyMaskReader


class NuclearAlignmentScore:
    def __init__(self, shape_yx: tuple[int, int], max_samples: int = 500_000):
        self.shape = tuple(int(v) for v in shape_yx)
        self.stride = min(4, max(1, int(math.ceil(math.sqrt(
            self.shape[0] * self.shape[1] / max(1, max_samples)
        )))))
        self.pos = np.zeros(256, dtype=np.int64)
        self.neg = np.zeros(256, dtype=np.int64)
        self.edge_sum = self.edge_n = self.core_sum = self.core_n = 0.0
        self.inner_sum = self.inner_n = self.outer_sum = self.outer_n = 0.0
        self.sampled_windows = 0

    def add(self, dapi: np.ndarray, labels: np.ndarray, y0: int, x0: int) -> None:
        if dapi.shape != labels.shape:
            raise ValueError("DAPI and nuclear labels are not on the same grid")
        sy = (-y0) % self.stride
        sx = (-x0) % self.stride
        values = np.asarray(dapi[sy::self.stride, sx::self.stride], dtype=np.float32)
        sampled_labels = labels[sy::self.stride, sx::self.stride]
        yy = y0 + sy + np.arange(values.shape[0]) * self.stride
        xx = x0 + sx + np.arange(values.shape[1]) * self.stride
        self.add_sampled(values, sampled_labels, yy, xx)

    def add_sampled(self, values: np.ndarray, labels: np.ndarray,
                    yy: np.ndarray, xx: np.ndarray) -> None:
        nucleus = np.asarray(labels > 0)
        if not values.size:
            return
        self.sampled_windows += 1
        # Only compare nuclei with the nearby extranuclear compartment. Blank
        # slide background would otherwise make a non-nuclear DAPI map look good.
        nearby = binary_dilation(nucleus, iterations=max(1, int(math.ceil(8 / self.stride)))) & ~nucleus
        finite = np.isfinite(values)
        bins = np.clip(np.floor(values * 255 + 0.5), 0, 255).astype(np.uint8)
        self.pos += np.bincount(bins[nucleus & finite], minlength=256)
        self.neg += np.bincount(bins[nearby & finite], minlength=256)

        # A diffuse bright region can have good pixel AUC while crossing many
        # nuclei. The narrow inside/outside nuclear shells detect that failure.
        inside = nucleus & ~binary_erosion(nucleus, iterations=2) & finite
        outside = binary_dilation(nucleus, iterations=2) & ~nucleus & finite
        self.inner_sum += float(values[inside].sum())
        self.inner_n += int(inside.sum())
        self.outer_sum += float(values[outside].sum())
        self.outer_n += int(outside.sum())

        edge_width = max(8, int(round(8 / 0.25)))  # 8 µm at model resolution
        edge = ((yy[:, None] < edge_width) | (xx[None, :] < edge_width)
                | (yy[:, None] >= self.shape[0] - edge_width)
                | (xx[None, :] >= self.shape[1] - edge_width))
        edge_nuclei = nucleus & finite & edge
        core_nuclei = nucleus & finite & ~edge
        self.edge_sum += float(values[edge_nuclei].sum())
        self.edge_n += int(edge_nuclei.sum())
        self.core_sum += float(values[core_nuclei].sum())
        self.core_n += int(core_nuclei.sum())

    def report(self, *, min_auc: float = 0.70, min_dice: float = 0.35,
               min_edge_ratio: float = 0.65,
               min_boundary_contrast: float = 0.05) -> dict:
        p, n = int(self.pos.sum()), int(self.neg.sum())
        result = {
            "schema_version": 1,
            "scope": "virtual_DAPI_vs_HE_derived_nuclear_mask",
            "sampling_stride_px": self.stride,
            "sampled_windows": self.sampled_windows,
            "nuclear_pixels": p,
            "nearby_extranuclear_pixels": n,
            "thresholds": {"min_auc": min_auc, "min_prevalence_matched_dice": min_dice,
                           "min_edge_to_core_nuclear_mean_ratio": min_edge_ratio,
                           "min_nuclear_boundary_contrast": min_boundary_contrast},
            "interpretation_limit": (
                "Agreement with H&E-derived nuclei does not establish measured DAPI accuracy. "
                "A poor StarDist mask can also lower this score. Inspect the overlay and mask."
            ),
        }
        if p < 500 or n < 500:
            result.update(status="insufficient", alignment_score=None, auc=None,
                          prevalence_matched_dice=None, edge_to_core_nuclear_mean_ratio=None,
                          nuclear_boundary_contrast=None)
            return result
        below = np.cumsum(self.neg) - self.neg
        auc = float(np.sum(self.pos * (below + 0.5 * self.neg)) / (p * n))
        selected = np.cumsum((self.pos + self.neg)[::-1])
        threshold = 255 - int(np.searchsorted(selected, p, side="left"))
        tp = int(self.pos[threshold:].sum())
        predicted = int((self.pos[threshold:] + self.neg[threshold:]).sum())
        dice = float(2 * tp / (p + predicted))
        edge_ratio = (float((self.edge_sum / self.edge_n) / (self.core_sum / self.core_n))
                      if self.edge_n >= 100 and self.core_n >= 500 and self.core_sum > 0 else None)
        boundary_contrast = (float(self.inner_sum / self.inner_n - self.outer_sum / self.outer_n)
                             if self.inner_n >= 50 and self.outer_n >= 50 else None)
        boundary_skill = float(np.clip(boundary_contrast / 0.20, 0, 1)) if boundary_contrast is not None else 0.0
        alignment_score = float(np.clip(0.5 * (2 * auc - 1) + 0.25 * dice + 0.25 * boundary_skill, 0, 1))
        alignment_pass = (auc >= min_auc and dice >= min_dice and
                          boundary_contrast is not None and boundary_contrast >= min_boundary_contrast)
        border_pass = edge_ratio is None or edge_ratio >= min_edge_ratio
        result.update(
            status="pass" if alignment_pass and border_pass else "fail",
            alignment_status="pass" if alignment_pass else "fail",
            border_shadow_status=("insufficient" if edge_ratio is None else
                                  "pass" if border_pass else "fail"),
            alignment_score=alignment_score,
            auc=auc,
            prevalence_matched_dice=dice,
            prevalence_matched_threshold=threshold / 255.0,
            nuclear_boundary_contrast=boundary_contrast,
            nuclear_inner_boundary_pixels=int(self.inner_n),
            nearby_outer_boundary_pixels=int(self.outer_n),
            edge_to_core_nuclear_mean_ratio=edge_ratio,
            edge_nuclear_pixels=int(self.edge_n),
            core_nuclear_pixels=int(self.core_n),
        )
        return result


def assess(image_path: str, nuclei_path: str, out_path: str, *, block_size: int = 1024,
           min_auc: float = 0.70, min_dice: float = 0.35,
           min_edge_ratio: float = 0.65,
           min_boundary_contrast: float = 0.05) -> dict:
    with LazyImageReader(image_path) as image, LazyMaskReader(nuclei_path) as nuclei:
        if "DAPI" not in image.channel_names:
            raise ValueError("GigaTIME output must include DAPI for nuclear QC")
        dapi_index = image.channel_names.index("DAPI")
        shape = (image.height, image.width)
        meta = image.storage_meta
        if not meta or tuple(meta.get("inference_shape_yx", ())) != shape:
            raise ValueError("GigaTIME inference-grid metadata are missing or inconsistent")
        if abs(float(meta.get("effective_mpp", 0.0)) - 0.25) > 1.0e-6:
            raise ValueError("GigaTIME nuclear QC requires the calibrated 0.25 µm/pixel grid")
        source_shape = tuple(meta.get("original_shape_yx", ()))
        if (nuclei.height, nuclei.width) not in (shape, source_shape):
            raise ValueError("StarDist mask shape does not match the source or inference grid")
        scorer = NuclearAlignmentScore(shape)
        preview_region = None
        preview_nuclei = -1
        y_starts = list(range(0, shape[0], block_size))
        x_starts = list(range(0, shape[1], block_size))
        if len(y_starts) * len(x_starts) > 64:
            y_starts = [y_starts[i] for i in sorted(set(np.linspace(0, len(y_starts)-1, 8, dtype=int)))]
            x_starts = [x_starts[i] for i in sorted(set(np.linspace(0, len(x_starts)-1, 8, dtype=int)))]
        for y0 in y_starts:
            y1 = min(shape[0], y0 + block_size)
            for x0 in x_starts:
                x1 = min(shape[1], x0 + block_size)
                yy = np.arange(y0 + (-y0) % scorer.stride, y1, scorer.stride, dtype=np.int64)
                xx = np.arange(x0 + (-x0) % scorer.stride, x1, scorer.stride, dtype=np.int64)
                if not yy.size or not xx.size:
                    continue
                if image.axes == "CYX":
                    values = np.asarray(image.arr.oindex[dapi_index, yy, xx], dtype=np.float32)
                elif image.axes == "YXC":
                    values = np.asarray(image.arr.oindex[yy, xx, dapi_index], dtype=np.float32)
                elif image.axes == "YCX":
                    values = np.asarray(image.arr.oindex[yy, dapi_index, xx], dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported GigaTIME axes: {image.axes}")
                if image.scale_max:
                    values /= image.scale_max
                mask_y = np.minimum(nuclei.height - 1, np.floor((yy + .5) * nuclei.height / shape[0]).astype(np.int64))
                mask_x = np.minimum(nuclei.width - 1, np.floor((xx + .5) * nuclei.width / shape[1]).astype(np.int64))
                if nuclei.singleton_axis == 0:
                    labels = np.asarray(nuclei.arr.oindex[0, mask_y, mask_x])
                elif nuclei.singleton_axis == 2:
                    labels = np.asarray(nuclei.arr.oindex[mask_y, mask_x, 0])
                else:
                    labels = np.asarray(nuclei.arr.oindex[mask_y, mask_x])
                scorer.add_sampled(values, labels, yy, xx)
                if int(np.count_nonzero(labels)) > preview_nuclei:
                    preview_nuclei = int(np.count_nonzero(labels))
                    preview_region = (y0, y1, x0, x1)
        result = scorer.report(min_auc=min_auc, min_dice=min_dice,
                               min_edge_ratio=min_edge_ratio,
                               min_boundary_contrast=min_boundary_contrast)
        result["sampled_stratified_windows"] = len(y_starts) * len(x_starts)
        if preview_region is not None:
            y0, y1, x0, x1 = preview_region
            dapi = np.clip(image.read_block(y0, y1, x0, x1)[dapi_index], 0, 1)
            labels = nuclei.read_block(y0, y1, x0, x1, shape) > 0
            gray = np.round(dapi * 255).astype(np.uint8)
            preview = np.repeat(gray[..., None], 3, axis=2)
            outline = labels & ~binary_erosion(labels)
            preview[outline] = (255, 45, 45)
            preview_path = Path(out_path).with_suffix(".png")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(preview).save(preview_path)
            result["preview"] = preview_path.name
            result["preview_region_yxyx"] = list(preview_region)
        result["image"] = str(Path(image_path).resolve())
        result["nuclei_mask"] = str(Path(nuclei_path).resolve())
        result["image_shape_yx"] = list(shape)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--nuclei", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--min-auc", type=float, default=0.70)
    parser.add_argument("--min-dice", type=float, default=0.35)
    parser.add_argument("--min-edge-ratio", type=float, default=0.65)
    parser.add_argument("--min-boundary-contrast", type=float, default=0.05)
    parser.add_argument("--mode", choices=("warn", "fail"), default="fail")
    args = parser.parse_args()
    result = assess(args.image, args.nuclei, args.output, block_size=args.block_size,
                    min_auc=args.min_auc, min_dice=args.min_dice,
                    min_edge_ratio=args.min_edge_ratio,
                    min_boundary_contrast=args.min_boundary_contrast)
    print(json.dumps(result, indent=2))
    if args.mode == "fail" and result["status"] != "pass":
        raise SystemExit("GigaTIME nuclear alignment QC did not pass; do not use cell markers")


if __name__ == "__main__":
    main()
