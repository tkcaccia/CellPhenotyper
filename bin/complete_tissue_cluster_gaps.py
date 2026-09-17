#!/usr/bin/env python3
"""Opt-in exploratory gap completion; never overwrites accepted segmentation.

Reuses raw KODAMA assignments where available, then H&E-gradient watershed
(a multi-label wand) for remaining stained gaps. Original abstentions are not
reclassified as confident. Not a replacement for GrandQC artifact assessment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed


def stained_pixels(rgb, minimum_od=0.08, minimum_saturation=0.05):
    rgb = np.asarray(rgb, dtype=np.float32)[..., :3]
    od = -np.log((rgb + 1) / 256).mean(axis=-1)
    saturation = (rgb.max(axis=-1) - rgb.min(axis=-1)) / np.maximum(rgb.max(axis=-1), 1)
    return (od >= minimum_od) & (saturation >= minimum_saturation)


def complete(rgb, original, tissue, raw, *, artifacts=None, trusted=None, pixel_um, max_distance_um=150,
             minimum_od=0.08, minimum_saturation=0.05):
    if rgb.shape[:2] != original.shape or tissue.shape != original.shape or raw.shape != original.shape:
        raise ValueError("All inputs must be spatially aligned")
    if pixel_um <= 0 or max_distance_um <= 0:
        raise ValueError("Physical scales must be positive")
    classes = np.unique(original[original > 0])
    if not len(classes) or not set(np.unique(raw[raw > 0])).issubset(set(classes)):
        raise ValueError("Raw assignments must use existing nonzero classes")
    artifacts = np.zeros(original.shape, bool) if artifacts is None else np.asarray(artifacts, bool)
    if artifacts.shape != original.shape:
        raise ValueError("Artifact mask must be aligned")
    # GrandQC artifacts are an inviolable exclusion, including for old labels.
    original = np.where(artifacts, 0, original)
    stain = stained_pixels(rgb, minimum_od, minimum_saturation)
    # Only enclosed tissue gaps; no unconstrained dilation into the slide.
    envelope = ndi.binary_fill_holes(tissue | (original > 0))
    near = ndi.distance_transform_edt(~(original > 0), sampling=pixel_um) <= max_distance_um
    eligible = (original == 0) & ~artifacts & (tissue | (stain & near & envelope))
    recovered = eligible & ~tissue
    result = original.copy()
    direct = eligible & (raw > 0)
    result[direct] = raw[direct]
    gray = np.asarray(rgb, dtype=np.float32)[..., :3].mean(axis=-1) / 255
    propagated = watershed(sobel(gray), markers=result.astype(np.int32),
                           mask=(original > 0) | eligible)
    inferred = eligible & (result == 0) & (propagated > 0)
    result[inferred] = propagated[inferred]
    disconnected = eligible & (result == 0)
    if disconnected.any():
        # Rare seedless islands: transfer an existing class using H&E appearance
        # plus weak spatial context, never geometry that crosses an artifact.
        # This is pseudo-label transfer, not independent biological validation.
        from scipy.spatial import cKDTree
        from skimage.color import rgb2lab
        lab = rgb2lab(np.asarray(rgb)[..., :3]).astype(np.float32)
        lab /= np.array([100,128,128],np.float32)
        yy,xx = np.nonzero(disconnected)
        donor = (original > 0) & ~artifacts
        if trusted is not None:
            donor &= trusted
        if not donor.any():
            raise ValueError("No trusted donors for disconnected tissue pockets")
        rng = np.random.default_rng(1729)
        donor_ids = []
        for value in classes:
            ids = np.flatnonzero(donor & (original == value))
            if len(ids)>10000:
                ids=rng.choice(ids,10000,replace=False)
            donor_ids.extend(ids.tolist())
        dy,dx=np.unravel_index(donor_ids, original.shape)
        def features(y,x):
            return np.column_stack((lab[y,x], .15*y/original.shape[0], .15*x/original.shape[1]))
        tree=cKDTree(features(dy,dx))
        _, nearest=tree.query(features(yy,xx), k=1)
        result[yy,xx]=original[dy[nearest],dx[nearest]]
    provenance = np.zeros(original.shape, np.uint8)
    provenance[original > 0] = 1
    provenance[direct] = 2
    provenance[inferred] = 3
    provenance[disconnected & (result > 0)] = 5
    provenance[eligible & (result == 0)] = 4
    return result, provenance, recovered, eligible


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "labels", "tissue-mask", "artifact-mask", "shift", "grid", "clusters", "outdir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mpp", type=float, required=True)
    p.add_argument("--step", type=int, default=8)
    p.add_argument("--max-distance-um", type=float, default=150)
    p.add_argument("--minimum-od", type=float, default=0.08)
    p.add_argument("--minimum-saturation", type=float, default=0.05)
    p.add_argument("--write-native", action="store_true")
    a = p.parse_args()
    if a.step < 1 or a.mpp <= 0:
        p.error("step and mpp must be positive")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=False)
    import pandas as pd
    import tifffile
    from PIL import Image
    from refine_grown_tissue_medsam import TiffWindowReader, ScaledBinaryMaskReader, read_stride_tiled
    image = TiffWindowReader(a.image, "H&E")
    labels = TiffWindowReader(a.labels, "original labels")
    shape = labels.spatial_shape()
    if image.spatial_shape() != shape:
        raise ValueError("Image and labels must share the crop coordinate frame")
    tissue_reader = ScaledBinaryMaskReader(a.tissue_mask, shape)
    print("Reading bounded-resolution H&E and labels", flush=True)
    rgb = read_stride_tiled(image, a.step)
    old = read_stride_tiled(labels, a.step)
    tissue = tissue_reader.read_stride(a.step)
    # GrandQC artifact TIFF is in full-slide coordinates, not crop coordinates.
    artifact_source = tifffile.imread(a.artifact_mask) != 0
    shift = json.loads(Path(a.shift).read_text())
    full = shift["full_size"]
    bbox = shift["crop_bbox_xyxy"]
    if (bbox["y1"]-bbox["y0"], bbox["x1"]-bbox["x0"]) != shape:
        raise ValueError("Shift metadata does not match native label dimensions")
    def artifact_window(ys, xs):
        ay = np.minimum(artifact_source.shape[0]-1, np.floor((ys+bbox["y0"])*artifact_source.shape[0]/full["height"]).astype(int))
        ax = np.minimum(artifact_source.shape[1]-1, np.floor((xs+bbox["x0"])*artifact_source.shape[1]/full["width"]).astype(int))
        return artifact_source[np.ix_(ay,ax)]
    artifacts = artifact_window(np.arange(0,shape[0],a.step), np.arange(0,shape[1],a.step))
    raw = np.zeros(old.shape, old.dtype)
    abstained = np.zeros(old.shape, bool)
    grid = pd.read_csv(a.grid)
    clusters = pd.read_csv(a.clusters)
    if grid.label.duplicated().any() or clusters.label.duplicated().any():
        raise ValueError("Duplicate observation identifiers")
    joined = grid.merge(clusters, on="label", validate="one_to_one", how="left")
    if joined.cluster.isna().any():
        raise ValueError("Grid observations missing raw assignments")
    for row in joined.itertuples():
        x0, x1 = (int(row.core_x0) + a.step - 1) // a.step, (int(row.core_x1) + a.step - 1) // a.step
        y0, y1 = (int(row.core_y0) + a.step - 1) // a.step, (int(row.core_y1) + a.step - 1) // a.step
        raw[y0:y1, x0:x1] = int(row.cluster)
        abstained[y0:y1, x0:x1] = str(row.is_abstained).lower() == "true"
    print("Completing stained gaps using KODAMA labels and gradient watershed", flush=True)
    new, provenance, recovered, eligible = complete(rgb, old, tissue, raw,
        artifacts=artifacts, trusted=(~abstained)&(old>0)&(old==raw),
        pixel_um=a.mpp*a.step, max_distance_um=a.max_distance_um,
        minimum_od=a.minimum_od, minimum_saturation=a.minimum_saturation)
    def metrics(region):
        zero = (old == 0) & region
        return {"pixels": int(region.sum()), "original_unassigned": int(zero.sum()),
            "unassigned_outside_tissue_mask": int((zero & ~tissue).sum()),
            "unassigned_grandqc_artifact": int((zero & artifacts).sum()),
            "unassigned_on_abstained_grid": int((zero & abstained).sum()),
            "added_raw_kodama": int(((provenance == 2) & region).sum()),
            "added_watershed": int(((provenance == 3) & region).sum()),
            "added_appearance_transfer": int(((provenance == 5) & region).sum()),
            "recovered_support_assigned": int((recovered & (new > 0) & region).sum()),
            "eligible_unresolved": int(((provenance == 4) & region).sum())}
    roi = np.zeros(old.shape, bool)
    h, w = old.shape
    roi[int(h*.24):int(h*.65), int(w*.74):int(w*.985)] = True
    report = {"method": "raw_KODAMA_then_HE_gradient_watershed_stained_enclosed_gaps",
        "exploratory": True, "biological_accuracy_validated": False,
        "parameters": vars(a), "shape_native": list(shape),
        "measurement_pixel_size_um": a.mpp*a.step,
        "whole_image": metrics(np.ones(old.shape, bool)), "approximate_circled_box": metrics(roi),
        "roi_note": "Approximate screenshot bounding box, used only for reporting, not assignment",
        "original_nonartifact_nonzero_changed": int(((old > 0) & ~artifacts & (old != new)).sum()),
        "artifact_pixels_assigned": int(((new > 0) & artifacts).sum()),
        "classes": np.unique(new[new > 0]).tolist(),
        "provenance_codes": {0:"unassigned_or_background",1:"original_assignment",2:"provisional_raw_KODAMA",3:"provisional_HE_watershed",4:"eligible_but_unresolved",5:"provisional_HE_appearance_transfer"},
        "limitations": ["Recovered tissue bypasses background classification but never the explicit artifact mask",
            "Original uncertainty flags remain unchanged in the source CSV",
            "No expert annotation used as training data", "Completion works at the stated physical resolution"]}
    tifffile.imwrite(out / "completed_preview_labels.tif", new, compression="zlib")
    tifffile.imwrite(out / "completion_preview_provenance.tif", provenance, compression="zlib")
    palette = np.array([[0,0,0],[230,90,145],[60,180,100],[100,140,230],[245,210,70],[220,140,95]], np.uint8)
    def overlay(mask):
        view = rgb[..., :3].copy()
        nz = mask > 0
        view[nz] = (.55*view[nz] + .45*palette[1+(mask[nz]-1) % (len(palette)-1)]).astype(np.uint8)
        return Image.fromarray(view)
    panels = []
    for mask in (old, new):
        panel = overlay(mask)
        panel.thumbnail((1400,1400))
        panels.append(panel)
    canvas = Image.new("RGB", (sum(i.width for i in panels), max(i.height for i in panels)), "white")
    canvas.paste(panels[0], (0,0)); canvas.paste(panels[1], (panels[0].width,0))
    canvas.save(out / "before_after.jpg", quality=92)
    roi_box = (int(w*.74), int(h*.24), int(w*.985), int(h*.65))
    panels = [overlay(mask).crop(roi_box) for mask in (old,new)]
    canvas = Image.new("RGB", (2*panels[0].width, panels[0].height), "white")
    canvas.paste(panels[0], (0,0)); canvas.paste(panels[1], (panels[0].width,0))
    canvas.save(out / "circled_area_before_after.jpg", quality=95)
    (out / "completion_report.json").write_text(json.dumps(report, indent=2)+"\n")
    if a.write_native:
        print("Writing native-resolution labels; preserving every original nonzero pixel", flush=True)
        counts = {"original_nonartifact_pixels_changed":0,"added_pixels":0,"native_white_candidates_rejected":0,"original_artifact_pixels_removed":0}
        def blocks(kind):
            x = np.minimum(np.arange(shape[1]) // a.step, w-1)
            for y0 in range(0, shape[0], 256):
                y1 = min(y0+256, shape[0])
                y = np.minimum(np.arange(y0,y1) // a.step, h-1)
                base = labels.read(y0,y1,0,shape[1]).copy()
                art = artifact_window(np.arange(y0,y1),np.arange(shape[1]))
                proposal = new[np.ix_(y,x)]
                codes = provenance[np.ix_(y,x)]
                stained = stained_pixels(image.read(y0,y1,0,shape[1]), a.minimum_od, a.minimum_saturation)
                candidate = (base == 0) & np.isin(codes,[2,3,5])
                clean = tissue_reader.read(y0,y1,0,shape[1])
                add = candidate & (stained | clean) & ~art
                if kind == "labels":
                    counts["original_artifact_pixels_removed"] += int(((base > 0) & art).sum())
                    counts["added_pixels"] += int(add.sum())
                    counts["native_white_candidates_rejected"] += int((candidate & ~stained).sum())
                    base[add] = proposal[add]
                    base[art] = 0
                    yield base
                else:
                    prov = np.zeros(base.shape, np.uint8)
                    prov[base > 0] = 1
                    prov[add] = codes[add]
                    prov[art] = 0
                    yield prov
        tile_shape = (256, ((shape[1]+15)//16)*16)
        def padded_tiles(kind):
            for block in blocks(kind):
                tile = np.zeros(tile_shape, dtype=block.dtype)
                tile[:block.shape[0], :block.shape[1]] = block
                yield tile
        for kind, dtype in (("labels", old.dtype), ("provenance", np.uint8)):
            tifffile.imwrite(out / ("completed_"+kind+".ome.tif"), padded_tiles(kind),
                shape=shape, dtype=dtype, tile=tile_shape, compression="zlib", bigtiff=True,
                metadata={"axes":"YX", "PhysicalSizeX":a.mpp,"PhysicalSizeY":a.mpp,
                          "PhysicalSizeXUnit":"µm", "PhysicalSizeYUnit":"µm"})
        report["native_completion"] = counts
    (out / "completion_report.json").write_text(json.dumps(report, indent=2)+"\n")
    image.close(); labels.close(); tissue_reader.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
