#!/usr/bin/env python3
"""Assign each segmented object (cell) to a GeoJSON polygon label (FAST + parallel).

v6: easier polygon *name* labels
--------------------------------
If your output labels are just numbers, it's usually because the GeoJSON feature
properties do not contain the key we used, so we fell back to the feature index.

This version adds:
- --label-prop <key> : choose the property key/path that contains the label name
  * supports dotted paths into nested dicts, e.g. "classification.name"
- Auto-detect label key if --label-prop is omitted (prefers string-like values)
- --list-props : print available property keys/paths (top-level + one-level nested) and exit
- Keeps support for --shift shift.json (crop->full coordinate offsets)

Dependencies: pandas, numpy, shapely, tqdm

Examples
--------
1) List candidate label properties:
   python assign_cells_to_geojson_polygons_v6.py --roi ROI.geojson --list-props

2) Use a specific property as label:
   python assign_cells_to_geojson_polygons_v6.py ... --label-prop name

3) Nested property:
   python assign_cells_to_geojson_polygons_v6.py ... --label-prop classification.name

4) Typical run with shift + parallel:
   singularity exec --bind "$PWD":"$PWD" --pwd "$PWD" $SINGULARITY \
     python assign_cells_to_geojson_polygons_v6.py \
       --objects out_stardist_roi/objects.csv \
       --roi Data/Visium_HD_Human_Colon_Cancer_290325.geojson \
       --shift out_stardist_roi/shift.json \
       --out out_stardist_roi/objects_assigned.csv \
       --out-col polygon_name \
       --workers ${SLURM_CPUS_PER_TASK:-16}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from shapely import wkb

# ---- multiprocessing globals ----
_G_POLYS: List[Any] = []
_G_LABELS: List[Any] = []
_G_AREAS: np.ndarray | None = None
_G_TREE: STRtree | None = None
_G_CHOOSE: str = "smallest"
_G_XOFF: float = 0.0
_G_YOFF: float = 0.0


def _guess_xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    candidates = [
        ("x_fullres", "y_fullres"),
        ("centroid_x", "centroid_y"),
        ("x", "y"),
        ("X", "Y"),
        ("centroid-1", "centroid-0"),  # skimage: col,row
        ("cx", "cy"),
        ("x_centroid", "y_centroid"),
        ("col", "row"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ]
    for x, y in candidates:
        if x in cols and y in cols:
            return x, y
    x_like = [c for c in cols if c.lower().endswith(("x", "col")) or ("x" in c.lower())]
    y_like = [c for c in cols if c.lower().endswith(("y", "row")) or ("y" in c.lower())]
    for x in x_like:
        for y in y_like:
            if x != y:
                return x, y
    raise ValueError(
        f"Could not infer x/y columns from objects.csv. Columns: {cols}. "
        f"Please provide --xcol and --ycol."
    )


def _load_geojson(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        geo = json.load(f)
    if geo.get("type") != "FeatureCollection" or "features" not in geo:
        raise ValueError(f"ROI file does not look like a FeatureCollection: {path}")
    return geo


def _get_nested(d: Any, path: str) -> Any:
    """Get a value from dict via dotted path, e.g. 'classification.name'."""
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _flatten_prop_paths(props: Dict[str, Any]) -> List[str]:
    """List candidate paths: top-level keys plus one-level nested dict keys."""
    paths = []
    for k, v in props.items():
        paths.append(k)
        if isinstance(v, dict):
            for kk in v.keys():
                paths.append(f"{k}.{kk}")
    return paths


def _score_label_path(features: List[Dict[str, Any]], path: str) -> Tuple[int, int]:
    """Return (n_nonempty, n_stringy) for this path."""
    nonempty = 0
    stringy = 0
    for feat in features:
        props = feat.get("properties") or {}
        val = _get_nested(props, path)
        if val is None:
            continue
        # unwrap some common structures
        if isinstance(val, dict) and "name" in val:
            val = val["name"]
        if isinstance(val, (list, tuple)) and len(val) == 1:
            val = val[0]
        # count
        if isinstance(val, str) and val.strip() != "":
            nonempty += 1
            stringy += 1
        elif isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
            nonempty += 1
        elif isinstance(val, bool):
            nonempty += 1
    return nonempty, stringy


def _auto_choose_label_path(features: List[Dict[str, Any]], user_fallbacks: List[str]) -> Optional[str]:
    """Pick a label path that exists for most features, preferring strings."""
    if not features:
        return None
    # build candidate set from actual properties + fallbacks
    cand = set(user_fallbacks)
    # also add observed paths from first few features
    for feat in features[: min(50, len(features))]:
        props = feat.get("properties") or {}
        for p in _flatten_prop_paths(props):
            cand.add(p)

    best = None
    best_score = (-1, -1)  # (nonempty, stringy)
    for p in cand:
        nonempty, stringy = _score_label_path(features, p)
        score = (nonempty, stringy)
        if score > best_score:
            best_score = score
            best = p

    # Require that at least one feature has the value
    if best_score[0] <= 0:
        return None
    return best


def _normalize_label(val: Any, fallback: Any) -> Any:
    if val is None:
        return fallback
    if isinstance(val, dict):
        # common: {"name": "..."}
        if "name" in val and isinstance(val["name"], str):
            return val["name"]
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return fallback
        if len(val) == 1:
            return _normalize_label(val[0], fallback)
        return json.dumps(val, ensure_ascii=False)
    return val


def _load_polygons_and_labels(roi_path: str, label_path: Optional[str], fallback_paths: List[str]) -> Tuple[List[Any], List[Any], Optional[str]]:
    geo = _load_geojson(roi_path)
    features = geo["features"]

    chosen_path = label_path
    if chosen_path is None:
        chosen_path = _auto_choose_label_path(features, fallback_paths)

    polys: List[Any] = []
    labels: List[Any] = []

    for i, feat in enumerate(features):
        geom = feat.get("geometry")
        if geom is None:
            continue
        g = shape(geom)
        if g.is_empty:
            continue

        props = feat.get("properties") or {}
        lab = None
        if chosen_path is not None:
            lab = _normalize_label(_get_nested(props, chosen_path), fallback=None)
        if lab is None:
            # try fallbacks explicitly (including nested)
            for p in fallback_paths:
                lab = _normalize_label(_get_nested(props, p), fallback=None)
                if lab is not None:
                    break
        if lab is None:
            lab = i  # final fallback to feature index

        polys.append(g)
        labels.append(lab)

    if not polys:
        raise ValueError(f"No valid geometries found in ROI file: {roi_path}")

    return polys, labels, chosen_path


def _load_shift(shift_path: str) -> Tuple[float, float]:
    with open(shift_path, "r") as f:
        d = json.load(f)

    # New pipeline key: offset_crop_to_original
    if "offset_crop_to_original" in d:
        off = d["offset_crop_to_original"]
        if isinstance(off, dict):
            for kx, ky in [("x", "y"), ("x0", "y0"), ("dx", "dy"), ("col", "row")]:
                if kx in off and ky in off:
                    return float(off[kx]), float(off[ky])
        if isinstance(off, (list, tuple)) and len(off) >= 2:
            return float(off[0]), float(off[1])

    # crop_bbox_xyxy: [x0,y0,x1,y1]
    if "crop_bbox_xyxy" in d:
        bb = d["crop_bbox_xyxy"]
        if isinstance(bb, (list, tuple)) and len(bb) >= 2:
            return float(bb[0]), float(bb[1])

    # Older styles
    for kx, ky in [
        ("x0", "y0"),
        ("origin_x", "origin_y"),
        ("x", "y"),
        ("dx", "dy"),
        ("shift_x", "shift_y"),
        ("crop_x0", "crop_y0"),
    ]:
        if kx in d and ky in d:
            return float(d[kx]), float(d[ky])

    if "shift" in d and isinstance(d["shift"], dict):
        sd = d["shift"]
        for kx, ky in [("x0", "y0"), ("x", "y"), ("dx", "dy")]:
            if kx in sd and ky in sd:
                return float(sd[kx]), float(sd[ky])

    raise ValueError(f"Could not find x/y shift keys in {shift_path}. Keys present: {list(d.keys())}")


def init_worker(poly_wkb_list: List[bytes], labels: List[Any], choose: str, xoff: float, yoff: float) -> None:
    global _G_POLYS, _G_LABELS, _G_AREAS, _G_TREE, _G_CHOOSE, _G_XOFF, _G_YOFF
    _G_POLYS = [wkb.loads(b) for b in poly_wkb_list]
    _G_LABELS = labels
    _G_AREAS = np.array([p.area for p in _G_POLYS], dtype=np.float64)
    _G_TREE = STRtree(_G_POLYS)
    _G_CHOOSE = choose
    _G_XOFF = float(xoff)
    _G_YOFF = float(yoff)


def _tree_query_indices(tree: STRtree, geom: Any):
    cand = tree.query(geom)
    if isinstance(cand, np.ndarray) and np.issubdtype(cand.dtype, np.integer):
        return cand.tolist()
    idx = []
    for g in cand:
        try:
            idx.append(_G_POLYS.index(g))
        except ValueError:
            pass
    return idx


def _assign_points(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if _G_TREE is None or _G_AREAS is None:
        raise RuntimeError("Worker not initialized.")

    out = np.empty(xs.shape[0], dtype=object)
    polys = _G_POLYS
    labels = _G_LABELS
    areas = _G_AREAS
    tree = _G_TREE
    choose = _G_CHOOSE
    xoff = _G_XOFF
    yoff = _G_YOFF

    for i in range(xs.shape[0]):
        p = Point(float(xs[i]) + xoff, float(ys[i]) + yoff)
        cand_idx = _tree_query_indices(tree, p)
        if len(cand_idx) == 0:
            out[i] = np.nan
            continue
        hits = []
        for j in cand_idx:
            if polys[j].contains(p):
                hits.append(j)
        if not hits:
            out[i] = np.nan
            continue
        if len(hits) == 1 or choose == "first":
            out[i] = labels[hits[0]]
        else:
            k = hits[int(np.argmin(areas[hits]))]
            out[i] = labels[k]
    return out


def worker_task(payload):
    k, xs, ys = payload
    return k, _assign_points(xs, ys)


def _bbox_of_polys(polys: List[Any]) -> Tuple[float, float, float, float]:
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return minx, miny, maxx, maxy


def _list_props(roi_path: str) -> None:
    geo = _load_geojson(roi_path)
    features = geo["features"]
    counts: Dict[str, int] = {}
    for feat in features:
        props = feat.get("properties") or {}
        for p in _flatten_prop_paths(props):
            val = _get_nested(props, p)
            if val is None:
                continue
            # count non-empty
            if isinstance(val, str) and val.strip() == "":
                continue
            counts[p] = counts.get(p, 0) + 1

    if not counts:
        print("[INFO] No properties found (or all empty).")
        return

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print("[INFO] Available property paths (non-empty counts):")
    for p, c in items[:200]:
        print(f"  {p}  ({c}/{len(features)})")

    # show example from first feature with props
    for feat in features:
        props = feat.get("properties") or {}
        if props:
            print("\n[INFO] Example properties from one feature:")
            print(json.dumps(props, indent=2, ensure_ascii=False)[:4000])
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", help="objects.csv with centroids")
    ap.add_argument("--roi", required=True, help="GeoJSON polygons")
    ap.add_argument("--out", help="output CSV")
    ap.add_argument("--shift", default=None, help="shift.json to map crop coords -> full coords")
    ap.add_argument("--xcol", default=None)
    ap.add_argument("--ycol", default=None)
    ap.add_argument("--label-prop", default=None, help="Property key/path for polygon label (e.g., name or classification.name)")
    ap.add_argument("--fallback-props", nargs="*", default=["label", "name", "Name", "id", "classification", "classification.name"])
    ap.add_argument("--out-col", default="polygon_label")
    ap.add_argument("--choose", choices=["smallest", "first"], default="smallest")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--chunk-rows", type=int, default=20000)
    ap.add_argument("--list-props", action="store_true", help="List property paths in ROI and exit.")
    args = ap.parse_args()

    if args.list_props:
        _list_props(args.roi)
        return

    if not args.objects or not args.out:
        raise SystemExit("[ERROR] --objects and --out are required unless --list-props is used.")

    objects_path = Path(args.objects)
    roi_path = Path(args.roi)
    out_path = Path(args.out)

    if not objects_path.exists():
        raise SystemExit(f"[ERROR] objects file not found: {objects_path}")
    if not roi_path.exists():
        raise SystemExit(f"[ERROR] roi file not found: {roi_path}")

    print(f"[INFO] Reading objects: {objects_path}")
    df = pd.read_csv(objects_path)

    if args.xcol and args.ycol:
        xcol, ycol = args.xcol, args.ycol
    else:
        xcol, ycol = _guess_xy_columns(df)

    if xcol not in df.columns or ycol not in df.columns:
        raise SystemExit(f"[ERROR] x/y columns not found: xcol={xcol}, ycol={ycol}")

    xs = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    ys = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not np.all(valid):
        bad = int(np.sum(~valid))
        print(f"[WARN] {bad} rows have non-finite x/y; assigned as NaN.")
        xs = xs.copy(); ys = ys.copy()
        xs[~valid] = 0.0
        ys[~valid] = 0.0

    xoff = 0.0
    yoff = 0.0
    if args.shift:
        xoff, yoff = _load_shift(args.shift)
        print(f"[INFO] Applying shift: xoff={xoff}, yoff={yoff} (from {args.shift})")

    print(f"[INFO] Loading polygons: {roi_path}")
    polys, labels, chosen_path = _load_polygons_and_labels(str(roi_path), args.label_prop, list(args.fallback_props))
    if chosen_path:
        nonempty, stringy = _score_label_path(_load_geojson(str(roi_path))["features"], chosen_path)
        print(f"[INFO] Using label path: '{chosen_path}' (non-empty={nonempty}, string-like={stringy})")
    else:
        print("[INFO] No label path detected; using fallbacks then feature index.")

    poly_wkb_list = [p.wkb for p in polys]

    # bbox diagnostics
    pminx, pminy, pmaxx, pmaxy = _bbox_of_polys(polys)
    xmin, ymin = float(np.nanmin(xs[valid]) + xoff), float(np.nanmin(ys[valid]) + yoff)
    xmax, ymax = float(np.nanmax(xs[valid]) + xoff), float(np.nanmax(ys[valid]) + yoff)
    print(f"[INFO] Polygon bbox : x=[{pminx:.1f},{pmaxx:.1f}] y=[{pminy:.1f},{pmaxy:.1f}]")
    print(f"[INFO] Points  bbox : x=[{xmin:.1f},{xmax:.1f}] y=[{ymin:.1f},{ymax:.1f}]")
    if (xmax < pminx) or (xmin > pmaxx) or (ymax < pminy) or (ymin > pmaxy):
        print("[WARN] Point bbox does NOT overlap polygon bbox. Coordinate mismatch likely.")
        print("       - Check if objects.csv is crop coords (needs --shift).")
        print("       - Or use correct --xcol/--ycol (fullres vs crop).")

    if args.workers and args.workers > 0:
        workers = args.workers
    else:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or (os.cpu_count() or 1)
    workers = max(1, int(workers))

    n = df.shape[0]
    chunk_rows = max(1000, int(args.chunk_rows))
    print(f"[INFO] Objects={n} | Polygons={len(polys)} | xcol={xcol} ycol={ycol} | workers={workers} chunk_rows={chunk_rows}")

    init_worker(poly_wkb_list, labels, args.choose, xoff, yoff)

    assigned = np.empty(n, dtype=object)
    assigned[:] = np.nan
    if np.any(~valid):
        assigned[~valid] = np.nan

    idx_valid = np.where(valid)[0]
    if idx_valid.size == 0:
        df[args.out_col] = assigned
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[INFO] Wrote: {out_path}")
        return

    idx_slices = []
    tasks = []
    for s in range(0, idx_valid.size, chunk_rows):
        e = min(idx_valid.size, s + chunk_rows)
        idx = idx_valid[s:e]
        idx_slices.append(idx)
        tasks.append((len(tasks), xs[idx], ys[idx]))

    if workers == 1:
        print("[INFO] Single-process assignment ...")
        for k, (_, xchunk, ychunk) in enumerate(tqdm(tasks, total=len(tasks))):
            labs = _assign_points(xchunk, ychunk)
            assigned[idx_slices[k]] = labs
    else:
        print("[INFO] Multiprocessing assignment ...")
        import multiprocessing as mp
        ctx_name = os.environ.get("MP_CONTEXT", "fork")
        try:
            ctx = mp.get_context(ctx_name)
        except ValueError:
            ctx = mp.get_context("fork")

        with ctx.Pool(processes=workers, initializer=init_worker, initargs=(poly_wkb_list, labels, args.choose, xoff, yoff)) as pool:
            for k, labs in tqdm(pool.imap_unordered(worker_task, tasks, chunksize=1), total=len(tasks)):
                assigned[idx_slices[k]] = labs

    df[args.out_col] = assigned
    assigned_count = int(pd.Series(assigned).notna().sum())
    print(f"[INFO] Assigned {assigned_count}/{n} objects ({assigned_count/n*100:.2f}%).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote: {out_path}")


if __name__ == "__main__":
    main()
