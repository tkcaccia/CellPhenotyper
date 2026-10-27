"""CPU-only, source-bound Bora expert ROI fit and same-slide holdout check.

This stage does not project a model over a slide, make GeoJSON predictions,
modify a CellPhenotyper project, or invoke a GPU. It requires a completed
Virchow2 chunk inventory and independently annotated native-pixel polygons.
The resulting evidence is same-slide development evidence, never independent
patient validation or ground truth for unannotated tissue.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion


FORMAT = "cellphenotyper-bora-semantic-roi-intake-v1"
REQUIRED_INPUTS = ("registration", "native_image", "working_image", "working_he",
                   "he_evidence", "working_support", "support_receipt",
                   "token_inventory", "virchow2_checkpoint", "expert_geojson")
SOURCE_MODULES = ("__init__.py", "admission.py", "chunk_plan.py",
                  "chunk_store.py", "token_grid.py", "local_semantic.py")
MIN_GAP_UM = 224.0
MIN_HOLDOUT_TOKENS = 100
MIN_HOLDOUT_ROIS = 2
MIN_MACRO_IOU = 0.70
MIN_CLASS_IOU = 0.60
MIN_CLASS_RECALL = 0.70
STRIDE = 14
CHANNELS = 1280
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_PAIR_RASTER_PIXELS = 16_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Missing or oversized JSON: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _bound_file(ref: dict, name: str) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{name} requires exact path and SHA-256")
    path = Path(ref["path"])
    expected = ref["sha256"]
    if (not path.is_absolute() or not path.is_file() or
            not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None or
            sha256(path) != expected):
        raise ValueError(f"{name} source is missing or differs from frozen SHA-256")
    return path


def _frame(frame: dict, image_path: Path) -> tuple[int, int, float, float]:
    import tifffile

    if not isinstance(frame, dict) or frame.get("sha256") != sha256(image_path):
        raise ValueError("Image SHA-256 differs from registered frame")
    with tifffile.TiffFile(image_path) as image:
        page = image.pages[0]
        h, w = int(page.imagelength), int(page.imagewidth)
    if (frame.get("height"), frame.get("width")) != (h, w):
        raise ValueError("Image dimensions differ from registered frame")
    if (frame.get("coordinate_space") != "crop-native-pixels" or
            frame.get("coordinate_convention") != "pixel-edge-origin_centres-in-native-pixels" or
            frame.get("origin_x") != 0 or frame.get("origin_y") != 0 or
            not isinstance(frame.get("frame_id"), str) or not frame["frame_id"]):
        raise ValueError("A full, zero-origin pixel-edge image frame is required")
    mx, my = frame.get("mpp_x"), frame.get("mpp_y")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (mx, my)):
        raise ValueError("Registered positive X/Y MPP is required")
    identity = {key: frame[key] for key in ("sha256", "width", "height", "mpp_x",
        "mpp_y", "origin_x", "origin_y", "coordinate_space", "coordinate_convention")}
    expected = "cellvit-frame-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    if frame["frame_id"] != expected:
        raise ValueError("Frame ID differs from calibrated image identity")
    return h, w, float(mx), float(my)


def _source_modules(root: Path) -> dict:
    if not root.is_absolute() or not (root / "bora").is_dir():
        raise ValueError("Bora source root must be an absolute src directory")
    files = {name: root / "bora" / name for name in SOURCE_MODULES}
    if any(not path.is_file() for path in files.values()):
        raise ValueError("Bora semantic/token-reader source is incomplete")
    sys.path.insert(0, str(root))
    semantic = importlib.import_module("bora.local_semantic")
    token_grid = importlib.import_module("bora.token_grid")
    if (Path(semantic.__file__).resolve() != files["local_semantic.py"].resolve() or
            Path(token_grid.__file__).resolve() != files["token_grid.py"].resolve()):
        raise ValueError("Imported Bora source differs from selected source root")
    return {"semantic": semantic, "token_grid": token_grid,
            "sha256": {name: sha256(path) for name, path in files.items()}}


def _ring(points, w: int, h: int) -> list[tuple[float, float]]:
    if not isinstance(points, list) or not 4 <= len(points) <= 2000:
        raise ValueError("Polygon ring requires 4–2000 vertices")
    ring = []
    for point in points:
        if (not isinstance(point, list) or len(point) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
            raise ValueError("Polygon coordinates must be finite [x,y] pairs")
        x, y = map(float, point)
        if not (0 <= x <= w and 0 <= y <= h):
            raise ValueError("Expert polygon leaves native image frame")
        ring.append((x, y))
    if ring[0] != ring[-1] or any(a == b for a, b in zip(ring, ring[1:])):
        raise ValueError("Polygon ring is unclosed or has a zero-length edge")
    if abs(sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(ring, ring[1:]))) < 1e-8:
        raise ValueError("Polygon ring has zero area")
    edges = list(zip(ring, ring[1:]))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def intersects(a, b, c, d):
        p, q, r, s = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
        if p*q < 0 and r*s < 0:
            return True
        return any(abs(v) < 1e-10 and min(e[0], f[0]) <= g[0] <= max(e[0], f[0])
                   and min(e[1], f[1]) <= g[1] <= max(e[1], f[1])
                   for v, e, f, g in ((p,a,b,c),(q,a,b,d),(r,c,d,a),(s,c,d,b)))
    for i, (a,b) in enumerate(edges):
        for j in range(i+1, len(edges)):
            if j == i+1 or (i == 0 and j == len(edges)-1):
                continue
            if intersects(a,b,*edges[j]):
                raise ValueError("Expert polygon ring self-intersects")
    return ring


def _geometry(value: dict, w: int, h: int) -> tuple[float,float,float,float]:
    from rasterio.features import is_valid_geom

    if (not isinstance(value, dict) or value.get("type") not in ("Polygon", "MultiPolygon")
            or not is_valid_geom(value)):
        raise ValueError("Expert geometry must be a valid Polygon or MultiPolygon")
    polygons = [value["coordinates"]] if value["type"] == "Polygon" else value["coordinates"]
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("Expert geometry has no polygons")
    if value["type"] == "MultiPolygon" or any(len(polygon) != 1 for polygon in polygons):
        try:
            from shapely.geometry import shape
        except ImportError as exc:
            raise ValueError("Holes and MultiPolygons require GEOS/Shapely topology validation") from exc
        geometry = shape(value)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Expert geometry fails GEOS topology validation")
    points = []
    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            raise ValueError("Expert polygon has no rings")
        for ring in polygon:
            points.extend(_ring(ring, w, h))
    return (min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points))


def _overlap(a: dict, b: dict, native_shape: tuple[int,int]) -> bool:
    from affine import Affine
    from rasterio.features import rasterize

    aa, bb = a["bbox"], b["bbox"]
    x0, y0 = math.floor(max(aa[0],bb[0])), math.floor(max(aa[1],bb[1]))
    x1, y1 = math.ceil(min(aa[2],bb[2])), math.ceil(min(aa[3],bb[3]))
    if x1 <= x0 or y1 <= y0:
        return False
    if (x1-x0)*(y1-y0) > MAX_PAIR_RASTER_PIXELS:
        raise ValueError("Potential expert polygon overlap exceeds bounded native raster check")
    transform = Affine.translation(x0,y0)
    shape = (y1-y0,x1-x0)
    left = rasterize([(a["geometry"],1)], out_shape=shape, transform=transform, dtype="uint8")
    right = rasterize([(b["geometry"],1)], out_shape=shape, transform=transform, dtype="uint8")
    return bool(np.any(left & right))


def _annotations(document: dict, registration: dict, inventory_sha: str) -> tuple[dict,list]:
    native = registration["cell_full_image"]
    source = document.get("source")
    expected = {"source_frame_id":native["frame_id"],"image_sha256":native["sha256"],
        "native_shape_yx":[native["height"],native["width"]],
        "native_mpp_xy":[native["mpp_x"],native["mpp_y"]],
        "token_inventory_sha256":inventory_sha,
        "coordinate_space":"native_image_pixel_edges"}
    if (document.get("type") != "FeatureCollection" or
            document.get("schema") not in
                ("cellphenotyper-expert-tissue-v1", "pm016-expert-tissue-v1") or
            not isinstance(source, dict) or any(source.get(k) != v for k,v in expected.items())):
        raise ValueError("Expert annotation is not bound to native H&E and complete token inventory")
    names = document.get("class_names")
    if (not isinstance(names, dict) or len(names) < 2 or
            any(not k.isdecimal() or not 1 <= int(k) <= 253 or
                not isinstance(v,str) or not v.strip() for k,v in names.items()) or
            len(set(names.values())) != len(names)):
        raise ValueError("At least two unique named tissue classes are required")
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Independently drawn expert polygons are required")
    rows, ids = [], set()
    for feature in features:
        props = feature.get("properties") if isinstance(feature,dict) and feature.get("type") == "Feature" else None
        if not isinstance(props,dict):
            raise ValueError("Every expert polygon needs Feature properties")
        aid, cid, split, roi = (props.get(key) for key in ("annotation_id","class_id","split","roi_id"))
        if (not isinstance(aid,str) or not aid.strip() or aid in ids or
                type(cid) is not int or str(cid) not in names or
                split not in ("train","holdout") or not isinstance(roi,str) or not roi.strip() or
                props.get("annotation_source") != "independent_expert" or
                not isinstance(props.get("annotator_id"),str) or not props["annotator_id"].strip()):
            raise ValueError("Expert class, split, annotation/ROI ID or annotator provenance missing")
        ids.add(aid)
        geom = feature.get("geometry")
        bbox = _geometry(geom, native["width"], native["height"])
        rows.append(dict(annotation_id=aid,class_id=cid,split=split,roi_id=roi,
                         annotator_id=props["annotator_id"],
                         bbox=bbox,geometry=geom))
    for index,a in enumerate(rows):
        for b in rows[index+1:]:
            if _overlap(a,b,(native["height"],native["width"])):
                raise ValueError(f"Expert polygons overlap: {a['annotation_id']} / {b['annotation_id']}")
            aa,bb=a["bbox"],b["bbox"]
            if min(aa[2],bb[2]) > max(aa[0],bb[0]) and min(aa[3],bb[3]) > max(aa[1],bb[1]):
                try:
                    from shapely.geometry import shape
                except ImportError as exc:
                    raise ValueError("Ambiguous subpixel polygon overlap requires GEOS/Shapely") from exc
                if shape(a["geometry"]).intersection(shape(b["geometry"])).area > 0:
                    raise ValueError(f"Expert polygons overlap below raster resolution: {a['annotation_id']} / {b['annotation_id']}")
    train = [r for r in rows if r["split"] == "train"]
    holdout = [r for r in rows if r["split"] == "holdout"]
    if ({r["class_id"] for r in train} != {int(k) for k in names} or
            {r["class_id"] for r in holdout} != {int(k) for k in names} or
            {r["roi_id"] for r in train} & {r["roi_id"] for r in holdout}):
        raise ValueError("Each declared class needs separate train/holdout ROIs")
    gaps = []
    for a in train:
        for b in holdout:
            aa,bb=a["bbox"],b["bbox"]
            dx=max(0.,aa[0]-bb[2],bb[0]-aa[2])*native["mpp_x"]
            dy=max(0.,aa[1]-bb[3],bb[1]-aa[3])*native["mpp_y"]
            gaps.append(math.hypot(dx,dy))
    if not gaps or min(gaps) < MIN_GAP_UM:
        raise ValueError("Train and holdout ROI bounds need a 224 µm spatial gap")
    return {int(k):v for k,v in names.items()}, rows


def _token_mask(geometry: dict, bbox: tuple, token_shape: tuple,
                working_shape: tuple, native_per_working: tuple) -> np.ndarray:
    """Only complete, 14x14 working-pixel interiors; then erode one token."""
    from affine import Affine
    from rasterio.features import rasterize

    th,tw=token_shape
    sx,sy=native_per_working
    x0=max(0,int(math.floor(bbox[0]/sx/STRIDE))-1)
    y0=max(0,int(math.floor(bbox[1]/sy/STRIDE))-1)
    x1=min(tw,int(math.ceil(bbox[2]/sx/STRIDE))+1)
    y1=min(th,int(math.ceil(bbox[3]/sy/STRIDE))+1)
    result=np.zeros((th,tw),bool)
    for ty in range(y0,y1,64):
        for tx in range(x0,x1,64):
            ny,nx=min(64,y1-ty),min(64,x1-tx)
            affine=Affine.translation(tx*STRIDE*sx,ty*STRIDE*sy) @ Affine.scale(sx,sy)
            pixels=rasterize([(geometry,1)],out_shape=(ny*STRIDE,nx*STRIDE),
                transform=affine,dtype="uint8",all_touched=False)
            whole=pixels.reshape(ny,STRIDE,nx,STRIDE).all(axis=(1,3))
            result[ty:ty+ny,tx:tx+nx]=whole
    # The final token may include synthetic white padding; it cannot receive
    # an expert label. A one-token erosion excludes rasterized edge ambiguity.
    valid_y=(np.arange(th)+1)*STRIDE<=working_shape[0]
    valid_x=(np.arange(tw)+1)*STRIDE<=working_shape[1]
    result &= valid_y[:,None] & valid_x[None,:]
    return binary_erosion(result,structure=np.ones((3,3),bool),border_value=0)


def _he_token_features(he: np.ndarray, token_shape: tuple) -> np.ndarray:
    th,tw=token_shape
    result=np.empty((th,tw,3),np.float32)
    for y in range(th):
        strip=np.full((STRIDE,tw*STRIDE,3),255,np.uint8)
        actual=max(0,min(STRIDE,he.shape[0]-y*STRIDE))
        if actual:
            strip[:actual,:he.shape[1]]=he[y*STRIDE:y*STRIDE+actual]
        result[y]=strip.reshape(STRIDE,tw,STRIDE,3).mean(axis=(0,2),dtype=np.float64)/255.
    return result


def _support_tokens(support: np.ndarray, token_shape: tuple) -> np.ndarray:
    th,tw=token_shape
    result=np.zeros((th,tw),bool)
    for y in range(th):
        if (y+1)*STRIDE>support.shape[0]:
            continue
        width=(support.shape[1]//STRIDE)*STRIDE
        row=support[y*STRIDE:(y+1)*STRIDE,:width]
        result[y,:width//STRIDE]=row.reshape(STRIDE,width//STRIDE,STRIDE).all(axis=(0,2))
    return result


def _read_vectors(grid, positions: np.ndarray) -> np.ndarray:
    result=np.empty((len(positions),CHANNELS),np.float32)
    if not len(positions):
        return result
    width=grid.shape[1]
    rows=positions//width
    cols=positions%width
    for y in np.unique(rows):
        selected=np.flatnonzero(rows==y)
        x0,x1=int(cols[selected].min()),int(cols[selected].max())+1
        block=grid.read_block(int(y),int(y)+1,x0,x1)[0]
        result[selected]=block[cols[selected]-x0]
    if not np.isfinite(result).all():
        raise ValueError("Selected spatial tokens are nonfinite")
    return result


def _validate_sources(spec: dict):
    import tifffile

    if spec.get("format") != FORMAT or set(spec.get("inputs",{})) != set(REQUIRED_INPUTS):
        raise ValueError("Complete source-bound semantic ROI specification required")
    inputs={name:_bound_file(spec["inputs"][name],name) for name in REQUIRED_INPUTS}
    registration=read_json(inputs["registration"])
    if (registration.get("format") != "cellphenotyper-dual-mpp-registration-v1" or
            registration.get("verified") is not True or
            registration.get("support_sampling") !=
                "absolute pixel centre, nearest, full-frame GrandQC mask"):
        raise ValueError("Verified dual-MPP registration with native support sampling required")
    native=registration.get("cell_full_image")
    working=registration.get("image")
    nh,nw,nmx,nmy=_frame(native,inputs["native_image"])
    wh,ww,wmx,wmy=_frame(working,inputs["working_image"])
    sx,sy=nw/ww,nh/wh
    expected=[[sx,0,0],[0,sy,0],[0,0,1]]
    affine=registration.get("tissue_to_cell_full_affine")
    if (not isinstance(affine,list) or any(not isinstance(row,list) or len(row)!=3 for row in affine) or
            any(not math.isclose(float(actual),wanted,rel_tol=1e-12,abs_tol=1e-9)
                for row,goal in zip(affine,expected) for actual,wanted in zip(row,goal)) or
            not math.isclose(nw*nmx,ww*wmx,rel_tol=1e-6,abs_tol=1e-6) or
            not math.isclose(nh*nmy,wh*wmy,rel_tol=1e-6,abs_tol=1e-6)):
        raise ValueError("Native/working pixel-edge affine or physical field differs")
    evidence=read_json(inputs["he_evidence"])
    coarse=read_json(inputs["support_receipt"])
    if (evidence.get("format") != "coordinator-bora-global-evidence-v1" or
            evidence.get("source_sha256",{}).get("registration") != spec["inputs"]["registration"]["sha256"] or
            evidence.get("output_sha256",{}).get("working_he") != spec["inputs"]["working_he"]["sha256"] or
            coarse.get("format") != "coordinator-bora-global-coarse-v1" or
            coarse.get("input_sha256",{}).get("registration") != spec["inputs"]["registration"]["sha256"] or
            coarse.get("output_sha256",{}).get("support") != spec["inputs"]["working_support"]["sha256"] or
            any(record.get("working_frame_id") != working["frame_id"] or
                record.get("working_image_sha256") != working["sha256"] or
                record.get("working_shape_yx") != [wh,ww]
                for record in (evidence,coarse))):
        raise ValueError("H&E or support receipt differs from registered working image")
    inventory=read_json(inputs["token_inventory"])
    if (inventory.get("format") != "coordinator-bora-token-chunk-inventory-v1" or
            inventory.get("working_frame_id") != working["frame_id"] or
            inventory.get("working_image_sha256") != working["sha256"] or
            inventory.get("checkpoint_sha256") != spec["inputs"]["virchow2_checkpoint"]["sha256"] or
            inventory.get("all_cleanup_verified") is not True or
            inventory.get("one_global_absolute_field") is not True or
            inventory.get("completed_case_count") != inventory.get("plan",{}).get("case_count") or
            re.fullmatch(r"[0-9a-f]{64}",str(inventory.get("checkpoint_sha256",""))) is None):
        raise ValueError("Complete lifecycle-qualified Virchow2 token inventory required")
    he=np.load(inputs["working_he"],mmap_mode="r",allow_pickle=False)
    support=np.load(inputs["working_support"],mmap_mode="r",allow_pickle=False)
    if (he.shape!=(wh,ww,3) or he.dtype!=np.uint8 or
            support.shape!=(wh,ww) or support.dtype not in (np.uint8,np.bool_) or
            not np.isin(support,[0,1]).all()):
        raise ValueError("Registered working H&E/support arrays are invalid")
    modules=_source_modules(Path(spec["bora_source_root"]))
    if spec.get("bora_source_sha256") != modules["sha256"]:
        raise ValueError("Bora semantic/token-reader source differs from frozen specification")
    for factor in (1,2,4):
        modules["token_grid"].ChunkGrid(inputs["token_inventory"].parent / f"scale-{factor}",
                                         factor=factor).verify_all()
    grid=modules["token_grid"].ChunkGrid(inputs["token_inventory"].parent / "scale-1",factor=1)
    token_shape=(math.ceil(wh/STRIDE),math.ceil(ww/STRIDE))
    if grid.shape!=(token_shape[0],token_shape[1],CHANNELS):
        raise ValueError("Completed token grid differs from registered 14-pixel H&E frame")
    annotations=read_json(inputs["expert_geojson"])
    names,features=_annotations(annotations,registration,
        spec["inputs"]["token_inventory"]["sha256"])
    return inputs,registration,inventory,modules,grid,he,support,names,features,(nh,nw,nmx,nmy),(wh,ww,wmx,wmy)


def run(spec_path: Path, output: Path) -> dict:
    spec_path=Path(spec_path).resolve(strict=True)
    output=Path(output)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError("Use a fresh absolute output directory under an existing parent")
    spec=read_json(spec_path)
    # An empty intake template is a scientific blocker. Reject it before
    # hashing whole images and every frozen token chunk.
    if isinstance(spec.get("inputs"),dict) and "expert_geojson" in spec["inputs"]:
        early=read_json(_bound_file(spec["inputs"]["expert_geojson"],"expert_geojson"))
        if not early.get("features"):
            raise ValueError("Independently drawn expert polygons are required")
    inputs,reg,inventory,modules,grid,he,support,names,features,native,working=_validate_sources(spec)
    if (output.resolve().is_relative_to(inputs["token_inventory"].parent.resolve()) or
            output.resolve().is_relative_to(Path(spec["bora_source_root"]).resolve())):
        raise ValueError("Semantic candidate output must be separate from frozen tokens and source code")
    token_shape=grid.shape[:2]
    _,_,nmx,nmy=native
    wh,ww,wmx,wmy=working
    sx,sy=native[1]/ww,native[0]/wh
    labels={split:np.zeros(token_shape,np.uint8) for split in ("train","holdout")}
    owners=np.full(token_shape,-1,np.int32)
    roi_tokens=defaultdict(set)
    per_feature=[]
    tissue=_support_tokens(support,token_shape)
    for feature_index,feature in enumerate(features):
        mask=_token_mask(feature["geometry"],feature["bbox"],token_shape,(wh,ww),(sx,sy))
        mask &= tissue
        split=feature["split"]
        if np.any(mask & ((labels["train"]>0)|(labels["holdout"]>0))):
            raise ValueError("Expert polygon token interiors overlap after registration")
        labels[split][mask]=feature["class_id"]
        owners[mask]=feature_index
        count=int(mask.sum())
        roi_tokens[(split,feature["class_id"],feature["roi_id"])].update(np.flatnonzero(mask))
        per_feature.append(dict(annotation_id=feature["annotation_id"],split=split,
                                class_id=feature["class_id"],roi_id=feature["roi_id"],
                                annotator_id=feature["annotator_id"],
                                retained_supported_interior_tokens=count))
    for cid in names:
        if not np.any(labels["train"]==cid):
            raise ValueError(f"Class {cid} has no unambiguous supported training interiors")
    he_features=_he_token_features(he,token_shape)
    frame=dict(format="virchow2-spatial-tokens-v1",
        encoder_sha256=inventory["checkpoint_sha256"],
        working_mpp_xy=[wmx,wmy],token_stride_yx=[STRIDE,STRIDE],
        token_origin_working_yx=[0,0],channel_order="YXC-1280")
    seed=spec.get("seed",314159)
    max_per_class=spec.get("max_samples_per_class",1024)
    if (type(seed) is not int or seed<0 or type(max_per_class) is not int or
            not 2<=max_per_class<=1024):
        raise ValueError("Training seed and 2..1024 samples/class must be fixed in the specification")
    rng=np.random.default_rng(seed)
    chosen=[]
    selected_counts={}
    for cid in sorted(names):
        available=np.flatnonzero(labels["train"]==cid)
        take=np.sort(rng.choice(available,size=min(len(available),max_per_class),replace=False))
        chosen.extend(take.tolist())
        selected_counts[str(cid)]=len(take)
    chosen=np.array(sorted(chosen),np.int64)
    selected_owner=owners.reshape(-1)[chosen]
    selected_by_polygon={feature["annotation_id"]:int(np.count_nonzero(selected_owner==index))
                         for index,feature in enumerate(features) if feature["split"]=="train"}
    selected_labels=labels["train"].reshape(-1)[chosen]
    train_tokens=_read_vectors(grid,chosen).reshape(-1,1,CHANNELS)
    train_he=he_features.reshape(-1,3)[chosen].reshape(-1,1,3)
    case=modules["semantic"].LocalSemanticCase(tokens=train_tokens,
        expert_labels=selected_labels.reshape(-1,1),specimen_group=reg["cell_full_image"]["frame_id"],
        frame=frame,split="train",annotation_source="independent_expert",
        patient_group=str(spec.get("patient_group") or reg["cell_full_image"]["frame_id"]),
        he_features=train_he)
    model=modules["semantic"].LocalSemanticModel().fit([case],class_names=names,
        max_samples_per_class=max_per_class,seed=seed)
    class_ids=list(map(int,model.classes_))
    confusion=np.zeros((len(class_ids),len(class_ids)),np.int64)
    holdout_positions=np.flatnonzero(labels["holdout"])
    predictions=np.empty(len(holdout_positions),np.uint8)
    for start in range(0,len(holdout_positions),1024):
        positions=holdout_positions[start:start+1024]
        vectors=_read_vectors(grid,positions).reshape(-1,1,CHANNELS)
        he_values=he_features.reshape(-1,3)[positions].reshape(-1,1,3)
        predicted,_=model.predict(vectors,frame=frame,he_features=he_values)
        predictions[start:start+len(positions)]=predicted.reshape(-1)
    truth=labels["holdout"].reshape(-1)[holdout_positions]
    for i,cid in enumerate(class_ids):
        for j,pid in enumerate(class_ids):
            confusion[i,j]=int(np.count_nonzero((truth==cid)&(predictions==pid)))
    class_report={}
    for i,cid in enumerate(class_ids):
        tp=int(confusion[i,i]); actual=int(confusion[i].sum()); predicted=int(confusion[:,i].sum())
        union=actual+predicted-tp
        roi_counts={roi:len(points) for (split,k,roi),points in roi_tokens.items()
                    if split=="holdout" and k==cid}
        class_report[str(cid)]=dict(name=names[cid],support=actual,
            holdout_roi_counts=roi_counts,
            precision=tp/predicted if predicted else None,
            recall=tp/actual if actual else None,
            iou=tp/union if union else None)
    values=[v["iou"] for v in class_report.values() if v["iou"] is not None]
    macro_iou=float(np.mean(values)) if values else 0.0
    gates=dict(
        holdout_material=all(v["support"]>=MIN_HOLDOUT_TOKENS and
            sum(count>0 for count in v["holdout_roi_counts"].values())>=MIN_HOLDOUT_ROIS
            for v in class_report.values()),
        macro_iou=macro_iou>=MIN_MACRO_IOU,
        per_class_iou=all(v["iou"] is not None and v["iou"]>=MIN_CLASS_IOU for v in class_report.values()),
        per_class_recall=all(v["recall"] is not None and v["recall"]>=MIN_CLASS_RECALL for v in class_report.values()),
        no_single_class_collapse=len(set(map(int,predictions)))>=2)
    if any(not v for v in gates.values()):
        status="same_slide_holdout_failed"
    else:
        status="same_slide_holdout_passed_independent_patient_validation_pending"
    source_hashes={name:spec["inputs"][name]["sha256"] for name in REQUIRED_INPUTS}
    source_hashes["spec"]=sha256(spec_path)
    source_hashes["bora_source_modules"]=modules["sha256"]
    report=dict(format="cellphenotyper-bora-semantic-roi-evaluation-v1",status=status,
        complete=True,
        source_sha256=source_hashes,source_frame_id=reg["cell_full_image"]["frame_id"],
        working_frame_id=reg["image"]["frame_id"],token_shape_yx=list(token_shape),
        token_stride_working_yx=[STRIDE,STRIDE],token_origin_working_yx=[0,0],
        token_label_rule="full_14x14_working_pixel_interior_then_one_token_erosion_and_full_support",
        he_feature_rule="registered_working_RGB_white_padded_to_token_grid_mean_14x14_divide_255",
        annotation_source="declared_independent_expert",same_slide_holdout=True,
        independent_patient_validation=False,probability_calibrated=False,
        train_holdout_minimum_gap_um=MIN_GAP_UM,training_sample_seed=seed,
        training_sample_cap_per_class=max_per_class,selected_train_tokens_by_class=selected_counts,
        selected_train_tokens_by_polygon=selected_by_polygon,
        per_polygon=per_feature,class_ids=class_ids,class_names=names,
        holdout_tokens=int(confusion.sum()),confusion=confusion.tolist(),
        per_class=class_report,macro_iou=macro_iou,gates=gates,
        gate_thresholds=dict(holdout_tokens_per_class=MIN_HOLDOUT_TOKENS,
            holdout_rois_per_class=MIN_HOLDOUT_ROIS,macro_iou=MIN_MACRO_IOU,
            class_iou=MIN_CLASS_IOU,class_recall=MIN_CLASS_RECALL),
        fullslide_projection_started=False,semantic_annotation_published=False)
    # One immutable result directory. A failed scientific gate still keeps its
    # model and exact held-out score record for review; it cannot launch inference.
    output.mkdir()
    model_path=output/"local_semantic_model.npz"
    model.save(model_path)
    report["model_sha256"]=sha256(model_path)
    grid.verify_all()
    if any(sha256(inputs[name])!=digest for name,digest in
           ((name,spec["inputs"][name]["sha256"]) for name in REQUIRED_INPUTS)):
        raise ValueError("Source file changed during semantic ROI evaluation")
    if modules["sha256"] != {name:sha256(Path(spec["bora_source_root"])/"bora"/name)
                             for name in SOURCE_MODULES}:
        raise ValueError("Bora source changed during semantic ROI evaluation")
    (output/"EVALUATION.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.spec,args.output),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
