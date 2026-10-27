# Bora expert ROI intake and same-slide evaluation

`bin/bora_semantic_roi_intake.py` is an opt-in, CPU-only stage that prepares a
**candidate** Bora `LocalSemanticModel` from independently annotated tissue
polygons. It never dispatches Virchow2, a GPU case, whole-slide inference, a
project export, or a tissue annotation. A passing same-slide spatial holdout is
development evidence; independent-patient biological validation remains open.

The command is:

```bash
python3 bin/bora_semantic_roi_intake.py \
  --spec /absolute/path/to/frozen-semantic-roi-spec.json \
  --output /absolute/path/to/new-semantic-roi-candidate
```

The output path must not exist. It contains only `local_semantic_model.npz` and
`EVALUATION.json`. A failed holdout gate retains these artifacts for review and
sets `status: same_slide_holdout_failed`; it never becomes a prediction. A
passing gate sets
`same_slide_holdout_passed_independent_patient_validation_pending`, which is
still **not** authorization for full-slide projection or annotation publication.

The specification has `format: cellphenotyper-bora-semantic-roi-intake-v1`, an
absolute `bora_source_root` pointing to the reviewed Bora `src` directory, a
`bora_source_sha256` object keyed by `__init__.py`, `admission.py`,
`chunk_plan.py`, `chunk_store.py`, `token_grid.py`, and `local_semantic.py`, an
integer `seed`, `max_samples_per_class` in 2–1024, and an `inputs` object.
Every input entry is `{ "path": "/absolute/path", "sha256": "64 lowercase hex" }`:

| Input | Required evidence |
| --- | --- |
| `registration` | Verified dual-MPP native/working pixel-edge frame and affine |
| `native_image`, `working_image` | Exact registered level-0 TIFF bytes, dimensions and MPP |
| `working_he`, `he_evidence` | Registered `uint8` H&E NPY and its Bora evidence receipt |
| `working_support`, `support_receipt` | Registered binary working support NPY and Bora coarse receipt |
| `token_inventory` | Complete `CHUNK_INVENTORY.json` with all scale-1/2/4 chunk hashes and lifecycle completion |
| `virchow2_checkpoint` | Encoder file whose SHA-256 matches the inventory checkpoint |
| `expert_geojson` | Native pixel-edge expert FeatureCollection bound to image frame and inventory SHA-256 |

The GeoJSON is a FeatureCollection with `schema:
cellphenotyper-expert-tissue-v1` (the pinned PM016
`pm016-expert-tissue-v1` template is also accepted). Its `source` must specify
`source_frame_id`, `image_sha256`,
`native_shape_yx`, `native_mpp_xy`, `token_inventory_sha256`, and
`coordinate_space: native_image_pixel_edges`. Its `class_names` maps integer
IDs 1–253 to distinct tissue names. Every Polygon/MultiPolygon Feature needs
`annotation_id`, `class_id`, `split` (`train` or `holdout`), `roi_id`,
`annotator_id`, and `annotation_source: independent_expert`. Do not derive these
polygons from an old Bora prediction. Define the actual tissue taxonomy with
the expert; do not force “tumor” onto a slide that lacks a confirmed tumor
region. The stage checks frame bounds, simple-ring intersections, overlapping
polygons, distinct ROI IDs, and a 224 µm gap between train and holdout bounds.
Holes, MultiPolygons, or ambiguous subpixel overlaps require GEOS/Shapely for
full topology validation; without it the stage stops.

Each eligible label must cover the entire 14×14 registered working-pixel
Virchow2 token interior. The selected token mask is eroded by one token and
must lie wholly in tissue support. The registered working H&E is white-padded
only at the true bottom/right image edge, then averaged over the exact 14×14
token footprints. The model input is 1,280 Virchow2 channels plus 3 H&E
channels; incomplete CellViT composition is excluded. Training samples are
selected only from `train` polygons, at most 1,024 per class. The holdout is
read after fitting and is scored with `predict`, **not** Bora's
independent-patient `evaluate()` method.

The frozen same-slide gates are at least 100 retained holdout tokens and two
distinct holdout ROIs per class, macro IoU ≥0.70, every class IoU ≥0.60 and
recall ≥0.70, and predictions containing more than one class. The report saves
the confusion matrix, per-class precision/recall/IoU, per-polygon retained
counts, selected training polygon counts, exact source hashes and model hash.
No label accuracy claim is possible until an expert has inspected the
annotation alignment on the full native H&E. A separate downstream stage and
admission review are required for bounded full-slide probabilities, H&E-guided
boundary refinement, global ≥100 µm² vector cleanup, and project publication.

The synthetic tests use the staged reviewed Bora source and run with:

```bash
python3 -m pytest -q tests/test_bora_semantic_roi_intake.py
```

Set `BORA_SEMANTIC_SOURCE_ROOT` to a different reviewed Bora `src` path when
running the tests elsewhere. If no Bora source is available, the tests skip;
the production command itself fails closed.
