# Draw PM016 expert tissue regions in wsiTools

This procedure creates **reviewer labels**, not a Bora prediction. It does not
assign a biological class automatically. Use a separate copy of the PM016
experiment and save the viewer session under a new name; the pipeline project,
H&E TIFF, existing predictions and CellViT files remain untouched.

1. In wsiTools Desktop, choose **Open project → Experiment or saved project
   (.json)** and open the copied `experiment.cellphenotyper.json`. Check that the
   displayed tissue is PM016 TLN1 H&E. The registered native image is
   23,904 × 14,762 pixels at approximately 0.25 µm/pixel.
2. Hide Bora/KODAMA predicted overlays so they do not become your annotation.
   In **Annotations**, choose an explicit tissue class name approved by the
   reviewer. Use **Draw polygon ROI** (`D`) or **Brush annotation** (`Ctrl+B`) to
   draw new regions on the H&E. Give each region a distinct name. Draw separate
   training and holdout regions for *every* declared class; keep their bounding
   boxes at least 224 µm apart. Avoid uncertain edges and verify the full slide
   location before saving. Do not relabel an imported predicted boundary.
3. Choose **Project → Save project** and save a *new* viewer-session file for
   review. In **Annotations**, select/check only the newly drawn regions, then
   choose **Export selected ROIs** (`Ctrl+E`) as GeoJSON. Keep this exact export;
   any later edit requires a new review manifest. The converter accepts only
   viewer `drawn_roi_*` / `brush_roi_*` Features with `source: drawn` / `brush`,
   the matching viewer class, and level-0 image coordinates. An imported Bora
   layer, renamed prediction, generic `annotation` class or uncertain source is
   refused.
4. Have the responsible reviewer explicitly declare the class taxonomy,
   train/holdout split, unique region ID, and annotator ID for **every** exported
   Feature ID. Save these declarations as `review.json` using the shape below.
   `class_names` values must *exactly* match the classes chosen in the viewer.
   Neither a class nor a split is inferred from ROI colour, filename, geometry,
   KODAMA group or Bora prediction.

```json
{
  "format": "cellphenotyper-wsitools-expert-review-v1",
  "source_geojson_sha256": "<SHA-256 of the exact exported GeoJSON>",
  "project_sha256": "<SHA-256 of the copied experiment.cellphenotyper.json>",
  "class_names": {"1": "<reviewer class 1>", "2": "<reviewer class 2>"},
  "reviewed_features": {
    "drawn_roi_1": {
      "class_id": 1,
      "split": "train",
      "roi_id": "<unique region ID>",
      "annotator_id": "<reviewer ID>",
      "annotation_source": "independent_expert"
    }
  }
}
```

List **all** exported Feature IDs under `reviewed_features`, including separate
`holdout` regions for every class. Get the hashes with `shasum -a 256` on macOS.
The review manifest is an assertion by the reviewer; this tool cannot prove
who drew a polygon. It can reject model/import provenance and mismatched files.

Run this CPU-only conversion after obtaining the pinned, complete PM016
`CHUNK_INVENTORY.json` and registered `registration.json` used by the intended
Bora token run:

```bash
python3 bin/prepare_bora_expert_geojson.py \
  --project /absolute/path/to/copied/experiment.cellphenotyper.json \
  --registration /absolute/path/to/registration.json \
  --image /absolute/path/to/copied/01_input/PM016_TLN1_HE.ome.tif \
  --inventory /absolute/path/to/CHUNK_INVENTORY.json \
  --geojson /absolute/path/to/reviewed.geojson \
  --review /absolute/path/to/review.json \
  --output /absolute/path/to/new/expert-tissue.geojson
```

The converter checks the copied experiment against the registered native frame,
rehashes the native image, verifies the export and project digests, preserves
native pixel-edge polygons and X/Y MPP, rejects ambiguous provenance, checks
class coverage in both splits, non-overlap and the 224 µm bounding-box gap. It
never overwrites an output. Complex overlapping bounding boxes need Shapely for
exact topology; the command stops if Shapely is absent. The result follows
`bin/bora_semantic_roi_intake.py`'s expert GeoJSON contract. Use its separate
source-bound intake and holdout evaluation before considering any full-slide
projection. Passing a same-slide holdout is development evidence, not
independent-patient validation.

Focused tests: `python3 -m pytest -q tests/test_prepare_bora_expert_geojson.py`.
