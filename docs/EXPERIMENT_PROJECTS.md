# Open a CellPhenotyper experiment in wsiTools

An experiment is one JSON index for one or several tissues. It references the
calibrated images, annotations and existing feature tables; it does not embed
images, weights or feature arrays. Exporting, validating and opening the index
do not run an encoder or submit a workflow.

For a current completed Virchow2 run, execution reporting automatically creates
`00_execution/experiment.cellphenotyper.json` when its image, grid and cached
features pass validation. The landing page links the project file. Choose
**Open Project** in a compatible wsiTools Desktop version and select that file.
Keep the index in its output tree so relative paths continue to resolve.

`00_execution/experiment_export_status.json` always explains availability.
Legacy runs without complete Virchow2 caches remain readable as ordinary
reports. Invalid or incomplete caches produce an explicit export error; they
do not trigger inference. If an older experiment index exists, a failed export
preserves it and records `previous_index_retained=true`; do not mistake that
older index for a successful new export.

## Export existing runs

From the repository folder:

```bash
python bin/export_experiment.py --run /path/to/results \
  --output /path/to/results/00_execution/experiment.cellphenotyper.json \
  --name "My tissue study"
```

Repeat `--run` to combine output trees in one experiment. No files are copied.
Move the containing directory tree together for portability. Files outside that
tree remain external dependencies, and relative-link Nextflow publishing still
requires its link targets. Use `--overwrite` only to replace an existing index;
source images/caches are never overwritten.

Automatic discovery checks `VIRCHOW2_COMPLETE.json` and its grid/table hashes,
matches embedding IDs and coordinates to the grid CSV, verifies physical
sampling and uses the published analysis crop. Ambiguous/missing images or
multiple unmatched shards require an explicit specification; no tissue is
silently paired by filename alone. Current automatic discovery supports the
producer's single pooled shard per tissue. It does not invent dense token caches.
Current producer completion records do not retain a source-image digest.
Export verifies the present image bytes/header, table/grid hashes and geometry,
but cannot retroactively prove which historical image bytes the encoder read.
This limitation is recorded in tissue provenance. Preserve original run inputs
and treat input-image identity instrumentation as a separate producer change.

## Register an explicit image or existing token cache

Copy and edit [the small specification example](../examples/experiment_spec.example.json).
Paths in the specification are relative to the specification file. Add further
objects to `tissues` for other images, or add an existing spatial feature table
to `feature_sets`. Then run:

```bash
python bin/export_experiment.py --spec /path/to/assets.json \
  --output /path/to/experiment.cellphenotyper.json
python bin/export_experiment.py --validate /path/to/experiment.cellphenotyper.json
```

Dimensions and asset SHA-256 hashes are filled from files. Supplying a hash is
an integrity assertion: a mismatch fails instead of replacing it. `mpp_x` and
`mpp_y` must be supplied from verified calibration. The optional input-only
`feature_columns_prefix` expands matching CSV header columns; exported v1
indices always contain the full explicit `feature_columns` allowlist. You may
provide that list directly instead. No labels, IDs, split fields, coordinates
or area metadata are predictor columns.

## Version 1 contract

The root contains `schema="cellphenotyper-experiment"`, `schema_version=1`, a
stable `experiment_id`, friendly `name`, `settings`, `provenance` and `tissues`.
Every tissue contains `tissue_id`, `name`, optional `specimen_id`, `image`,
`artifacts` and `feature_sets`. IDs are independent of the containing folder;
local observation IDs may repeat across tissues, so consumers must namespace
them by tissue ID. Provide explicit IDs when two separately identified tissues
have the same name and source bytes. Names alone are not identities.

`image` contains a relative path, SHA-256, width, height, MPP X/Y and `frame_id`.
It is the exact calibrated image used by the features, which can be an analysis
crop. It must not silently stand in for the original WSI. Automatic export
records the crop-to-original affine in tissue provenance when `shift.json`
provides it; consumers must explicitly apply that affine when changing frames.

Coordinates use native pixels of the referenced image, with the origin at the
top-left pixel edge, X rightwards and Y downwards. CSV X/Y values are observation
centre/representative positions, within the image. `grid.origin_x_px` and
`origin_y_px` are **edge origins**, not first-observation centre coordinates.
Grids describe cell/footprint sizes and spacing; edge padding remains part of
feature preprocessing provenance. Current pooled exports preserve the grid
producer's in-image representative coordinate for a partial edge tile.

Compatible tissues share a semantic `feature_set_id`, such as
`virchow2-pooled`. Its assets/observations are namespaced by
`(tissue_id, feature_set_id)`, not by feature ID alone. IDs must be unique within
each tissue; matching IDs across tissues also require matching encoder and
preprocessing definitions before joint analysis.

Every feature set has its kind/unit, relative CSV/CSV.GZ path and hash,
explicit predictor columns, ID/X/Y columns, `coordinate_frame="image"`, encoder
revision/checkpoint digest, preprocessing, grid and verified row count:

| Kind | Unit | Width | Definition |
|---|---|---:|---|
| `virchow2-pooled` | `tile` | 2560 | CLS concatenated with mean spatial-patch features for one model tile |
| `virchow2-spatial` | `token` | 1280 | Spatial-patch token features; CLS and four registers excluded |

Model tiles cover 224 pixels at 0.5 µm/pixel in the current pipeline, with no
overlap. Spatial tokens occupy a 14-model-pixel lattice. Pooled tiles are not
cell observations and cannot reconstruct the dense token field. A different
scale, halo/window origin or preprocessing recipe defines a different feature
set even if the model hash agrees. Do not pool incompatible feature definitions
across tissues. Labels, training membership, fitted scaling and RGB recipe scope
belong in separate state/artifacts, never in the predictor matrix.

## CPU-only synthetic integration demo

```bash
python examples/create_cached_experiment_demo.py /path/to/new-demo-folder
```

This creates three 448×448 synthetic PNG images, two-class annotation GeoJSON,
both full-width feature formats and a ready experiment. Local observation and
ROI IDs intentionally repeat between tissues. `labels.csv` records training
tissues 1/2 and query tissue 3 separately from the predictors. Every provenance
record marks the fixture as synthetic; the encoder/checkpoint identifiers are
all-zero sentinels. The analytic feature values are **not Virchow2 inference**
and cannot support biological accuracy or model performance claims. The
generator refuses to overwrite an existing destination.

## Python API

Use Python 3 with Pillow for PNG/JPEG/WebP headers and tifffile for TIFF headers;
both header libraries are already listed in `requirements-dev.txt`. No model
libraries are required for export or validation. The CPU tests use the existing
pytest, NumPy and tifffile test dependencies.

Add the repository's `bin` folder to the Python module path, then use
`export_experiment(spec, output, base_dir=..., overwrite=False)`,
`export_runs(run_roots, output, name=..., overwrite=False)` or
`validate_experiment(path)` from `export_experiment`. Validation streams feature
rows and image headers; it does not load full WSI pixels or use GPU libraries.
Validation errors raise `ExperimentError`. Writes use a temporary file, flush
and atomic rename so failed export cannot replace a valid previous index.
