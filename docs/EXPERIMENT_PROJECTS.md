# Open a CellPhenotyper experiment in wsiTools

An experiment is one JSON index for one or several tissues. It references the
calibrated images, annotations and existing feature tables; it does not embed
images, weights or feature arrays. Exporting, validating and opening the index
do not run an encoder or submit a workflow.

This exporter and report hook do not add a Virchow2 inference route to the
Nextflow pipeline in this revision. Its existing UNI-2 outputs cannot be
relabeled as Virchow2. Use `--spec` for independently generated compatible
features, or `--run` for an existing output tree matching the completion-record
contract below. The Virchow2 producer integration is separate work.

For a run that already contains compatible Virchow2 caches, execution reporting creates
`00_execution/experiment.cellphenotyper.json` when its image, grid and cached
features pass validation. `00_execution/final_report.md` links the project file. Choose
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
The supported completion-record layout does not retain a source-image digest.
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

The automatic-discovery contract requires model tiles covering 224 pixels at
0.5 µm/pixel, with no overlap. Spatial tokens occupy a 14-model-pixel lattice. Pooled tiles are not
cell observations and cannot reconstruct the dense token field. A different
scale, halo/window origin or preprocessing recipe defines a different feature
set even if the model hash agrees. Do not pool incompatible feature definitions
across tissues. Labels, training membership, fitted scaling and RGB recipe scope
belong in separate state/artifacts, never in the predictor matrix.

## Keep a spatial transcriptomics object in the project

In a compatible wsiTools version, choose **Add to project** after opening a
trusted local Seurat or compatible spatial RDS file. Select its tissue,
metadata columns and reductions, and confirm the same-section coordinate
mapping against the exact H&E image. Saving stores a byte-identical RDS copy
and a compact JSON review bundle with the project, then adds references to the
experiment index. The original source file remains untouched. Keep the stored
RDS, image, bundle and index together. Paths in the index are relative, so moving the
containing tree preserves the project. Assets outside the tree remain external
dependencies.

Reopening **Open Project** reconstructs the bounded coordinates, categorical
metadata and reductions from the JSON bundle. It does not automatically
deserialize the RDS, run CellPhenotyper, train a model or start Bora. The RDS
remains the intact original object for an explicit trusted-file action. Python
validation checks its bytes and SHA-256 only; it cannot certify that its object
class, expression data or biological annotations are valid. Spots and cells
keep their own observation units, separate from Virchow2 tiles and tokens.

Before replacing an existing index, wsiTools keeps a local backup. New attachment
assets must use immutable/versioned filenames and be completely written and
checked before publishing the index atomically. A failed save retains the old
index and its referenced assets. Do not edit a bundle already referenced by an
older index in place. The Python exporter replaces only the index and never
copies, rewrites or deserializes the original RDS. Export/validation failure
leaves a previous index unchanged.

Each attached tissue/dataset uses two linked `artifacts` records:

| Role | Required metadata |
|---|---|
| `spatial_transcriptomics_object` | `format="rds"`, `artifact_id="spatial:<dataset>:object"`, `dataset_id`, `observation_unit`, `frame_id`, `image_sha256`, `row_count`, `review_bundle_artifact_id` |
| `spatial_review_bundle` | `format="json"`, `artifact_id="spatial:<dataset>:review"`, the same dataset/unit/frame/image/count fields, `source_artifact_id`, `source_object_sha256` |

Both records also carry their normal relative `path` and full-file `sha256`.
The object and review records must link to each other, and the bundle's source
hash must match the object artifact. Dataset IDs use letters, digits, dots,
colons, underscores and hyphens, beginning with a letter or digit. A tissue
cannot attach the same dataset twice or reuse its ID for a `review_observations`
CSV. Other existing artifact roles remain compatible.

The JSON bundle declares
`schema="cellphenotyper-spatial-review-bundle", schema_version=1`, the
experiment/tissue/dataset/unit/frame identities, `image_sha256`,
`source_artifact_id`, `source_object_sha256`, `coordinate_mapping`,
`observations` and `reductions`. All file locations belong in the index;
internal paths are forbidden in the bundle. The bundle is limited to 16 MiB
and 10000 observations. Large objects need an explicitly bounded metadata
export; validation never samples rows silently.

The mapping must explicitly declare `confirmed=true`, `same_section=true`,
the exact image hash/frame, a coordinate space (`native_pixels`,
`level0_slide_pixels` or `source_pixels`) and an invertible finite 3×3 affine
with final row `[0,0,1]`. Only `source_pixels` may declare a nonidentity affine.
Bundle observation X/Y values are already normalized to native centres in the
target image. The mapping records how they were obtained; consumers must not
apply it a second time. No crop offset, pixel scale, axis swap or cross-section
registration is guessed.

Every observation has canonical `id`, separate `original_id`, dataset/tissue/
frame/unit fields, finite in-image `x` and `y`, and categorical `cluster_values`.
Optional footprint dimensions or native edge rectangles retain their original
extent, including portions outside the image. Each reduction has a canonical
`id`, dataset/frame/unit metadata and ID-keyed `points` with finite embedding
X/Y values. Its ID set must exactly match the dataset observations; shuffled
rows are valid. Canonical IDs follow wsiTools' existing URL-encoded
`experiment::tissue::dataset::original` convention, so repeated local barcodes
across tissues stay distinct. Source cluster labels remain separate from
subsequent human review labels and decisions.

The current wsiTools **Add to project** workflow registers transcriptomic spots.
The artifact contract also reserves `cell` as a separate explicit unit for
compatible consumers; it does not promote spots into cell observations.

The CLI validates the pair, bundle contents and all file hashes on both export
and reopen. Missing pairs, changed images/objects, inconsistent mappings,
duplicate/unknown IDs, nonfinite values and incompatible units fail with an
explicit error before the index is published.

## Register existing GigaTIME marker images and cell scores

Existing GigaTIME outputs can be opened as virtual-marker image layers with
per-cell score information. To add them to an existing verified experiment,
first inspect the marker names and pages:

```bash
python bin/register_gigatime.py --inspect-marker-image /path/to/gigatime_probs.ome.tif
```

Then explicitly select the tissue, channels, placement and score columns.
This example confirms that the marker raster covers the entire selected H&E
image and that the CSV contains the producer's raw pixel-index centroids:

```bash
python bin/register_gigatime.py \
  --experiment /path/to/project/experiment.cellphenotyper.json \
  --tissue-id sample-a --marker-image /path/to/gigatime_probs.ome.tif \
  --channel CD3 0 --channel CD8 1 --full-image \
  --scores /path/to/sample-a_nuclei_gigatime_quantification.csv \
  --mask /path/to/labels.tif --mask-name nuclei --segmentation-unit nucleus \
  --score-column CD3 CD3__mean --score-column CD8 CD8__mean \
  --source-coordinate-convention zero-based-pixel-index \
  --output /path/to/project/with-gigatime.cellphenotyper.json
```

Choose **Open Project** in wsiTools and open the new output index. The helper
retains every tissue, transcriptomics attachment, feature set and existing
metadata. It reads calibration, image identity and frame from the verified
project, calculates hashes and the declared coordinate transform, and writes
only the new index. Original assets are referenced with rebased relative paths;
they are not copied or rewritten. Keep those assets available when moving the
index. Existing output files, directories, aliases and dangling symlinks are
refused; there is no in-place or overwrite mode. To register another independent
layer group, choose a different `--artifact-prefix` instead of replacing one.

Use `--extent X Y WIDTH HEIGHT` for a marker raster covering an explicitly known
crop in native H&E pixel edges. Use `native-image-pixel-centres` only when the
CSV already contains native H&E centres; the helper then uses identity. It never
guesses crop offsets or coordinate conventions. Floating marker pages require
an explicit display range, for example `--display-max CD3 1 --display-max CD8 1`;
this does not rescale cell scores. For marker-image-only registration, omit all
score/mask options. The default CSV columns are `label_id`, `centroid_x_px` and
`centroid_y_px`; use `--id-column`, `--x-column` and `--y-column` for other names.

An explicit `--spec` remains available for building a new project or advanced
registration below. Both routes work with `feature_sets: []`; no Virchow2
cache, encoder execution, Nextflow run or GPU is required. Automatic `--run`
discovery remains the existing Virchow2 route and does not guess legacy
GigaTIME image/table/mask pairings. Invalid input reports its tissue, artifact
and file; CSV value errors also report the line and column. Validation and
export failures do not publish an index or modify the source project.

Label these layers **GigaTIME virtual markers** and retain the declaration
`score_semantics="uncalibrated_virtual_marker_score"`. The compatibility name
`gigatime_probs.*` does not establish probability calibration. Cell tables
retain their native finite nonnegative exported values, including uint8-scale
values above one and sums above the per-pixel range. Export and reopen never
rescale these values, interpret them as measured protein abundance or infer
positive/negative thresholds. Display contrast operates separately from cell
table values.

Three artifact roles describe the existing files:

| Role | Explicit contract |
|---|---|
| `gigatime_marker_image` | `format="ome-tiff"`, `channels: [{marker,page}]`, native H&E edge `extent: {x,y,width,height}`, and fixed `score_semantics` |
| `cell_segmentation_mask` | `format="tiff"` or `"ome-tiff"`, `mask_name`, `segmentation_unit="cell"` or `"nucleus"`, and native `width,height` |
| `gigatime_cell_scores` | CSV/CSV.GZ with explicit `id_column`, `x_column`, `y_column`, `marker_columns: [{marker,column}]`, `row_count`, `mask_name`, `mask_artifact_id`, `source_artifact_id` pointing to the marker image, `source_coordinate_convention`, `source_to_image_affine`, and fixed `score_semantics` |

Every artifact has its normal stable `artifact_id`, relative `path`, SHA-256,
the tissue's `frame_id`, and the exact H&E `image_sha256`. Hashes identify the
present files. Registering existing outputs asserts their association; it
cannot retroactively prove which historical image or label mask generated the
table. The validator checks integer single-plane mask headers and representable
label IDs, without scanning every mask pixel to establish label membership or
cell lineage. A grid/tissue-cluster mask cannot be relabeled as cell segmentation.

Channel `page` values are zero-based TIFF page indices, and their marker names
must match the OME channel names exactly. Integer marker pages use their storage
range for display by default; an optional positive channel `display_max` sets
an explicit display range. Floating-point marker pages require `display_max`.
This display setting leaves the CSV untouched. Register only the score columns
whose markers are explicitly included in the linked image. Other source CSV
columns remain in the original file; no mean/sum/max column is chosen implicitly.

The current mask contract requires a single-plane integer mask on the full
native H&E grid, with the same width and height. Mask dimensions and score
`row_count` may be omitted in an input specification: export fills them from
the TIFF header and streamed CSV, rejecting supplied values that disagree.
They are mandatory in the published index. Cell IDs are unique positive decimal
label strings within a table; tissues may reuse local labels. A present CSV
`mask_name` column must match the declared linked mask on every row.

Current producer CSVs use zero-based **pixel indices** for `centroid_x_px` and
`centroid_y_px` in the GigaTIME inference raster. For a marker raster of width
`W` and height `H` placed in the H&E extent `(x0,y0,Ew,Eh)`, explicitly declare
`source_coordinate_convention="zero-based-pixel-index"` and the affine:

```text
[[Ew/W, 0, x0 + 0.5*Ew/W],
 [0, Eh/H, y0 + 0.5*Eh/H],
 [0, 0, 1]]
```

This places observation centres correctly in the native H&E frame. A table
already containing native H&E pixel centres declares
`source_coordinate_convention="native-image-pixel-centres"` and the identity
affine. The validator rejects a missing half-pixel shift, an inconsistent
scale/extent or a transformed centroid outside its linked image. Arbitrary
rotated marker placement is outside this extent-based contract. Area and
bounding-box columns retain their source-grid meaning; the XY transform does
not convert those measurements automatically.

For example, this input specification registers an existing 32×24 marker raster
and its raw centroid table against a 64×48 calibrated H&E image. Replace paths,
names, dimensions, calibration and `HE_SHA256` with verified values for your data:

```json
{
  "name": "My GigaTIME review",
  "tissues": [{
    "tissue_id": "sample-a", "name": "Sample A",
    "image": {"path": "sample-a/he.png", "mpp_x": 0.5, "mpp_y": 0.5,
              "frame_id": "sample-a:image"},
    "feature_sets": [],
    "artifacts": [
      {"artifact_id": "gigatime:markers", "role": "gigatime_marker_image",
       "format": "ome-tiff", "path": "sample-a/gigatime_probs.ome.tif",
       "frame_id": "sample-a:image", "image_sha256": "HE_SHA256",
       "score_semantics": "uncalibrated_virtual_marker_score",
       "channels": [{"marker": "CD3", "page": 0}, {"marker": "CD8", "page": 1}],
       "extent": {"x": 0, "y": 0, "width": 64, "height": 48}},
      {"artifact_id": "segmentation:nuclei", "role": "cell_segmentation_mask",
       "format": "tiff", "path": "sample-a/labels.tif", "mask_name": "nuclei",
       "segmentation_unit": "nucleus", "frame_id": "sample-a:image",
       "image_sha256": "HE_SHA256"},
      {"artifact_id": "gigatime:nuclei", "role": "gigatime_cell_scores",
       "format": "csv", "path": "sample-a/sample-a_nuclei_gigatime_quantification.csv",
       "frame_id": "sample-a:image", "image_sha256": "HE_SHA256",
       "id_column": "label_id", "x_column": "centroid_x_px", "y_column": "centroid_y_px",
       "marker_columns": [{"marker": "CD3", "column": "CD3__mean"},
                          {"marker": "CD8", "column": "CD8__mean"}],
       "mask_name": "nuclei", "mask_artifact_id": "segmentation:nuclei",
       "source_artifact_id": "gigatime:markers",
       "source_coordinate_convention": "zero-based-pixel-index",
       "source_to_image_affine": [[2,0,1],[0,2,1],[0,0,1]],
       "score_semantics": "uncalibrated_virtual_marker_score",
       "value_encoding": "native_export_values"}
    ]
  }]
}
```

Then register and check the existing assets:

```bash
python bin/export_experiment.py --spec /path/to/gigatime-spec.json \
  --output /path/to/project/experiment.cellphenotyper.json
python bin/export_experiment.py --validate /path/to/project/experiment.cellphenotyper.json
```

No source file is rewritten. Move the containing image/mask/table/index tree
together, then choose **Open Project** in a compatible wsiTools version.
Python streams full score tables for validation; the viewer applies explicit
display bounds and reports oversized tables without silently sampling them.

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
Validation errors raise `ExperimentError`. Writes use a temporary file and
flush before atomic publication. New-output publication refuses concurrent
existing files; an explicit overwrite uses atomic replacement. Failed export
cannot replace a valid previous index.

Original RDS bytes are preserved. If an RDS uses external backing files for its
assays, those files are not automatically discovered or copied. Retain them for
subsequent RNA analysis; the checked display bundle opens independently.
