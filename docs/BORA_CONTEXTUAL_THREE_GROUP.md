# Standard contextual Bora continuation for three KODAMA groups

`bora_contextual_fullslide.nf` is the explicit CellPhenotyper continuation for
the registered **Virchow2 → KODAMA (three direct groups) → contextual Bora**
route. It starts after a complete 0.5 µm/pixel Virchow2 spatial-token inventory
has been projected to one 32-channel PCA lattice on the same image. It does
not silently use the earlier `main.nf` tissue route. That main workflow has
not yet been upgraded to produce the complete three-scale spatial-token
inventory or this KODAMA binding, so it remains unchanged.

Use Bora `main` at commit `fa3e3d41f7b61500a442d9c0c52f3f86b739ff9d`
or a later compatible revision. This continuation calls its public registered
three-group fit, tiled score, native refinement, compact label assembly, and
guarded contour APIs.

The current standard is a method, not PM016's fitted weights. Every new slide
must fit its own three-group numeric head from its KODAMA-labeled tile
representatives or supply a head receipt pinned to the *same image* and PCA
lattice. PM016's fitted coefficients, 2.9371-token radius and 0.7806 prior
exponent are not defaults and are not included in Git. KODAMA group IDs are
anonymous domains; the output does not claim tumor/stroma identity.

## Frozen input contract

Start with [the manifest template](../params/bora_contextual_fullslide.example.json).
Every file is an ordinary absolute path paired with its SHA-256. The manifest
identifies one CellPhenotyper base project, the exact native RGB TIFF and frame,
native and token tissue-support arrays, the complete float32 `[Y,X,32]` PCA
lattice, the frozen PCA projection file, and the direct three-group KODAMA binding. The working grid must be
0.5 µm/pixel, with a spatial token every 14 working pixels; the native slide
MPP and physical extent must agree with it. The binding must pin the native
image, frame, PCA field, token support, working shape and three group IDs.
It also verifies and binds the frozen PCA projection file's SHA-256; an unrelated projection cannot
reuse the fitted model even if its array dimensions match.

The recommended `head_mode` is `fit_from_kodama_tiles`. Its bound UTF-8 CSV
has the exact columns `tile_id,native_x_px,native_y_px,kodama_group`, one
registered representative point per tile, group IDs exactly 1, 2 and 3, and
at least ten tiles per group. The KODAMA binding must record the CSV SHA-256.
The action samples the **contextualized** PCA lattice at those points, fits
Bora's balanced three-class logistic head, and saves both
`model/numeric_head.npz` and Bora's `model/FIT_COMPLETE.json` in this slide's
result folder.
The declared training prior must equal the three CSV group fractions.

For a deterministic replay, `head_mode: pinned_same_slide_head` replaces the
tile CSV with `numeric_head.npz` and Bora's `FIT_COMPLETE.json` from one
existing model folder. The receipt must bind this native image and frame,
PCA projection, KODAMA binding and model hash. A head from PM016 cannot be
passed for another image.

The context radius, prior correction, regularization, core size and halo are
explicit in the manifest. The recovered native H&E boundary-refinement
specification is versioned and checked exactly. The source tree of Bora is
hashed (every Python module and `pyproject.toml`) and must match the launch
pin. Install the matching Bora source and CPU dependencies in the actual
runtime; use no Docker/Singularity profile for this entrypoint yet.

## Run one slide

```bash
python3 bin/bora_contextual_fullslide.py identity \
  --bora-source /absolute/reviewed/Bora

nextflow run bora_contextual_fullslide.nf \
  --bora_contextual_manifest /absolute/slide/bora-contextual.json \
  --bora_contextual_source /absolute/reviewed/Bora \
  --bora_contextual_source_sha256 HASH_FROM_IDENTITY \
  --bora_contextual_output /absolute/new/CellPhenotyper_slide_result
```

This uses two CPUs and a local CPU Python environment. Its default application
deadline is 14,400 seconds and its declared disk budget is 32 GiB. Set
`--bora_contextual_deadline_seconds` and
`--bora_contextual_disk_budget_bytes` explicitly for an admitted larger slide.
The caller must still enforce an outer bounded lifetime and terminal cleanup
when running on a remote host. UCT GPU work remains paused by coordinator
policy. The entrypoint never starts a GPU job or retries a failed result.
Use a fresh output folder for each corrected run; failed folders retain
`RUN_FAILED.json` and their completed evidence.

## Products and acceptance

The action scores one complete, pooled probability lattice across encoder
patch seams. Every native core is projected at **absolute image pixel
centres**, refined against native H&E with finite context, and written as
compact uint8 labels with overlapping halos. Bora's global assembler requires exact
adjacent-halo agreement and one complete source-frame mask. It then creates
the authoritative cleaned mask and global exact GeoJSON, removing parts and
holes strictly below **1,000 µm²**. Finally Bora chooses a guarded whole
boundary simplification that reduces stair-step segments while preserving
component identities and passing class-IoU, displacement and separate
lost/gained-support checks. No macroregion polygons are independently fitted.

The result folder contains `token_scores/three_group_probability.npy`,
`model/FIT_COMPLETE.json` when fitted there, `native_shards/`,
`assembly/`, `contours/global_vectors/exact/`, and
`contours/tissue_annotations/BORA_TISSUE_ANNOTATIONS.geojson`. The new
`experiment.cellphenotyper.json` imports the latter as an ordinary editable
`tissue_annotations` layer, with `predicted_not_expert_annotation: true` and
`training_eligible: false`. Existing base-project assets are referenced by
rebased relative paths, not copied. This includes nested indexed CellViT
classification, SQLite index, feature-manifest and registered-tissue references.
It also rebases hashed provenance references at project root. The exporter
checks the hashes of every referenced file and the declared cell
population before publishing the combined project; the cell source and feature
vectors stay in their original location. The base project's full-resolution
H&E must have the same bytes, frame, dimensions and MPP as Bora's native H&E;
the exporter will not merge a partial crop or guess a registration. wsiTools
performs its own indexed-cell checks when **Open Project** loads the result.
`RUN_COMPLETE.json` is written only after the project validates. Keep the
original project and exact mask for audit.
Open the project from the canonical `--bora_contextual_output` folder. The
Nextflow task work directory contains only a completion receipt; it does not
publish a detached copy of the project with broken relative asset paths.

The PM016 case that motivated this method produced 49 components and 83.1%
diagonal segments. Those counts are **not** universal acceptance thresholds.
The small integration test uses a synthetic registered image; it tests
identity, same-slide fitting, native assembly, guarded geometry and project
import. It does not establish biological accuracy on another specimen.

## Combine a previously accepted Bora result with complete CellViT

When the accepted Bora GeoJSON already exists, the CPU-only
`bin/combine_cellvit_bora_project.py` action can attach those exact bytes to a
new, complete indexed CellViT project without rerunning either model. It
requires the CellViT full-slide receipt, its partition manifest and SQLite
index, all eight available heads, retained 1,280-dimensional features, and
equal planned/completed shard counts. For PM016, pass `--expected-shards 104`;
the known partial 2/104 result is rejected. The accepted Bora receipt and
GeoJSON must agree by SHA-256 and bind the exact CellViT H&E image bytes, frame,
dimensions and MPP. The accepted GeoJSON is copied into the output folder as
one ordinary editable `tissue_annotations` layer, with anonymous group names
and `training_eligible: false`. The small CellViT completion receipt and Bora
receipt are copied into the same new output folder. Existing images, cell
indexes and feature vectors stay referenced by verified relative paths. The
exporter builds and validates a sibling staging folder, then publishes the
whole `combined` folder by one rename; a failed check leaves no final project.

```bash
python bin/combine_cellvit_bora_project.py \
  --cell-project /path/to/full-cellvit/experiment.cellphenotyper.json \
  --cell-receipt /path/to/full-cellvit/CELLVIT_PROJECT_COMPLETE.json \
  --bora-geojson /path/to/accepted/BORA_TISSUE_ANNOTATIONS.geojson \
  --bora-receipt /path/to/accepted/OUTER_GUARDED_FINAL_COMPLETE.json \
  --expected-bora-sha256 94d261e3bd68185ea9b01e283b9b297f8d4ce8964efa67a404848990da49d4e8 \
  --expected-shards 104 \
  --output /path/to/full-cellvit/combined/experiment.cellphenotyper.json
```

Do not create the `combined` folder first; the exporter creates it only after
validation succeeds. Open that project's
`experiment.cellphenotyper.json` in wsiTools after CellViT has completed all
104 shards and the combined project has passed wsiTools' own indexed-cell
reader. No viewer is launched for an incomplete population.

## Predict tissue names with the frozen Neighbor-context PLS model

Boundary refinement and tissue identity are separate steps. A same-slide
three-group numeric head predicts KODAMA domains; its output names remain
anonymous until the frozen pretrained tissue classifier is actually run.
Use Bora commit `6c9d7b05f2cd5d52a7694490fdafb074aa7eb81b` or later for
`classify-regions`. This CPU action pools all **1,280 Virchow2 spatial-token
channels** over each accepted source group and its physical boundary-weighted
neighbors, then applies the frozen Neighbor-context fastPLS model. The
32-channel PCA lattice used for boundary refinement cannot replace those
inputs. Supply the exact trained model separately; no tissue-model weights,
slide data or PM016 coefficients are bundled here.

```bash
bora classify-regions /path/to/accepted/BORA_TISSUE_ANNOTATIONS.geojson \
  /path/to/native-region-registration.json \
  --tokens /path/to/registered-full-1280-channel-tokens.npy \
  --model /path/to/frozen-neighbor-context-model.npz \
  --output /path/to/new/tissue-identity

python bin/export_bora_region_identity_project.py \
  --base-project /path/to/combined/experiment.cellphenotyper.json \
  --identity-dir /path/to/new/tissue-identity \
  --output /path/to/new/named-project/experiment.cellphenotyper.json
```

The registration JSON uses `format: bora-region-registration-v1`,
`source_frame_id`, `native_shape_yx`, `working_shape_yx`, `native_mpp_xy`,
`pixel_aligned: true`, `registration_method: native_same_grid`,
`token_stride: 14`, and exact lowercase SHA-256 values under
`image_sha256`, `contours_sha256` and `tokens_sha256`. It is an explicit
assertion of native-pixel/working-grid registration, not an estimated alignment.
The scorer checks those bytes and rejects incompatible 32-channel inputs.
Its stripe memory limit bounds descriptor accumulation without loading the
whole native slide into RAM; it does not change polygon geometry.

The result directory contains `tissue-annotations.geojson`, `descriptors.npz`
and `report.json`, binding the source contours, tokens, model and registration.
One score is computed per original source group, including its disconnected
components. Every feature preserves `source_group_id` and `source_group_name`
while `classification.name` receives the predicted tissue type. Geometry,
feature order, holes and smooth diagonal boundaries remain unchanged. PLS
class-indicator responses are not calibrated probabilities. The transfer
remains biologically unvalidated until compared with expert labels on the
new specimen; execution success alone does not establish accuracy.

The CellPhenotyper exporter verifies the scorer's report/output hashes,
compares every geometry with the exact accepted source annotation, checks
image/frame/dimensions/MPP and preserves all indexed cells and feature paths.
It creates a fresh project folder atomically and imports the named regions
as ordinary editable tissue annotations. The original anonymous-group project
and its receipts remain intact. Open this **named** project's JSON in wsiTools
to see predicted tissue names alongside the cell classifications.
