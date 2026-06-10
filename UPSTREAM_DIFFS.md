# Pipeline Step Differences vs Upstream Tools

This note summarizes where CellPhenotyper follows an upstream tool directly and where it deliberately changes behavior.

The goal is to make the GitHub repository explicit about:

- which pipeline stages are custom CellPhenotyper orchestration
- which stages wrap third-party tools
- what was modified relative to the original tool behavior

## Scope

This document reflects the current repository behavior, not just the defaults from the original third-party projects.

Upstream tools referenced here include:

- GrandQC
- StarDist
- GigaTIME
- UNI-2
- KODAMA
- MedSAM

## Step-by-step summary

| Stage | Pipeline step | Upstream/base tool | Main differences in CellPhenotyper |
| --- | --- | --- | --- |
| `01_input` | Input conversion | CellPhenotyper custom | Converts source images to pipeline-ready OME-TIFF, supports region-specific handling for CZI inputs, and standardizes downstream coordinates and sample naming. |
| `01a_grandqc` | Artifact QC | GrandQC | Uses official GrandQC checkpoints, but wraps OME-TIFF reading, auto-selects artifact model magnification from image MPP, adapts tissue thumbnail size for small high-resolution fields, supports overlap-based artifact inference, suppresses low-confidence artifact calls, and applies a small-FOV foreign-object refinement step for zoomed-in images. |
| `02_stardist` | Nuclei segmentation | StarDist | Uses StarDist segmentation with pipeline-specific ROI cropping, auto tissue ROI preparation, large-WSI blockwise execution, adaptive memory-aware fallback, chunked full-label export (`zarr`) instead of dense TIFF by default, and reduced-memory QC preview generation. |
| `03_gigatime` | Virtual mIHC prediction | GigaTIME | Adds blockwise slide execution, selectable persisted output channels, direct pyramidal OME-TIFF export with JPEG compression, support for native-resolution execution on ROI crops, and output modes that avoid saving intermediate tile trees or zarr stores when not needed. |
| `04_tissue_mask` | Tissue mask build | CellPhenotyper custom | This is a CellPhenotyper stage, not a direct third-party wrapper. It builds the downstream tissue mask used by later assignment, growth, and refinement stages. |
| `05_roi` | ROI preparation | CellPhenotyper custom | Accepts provided ROI GeoJSON, supports auto-generated ROI when no manual ROI exists, and keeps crop/original coordinate transforms explicit for registration. |
| `06_roi_mask` | ROI rasterization | CellPhenotyper custom | Rasterizes ROI polygons into crop-aligned masks and preserves class labels when present in annotation metadata. |
| `06_marker_quantification` | GigaTIME marker quantification | GigaTIME + CellPhenotyper custom quantification | Quantification is performed during tile generation rather than after reconstructing the full image, supports both nuclei and cytoplasm masks, and handles overlapping GigaTIME tiles during accumulation. |
| `07_cell_assignments` | Object-to-ROI assignment | CellPhenotyper custom | Assigns segmented objects to ROI/tissue space for downstream embedding and clustering. |
| `08_cytoplasm` | Cytoplasm expansion | CellPhenotyper custom | Expands nuclei labels into cytoplasm labels for paired nuclei/cytoplasm quantification and downstream per-cell views. |
| `10_embeddings` | Morphology embeddings | UNI-2 | Uses the UNI-2 encoder but changes execution topology: defaults were narrowed to `tile` + `inner_square`, supports a shared extraction path that emits both outputs from one embedding task, and uses a shared Hugging Face cache to avoid per-run model duplication. |
| `11_kodama` | Latent manifold analysis | KODAMA | KODAMA input loading is selective: CellPhenotyper now supports running on only the requested embedding families, and the current default is `tile,inner_square` rather than all available embedding views. |
| `13_clustering` | Cluster assignment | CellPhenotyper custom + R clustering code | Produces both `standard` and `fine` clustering variants rather than a single downstream solution. |
| `15_cluster_mask` | Cluster mask generation | CellPhenotyper custom | Propagates both clustering variants into mask outputs. |
| `16_grown_tissue` | Cluster-guided tissue growth | CellPhenotyper custom | Grows cluster masks into tissue space, with both `standard` and `fine` variants propagated independently. |
| `17_medsam_refined_tissue` | Tissue refinement | MedSAM + CellPhenotyper orchestration | Uses MedSAM as a refinement backend, but runs it on CellPhenotyper-generated masks and carries both clustering variants through to refined tissue outputs. |
| `18_cluster_geojson` | Final polygon export | CellPhenotyper custom | Converts the final refined masks to GeoJSON, preserving the pipeline’s variant structure and sample-specific naming. |
| `00_execution` | Execution report | CellPhenotyper custom | Adds a project-level output manifest, deterministic `output_id` values for every output, and per-process runtime summaries from the Nextflow trace. |

## Detailed notes by wrapped tool

### GrandQC

Files:

- `bin/run_grandqc_artifact_analysis.py`
- `modules/run_grandqc_artifact_analysis.nf`
- `nextflow.config`

Changes relative to upstream GrandQC:

1. Uses CellPhenotyper OME-TIFF reading instead of the original repository scripts directly.
2. Auto-selects the artifact model (`1.0`, `1.5`, `2.0`) from image MPP instead of requiring manual shell-script edits.
3. Uses adaptive tissue thumbnail sizing so small high-magnification fields do not collapse during tissue detection.
4. Includes a heuristic fallback tissue mask if the learned tissue detector returns near-zero tissue on a clearly nonblank image.
5. Uses overlap-based artifact inference with smooth blending for small-FOV cases instead of strict hard tile stitching.
6. Suppresses low-confidence artifact calls after score blending.
7. Applies a small-FOV foreign-object refinement stage that collapses broad false-positive artifact fields onto the dominant dark foreign-object structure when appropriate.
8. Bundles model/cache handling and preview/summary generation into the pipeline stage.

### StarDist

Files:

- `bin/run_stardist_roi_segmentation.py`
- `modules/run_stardist_roi_segmentation.nf`

Changes relative to standard StarDist usage:

1. Runs on a crop/ROI path prepared by the pipeline.
2. Supports a large-image blockwise execution path automatically.
3. Uses memory-aware fallback and smaller blocks for difficult WSIs.
4. Exports full labels as chunked `zarr` by default for large images.
5. Reduces preview-memory pressure for large WSI QC artifacts.

### GigaTIME

Files:

- `bin/run_gigatime_on_crop.py`
- `bin/export_gigatime_store_to_ometiff.py`
- `bin/quantify_gigatime_intensity.py`
- `modules/run_gigatime_on_crop.nf`
- `modules/quantify_gigatime_intensity.nf`

Changes relative to straightforward GigaTIME inference:

1. Blockwise ROI/full-slide execution.
2. Native-resolution enforcement on validated paths instead of coarse fallback.
3. Selectable persisted marker subsets.
4. Direct pyramidal OME-TIFF export with JPEG compression.
5. Optional no-tile/no-zarr final-output modes.
6. Single-cell quantification performed during tile generation.
7. Quantification over both nuclei and cytoplasm masks.
8. Overlap-aware accumulation during tile-time quantification.

### UNI-2

Files:

- `bin/extract_uni2_embeddings.py`
- `modules/extract_uni2_embeddings.nf`
- `modules/extract_uni2_embeddings_shared.nf`

Changes relative to naïve UNI-2 usage:

1. Tiles are cell-centered on StarDist labels rather than generic image tiles.
2. Default embedding families were reduced to `tile` and `inner_square`.
3. A shared extraction path can emit both `tile` and `inner_square` outputs from one stage.
4. The pipeline uses a shared Hugging Face cache path to avoid duplicate model downloads across runs/repos.

### KODAMA

Files:

- `bin/load_kodama_rawdata.R`
- `bin/run_kodama_analysis.R`
- `modules/run_kodama_analysis.nf`

Changes relative to the previous pipeline behavior:

1. KODAMA no longer assumes all embedding families must exist.
2. It now loads only the selected embedding families.
3. The current default is `tile,inner_square`.
4. Placeholder handling was added so missing unused embedding families do not block execution.
5. For large WSI runs, KODAMA uses a memory-bounded landmark/projection path after PCA rather than constructing the full all-cell KODAMA network in RAM.
6. The runtime image must include the R nearest-neighbor/projection packages `BiocNeighbors`, `RANN`, `RcppHNSW`, `RcppAnnoy`, and `uwot`; these are installed into the micromamba R library, not a user-home R library, because Singularity tasks run with `--no-home`.
7. The pipeline explicitly sets KODAMA's internal PLS component count with `kodama_ncomp = 2`. This is separate from `kodama_dims_to_run = 20`: KODAMA still receives 20 PCA dimensions, but the native stochastic PLS optimization avoids transient high-component class states that can emit `Mat::col(): index out of bounds` on Linux.

### MedSAM

Files:

- pipeline orchestration in `main.nf` and corresponding modules

Changes relative to direct standalone MedSAM use:

1. MedSAM is used as a downstream refinement stage, not as a first-pass segmenter.
2. It refines masks produced by the CellPhenotyper cluster-growth stages.
3. Both `standard` and `fine` cluster branches are carried through independently.

## Steps with no external upstream counterpart

These are primarily CellPhenotyper-specific orchestration or geometry stages:

- `04_tissue_mask`
- `05_roi`
- `06_roi_mask`
- `07_cell_assignments`
- `08_cytoplasm`
- `13_clustering`
- `15_cluster_mask`
- `16_grown_tissue`
- `18_cluster_geojson`
- `00_execution`

For these, the relevant comparison is not “tool vs upstream,” but “current pipeline behavior vs earlier internal pipeline behavior.”

## Related documentation

- `README.md`
- `PARAMETERS.md`
- `OUTPUT.md`
