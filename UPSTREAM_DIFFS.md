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
- HoVer-Net MoNuSAC
- CellViT++
- TITAN / CONCH v1.5
- PathoFMPred

## Step-by-step summary

| Stage | Pipeline step | Upstream/base tool | Main differences in CellPhenotyper |
| --- | --- | --- | --- |
| `01_input` | Input conversion | CellPhenotyper custom | Validates native MPP and anisotropy before conversion, converts source images to pipeline-ready OME-TIFF, verifies post-conversion MPP preservation, supports region-specific handling for CZI inputs, and standardizes downstream coordinates and sample naming. |
| `02_grandqc` | Artifact QC | GrandQC | Uses official GrandQC checkpoints, but wraps OME-TIFF reading, auto-selects artifact model magnification from image MPP, adapts tissue thumbnail size for small high-resolution fields, uses hardware-aware fully convolutional artifact context with smooth overlap blending, and keeps the official tissue-detector patch geometry separate. |
| `03_stardist` | Primary nuclei segmentation | StarDist | Uses StarDist segmentation with pipeline-specific ROI cropping, auto tissue ROI preparation, large-WSI blockwise execution, adaptive memory-aware fallback, chunked full-label export (`zarr`) instead of dense TIFF by default, and reduced-memory QC preview generation. |
| `03b_hovernet_monusac` | Typed nuclei detection | HoVer-Net MoNuSAC | Runs the official fast MoNuSAC model, normalizes its numeric classes to explicit names, and converts detections into the shared StarDist crop coordinate frame. |
| `03c_cellvitpp` | Typed nuclei detection | CellViT++ | Runs the official inference package with explicit MPP and normalizes numeric classes to named `neoplastic`, `inflammatory`, `connective`, `dead`, and `epithelial` labels. |
| `03d_cell_consensus` | Consensus cell identification | CellPhenotyper custom | Performs one-to-one physical-distance matching across StarDist, HoVer-Net, and CellViT++, requires configurable multi-model support, retains named detector classes, and emits canonical cells, geometry, provenance, and QC. |
| `04_TMA` | TMA detection and spot assignment | CellPhenotyper custom | Detects whether the image is a tissue microarray; for TMA images, exports spot GeoJSON and associates detected cells with spot IDs. |
| `04_tissue_mask` | Tissue mask build | CellPhenotyper custom | This is a CellPhenotyper stage, not a direct third-party wrapper. It builds the downstream tissue mask used by later assignment, growth, and refinement stages. |
| `05_gigatime` | Virtual mIHC and marker quantification | GigaTIME + CellPhenotyper custom quantification | Adds exact MPP-aware blockwise inference, selectable persisted channels, all-marker integrated single-cell quantification, and direct multichannel pyramidal OME-TIFF export. |
| `06_roi` | ROI preparation and rasterization | CellPhenotyper custom | Accepts provided or automatically generated ROI GeoJSON, keeps coordinate transforms explicit, and rasterizes polygons into crop-aligned labeled masks. |
| `07_cell_assignments` | Object-to-ROI assignment | CellPhenotyper custom | Assigns segmented objects to ROI/tissue space for downstream embedding and clustering. |
| `08_cytoplasm` | Cytoplasm expansion | CellPhenotyper custom | Expands nuclei labels into cytoplasm labels for paired nuclei/cytoplasm quantification and downstream per-cell views. |
| `09_embeddings` | Morphology embeddings | UNI-2 | Uses cell-centered `tile` and fixed 90-pixel `inner_square` views and a shared token-subset forward pass, with checkpointed buckets and a shared Hugging Face cache. |
| `10_kodama` | Latent manifold analysis | KODAMA | Loads only selected embedding families, uses 20 PCA components and 1,000 KODAMA landmarks by default, and frees raw features after PCA to bound memory. |
| `11_clustering` | Cluster assignment | CellPhenotyper custom + R clustering code | Uses inverse-neighbor-distance landmark sampling (`p=2`), Leiden clustering on landmarks, and nearest-neighbor assignment of remaining cells. |
| `12_cluster_mask` | Cluster mask generation | CellPhenotyper custom | Converts per-cell cluster labels into a spatial mask. |
| `13_grown_tissue` | Cluster-guided tissue growth | CellPhenotyper custom | Grows cell-supported clusters into tissue space while retaining class identity. |
| `14_medsam_refine_tissue` | Tissue refinement | MedSAM + CellPhenotyper orchestration | Runs GPU MedSAM only in an editable border band around grown clusters and publishes full-resolution random-region QC plus raw-versus-final panels. |
| `15_cluster_geojson` | Final polygon export | CellPhenotyper custom | Converts the final refined section masks to GeoJSON with sample and class metadata. |
| `16_neoplastic_section` | Neoplastic-enriched section selection | CellPhenotyper custom | Splits final tissue polygons into connected sections, counts named neoplastic consensus cells, and deterministically exports the section with the largest count. |
| `17_titan` | Section representation | TITAN / CONCH v1.5 | Applies the official TITAN slide aggregator to MPP-correct CONCH patch features from the selected section and emits a validated 768-feature vector with provenance. |
| `18_pathofmpred` | Research endpoint prediction | PathoFMPred | Applies an explicit cancer-specific private registry to the TITAN vector and publishes predictions, QC plots, and a research report without clinical-calibration claims. |
| `00_execution` | Execution report | CellPhenotyper custom | Adds a project-level output manifest, stable unique `output_id` values, preserved full-run and targeted traces, and per-process runtime/memory summaries. |

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
5. Separates the official 512-pixel tissue-detector patches from hardware-aware artifact context tiles; CUDA devices with at least 8 GiB VRAM use 1024-pixel fully convolutional context to suppress checkpoint padding-position bands without changing physical MPP.
6. Uses overlap-based artifact inference with smooth probability blending instead of strict hard tile stitching.
7. Suppresses low-confidence artifact calls after score blending.
8. Applies a small-FOV foreign-object refinement stage that collapses broad false-positive artifact fields onto the dominant dark foreign-object structure when appropriate.
9. Bundles model/cache handling and preview/summary generation into the pipeline stage.

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
2. Exact MPP-aware floating-point resampling, including lazy pyvips upsampling when the source is coarser than the model target, instead of integer scale rounding or coarse fallback.
3. Selectable persisted marker subsets.
4. Direct pyramidal OME-TIFF export with JPEG compression.
5. Optional no-tile/no-zarr final-output modes.
6. Single-cell quantification performed during tile generation.
7. Quantification over both nuclei and cytoplasm masks.
8. Overlap-aware accumulation during tile-time quantification.
9. GPU-first automatic batch sizing from live free VRAM, with host-RAM-bounded block buffers and in-process CUDA OOM batch reduction.
10. Coarse background-block skipping that is vetoed whenever a nucleus or cytoplasm label is present, preserving complete single-cell quantification.

### UNI-2

Files:

- `bin/extract_uni2_embeddings.py`
- `modules/extract_uni2_embeddings.nf`
- `modules/extract_uni2_embeddings_shared.nf`

Changes relative to naïve UNI-2 usage:

1. Tiles are cell-centered on canonical cell labels rather than generic image tiles.
2. Default embedding families were reduced to `tile` and `inner_square`.
3. Rounded extraction coordinates and grid assignment use the same coordinates, eliminating boundary-cell omissions.
4. Per-grid and whole-stage manifests enforce exact one-row-per-mask-label coverage and reject incomplete resume shards.
5. A shared extraction path can emit both `tile` and `inner_square` outputs from one stage.
6. The pipeline uses a shared Hugging Face cache path to avoid duplicate model downloads across runs/repos.

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
6. The runtime image includes the R nearest-neighbor/projection packages `BiocNeighbors`, `RANN`, `RcppHNSW`, `RcppAnnoy`, and `uwot` in micromamba libraries. Each R task explicitly disables host `.Renviron`/`.Rprofile` files and selects the corresponding bundled library so Singularity cannot load ABI-incompatible packages from `~/R`.
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
- `03d_cell_consensus`
- `04_TMA`
- `06_roi`
- `07_cell_assignments`
- `08_cytoplasm`
- `11_clustering`
- `12_cluster_mask`
- `13_grown_tissue`
- `15_cluster_geojson`
- `16_neoplastic_section`
- `00_execution`

## Neoplastic-section selection, TITAN, and PathoFMPred

- **Section selection:** Custom CellPhenotyper code splits the final Polygon/MultiPolygon output into connected sections and counts consensus cells by exact point-in-polygon membership. Neoplastic status comes from the normalized named CellViT++ class, not a hard-coded unexplained integer. Selection and tie-breaking are deterministic, and the full-resolution crop retains an explicit coordinate transform.
- **TITAN:** CellPhenotyper does not reimplement the slide encoder. It loads the official gated `MahmoodLab/TITAN` code with `trust_remote_code=True`, obtains the official CONCH v1.5 image encoder through `return_conch()`, and calls `encode_slide_from_patch_features()`. Pipeline additions are MPP-aware WSI patch extraction, section masking, adaptive GPU batching, HDF5 provenance, strict 768-feature validation, and restartable external model caching.
- **PathoFMPred:** The prediction model remains in the separate protected PathoFMPred package. CellPhenotyper validates the exact named TITAN feature contract, supplies an explicit cancer code, runs with R 4.6 and the package's pinned `fastPLS` dependency, and publishes predictions/QC reports. The pipeline adds no clinical-calibration claim and labels these outputs as TCGA-derived research estimates.

For these, the relevant comparison is not “tool vs upstream,” but “current pipeline behavior vs earlier internal pipeline behavior.”

## Related documentation

- `README.md`
- `PARAMETERS.md`
- `OUTPUT.md`
# Multi-Model Cell Consensus

- **HoVer-Net:** The official PyTorch inference code and official fast MoNuSAC checkpoint are used for inference. CellPhenotyper adds MPP-aware input normalization, an Aperio-compatible pyramidal TIFF carrying `AppMag` and `MPP` because upstream requires objective-power metadata, the explicit MoNuSAC type map because upstream parses its empty default as a filename, cache cleanup, support for upstream's `nuc` JSON key, conversion of coordinates back into the StarDist crop frame, and runtime aliases for NumPy scalar names removed after the versions pinned by upstream. Post-processing workers are bounded by allocated RAM. The unmodified upstream repository is copied into a task-local writable runtime because upstream creates `debug.log` in its current directory and a Singularity image is immutable. The build recipes make the bundled checkpoint world-readable and the wrapper verifies readability before expensive WSI normalization. For recovery after a post-processing-only failure, the isolated copy of upstream `wsi.py` can be patched to reopen a validated completed `pred_map.npy` and skip raw inference; the installed upstream source and model are not modified.
- **CellViT++:** The official `cellvit-inference` package is used unchanged and pinned to `cellvit==1.0.9`. CellPhenotyper adds pyramidal crop preparation, passes the source MPP explicitly, clamps batch size to the upstream-supported range `2..48`, supplies a nonzero Ray worker count to avoid the upstream zero-worker modulo failure, and normalizes output discovery into a stable `cellvit_cells.json` artifact. Before creating the WSI pyramid, the wrapper verifies that the selected bundled checkpoint is readable. Ray temporary files, Matplotlib state, and runtime caches are redirected into task-local scratch so large runs do not fill the host root filesystem or write into the immutable image.
- **Consensus:** This is CellPhenotyper-specific code. It performs distance-gated, one-to-one matching between methods; prevents a consensus component from containing duplicate predictions from one method; retains cells supported by at least two methods; assigns a canonical sequential ID; records detector provenance; and writes a tiled label TIFF without holding three WSI masks in RAM. The writer reserves a unique centroid-near seed pixel for every canonical ID so overlapping contours cannot erase a cell, then streams a full label-coverage validation before publication.
