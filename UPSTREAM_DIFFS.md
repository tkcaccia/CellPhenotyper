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
| `02_grandqc` | Tissue and artifact QC | GrandQC | Uses official checkpoints and 512-pixel FP32 inference, wraps OME-TIFF reading, selects the artifact checkpoint from MPP with a validated conventional-WSI override, adapts tissue thumbnails, uses invariant disk-backed probability blending for every image size, emits the mandatory clean-tissue mask, and records both Zenodo artifact identities and checkpoint SHA-256 values. |
| `03_stardist` | Shared analysis crop and primary nuclei segmentation | StarDist + CellPhenotyper coordinates | First crops to GrandQC clean tissue intersected with the optional ROI, then uses GrandQC centroid filtering, large-WSI blockwise execution, adaptive memory-aware fallback, chunked full-label export (`zarr`) and reduced-memory QC previews. The complete cached model bundle, including configuration and thresholds, is hashed when StarDist is actually used. |
| `03b_hovernet_monusac` | Typed nuclei detection | HoVer-Net MoNuSAC | Runs the official fast MoNuSAC model in parallel with the other detectors, supplies the shared GrandQC mask during WSI inference, normalizes its numeric classes to explicit names, records checkpoint/source provenance and non-exhaustive class scope, converts detections into the shared crop frame, and removes detections outside GrandQC clean tissue. |
| `03c_cellvitpp` | Typed nuclei detection | CellViT++ | Runs the official inference package in parallel with the other detectors, passes explicit MPP, normalizes numeric classes to names, removes detections outside GrandQC clean tissue, and records the installed package version plus checkpoint SHA-256. |
| `03d_cell_consensus` | Multi-detector instance fusion | CellPhenotyper custom | Performs one-to-one physical-distance matching across StarDist, HoVer-Net, and CellViT++, requires broad-scope StarDist/CellViT++ agreement by default, retains MoNuSAC as scoped support, preserves named detector classes separately, and emits canonical cells, abstentions, provenance, scope-aware count/spatial QC and pairwise agreement/boundary benchmarks. |
| `04_TMA` | TMA detection and spot assignment | CellPhenotyper custom | Detects whether the image is a tissue microarray; for TMA images, exports spot GeoJSON and associates detected cells with spot IDs. |
| `04_tissue_mask` | Crop-aligned clean-tissue mask | GrandQC + CellPhenotyper coordinates | Crops the full-slide GrandQC normal-tissue mask into the shared detector coordinate frame; no second heuristic tissue detector is run. |
| `05_gigatime` | Virtual mIHC and marker quantification | GigaTIME + CellPhenotyper custom quantification | Adds exact MPP-aware blockwise inference, selectable persisted channels, all-marker integrated single-cell quantification, direct multichannel pyramidal OME-TIFF export, an independent marker-feature KODAMA branch, explicit uncalibrated-score metadata, and sampled per-slide marker QC. |
| `06_roi` | ROI preparation and rasterization | CellPhenotyper custom | Accepts provided or automatically generated ROI GeoJSON, keeps coordinate transforms explicit, and rasterizes polygons into crop-aligned labeled masks. |
| `07_cell_assignments` | Object-to-ROI assignment | CellPhenotyper custom | Assigns segmented objects to ROI/tissue space for downstream embedding and clustering. |
| `08_cytoplasm` | Cytoplasm expansion | CellPhenotyper custom | Expands nuclei labels into cytoplasm labels for paired nuclei/cytoplasm quantification and downstream per-cell views. |
| `09_embeddings` | Morphology embeddings | UNI-2 | Supports cell-centred observations, an overlapping regular grid whose 90-pixel inner cores are adjacent, or both; tile and inner-square features use one token-subset forward pass with checkpointed buckets and a shared model cache. |
| `10_kodama` | Latent manifold analysis | KODAMA | Runs on UNI-2 and independently on all GigaTIME nuclei/cytoplasm marker means, uses 20 PCA components and 10,000 landmarks by default, and frees raw features after PCA. |
| `11_clustering` | Cluster assignment | CellPhenotyper custom + R clustering code | Uses inverse-neighbor-distance landmark sampling (`p=2`), Leiden clustering on landmarks, and nearest-neighbor assignment of remaining cells. |
| `12_cluster_mask` | Cluster mask generation | CellPhenotyper custom | Converts per-observation cluster labels into a spatial mask and overlays the result on a bounded-memory tiled-TIFF H&E preview. |
| `13_grown_tissue` | Cell-guided tissue growth | CellPhenotyper custom | Runs only for sparse cell-centred cluster masks. Grid masks already consist of adjacent dense inner cores and bypass this stage. For cell masks, arbitrary-resolution GrandQC support masks are resampled categorically onto a bounded work grid while retaining class identity. |
| `14_medsam_refine_tissue` | Tissue refinement | MedSAM + CellPhenotyper orchestration | Runs GPU MedSAM in editable external tissue-border bands, treats GrandQC background/artifact pixels as hard negatives during dilation, prompting, prediction cleanup and hole filling, then uses a bounded full-resolution marker-controlled watershed for internal inter-cluster boundaries; asserts zero support leakage and publishes separate change, empty-exclusion and native-resolution QC. |
| `15_cluster_geojson` | Final polygon export | CellPhenotyper custom | Converts the final refined section masks to GeoJSON with sample and class metadata. |
| `16_neoplastic_section` | Neoplastic-enriched section selection | CellPhenotyper custom | Splits final tissue polygons into connected sections, counts named neoplastic consensus cells, and deterministically exports the section with the largest count. |
| `17_titan` | Section representation | TITAN / CONCH v1.5 | Applies the official TITAN slide aggregator to MPP-correct CONCH patch features from the selected section and emits a validated 768-feature vector with provenance. |
| `18_pathofmpred` | Research endpoint prediction | PathoFMPred | Applies an explicit cancer-specific private registry to the TITAN vector and publishes predictions, QC plots, and a research report without clinical-calibration claims. |
| `00_execution` | Execution report | CellPhenotyper custom | Adds a project-level output manifest, stable unique `output_id` values, preserved full-run and targeted traces, per-process runtime/memory summaries, a stage-complete uncertainty register, a bounded-memory specimen atlas, a pre-execution per-filesystem capacity estimate, and a learned-model release inventory that fails closed on absent hashes, mutable revisions or unresolved license terms. |

## Detailed notes by wrapped tool

### GrandQC

Files:

- `bin/run_grandqc_artifact_analysis.py`
- `bin/grandqc_mask.py`
- `bin/crop_grandqc_clean_mask.py`
- `modules/run_grandqc_artifact_analysis.nf`
- `modules/crop_grandqc_clean_mask.nf`
- `nextflow.config`

Changes relative to upstream GrandQC:

1. Uses CellPhenotyper OME-TIFF reading instead of the original repository scripts directly.
2. Auto-selects the artifact model (`1.0`, `1.5`, `2.0`) from image MPP instead of requiring manual shell-script edits.
3. Uses adaptive tissue thumbnail sizing so small high-magnification fields do not collapse during tissue detection.
4. Includes a heuristic fallback tissue mask if the learned tissue detector returns near-zero tissue on a clearly nonblank image.
5. Uses the official 512-pixel geometry for both tissue and artifact inference on every device; larger experimental artifact tiles require an explicit override.
6. Uses overlap-based artifact inference with smooth probability blending instead of strict hard tile stitching. Class-score and weight accumulators are disk-backed memory maps for every image size, eliminating the former large-image switch to a different merge algorithm.
7. Suppresses low-confidence artifact calls after score blending.
8. Applies a small-FOV foreign-object refinement stage that collapses broad false-positive artifact fields onto the dominant dark foreign-object structure when appropriate.
9. Bundles model/cache handling and preview/summary generation into the pipeline stage.
10. Publishes a clean-tissue mask containing only normal tissue and uses it as the single downstream tissue gate.

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
6. Filters object centroids and label pixels against the crop-aligned GrandQC clean-tissue mask before publication.

### GigaTIME

Files:

- `bin/run_gigatime_on_crop.py`
- `bin/gigatime_seam_qc.py`
- `bin/export_gigatime_store_to_ometiff.py`
- `bin/quantify_gigatime_intensity.py`
- `bin/load_gigatime_kodama_rawdata.R`
- `modules/run_gigatime_on_crop.nf`
- `modules/quantify_gigatime_intensity.nf`
- `modules/run_gigatime_kodama.nf`

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
10. Optional coarse background-block skipping checks a patch-sized context halo and is vetoed whenever a nucleus or cytoplasm label is present. It is disabled by default because rectangular zero filling produced visible marker discontinuities.
11. Every persisted channel is assessed at every block boundary against neighboring within-block gradients; JSON/PNG QC records zero-fill transitions and can fail the process.
12. Builds an independent PCA/KODAMA representation from all matched nuclei and cytoplasm marker means.
13. Marks every virtual-marker value as an uncalibrated score in sidecars, Zarr metadata, quantification rows, summaries and QC figures while retaining historical filenames for compatibility.
14. Computes deterministic sampled per-marker distribution, saturation, near-zero, GrandQC clean-tissue/background and channel-correlation QC during the existing tiled inference pass.
15. Normalizes both `uint8` and `uint16` persisted stores back to the same `[0,1]` score scale before restart-only quantification.

### UNI-2

Files:

- `bin/extract_uni2_embeddings.py`
- `bin/build_uni2_spatial_grid.py`
- `modules/build_uni2_spatial_grid.nf`
- `modules/extract_uni2_embeddings_shared.nf`
- `subworkflows/run_auxiliary_cell_uni2.nf`
- `bin/compare_uni2_routes.R`
- `modules/compare_uni2_routes.nf`

Changes relative to naïve UNI-2 usage:

1. Supports either cell-centred observations or regular overlapping spatial tiles whose fixed inner cores meet edge-to-edge.
2. Default embedding families were reduced to `tile` and `inner_square`.
3. Rounded extraction coordinates and grid assignment use the same coordinates, eliminating boundary-cell omissions.
4. Per-grid and whole-stage manifests enforce exact one-row-per-mask-label coverage and reject incomplete resume shards.
5. A shared extraction path can emit both `tile` and `inner_square` outputs from one stage.
6. The pipeline uses a shared Hugging Face cache path to avoid duplicate model downloads across runs/repos.
7. `both` mode serializes a secondary cell-centred comparison after primary grid UNI-2 so two model instances cannot contend for one GPU.
8. `both` mode compares routes on matched grid units using Procrustes alignment, distance and neighborhood agreement, spatial coherence, and optional cross-validated virtual-marker proxy endpoints; it never chooses a route automatically.

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
8. The default landmark count is 10,000 for both UNI-2 and GigaTIME marker representations.
9. Downstream graph clustering repeats landmark selection and clustering across configurable seeds, reports adjusted Rand indices and per-observation stability, and can abstain from ambiguous landmark votes or unstable assignments.
10. Each clustering variant produces descriptive spatial-coherence and virtual-marker-enrichment tables plus a reproducible blinded-review packet with a separately stored cluster key. Biological interpretation still requires independent review or assays.

### MedSAM

Files:

- `bin/refine_grown_tissue_medsam.py`
- `bin/medsam_border_refine.py`
- `modules/refine_grown_tissue_medsam.nf`

Changes relative to direct standalone MedSAM use:

1. MedSAM is used as a downstream refinement stage, not as a first-pass segmenter.
2. It refines the grown sparse mask on the cell-centred route and the dense Stage 12 cluster mask directly on the grid route.
3. Both `standard` and `fine` cluster branches are carried through independently. Cell-centred masks arrive after tissue growth; dense grid masks bypass growth and are refined directly.
4. Internal boundaries are refined after MedSAM cleanup at full pixel resolution with a marker-controlled watershed limited to a configurable band.
5. The internal stage preserves foreground/background exactly and is a no-op when a tile contains only one cluster.
6. QC and summaries separate MedSAM changes from the image-guided internal-boundary changes.
7. The crop-aligned GrandQC clean-tissue/ROI mask is passed into the refinement implementation as an allowed-support mask. Seeds, baselines, dilation envelopes, prompts, editable bands, predictions and hole filling are prevented from entering GrandQC background or artifact areas.
8. Full-memory, streamed and resumed runs assert that final labels have zero pixels outside GrandQC support and publish a dedicated empty-area exclusion map plus pixel counts.

## Steps with no external upstream counterpart

These are primarily CellPhenotyper-specific orchestration or geometry stages. The
crop-aligned `04_tissue_mask` is not listed because its semantic source is the
upstream GrandQC normal-tissue class, although CellPhenotyper performs the crop and
coordinate transformation.

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

- **HoVer-Net:** The official PyTorch inference code and official fast MoNuSAC checkpoint are used for inference. CellPhenotyper launches it in parallel with StarDist and CellViT++, adds MPP-aware input normalization, an Aperio-compatible pyramidal TIFF carrying `AppMag` and `MPP` because upstream requires objective-power metadata, and passes a correctly named PNG version of the shared GrandQC mask through upstream's `--input_mask_dir` interface instead of allowing an independent 1.25x Otsu mask. It adds the explicit MoNuSAC type map because upstream parses its empty default as a filename, cache cleanup, support for upstream's `nuc` JSON key, conversion into the shared crop frame, and a final GrandQC centroid invariant. Metadata records the checkpoint SHA-256, upstream Git revision, inference-mask source, and the upstream-documented limitation that MoNuSAC positive classes do not span all nuclei. Post-processing workers are bounded by allocated RAM. The unmodified upstream repository is copied into a task-local writable runtime because upstream creates `debug.log` in its current directory and a Singularity image is immutable. The build recipes make the bundled checkpoint world-readable and the wrapper verifies readability before expensive WSI normalization. For recovery after a post-processing-only failure, the isolated copy of upstream `wsi.py` can be patched to reopen a validated completed `pred_map.npy` and skip raw inference; the installed upstream source and model are not modified.
- **CellViT++:** The official `cellvit-inference` package is used unchanged and pinned to `cellvit==1.0.9`. CellPhenotyper launches it in parallel with the other detectors, adds pyramidal crop preparation, passes the source MPP explicitly, clamps batch size to the upstream-supported range `2..48`, supplies a nonzero Ray worker count to avoid the upstream zero-worker modulo failure, normalizes output discovery into a stable `cellvit_cells.json` artifact, and applies the same GrandQC clean-tissue gate. Before creating the WSI pyramid, the wrapper verifies that the selected bundled checkpoint is readable. Ray temporary files, Matplotlib state, and runtime caches are redirected into task-local scratch so large runs do not fill the host root filesystem or write into the immutable image.
- **Multi-detector instance fusion:** This is CellPhenotyper-specific code. The explicit `cell_detection_mode` makes the scientific route independent of hardware: fusion requires GPU execution and never silently falls back to StarDist. It performs distance-gated, one-to-one matching between methods and prevents duplicate predictions from one detector in a component. The default role-aware policy requires direct StarDist/CellViT++ broad-scope agreement; MoNuSAC is scoped supporting evidence and does not increase the broad agreement score or shift the canonical centroid. The former two-of-any-three behavior remains an explicit sensitivity-analysis option. The stage assigns canonical IDs, records every acceptance or abstention decision, and writes a tiled label TIFF without holding three WSI masks in RAM. It reports pairwise match fractions, centroid displacement, polygon IoU and Hausdorff distance as inter-detector agreement, not reference-standard accuracy. Scope-aware count QC separates the all-detector ratio from the broad-scope ratio, records detector roles, exact combinations and acceptance counts, and checks whether any detector omitted entire image regions. The writer reserves a unique centroid-near seed pixel for every canonical ID and streams full label-coverage validation before publication.
