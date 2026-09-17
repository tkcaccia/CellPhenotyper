# CellPhenotyper Pipeline Audit - 2026-09-02

Follow-up (2026-09-03): the GigaTIME block-discontinuity finding below has been corrected and technically revalidated on the same cached WSI. The archived skip-enabled output failed 21 of 180 boundary tests; the corrected no-skip output failed 0 of 140 execution-block tests, 0 of 180 tests at the archived boundaries, and 0 of 1,120 tests across the inference-patch grid. See `audits/gigatime_seam_validation_20260903/`. Biological marker validation remains outstanding.

## Scope

This audit covered stage order and restart wiring, GrandQC tiling and memory behavior, crop-coordinate propagation, cell-detection semantics, GPU allocation, consensus QC, MedSAM background handling, output reporting, automated tests, and manuscript/software consistency.

## High-Risk Findings Addressed

1. GrandQC was configurable as optional even when downstream stages were requested. It is now mandatory for every analysis stage after conversion.
2. The shared image crop was ROI-derived rather than tissue-derived. It is now the padded level-0 bounding box of `GrandQC clean tissue intersection ROI`; the same crop and shifted support are used downstream.
3. GrandQC changed its overlap-merging algorithm above an image-size threshold. Every size now uses smooth probability blending through disk-backed float32 score and weight accumulators finalized in bounded row blocks.
4. Consensus was silently replaced by StarDist on CPU hardware. `cell_detection_mode` is now explicit; `consensus` requires GPU execution and `stardist` must be requested deliberately.
5. GPU processes could all see every device and had no host-wide memory admission. A shared scheduler now selects one GPU from live free memory, obtains task and VRAM-token locks, and exports only that device through `CUDA_VISIBLE_DEVICES`.
6. MedSAM could retain or create labels in excluded background. Initial labels, tile inputs, model outputs, optional watershed results, resumed checkpoints, and final masks are now clamped to the GrandQC tissue/ROI support.
7. A requested stage could close an empty channel without a direct error. Requested outputs now use fail-fast channel guards, and exact GigaTIME and marker restart prerequisites are loaded explicitly.
8. Consensus reported membership but not quantitative pairwise agreement. It now writes detection match fractions, centroid distances, polygon IoU, and Hausdorff distances with an explicit warning that these are not ground-truth accuracy.
9. The shared-crop fallback assumed that `tifffile.aszarr()` always returned an array. A pyramidal BTF-derived OME-TIFF returned a multiscale group and exposed a decoder-specific libvips failure. The fallback now resolves level 0 explicitly, reads bounded row blocks into a disk-backed RGB buffer, and removes that buffer after writing the tiled crop.
10. The large-image grow branch assumed that the GrandQC tissue mask and cluster seeds shared either the native or requested work grid. It now resamples any independent tissue-mask grid directly to the bounded categorical work grid, avoiding both the shape failure and a full-resolution temporary allocation.
11. Cluster and grown-mask QC previews silently substituted a white background when the compressed tiled H&E crop could not be memory-mapped. A shared Zarr-backed TIFF preview reader now downsamples tiled data without materializing the full image, and both process fingerprints include that helper.

## Verification Completed

- `pytest`: 110 passed, 13 skipped, including independent-grid large-image growth, optional-Zarr crop and non-memory-mappable tiled-preview regression tests.
- Python compilation: passed with the bundled document runtime.
- Shell syntax for GPU admission: passed.
- Nextflow flattened configuration: passed.
- Full GPU-configured consensus stub route: 23 core tasks completed.
- Exact GigaTIME restart: launched GigaTIME and OME-TIFF export only.
- Exact marker restart: launched nuclei and cell-associated quantification only.
- Explicit CPU StarDist route: completed without attempting consensus.
- Chiamaka scheduler probe: `/usr/bin/flock` available; one GPU slot and memory token acquired; NVIDIA GeForce RTX 5060 Ti GPU 0 selected; `CUDA_VISIBLE_DEVICES=0` exported.
- Chiamaka BTF crop probe: the real pyramidal Visium HD input exercised the Zarr-group fallback and wrote a 13,172 x 13,098 crop in bounded memory; the full rerun then passed the `analysis_crop` output contract.
- Chiamaka full route: `convert -> pathofmpred` completed successfully on the Visium HD ROI with consensus cell detection, grid UNI-2, two-cluster Leiden, GigaTIME, MedSAM, TITAN and PathoFMPred.
- Representative task times: UNI-2 15 min 44 s; GigaTIME inference 9 min 36 s; MedSAM 11 min 42 s; HoVer-Net 5 min 31 s; KODAMA 1 min 42 s.
- Cell consensus: StarDist 30,785; CellViT++ 27,304; HoVer-Net 4,991; consensus 20,317, including 17,699 two-detector and 2,618 three-detector cells.
- Grid route: 25,760 candidate locations and 19,330 retained tissue tiles at 0.25 micrometres per pixel; KODAMA/Leiden yielded 11,149 and 8,181 tiles after the test-specific two-cluster merge.
- Cluster-map QC: the canonical preview was regenerated over the true H&E crop after fixing the tiled-TIFF fallback; the earlier flat pink/green preview represented correct cluster labels over an incorrect white preview background.
- MedSAM: all 16 CUDA tiles completed with no failure; 449,482 foreground pixels were added, 872 removed and 15,326,067 relabelled after the combined MedSAM and image-guided refinement.
- Manuscript: 14 rendered pages inspected.
- Supplementary methods: 4 rendered pages inspected.

## Scientific and Product Findings from the Completed Run

1. The consensus is not yet a balanced three-model ensemble. HoVer-Net detected about one sixth as many cells as StarDist, so most accepted cells are effectively supported by StarDist and CellViT++. HoVer-Net calibration and expert-labelled validation are required before the ensemble can support an accuracy claim.
2. The forced two-cluster solution is well separated in KODAMA space, but its tissue map is highly interdigitated. Cluster number should be selected by repeated-seed stability, spatial coherence, reproducibility on held-out slides, marker enrichment and pathologist interpretation rather than visual embedding separation alone.
3. MedSAM demonstrably modifies the result, but its edit budget is substantial: 19.4 million pixels changed in the image-guided editable area. The pipeline needs boundary-level validation and an abstention/QC threshold, not only evidence that the model ran.
4. GigaTIME uses the correct 0.25-micrometre physical resolution and writes five channels correctly, but rectangular intensity discontinuities remain. Stored Zarr seam maxima reached 36.3 levels for DAPI and 29.0 for CD8 on an 8-bit scale, so blockwise background handling and seam QC require correction before biological use.
5. PathoFMPred outputs are technically complete but remain research predictions. They are uncalibrated TCGA-relative ranks, not probabilities, and several endpoints are site-sensitive; the report must preserve these limitations prominently.
6. The current folder tree is comprehensive but does not yet answer a biological question directly. A specimen-level phenotype atlas should combine tissue domains, consensus-cell phenotypes, GigaTIME marker enrichment, neighbourhood statistics, representative H&E regions and uncertainty in one reviewable report.

## Residual Risks and Recommended Experiments

### Priority 1 - Required Before Updated Scientific Claims

1. Correct GigaTIME block discontinuities, add a fail-fast seam metric and rerun GigaTIME plus dependent outputs on the cached Visium HD analysis.
2. Benchmark GrandQC against pathologist-labelled artifacts across multiple tissues, scanners, and MPP values.
3. Benchmark each detector and consensus against expert nuclear instances. Pairwise detector agreement is not accuracy.
4. Benchmark grown masks versus tissue-constrained MedSAM using expert outer and internal boundaries, including empty background.
5. Validate GigaTIME marker intensity and spatial concordance against matched multiplex imaging.
6. Compare cell-centred and grid UNI-2 routes on predeclared biological endpoints rather than relying on embedding appearance.
7. Evaluate domain number with stability, spatial coherence, marker enrichment and blinded pathologist review; retain forced cluster counts only as sensitivity analyses.

### Priority 2 - Required Before Performance Claims

1. Compare at least two NVIDIA memory classes with fixed inputs and seeds, reporting wall time, utilization, peak VRAM, peak RSS, concurrent schedule, and output equivalence.
2. Run controlled interruption/resume tests for GrandQC, detectors, UNI-2, GigaTIME, and MedSAM, comparing checksums against uninterrupted outputs.
3. Measure scratch demand from GrandQC probability memory maps and configure capacity checks before inference on multi-sample runs.

### Priority 3 - Reproducibility and Maintainability

1. Make the automatic GrandQC heuristic tissue fallback an explicit policy (`fail`, `model_only`, or `heuristic_fallback`) because it can alter the shared analysis area.
2. Replace deprecated `cell_consensus_enable` after one compatibility release.
3. Add Linux CI for the GPU admission helper with fake `nvidia-smi` and lock contention; retain the live host smoke probe as a release check.
4. Add a multi-sample stub fixture to test GPU lock-directory sharing and unique output contracts across samples.
5. Tag the corrected pipeline, pin the runtime by digest, publish model checksums, and update the manuscript Code availability statement with the resulting commit.

## Submission Status

The revised Visium HD ROI route completed end to end and establishes engineering feasibility across all 18 stages. It does not yet establish biological accuracy. The detector imbalance, unvalidated two-domain interpretation, substantial MedSAM edit budget and measured GigaTIME block discontinuities prevent this run from replacing the manuscript's archived scientific results. Nature Methods accuracy and performance claims should remain limited until the Priority 1 validation experiments are completed.
