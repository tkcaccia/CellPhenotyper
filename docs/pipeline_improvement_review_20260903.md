# CellPhenotyper Pipeline Improvement Review

Date: 2026-09-03

## Overall Assessment

CellPhenotyper has unusually broad coverage, but breadth must not be confused with scientific validity. The software should be presented as a set of explicitly selected research workflows rather than one universal pipeline. The most important product improvement is to help a user understand what question is being answered, what unit is being analyzed, where the evidence is weak, and which outputs require expert review.

The review uses the DOME data/optimization/model/evaluation reporting structure and CLAIM 2024 terminology where applicable. In particular, it distinguishes development, tuning, internal testing and external testing, uses “reference standard” for an expert benchmark, and treats a successful software run as distinct from evidence of biological or clinical validity.

- DOME: https://doi.org/10.1038/s41592-021-01205-4
- CLAIM 2024: https://doi.org/10.1148/ryai.240300

Priority meanings:

| Priority | Meaning |
|---|---|
| P0 | Can invalidate the scientific result or make a successful run misleading. |
| P1 | Required before strong accuracy, reproducibility, or biological claims. |
| P2 | Important for usability, scale, maintenance, or publication quality. |
| P3 | Desirable extension after the core validation is complete. |

Status meanings:

| Status | Meaning |
|---|---|
| Implemented | Present in the current code and covered by targeted tests. |
| Validated on one WSI | Passed a documented real-WSI technical check, but does not establish biological validity or generalization. |
| Validate | Implemented, but requires a real-data rerun or external reference. |
| Engineering | Can be addressed without a new scientific experiment. |
| Discuss experiment | Requires a predefined dataset, endpoint, sample size, or expert annotation plan before execution. |

## Scientific Purpose And Scope

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| S01 | P0 | Implemented | Require an explicit `analysis_intent`; incompatible observation routes now fail before execution. |
| S02 | P0 | Implemented for software; manuscript alignment remains | The documented primary use is research-only exploratory multimodal characterization of H&E whole-slide images. Virtual staining, cell phenotyping, tissue-domain discovery, and outcome prediction remain separate workflows and must remain separate secondary claims in the manuscript. |
| S03 | P0 | Implemented | Keep cell-centred and grid observations as distinct scientific routes; never describe grid tiles as cells or cell-centred tiles as unbiased tissue sampling. |
| S04 | P1 | Implemented | `docs/ANALYSIS_ROUTE_GUIDE.md` maps each biological question to its route, observation unit, outputs, validation evidence, inapplicable stages and prohibited claims. |
| S05 | P1 | Discuss experiment | Predefine one primary biological endpoint for each route before comparing them. Do not select a route from whichever visualization looks cleaner. |
| S06 | P1 | Implemented | The README, route guide, scientific contract, reports and prediction-output documentation state prominently that outputs are research-use-only and distinguish predictions from measurements and calibrated assays. |
| S07 | P2 | Implemented | `docs/COMPATIBILITY_MATRIX.md` separates intake acceptance, technical demonstration and scientific validation across specimens, stains, scanners, resolution, formats, ROIs, routes and hardware. |
| S08 | P2 | Implemented | The optional study manifest distinguishes exploratory, development, internal-testing, and external-testing declarations; the report assigns a conservative claim ceiling. |

## Input Data And Metadata

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| D01 | P0 | Implemented | Strict-MPP stages record the physical-resolution source and fail when required calibration cannot be resolved. |
| D02 | P0 | Implemented | Source and converted input-QC reports now record dimensions, axes, dtype, bit depth, channel/color interpretation, compression, tiling, pyramid levels, MPP candidates and metadata conflicts; the final report surfaces the converted-image result. |
| D03 | P1 | Discuss experiment | Quantify sensitivity to incorrect MPP metadata by intentionally perturbing scale across representative slides. |
| D04 | P1 | Implemented | Strict input validation rejects implausible, anisotropic or conflicting MPP metadata; an explicit override is accepted only as visible run provenance. |
| D05 | P1 | Partially implemented | Source/converted images and ROI GeoJSON now receive full SHA-256 hashes, and ROI QC binds both identities. Propagate both hashes into every downstream stage-specific provenance record. |
| D06 | P1 | Implemented | ROI intake now validates FeatureCollection/polygon type, validity, finite coordinates, holes, exact level-0 bounds, coordinate-space/CRS declarations and source-image association without silent repair or clipping. |
| D07 | P2 | Engineering | Add scanner/manufacturer metadata to the project report to support later site-effect analysis. |
| D08 | P2 | Engineering | Record whether the input was losslessly converted, recompressed, color transformed, or pyramid-resampled. |
| D09 | P2 | Engineering | Reject accidental composite figures, thumbnails, and low-resolution screenshots using minimum physical field-of-view and information-content checks. |
| D10 | P3 | Engineering | Support a sample sheet with subject, block, stain, batch, site, scanner, and split identifiers instead of inferring all identity from filenames. |

## GrandQC And Analysis Support

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| Q01 | P0 | Implemented | GrandQC is mandatory before downstream analysis and defines shared clean-tissue support. |
| Q02 | P0 | Validated on one WSI; multi-cohort validation remains | The corrected no-skip GigaTIME path processed all 225 blocks and removed the GrandQC-associated rectangular zero-fill transitions on the cached Visium breast crop. |
| Q03 | P1 | Discuss experiment | Benchmark artifact masks against blinded pathologist annotations across tissue, scanner, stain, and MPP strata. |
| Q04 | P1 | Engineering | Publish per-class score maps, threshold values, connected-component tables, and excluded-area fractions, not only a colored overlay. |
| Q05 | P1 | Engineering | Add explicit `fail`, `model_only`, and `heuristic_fallback` policies for any non-model tissue fallback. |
| Q06 | P1 | Engineering | Warn or abstain when the selected GrandQC checkpoint is outside its validated physical-resolution range. |
| Q07 | P2 | Discuss experiment | Measure whether GrandQC exclusions systematically remove necrotic, hemorrhagic, adipose, folded, or low-cellularity tissue that remains biologically relevant. |
| Q08 | P2 | Engineering | Add a compact gallery of the largest excluded regions at native resolution for rapid expert review. |

## Cell Identification And Consensus

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| C01 | P0 | Implemented | Instance fusion is separated from phenotype evidence; incompatible detector taxonomies are not voted into a synthetic cell type. |
| C02 | P0 | Implemented | Fusion requires direct broad-scope StarDist/CellViT++ agreement by default. MoNuSAC is scoped support and does not inflate the canonical agreement score or shift the canonical centroid; all abstentions remain auditable. |
| C03 | P0 | Tooling implemented; discuss experiment | `bin/benchmark_cell_detectors.py` evaluates detector and consensus geometry against adjudicated polygons with patient/slide separation. The cohort and annotations remain to be agreed; pairwise model agreement is not accuracy. |
| C04 | P1 | Tooling implemented; discuss experiment | The benchmark reports precision, recall, F1, panoptic quality, AJI, Dice, boundary error, split rate and merge rate, plus prespecified-stratum and paired-detector tables. No biological performance result exists until the agreed reference study is run. |
| C05 | P1 | Tooling implemented; discuss experiment | The manifest locks physical resolution per ROI and supports separate fixed-scale sensitivity conditions. The scale conditions and reference cohort must be agreed before execution. |
| C06 | P1 | Engineering | Expose detector-specific minimum probability and size thresholds in the report with retained/removed count curves. |
| C07 | P1 | Engineering | Flag cells near crop, ROI, GrandQC, and tile boundaries because their geometry has asymmetric context. |
| C08 | P1 | Engineering | Preserve unmatched high-confidence detector instances in an audit table even when they are excluded from canonical consensus. |
| C09 | P2 | Discuss experiment | Stratify agreement by nucleus size, elongation, stain intensity, crowding, cell type, and tissue region. |
| C10 | P2 | Engineering | Add native-resolution disagreement galleries for two-model-only, three-model, split, merge, and unmatched cases. |
| C11 | P2 | Implemented | Scientific text and the metro map now use “role-aware multi-detector instance fusion”; the compatibility folder ID remains stable for restart support. |
| C12 | P3 | Discuss experiment | Learn detector-specific calibration or fusion weights only on a training cohort and evaluate them unchanged on held-out sites. |

### Detector-Imbalance Investigation On The Visium Breast Run

The copied completed run supports a scope/calibration explanation rather than a simple crop or GrandQC failure:

| Check | StarDist | CellViT++ | HoVer-Net MoNuSAC |
|---|---:|---:|---:|
| Retained instances | 30,785 | 27,304 | 4,991 |
| Removed by GrandQC | 2,065 of 32,850 | 1,853 of 29,157 | 270 of 5,261 |
| Instances in an accepted multi-detector component | 19,978 | 19,396 | 3,878 |
| Instances in a three-detector component | 2,618 | 2,618 | 2,618 |

HoVer-Net detections occur in every occupied image quadrant, so the six-fold count difference is not explained by a missing spatial strip. The run used the same crop and GrandQC support, a recorded source scale of 0.2738 micrometres per pixel, and HoVer-Net resampling to 0.25 micrometres per pixel. GrandQC removed only 5.1% of its raw detections, which cannot explain the difference.

The substantive problem is model comparability. The selected MoNuSAC checkpoint targets epithelial, lymphocyte, macrophage and neutrophil nuclei and is not an exhaustive general-nucleus detector; it omits fibroblasts and other nuclei outside that annotation scope. In this breast image, it labelled 3,424 of 4,991 retained instances as macrophage. Among accepted CellViT++/HoVer-Net pairs, 1,561 HoVer-Net macrophage labels corresponded to CellViT++ neoplastic labels. These taxonomies are not interchangeable, and neither is an independent expert reference standard.

Therefore the default fusion policy now requires StarDist/CellViT++ broad-scope agreement, retains HoVer-Net as detector-specific supporting evidence, keeps each phenotype field separate, reports unmatched detections, and avoids claiming a balanced three-model ensemble. Applied to the existing component table, this role-aware rule would retain 19,057 canonical components and abstain from 1,260 components that were previously accepted only through one broad detector plus MoNuSAC. This is a scientifically motivated scope correction, not evidence that 19,057 is the true cell count. Whether MoNuSAC improves instance recall or precision requires the expert-labelled benchmark in C03-C05; changing thresholds from this one slide would be post hoc tuning.

An additional code-compatibility rerun used an older archived detector set before the final shared-GrandQC wrapper metadata. The new implementation completed on the full crop and reduced 22,961 old any-two instances to 21,551 broad-pair instances: 1,329 components lacked both broad detectors and 81 components joined the broad detections only transitively beyond the physical match radius. The complete audit is in `audits/detector_role_fusion_20260903/`. This confirms implementation behavior, not segmentation accuracy.

## TMA Handling

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| T01 | P1 | Engineering | Report the evidence for the TMA/non-TMA decision, not only the binary result. |
| T02 | P1 | Discuss experiment | Validate core detection on regular arrays, irregular arrays, missing cores, fragmented cores, tilted arrays, and non-TMA mimics. |
| T03 | P1 | Engineering | Abstain from TMA assignment when grid regularity or core separation is insufficient. |
| T04 | P2 | Engineering | Preserve core labels from an input map and quantify conflicts with automatically detected cores. |
| T05 | P2 | Engineering | Summarize per-core tissue area, artifact burden, cell count, marker coverage, and analysis exclusions. |
| T06 | P3 | Engineering | Support replicate-core linkage and patient-level grouping without encoding patient identity in filenames. |

## GigaTIME And Marker Quantification

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| G01 | P0 | Implemented | Whole-block background skipping is disabled by default because it caused rectangular zero-fill discontinuities. |
| G02 | P0 | Implemented | Persisted channels now receive a block-boundary seam metric, diagnostic heatmap, and configurable fail gate. |
| G03 | P0 | Validated on one WSI; multi-cohort validation remains | The cached Visium rerun passed at its 1,024-pixel block grid, at every archived 768-pixel boundary and at every 128-pixel inference-patch line; visual review confirmed removal of the former large rectangular sections. |
| G04 | P0 | Tooling implemented; discuss experiment | `docs/VIRTUAL_MARKER_VALIDATION_PROTOCOL.md` and `bin/benchmark_virtual_markers.py` define registered measured-protein evaluation with patient-level intervals, cellularity adjustment, spatial endpoints, registration/matching gates and discordance review. The paired cohort and acceptance margin remain to be agreed and run. |
| G05 | P1 | Implemented | `gigatime_marker_score_qc.{json,tsv,png}` records deterministic sampled dynamic range, saturation, near-zero fraction, GrandQC clean-tissue/background contrast and channel correlation during the existing tiled pass. |
| G06 | P1 | Partially implemented | OME channel names, physical scale and score semantics are invariant in the writer and reader tests; a recorded QuPath interoperability test remains required before release. |
| G07 | P1 | Implemented | Sidecar/Zarr metadata, quantification rows, summaries, reports and QC figures identify values as uncalibrated virtual-marker scores. Legacy `gigatime_probs` and `intensity` filenames are documented as compatibility names only. |
| G08 | P1 | Discuss experiment | Test stain normalization, scanner shift, compression, blur, and color perturbation sensitivity. |
| G09 | P2 | Engineering | Add positive and negative representative patch galleries for each persisted marker, with H&E and prediction side by side. |
| G10 | P2 | Engineering | Quantification should report area, mean, median, quantiles, maximum, sums, and the fraction above a prespecified threshold; thresholds must not be chosen per result. |
| G11 | P2 | Implemented | Every row and summary records the named segmentation compartment and states that scores are summarized only over pixels assigned to that compartment. |
| G12 | P3 | Discuss experiment | Evaluate marker-specific uncertainty or conformal intervals on an independent multiplex cohort. |

### Corrected GigaTIME Continuity Validation

An isolated GPU rerun on the cached Visium HD breast crop resampled 13,172 x 13,098 pixels at 0.2737744 micrometres per pixel to 14,425 x 14,344 pixels at exactly 0.25 micrometres per pixel. It processed all 225 blocks with background skipping disabled, completed in 8 min 45.17 s without an OOM retry, and exported a verified five-channel, four-level pyramidal OME-TIFF.

The archived skip-enabled image failed 21 of 180 channel-boundary tests; its worst DAPI boundary had p95 excess 0.33725 and affected 21.2% of sampled positions. The corrected image failed 0 of 140 tests at its execution blocks, 0 of 180 tests at the exact archived 768-pixel boundaries, and 0 of 1,120 tests across the complete 128-pixel inference-patch grid. Native-resolution review confirms that the old zeroed rectangle is continuous in the corrected image. Full evidence and reproducible commands are in `audits/gigatime_seam_validation_20260903/`.

This validates technical continuity on one WSI only. G04 remains the required biological validation against registered measured multiplex imaging.

## UNI-2 Representation Routes

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| U01 | P0 | Implemented | Grid and cell-centred routes remain separate and use explicit observation metadata. In `both` mode they now continue through independent clustering, route-appropriate mask handling, MedSAM and GeoJSON outputs. |
| U02 | P0 | Implemented | `both` mode now compares routes after aggregating cells to common grid cores rather than correlating arbitrary independent axes. |
| U03 | P0 | Implemented | The comparison includes coverage, Procrustes agreement, distance correlation, neighborhood overlap, spatial coherence, and optional virtual-marker proxy prediction. |
| U04 | P0 | Tooling implemented; discuss experiment | `docs/UNI2_ROUTE_VALIDATION_PROTOCOL.md` and `bin/benchmark_uni2_routes.py` compare locked cell and grid masks with adjudicated tissue domains using label-invariant, abstention-aware, patient-level and physical-boundary endpoints. The independent cohort and acceptance margin remain to be agreed and run. |
| U05 | P1 | Engineering | Record tile physical size, context size, inner-square physical size, source crop size, interpolation, encoder revision, and pooling definition per run. |
| U06 | P1 | Discuss experiment | Compare `token_subset` with the legacy masked inner-square forward on a predefined equivalence margin and downstream endpoints. |
| U07 | P1 | Engineering | Add tile-content QC for blur, pen, blank fraction, stain extremes, and padding fraction; allow abstention before encoding. |
| U08 | P1 | Engineering | Quantify representation duplication caused by dense overlapping context and report effective independent sample size. |
| U09 | P2 | Discuss experiment | Evaluate route robustness across tile sizes and target MPP without reusing the test cohort for parameter choice. |
| U10 | P2 | Engineering | Persist a small deterministic audit sample of tiles even when production tile saving is disabled. |
| U11 | P2 | Engineering | Report missing, resumed, duplicated, and checksum-mismatched embedding shards as separate counters. |
| U12 | P3 | Discuss experiment | Compare UNI-2 against at least one pathology foundation-model baseline under matched tiling and compute budgets. |

## KODAMA And Clustering

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| K01 | P0 | Implemented | Clustering retains the selected inverse-distance landmark strategy with `p=2` and reports the exact seed and neighbor settings. |
| K02 | P0 | Implemented | Full landmark selection, graph clustering, and KNN assignment repeat across seeds; ARI and per-observation stability are saved. |
| K03 | P0 | Implemented | KNN winner fraction and winner-minus-runner-up margin are reported; ambiguous or unstable observations can abstain. |
| K04 | P0 | Implemented | Nonzero `cluster_target_clusters` requires explicit sensitivity acknowledgement. Assignments, summaries, plots and the run landing page identify the result as sensitivity-only and prohibit presenting it as the graph-derived primary partition. |
| K05 | P1 | Implemented + discuss experiment | The pipeline now creates seed stability, spatial coherence, virtual-marker enrichment and blinded-review outputs for every variant. Selecting graph resolution and domain count on a training cohort remains an experiment requiring agreement. |
| K06 | P1 | Engineering | Report cluster size, spatial fragmentation, boundary length, isolated-component count, and nearest-cluster mixing. |
| K07 | P1 | Engineering | Add rare-cluster protection so biologically coherent small groups are not automatically merged solely by size. |
| K08 | P1 | Discuss experiment | Evaluate whether inverse-distance landmarking overrepresents dense common states and underrepresents rare phenotypes. |
| K09 | P1 | Implemented | KODAMA membership plots render abstentions in neutral gray; dedicated KODAMA and tissue-space figures distinguish assignment ambiguity from seed instability, while an observation-level GeoJSON preserves locations and uncertainty fields. |
| K10 | P2 | Engineering | Save cluster-level representative tiles, marker summaries, detector composition, and spatial regions for interpretation. |
| K11 | P2 | Engineering | Add a cluster-label transfer model evaluated on held-out slides if cross-slide labels are claimed. |
| K12 | P2 | Discuss experiment | Quantify batch/site separation before interpreting clusters as biology. |
| K13 | P3 | Engineering | Add consensus clustering across algorithms only as a sensitivity analysis; avoid an unprincipled menu of visually selected outputs. |

## Spatial Masks And MedSAM

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| M01 | P0 | Implemented | Cell-mask growth is bypassed for dense grid tiles because it is scientifically obsolete on that route. |
| M02 | P0 | Implemented | GrandQC background and artifact regions are hard-negative support during MedSAM refinement. |
| M03 | P0 | Validate | Confirm on real outputs that MedSAM’s enlarged edit border improves boundaries without erasing thin or isolated tissue. |
| M04 | P1 | Discuss experiment | Benchmark outer tissue boundaries and internal cluster boundaries against expert annotations separately. |
| M05 | P1 | Engineering | Fail or abstain when the editable fraction, relabelled fraction, component loss, or topology change exceeds prespecified limits. |
| M06 | P1 | Engineering | Report per-cluster additions, removals, relabeling, holes, fragmentation, and leakage rather than only global pixel totals. |
| M07 | P1 | Engineering | Add before/after native-resolution galleries sampled from high-change, low-change, and random regions. |
| M08 | P2 | Discuss experiment | Compare MedSAM refinement against morphology-only, watershed-only, and no-refinement baselines. |
| M09 | P2 | Engineering | Preserve the pre-refinement mask as a first-class output so downstream differences remain attributable. |
| M10 | P3 | Engineering | Add topology-aware constraints for TMA cores and narrow tissue bridges if these are required by the intended use. |

## Validation And Statistical Design

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| V01 | P0 | Implemented declaration; discuss experiment | `study_manifest` now requires study phase and cohort split metadata. The actual cohorts and sample sizes still require agreement before execution. |
| V02 | P0 | Discuss experiment | Prevent slides from the same patient, block, TMA, or site appearing across train and test splits. |
| V03 | P0 | Implemented declaration; discuss experiment | The manifest requires one primary endpoint, statistical unit and metric. Acceptance margins, exclusions and failure handling still require study-specific agreement. |
| V04 | P1 | Discuss experiment | Report confidence intervals with patient/slide-level resampling rather than treating millions of cells or tiles as independent samples. |
| V05 | P1 | Discuss experiment | Include at least one external site and scanner for generalization claims. |
| V06 | P1 | Discuss experiment | Measure interobserver variability so model performance is interpreted relative to expert disagreement. |
| V07 | P1 | Discuss experiment | Conduct ablations for GrandQC, detector fusion, inner-square features, route choice, clustering stability, and MedSAM. |
| V08 | P1 | Discuss experiment | Perform failure-enriched evaluation, including folds, blur, pen, low tissue, necrosis, staining extremes, and metadata errors. |
| V09 | P2 | Engineering | Produce a model/data flow diagram that distinguishes learned models, deterministic transforms, and human decisions. |
| V10 | P2 | Discuss experiment | Register all post hoc analyses and clearly label them exploratory in the manuscript. |

## Performance, Memory, And Hardware

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| P01 | P0 | Implemented | GPU-capable stages use a pipeline-wide hardware policy and shared GPU admission rather than independent uncontrolled allocation. |
| P02 | P0 | Implemented | `bin/storage_preflight.py` runs before image task submission, probes source dimensions and discoverable ROI bounds, estimates active-stage durable output and retained/transient work, includes relevant model/runtime cache growth, distinguishes `copy` from `rellink`, aggregates shared filesystems, preserves a configurable free-space reserve and reports both expected and restart-worst-case demand in `00_execution/storage_preflight.json`. Expected insufficiency fails by default; restart-only insufficiency is a review warning. |
| P03 | P1 | Discuss experiment | Benchmark at least two GPU-memory classes with identical inputs and seeds; report wall time, utilization, VRAM, RSS, I/O, and output equivalence. |
| P04 | P1 | Engineering | Record effective auto-selected batch, tile, worker, thread, and memory settings for every stage in one table. |
| P05 | P1 | Engineering | Add checksum-based interruption/resume tests for GrandQC, detectors, GigaTIME, UNI-2, KODAMA, and MedSAM. |
| P06 | P1 | Engineering | Prevent simultaneous CPU-heavy R and image writers from exhausting host RAM even when GPU memory is available. |
| P07 | P2 | Engineering | Report compute normalized by tissue area, cells, and retained grid observations, not only total runtime. |
| P08 | P2 | Engineering | Separate model inference, preprocessing, transfer, quantification, serialization, and pyramid-writing time. |
| P09 | P2 | Engineering | Add multi-sample scheduling tests for GPU lock fairness and starvation. |
| P10 | P3 | Engineering | Evaluate asynchronous prefetch and pinned-memory loading only after correctness checks establish output equivalence. |

## Reproducibility, Provenance, And Maintenance

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| R01 | P0 | Engineering | Pin runtime containers by immutable digest in release configurations. |
| R02 | P0 | Implemented as a fail-closed release gate | `resources/model_registry.json`, runtime wrapper metadata and `bin/build_model_inventory.py` produce `00_execution/model_inventory.{json,tsv}` with source, requested/resolved revision, checkpoint SHA-256, cache path, license status and training domain for every learned stage. Prediction files without metadata are failures. Legacy runs remain partial, and unresolved StarDist/HoVer-Net/MedSAM checkpoint terms, the GigaTIME license conflict and incomplete private PathoFMPred metadata remain explicit release blockers rather than inferred values. |
| R03 | P1 | Implemented for public critical parameters | `nextflow_schema.json` and `bin/validate_pipeline_params.py` enforce scientific-route, evidence, input-QC, hardware, uncertainty and publication-critical types, enums, ranges, conditionals and cross-field invariants; advanced resource overrides remain documented in `PARAMETERS.md`. |
| R04 | P1 | Engineering | Add nf-test coverage for stage contracts, exact restart points, resume, optional branches, and multi-sample inputs. |
| R05 | P1 | Engineering | Publish a `CITATION.cff`, software license, changelog, contribution guide, and versioning policy. |
| R06 | P1 | Engineering | Enable build provenance/attestation and publish an SBOM for release images. |
| R07 | P1 | Engineering | Add dependency and container vulnerability scanning without silently upgrading model-critical libraries. |
| R08 | P2 | Partially implemented | The complete auxiliary cell route has moved into route-focused UNI-2 and spatial subworkflows, bringing `main.nf` back below the Groovy generated-string limit. Continue extracting the primary grid/cell and detector routes and add formal typed tuple contracts. |
| R09 | P2 | Engineering | Define a deprecation window for compatibility parameters and remove ambiguous aliases after one release. |
| R10 | P2 | Engineering | Add deterministic miniature fixtures representing WSI, ROI, TMA, grid, cells, artifacts, and missing metadata. |
| R11 | P2 | Engineering | Verify every published symlink or copy and include its checksum in `project_outputs`. |
| R12 | P3 | Engineering | Add a release checklist that requires CPU smoke, GPU smoke, resume equivalence, output-schema comparison, and viewer compatibility. |

## Reports, Figures, And User Experience

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| X01 | P0 | Implemented | The final report surfaces study validation readiness and claim ceiling, role-aware detector fusion, GigaTIME seam status, clustering stability/abstention, independent-review status, route-comparison limitations, and a stage-complete uncertainty register that explicitly identifies missing estimates. |
| X02 | P0 | Implemented | `00_execution/specimen_atlas.{html,json}` provides a bounded-memory, specimen-level review sequence spanning H&E context, tissue/artifact/ROI/TMA support, detector/fusion output, uncalibrated virtual-marker summaries, grid or cell-centred domains, explicit uncertainty/abstention, MedSAM refinement and optional research endpoints. Each asset retains route, observation unit, semantic limit and stable source-output ID; missing layers remain visible. |
| X03 | P1 | Engineering | Make every figure show sample ID, physical scale bar, route, observation unit, model/checkpoint, and whether values are measured or predicted. |
| X05 | P1 | Implemented | Membership figures show abstentions in neutral gray; dedicated categorical uncertainty overlays and rasters prevent abstentions from being mistaken for empty tissue. |
| X06 | P1 | Implemented | `00_execution/index.html` is a self-contained review-first landing page linking the claim ceiling, ordered QC signals, bounded preview gallery, stage storage, slowest processes, reports and output-ID manifests. |
| X07 | P2 | Engineering | Distinguish user-actionable warnings from informational logs and collapse repeated progress messages. |
| X08 | P2 | Engineering | Add QuPath project generation with channel names, image calibration, GeoJSON overlays, and stable output IDs. |
| X09 | P2 | Engineering | Generate publication figures from scripts with stored parameters rather than manually edited screenshots. |
| X10 | P2 | Engineering | Report both absolute and relative paths while avoiding machine-specific paths in manuscript-facing artifacts. |
| X11 | P3 | Engineering | Add an interactive HTML QC viewer only after static artifacts and schemas are stable. |

## Ethics, Governance, And Translation

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| E01 | P0 | Implemented | Reports, pipeline card, route guide, GigaTIME metadata/quantification and PathoFMPred documentation identify these as research predictions rather than diagnostic probabilities or measured biomarkers. |
| E02 | P1 | Discuss experiment | Evaluate performance and failure rates across relevant demographic, geographic, tissue, scanner, and site subgroups where metadata permit. |
| E03 | P1 | Implemented as minimum policy | `docs/DATA_GOVERNANCE.md` defines sensitive-data classification, de-identification, least privilege, encrypted transfer/storage, credentials, scratch handling, project-specific retention, audit records, sharing and incident requirements; institutional approval remains mandatory. |
| E04 | P1 | Engineering | Document licenses and access restrictions for every model, checkpoint, dataset, and redistributed artifact. |
| E05 | P1 | Implemented | `docs/HUMAN_REVIEW_POLICY.md` defines reviewer roles, route-specific sign-off, failure criteria, prespecified sampling, blinding, adjudication, privacy and auditable decisions using `resources/human_review.template.json`. |
| E06 | P2 | Discuss experiment | Evaluate whether artifact exclusion or detector imbalance is differential across tissue phenotypes or patient groups. |
| E07 | P2 | Implemented at pipeline inventory level | `docs/PIPELINE_CARD.md` documents intended/unsupported use, component roles and scopes, observation units, evidence, uncertainty, failure modes, subgroup limitations and revalidation triggers; unresolved immutable license/checkpoint metadata remain visible release blockers. |
| E08 | P3 | Engineering | Define version-locking and revalidation requirements before any prospective or regulated deployment. |

## Manuscript And Publication

| ID | Priority | Status | Reviewer comment and required action |
|---|---|---|---|
| N01 | P0 | Engineering | Narrow the title, abstract, and claims to the validated primary contribution rather than listing all integrated tools. |
| N02 | P0 | Discuss experiment | Add independent validation for every accuracy claim; technical completion and visual plausibility are insufficient. |
| N03 | P1 | Engineering | Use a schematic that separates input/ROI routes, parallel detectors, cell and grid feature routes, learned models, deterministic transforms, and optional outputs. |
| N04 | P1 | Engineering | Report sample counts at patient, slide, ROI, core, cell, and grid levels; do not imply that cells are independent biological replicates. |
| N05 | P1 | Engineering | Add a complete Methods table for software version, model revision, MPP, tile size, stride, overlap, batch, thresholds, seeds, landmarks, graph k, resolution, and abstention. |
| N06 | P1 | Engineering | Present failure cases and abstentions in the main or supplementary figures, not only successful examples. |
| N07 | P1 | Engineering | Separate measured markers, virtual markers, detector labels, consensus instances, latent representations, and inferred domains in terminology. |
| N08 | P1 | Engineering | Ensure the Code availability statement points to an immutable release and container digest matching reported experiments. |
| N09 | P2 | Discuss experiment | Include reader studies or blinded pathology review only with prespecified tasks, sampling, and agreement analysis. |
| N10 | P2 | Engineering | Add reporting checklists and a claim-to-evidence table linking each manuscript statement to a result, figure, dataset, and code version. |

## Implemented In This Pass

| Requirement | Implementation |
|---|---|
| Primary intended use | `analysis_intent` and `analysis_contract.json` define the research question and reject incompatible routes. |
| Separate cell and grid workflows | Observation units remain explicit; grid masks bypass obsolete cell-mask growth. |
| Detector imbalance | The Visium run was audited: the discrepancy is not explained by spatial coverage or GrandQC filtering, while MoNuSAC scope and phenotype discordance prevent treating it as an equal exhaustive detector. Fusion reports geometry agreement and abstentions without fabricating phenotype consensus. |
| Detector validation readiness | `docs/DETECTOR_VALIDATION_PROTOCOL.md`, a long-form manifest template and `bin/benchmark_cell_detectors.py` define blinded adjudication, leakage checks, geometry-only endpoints, patient/slide bootstrap intervals, paired comparisons, subgroup tables, error-case GeoJSON and immutable input hashes. This is evaluation infrastructure, not completed biological validation. |
| Consensus semantics | Instance geometry and detector-specific phenotype evidence are separate output fields. |
| GigaTIME discontinuities | Default whole-block skipping is disabled; safer halo logic and a fail-capable seam QC report were added. A corrected five-channel GPU rerun passed execution-block, archived-boundary and full inference-patch-grid continuity checks on the cached Visium WSI. |
| GigaTIME score semantics and slide QC | Historical output names remain compatible, but metadata, quantification rows, summaries and figures now state `uncalibrated_virtual_marker_score`. Every tiled run emits deterministic sampled all-channel distribution, saturation, near-zero, clean-tissue/background and correlation QC without a second WSI read. Restart-only quantification now normalizes both `uint8` and `uint16` stores to `[0,1]`. |
| GigaTIME validation readiness | A long-form manifest, registered-reference protocol and standalone benchmark separate technical continuity from measured-marker concordance and prevent pooled-cell, post hoc threshold or unadjusted-cellularity claims. This is evaluation infrastructure, not completed biological validation. |
| Cluster selection | Repeated seeds, ARI, landmark vote margins, per-observation stability, abstention, spatial coherence, virtual-marker enrichment and a blinded-review packet replace purely visual confidence. |
| Forced cluster count | A nonzero requested cluster count fails unless explicitly acknowledged as a sensitivity analysis; accepted outputs and figures are stamped `sensitivity_only`. |
| UNI-2 route comparison | Cell observations are aggregated to common grid cores and evaluated with predefined representation, spatial, and marker-proxy endpoints. |
| UNI-2 independent validation readiness | A paired route manifest and standalone benchmark now add adjudicated-domain ARI/NMI, coverage, physical boundary metrics, patient bootstrap differences, prespecified decisions, QC panels and immutable hashes. These tools do not make the missing biological comparison complete. |
| Uncertainty | The scientific contract and final report distinguish descriptive agreement, uncalibrated scores, hard failures, and abstentions. `00_execution/uncertainty_register.{json,tsv}` covers every stage and states explicitly where uncertainty is not quantified. |
| Study evidence | `study_manifest` and `validation_readiness.json` encode DOME/CLAIM-oriented design fields and prevent technical completion from being reported as validation. |
| Input integrity | Source and converted image reports now include full hashes, image-layout/QC metadata and conflict-aware physical calibration; the final report exposes the result. |
| ROI integrity | Every ROI is immutable, hashed and checked against the converted image for geometry validity, holes, bounds, pixel coordinate space and declared image association before cropping. |
| Route selection and interpretation | `docs/ANALYSIS_ROUTE_GUIDE.md` links primary questions to observation units, required settings, relevant outputs, minimum evidence and prohibited claims. |
| Compatibility and limitations | `docs/COMPATIBILITY_MATRIX.md` prevents readable formats or successful runs from being mistaken for scanner, specimen, biological or clinical validation. |
| Parameter usability | `nextflow_schema.json` supplies machine-readable help and constraints, while CI and `bin/validate_pipeline_params.py` reject incompatible public parameter files. |
| Human review | `docs/HUMAN_REVIEW_POLICY.md` and `resources/human_review.template.json` define review roles, sampling, unacceptable failures, sign-off and an auditable decision record surfaced by the run report. |
| Governance and pipeline card | `docs/DATA_GOVERNANCE.md` and `docs/PIPELINE_CARD.md` centralize privacy, retention, oversight, model roles, evidence ceilings, failure modes and revalidation triggers. |

## Experiments Requiring Agreement Before Execution

The following should not be started until the cohort, reference standard, split, primary endpoint, acceptable margin, and statistical unit are agreed:

1. Expert-labelled GrandQC artifact benchmark.
2. Expert-labelled detector and instance-fusion benchmark.
3. Registered H&E/multiplex validation of GigaTIME markers.
4. Grid-versus-cell UNI-2 biological endpoint comparison.
5. Cluster-number and resolution selection study.
6. MedSAM outer- and internal-boundary benchmark.
7. External-site and scanner generalization study.
8. GPU performance and output-equivalence benchmark.
9. Demographic/site subgroup and bias analysis.
10. Blinded pathologist interpretation or reader study.
