# CellPhenotyper compatibility and limitations matrix

Matrix revision: 2026-09-03
Applies to: current development branch; use the commit and container digests recorded by each run as the authoritative software identity.

This matrix separates three questions that must not be conflated:

| Evidence term | Meaning |
|---|---|
| Intake accepted | The workflow recognizes the input and attempts conversion or validation. |
| Technically demonstrated | The current implementation completed a documented software or real-image check. |
| Scientifically validated | Performance was measured against a prespecified independent reference standard on an appropriate held-out cohort. |

No current CellPhenotyper route has completed a multi-site external biological validation. A successful run is evidence of technical execution, not accuracy or clinical validity.

## Intended and unsupported use

| Area | Current contract | Evidence and limitation |
|---|---|---|
| Primary material | Brightfield H&E tissue images | This is the intended image domain. Organ- and specimen-specific accuracy remains unestablished. |
| Primary use | Research-only exploratory segmentation, phenotype representation, tissue-domain discovery, virtual staining, or outcome research | One primary `analysis_intent` must be selected. These are separate claims and observation units. |
| Clinical use | Unsupported | Outputs are not diagnoses, treatment recommendations, calibrated assays, or validated clinical probabilities. |
| Other stains | Unsupported by the scientific contract | IHC, special stains, fluorescence and multiplex images may be useful as external references but are not valid H&E pipeline inputs without a separate validation. |
| Image channels | Three-channel RGB-compatible color | JPEG-in-TIFF YCbCr storage is accepted because standard WSI readers decode it to RGB. Grayscale, multispectral and arbitrary multichannel analysis inputs are not supported by the strict default. |
| Image dimensionality | One two-dimensional brightfield scene per resolved sample | Z-stacks, time series, focal stacks, volumetric images and serial-section registration are not validated routes. |
| Human oversight | Required for biological interpretation | GrandQC exclusions, detector disagreements, virtual markers, clusters, MedSAM edits and outcome scores require review appropriate to the intended claim. |

## Specimen compatibility

| Specimen or preparation | Intake status | Scientific status | Required action |
|---|---|---|---|
| FFPE resection H&E | Intended | No multi-site reference-standard validation | Validate per organ, laboratory, scanner and endpoint. |
| Core or endoscopic biopsy H&E | Technically plausible, not separately gated | Not validated | Evaluate small-fragment loss, edge effects, crush artifact and limited tissue area. |
| Tissue microarray H&E | Optional automatic TMA route | TMA decision and core assignment not reference-standard validated | Review the TMA decision and core polygons; validate regular, irregular, missing, tilted and fragmented cores. |
| Frozen-section H&E | Intake may accept the file | Unsupported scientifically | Establish a separate frozen-section cohort and artifact policy before use. |
| Decalcified tissue | Intake may accept the file | Unsupported scientifically | Validate morphology, stain shift and cell detection separately. |
| Cytology, blood film or cell block | Intake may accept an RGB file | Unsupported scientifically | Do not interpret tissue-domain or cell-model outputs without a dedicated study. |
| Non-human tissue | Intake may accept the file | Unsupported scientifically | Upstream model domains and morphology differ; a species-specific benchmark is required. |
| Synthetic, composite or screenshot image | Extension may be accepted | Unsupported as a production WSI | Strict MPP/pyramid checks should normally fail; do not bypass them without independently verified provenance. |

No organ type currently carries a CellPhenotyper accuracy claim. The cached Visium HD breast image supports limited technical checks only and must not be treated as a breast-cancer validation cohort.

## Physical resolution and scale

| Property | Strict default | Interpretation |
|---|---|---|
| Accepted native MPP sanity interval | 0.05 to 0.50 µm/px | Values outside this interval fail. The upper limit bounds upsampling to 2x for 0.25-µm cell-model inputs. |
| MPP anisotropy | Maximum 5% relative X/Y difference | Larger differences fail rather than silently assuming square pixels. |
| Metadata-source agreement | Maximum 2% conflict among OME, TIFF-tag and OpenSlide/pyvips sources | Conflicting sources fail unless an independently verified explicit override is recorded. |
| Conversion drift | Maximum 2% between source and converted effective MPP | Larger drift fails. |
| Cell detectors, GigaTIME and UNI-2 | Resampled to 0.25 µm/px where configured | Upsampling standardizes tensor geometry but cannot recover missing spatial detail. |
| TITAN | 0.5 µm/px with 512-pixel patches | Applies only to the explicitly selected tissue section. |
| GrandQC artifact checkpoint | Auto-selects 1.0, 1.5 or 2.0-MPP model by source MPP | Checkpoint selection is recorded, but accuracy across the accepted MPP interval still requires stratified validation. |

Magnification labels such as 20x or 40x are not accepted as a substitute for physical pixel size. Use measured or trustworthy image metadata; an MPP override is a visible exception, not a metadata-repair mechanism.

## File and reader compatibility

| Input family | Intake accepted | Current evidence level | Important limitations |
|---|---:|---|---|
| Pyramidal OME-TIFF (`.ome.tif`, `.ome.tiff`) | Yes | Real-image and repository smoke checks | Must resolve RGB layout, pyramid and MPP under strict defaults. |
| BigTIFF/BTF (`.btf`) | Yes | Conversion path exercised on public Visium HD input | File suffix does not prove valid OME metadata or biological suitability. |
| Aperio SVS (`.svs`) | Yes | Reader/conversion route available | No scanner-model or site generalization claim. |
| Hamamatsu NDPI (`.ndpi`) | Yes | Reader/conversion route available | No scanner-model or site generalization claim. |
| Zeiss CZI (`.czi`) | Yes | Multi-region routing implemented | Region-specific files should use `<image>.czi - ScanRegionN.geojson`; multi-plane and arbitrary scene semantics are not validated. |
| Olympus VSI (`.vsi`) | Yes | Real VSI conversion and explicit-RGB runtime checks | Requires the sibling `_<sample>_` companion directory. Three planar brightfield channels are joined into canonical RGB; source order defaults to `RGB` and is configurable through `convert_channel_order`. |
| Leica SCN and MRXS (`.scn`, `.mrxs`) | Yes | Parser route only unless a run records otherwise | Reader support and scientific validation must be demonstrated on representative files. |
| Olympus VMS/VMU (`.vms`, `.vmu`) | Yes | Parser route only unless a run records otherwise | Companion-file layouts and reader behavior require explicit testing. |
| Generic TIFF (`.tif`, `.tiff`) | Yes | Conversion/validation route available | Ambiguous axes, photometric interpretation and MPP are rejected by strict QC. |
| PNG/JPEG (`.png`, `.jpg`, `.jpeg`) | Yes | Intended for small controlled inputs and debugging | Usually lacks trustworthy physical calibration and a source pyramid; screenshots and composite figures are unsupported. |

“Intake accepted” means the suffix is recognized. Actual decoding depends on the pinned Bio-Formats, OpenSlide, tifffile and pyvips runtime and must be verified by the source and converted reports in `01_input`.

## ROI compatibility

| ROI property | Current behavior |
|---|---|
| Omitted ROI | A full-image level-0 pixel ROI is generated, then intersected with GrandQC clean tissue. |
| Geometry | Non-empty GeoJSON `FeatureCollection` containing `Polygon` or `MultiPolygon`. |
| Coordinate space | Level-0 image pixels. Geographic/world CRS declarations are rejected. |
| Integrity | Non-finite, self-intersecting, empty or out-of-bounds geometry fails under the default policy. Geometry is not silently repaired or clipped. |
| Holes | Preserved and counted. |
| Identity | ROI SHA-256 and converted-image SHA-256 are bound in the ROI QC report. |
| Legacy metadata | Missing coordinate-space or source-image declarations produce visible warnings; explicit declarations are preferred. |
| Unsupported geometry | Point, MultiPoint, LineString and nonpolygonal annotation-only files are not analysis ROIs. |

## Route-specific evidence ceiling

| Route or component | Current evidence | Claim ceiling until further validation |
|---|---|---|
| Input conversion and ROI | Hash-, geometry-, layout- and MPP-checked; real and synthetic technical checks | Reproducible input preparation, not biological validity |
| GrandQC | Model integration, overlap blending, clean-support propagation and QC outputs implemented | Artifact/tissue screening for research review; no CellPhenotyper accuracy claim |
| StarDist and CellViT++ broad-detector fusion | Role-aware implementation audited on one archived WSI and covered by tests | Inter-detector agreement only; no true-cell-count or segmentation-accuracy claim |
| HoVer-Net MoNuSAC | Scoped support with original cell-type evidence retained separately | Not an exhaustive cellularity estimate and not a phenotype consensus |
| TMA detection | Heuristic decision, spot GeoJSON and cell assignment implemented | Candidate core map requiring review; no automated TMA accuracy claim |
| GigaTIME | Five-channel GPU output continuity demonstrated on one WSI; seam gate implemented | Uncalibrated virtual-marker scores; no measured-abundance or marker-accuracy claim |
| Cell-centred UNI-2 | Calibrated whole-context and central 90-pixel token-subset features | Contextual morphological representation; clusters are not established cell types |
| Grid UNI-2 | Adjacent inner cores with overlapping context; route comparison implemented | Exploratory tissue domains; no preferred-route claim |
| KODAMA and Leiden | Repeated seeds, assignment margins, stability, spatial metrics and abstention implemented | Exploratory partitions requiring independent biological interpretation |
| Cell-mask growth | Restricted to sparse cell-centred masks | Deterministic support propagation, not learned ground truth |
| MedSAM and boundary refinement | GrandQC-constrained edits and native-resolution QC implemented | Candidate refined boundaries; no expert-level accuracy claim |
| TITAN | Official section-embedding path and provenance implemented | Research representation of a selected section |
| PathoFMPred | Protected cancer-specific model path and research report implemented | TCGA-derived research estimates, not calibrated probabilities or clinical predictions |

## Hardware and execution compatibility

| Environment | Current status | Limitation |
|---|---|---|
| Linux amd64 with NVIDIA GPU | Primary high-throughput path | Exact CUDA compatibility depends on the pinned image and GPU compute capability. |
| Linux amd64 CPU | Supported only for CPU-compatible routes | Consensus detection and other GPU-only configured stages must fail rather than silently change the scientific method. |
| macOS with Docker CPU | Development and small-fixture path | Not a production WSI performance target; Apple GPU access is not equivalent to NVIDIA container execution. |
| Linux arm64 CPU | Architecture-specific image available | Package and model coverage must be checked for the selected route. |
| Linux arm64 NVIDIA GPU | Conditional | Requires an explicitly compatible arm64 CUDA/PyTorch image; legacy images may not support newer compute capabilities. |
| Multi-GPU host | Shared per-device admission implemented | Output equivalence and scaling across GPU classes still require a formal benchmark. |

Automatic hardware tuning may change batch size, worker count, tile size within validated algorithmic constraints, and concurrency. It must not change `analysis_intent`, `cell_detection_mode`, observation sampling, model identity, clustering policy, or output interpretation.

## Required evidence before expanding support

1. Lock the intended use, endpoint, statistical unit, reference standard and acceptance margin.
2. Use patient-level independent splits, including a genuinely external site and scanner for generalization claims.
3. Include failure-enriched cases and report exclusions and non-evaluable runs.
4. Measure subgroup performance where demographic and acquisition metadata permit.
5. Compare results with interobserver variability and report uncertainty intervals at patient or slide level.
6. Freeze the pipeline commit, container digest, model revisions and parameters before testing.
7. Update this matrix only after evidence is archived and linked to the relevant claim.
