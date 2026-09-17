# Choosing a CellPhenotyper analysis route

CellPhenotyper contains several analyses, but they do not answer the same scientific question. Select one primary `analysis_intent` per run before examining results. Hardware availability may change safe batch sizes and concurrency, but it must never change the scientific route, detector population, observation unit, or interpretation.

## Route decision

| Primary question | Required intent and route | Observation unit | Main outputs to inspect | Evidence needed for a strong claim | Do not claim |
|---|---|---|---|---|---|
| Where are the nuclei? | `analysis_intent=cell_segmentation`; normally `cell_detection_mode=consensus` | Nuclear instance | `03_stardist`, `03b_hovernet_monusac`, `03c_cellvitpp`, `03d_cell_consensus` | Expert-labelled nuclei from held-out slides, with detection and boundary metrics | Detector agreement proves accuracy, or MoNuSAC counts total cellularity |
| Which H&E morphology contexts distinguish detected cells? | `analysis_intent=cell_phenotyping`; requires `uni2_sampling_mode=cells` | Contextual cell: a nucleus-centred 224-pixel tile and its central 90-pixel inner square | `09_embeddings`, `10_kodama`, `11_clustering`, `12_cluster_mask`, `13_grown_tissue`, `14_medsam_refine_tissue`, `15_cluster_geojson` | Repeated-seed stability plus blinded pathological review and an independent biological endpoint | An unsupervised cluster is a known cell type, diagnosis, or independent cell replicate |
| Which spatial tissue domains are present without conditioning sampling on nuclei? | `analysis_intent=tissue_domain_discovery`; requires `uni2_sampling_mode=grid` or `both` | Tissue-grid core | `09_grid_tiles`, `09_embeddings`, `10_kodama`, `11_clustering`, `12_cluster_mask`, `14_medsam_refine_tissue`, `15_cluster_geojson` | Expert region annotations or a prespecified independent spatial endpoint on held-out slides | Grid tiles are cells, or `13_grown_tissue` should be applied to the dense grid mask |
| What virtual marker patterns does GigaTIME predict? | `analysis_intent=virtual_staining`; requires `gigatime_enable=true` | Image pixel and quantified nuclear/cytoplasmic region | `05_gigatime`, including pyramidal OME-TIFF, Zarr, compartment-aware quantification, seam QC and marker-score distribution QC | Registered measured multiplex images with marker-wise calibration and held-out evaluation | Virtual scores are measured protein abundance, calibrated probabilities, or clinical assays |
| What section-level research prediction is produced? | `analysis_intent=outcome_prediction`; requires `pathofmpred_enable=true` and an explicit cancer type | Selected tissue section; patient is the inferential unit in a study | `16_neoplastic_section`, `17_titan`, `18_pathofmpred` | Locked patient-level protocol and external cohort appropriate to the cancer type | A TCGA-derived score is a calibrated probability, diagnosis, or validated clinical prediction |
| Is broad hypothesis generation the only objective? | `analysis_intent=exploratory`; choose `cells`, `grid`, or `both` explicitly | Route-dependent | All enabled outputs plus `00_execution` | Transparent exploratory declaration; confirmatory testing on independent data | Exploratory findings are validation or proof of generalization |

## Practical decision tree

1. If the endpoint is a patient or outcome, use `outcome_prediction` and design the study at patient level.
2. If the endpoint is a predicted molecular image or marker value, use `virtual_staining` and plan registered multiplex validation.
3. If the endpoint is nuclear detection, use `cell_segmentation` and stop at the detector or consensus output unless downstream analysis is separately justified.
4. If the endpoint concerns cells and their immediate morphology, use `cell_phenotyping` with `uni2_sampling_mode=cells`.
5. If the endpoint concerns continuous tissue architecture, use `tissue_domain_discovery` with `uni2_sampling_mode=grid`.
6. If the aim is to compare cell-centred and grid representations, use `exploratory` with `uni2_sampling_mode=both`, prespecify comparison endpoints, and do not choose the winner from the most attractive plot.

`uni2_sampling_mode=both` makes the grid route primary and executes a second complete cell-centred route under the `<sample>__cells` namespace. Both routes run independent UNI-2, KODAMA, Leiden, mask, MedSAM and GeoJSON stages. Only the cell route performs sparse-mask tissue growth; the dense grid mask goes directly to MedSAM. This engineering symmetry enables comparison but does not establish that the two routes answer the same biological question or that either route is superior.

## What the routes share

All routes begin with immutable input identity, physical-resolution QC, GrandQC, and a shared analysis crop. Without an input GeoJSON, the crop contains GrandQC clean tissue. With an input GeoJSON, it contains the intersection of clean tissue and the validated ROI. The routes then diverge:

| Component | Scientific role |
|---|---|
| GrandQC | Defines usable image support. Artifact/background exclusion is a gate, not a biological label and not an accuracy estimate. |
| Multi-detector fusion | Builds canonical nuclear instances from broad-detector agreement while retaining MoNuSAC as scoped evidence. Detector taxonomies stay separate. |
| Cell-centred UNI-2 | Characterizes detected cells in local morphological context. Whole-tile and central inner-square features come from one calibrated forward pass. |
| Grid UNI-2 | Samples tissue independently of nucleus centres. Overlapping context tiles have adjacent 90-pixel inner cores. |
| KODAMA and Leiden | Produce latent coordinates and unsupervised partitions. Stability, uncertainty, spatial coherence, and blinded review are required for interpretation. |
| Mask growth | Applies only to sparse cell-supported masks. It is intentionally bypassed for the already-dense grid mask. |
| MedSAM | Refines borders within clean-tissue support. Edit maps and native-resolution QC crops must be reviewed; model edits are not ground truth. |

## Before starting a study

1. Define one primary endpoint, one statistical unit, one direction of effect, and one acceptance threshold.
2. Split development, tuning, internal-test, and external-test subjects at patient level before optimization.
3. Define the reference standard, annotator expertise, blinding, adjudication, and inter-rater analysis.
4. Record supported tissue, stain, scanner, magnification, file format, and exclusion criteria.
5. Use `resources/study_manifest.template.json`; set `evidence_gate_mode=fail` for a locked testing run.
6. Keep tiles, cells, ROIs, cores, and sections as repeated observations within a slide or patient, not independent biological replicates.
7. Freeze pipeline release, container digest, model revisions, parameters, random seeds, and the study manifest before final testing.

## Before interpreting a run

| Review order | Required evidence |
|---|---|
| 1. Run identity | `00_execution/analysis_contract.json`, `validation_readiness.json`, hardware plan, trace, software/model provenance, and output checksums |
| 2. Input | Source/converted SHA-256, RGB layout, pyramid, MPP source and conflict status in `01_input` |
| 3. ROI and support | ROI geometry/image association in `06_roi`; GrandQC masks and artifact overlays in `02_grandqc` |
| 4. Cells | Per-detector counts, spatial coverage, unmatched detections, geometry agreement, and abstentions in `03d_cell_consensus` |
| 5. Virtual markers | OME channel identity, MPP, seam gate, native-resolution appearance, and registered-reference status in `05_gigatime` |
| 6. Representations | Observation type, tile geometry, coverage, route comparison, and embedding completion manifests in `09_embeddings` and `10_kodama` |
| 7. Clusters | Repeated-seed agreement, assignment margins, abstentions, spatial coherence, marker-proxy limits, and blinded review in `11_clustering` |
| 8. Spatial output | Mask topology, GrandQC leakage, MedSAM edits, native-resolution crops, and GeoJSON alignment in stages 12-15 |
| 9. Predictions | Section selection, cancer-type configuration, model provenance, and external-validation status in stages 16-18 |

A successful Nextflow run establishes technical completion only. It does not establish biological accuracy, clinical validity, fairness, scanner generalization, or suitability for patient care.
