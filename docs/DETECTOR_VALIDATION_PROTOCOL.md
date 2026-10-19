# Reference-standard detector validation protocol

## Purpose

This protocol addresses the different cell counts produced by StarDist, CellViT++, HoVer-Net MoNuSAC and role-aware instance fusion. Pairwise detector agreement is not accuracy. The benchmark requires an expert reference standard and evaluates **instance geometry only**. Phenotype-label reconciliation is a separate study because MoNuSAC and PanNuke do not share an exhaustive ontology.

The design must be reviewed and frozen before internal-test or external-test annotations are inspected.

## Intended claim

Primary intended use: research-grade nucleus instance segmentation in H&E whole-slide image regions at a declared physical resolution.

Permitted conclusion: performance of each fixed detector configuration and the fixed fusion rule within the prespecified tissues, compartments, quality strata, sites and scanners.

Not permitted: clinical accuracy, exhaustive cell typing, calibrated cell-type probabilities, performance outside the sampled domain, or evidence that agreement among models is a reference standard.

## Statistical units and splits

- The biological unit is the patient. Slide-level resampling is appropriate only when each patient contributes one slide.
- All ROIs from a patient, block, TMA or slide must remain in one split.
- `development` may be used to define thresholds and the fusion rule.
- `internal_test` is locked before evaluation and must not be used for tuning.
- `external_test` must come from a genuinely independent site or scanner and remain untouched until the analysis is frozen.
- Cells are outcomes within a sampling unit, not independent biological replicates.

## Reference standard

1. Select ROIs before viewing detector outputs, using a prespecified stratified sampling procedure.
2. Include relevant tissue compartments and failure-enriched strata such as crowding, blur, folds, necrosis, weak staining, stain extremes and boundary regions.
3. Have at least two qualified annotators independently delineate every nucleus while blinded to detector identity and output.
4. Adjudicate disagreements using a documented rule and record `reference_status=adjudicated` at manifest and feature level.
5. Keep phenotype labels optional and separate. The geometry benchmark does not score them.
6. Record annotation software, version, display scale, physical calibration, instructions, annotator role, blinding and adjudication date.
7. Determine sample size from the desired confidence-interval width or a prespecified noninferiority margin before annotation begins. Do not choose the ROI count from an apparent result.

Each reference GeoJSON must be a `FeatureCollection` containing one Polygon or MultiPolygon per nucleus. Every feature requires a unique `instance_id` and `reference_status: "adjudicated"`. Coordinates must be crop level-0 pixels matching the prediction frame exactly.

## Prespecified endpoints

Recommended primary endpoint: panoptic quality using one-to-one IoU >= 0.50, aggregated within the locked test split.

Secondary endpoints:

- precision, recall and F1 at IoU 0.50 and 0.75;
- detection and segmentation quality components of panoptic quality;
- aggregated Jaccard index and aggregated instance Dice;
- matched-instance IoU and Dice;
- centroid, Hausdorff and average symmetric boundary error in micrometres;
- absolute and relative count error;
- split and merge rates using a prespecified overlap threshold;
- performance by tissue compartment, image-quality stratum, site and scanner;
- failure rates and abstentions, including cases excluded by GrandQC or fusion policy.

Do not select an IoU threshold, detector threshold or fusion rule from a locked-test result. Label every post hoc analysis exploratory.

## Manifest

Use `resources/detector_validation_manifest.template.csv`. It is long-form: each row describes one detector applied to one ROI. Repeated rows for an ROI must have identical reference and acquisition metadata.

Required fields include study, patient, slide, ROI, split, site, scanner, tissue, compartment, quality stratum, MPP, crop dimensions, coordinate space, reference path, detector, analysis condition, prediction path and prediction format. Use `condition` to distinguish prespecified settings such as a common fixed MPP and a detector-recommended MPP; never encode these as if they were different detector identities. Patient and slide IDs are namespaced by study internally, but must still be stable and unique within that study.

Supported prediction formats:

- `labels_tif`: bounded 2D integer label mask, intended for StarDist validation ROIs;
- `cell_json`: normalized HoVer-Net or CellViT++ JSON containing a `cells` list and contours;
- `geojson`: one Polygon/MultiPolygon feature per predicted instance, including consensus output.

## Execution

```bash
python bin/benchmark_cell_detectors.py \
  --manifest detector_validation_manifest.csv \
  --outdir detector_validation_results \
  --iou-thresholds 0.50,0.75 \
  --primary-iou-threshold 0.50 \
  --bootstrap-unit patient_id \
  --bootstrap-replicates 2000 \
  --seed 2026
```

The tool refuses patient leakage across splits, inconsistent ROI metadata, missing physical calibration, coordinate-space ambiguity, non-adjudicated references, reference features without explicit instance IDs, duplicate detector/ROI rows, invalid reference polygons and oversized label-mask inputs. Physical distance metrics scale the two coordinate axes separately by `mpp_x` and `mpp_y`; anisotropic pixels are not approximated with a mean MPP.

## Outputs

- `detector_metrics_per_roi.csv`: metrics for each detector, ROI and IoU threshold.
- `detector_metrics_aggregate.csv`: split-level estimates and patient/slide bootstrap confidence intervals.
- `detector_metrics_by_stratum.csv`: descriptive tissue, compartment, quality, site and scanner results; strata with few independent units must not be overinterpreted.
- `detector_paired_differences.csv`: paired aggregate detector differences on common ROIs with patient/slide bootstrap confidence intervals.
- `detector_scale_sensitivity.csv`: paired within-detector differences between prespecified physical-resolution or other analysis conditions.
- `instance_outcomes.csv`: matched, false-negative and false-positive instances at the primary threshold; phenotype columns are explicitly unscored.
- `error_cases.geojson`: false-negative reference and false-positive prediction polygons for review.
- `detector_benchmark_provenance.csv`: paths, hashes, formats and instance counts.
- `detector_benchmark_summary.json`: evaluation contract and claim limits.
- `detector_benchmark_report.html`: review-first summary.

## Acceptance and reporting

The acceptance margin, expected direction, multiplicity policy and handling of failed slides must be written before evaluation. Report every attempted ROI, including failures and exclusions. Present patient/slide-level confidence intervals and paired detector differences; do not report only pooled cell-level percentages. Subgroup tables are descriptive unless the subgroup hypotheses, minimum independent-unit counts and multiplicity handling were prespecified.

The manuscript must separate reference-standard segmentation accuracy, detector agreement, instance-fusion acceptance or abstention, and detector-specific phenotype evidence. These quantities answer different questions and cannot substitute for one another.
