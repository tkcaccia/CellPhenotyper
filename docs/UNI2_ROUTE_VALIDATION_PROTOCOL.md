# UNI-2 Route Validation Protocol

## Question and claim boundary

Cell-centred and grid UNI-2 routes answer different scientific questions:

- the cell-centred route represents a nucleus-centred morphology-and-context observation and may support contextual cell phenotyping;
- the grid route represents a regular tissue region and may support tissue-domain discovery.

They are not interchangeable technical variants. The existing `COMPARE_UNI2_ROUTES` process measures matched-grid representation agreement, spatial structure and same-image GigaTIME proxy prediction. Those quantities are useful technical diagnostics but cannot select a biologically superior route.

This protocol compares the final tissue partitions from both routes with independently adjudicated morphological tissue-domain labels. It does not validate cell phenotypes, assign biological names to clusters or establish clinical utility.

## Reference-standard study

1. Select slides and ROIs without viewing route outputs.
2. Split at patient level into development, internal-test and genuinely external-test cohorts.
3. Have at least two qualified reviewers independently annotate morphological tissue domains while blinded to route identity, then adjudicate disagreements.
4. Record domain definitions, annotation software and version, physical scale, excluded regions, reviewer expertise, interrater agreement and adjudication date.
5. Include common compartments and failure-enriched regions, including transitions, necrosis, low cellularity, inflammation, folds, blur and stain extremes.
6. Freeze the CellPhenotyper commit, container digest, UNI-2 revision, physical scale, tile geometry, feature definition, KODAMA settings, landmark policy, clustering settings, abstention thresholds and MedSAM settings before testing.
7. Estimate sample size from patient-level paired-difference precision or a prespecified superiority/equivalence margin, not from pixel count.

The reference must be a 2D integer label TIFF aligned to the pipeline crop. Label `0` means outside evaluation; positive integers identify adjudicated domains. A separate tissue mask defines valid tissue support.

## Fair route comparison

- Run both routes on the identical immutable crop, ROI and GrandQC support.
- Use the same encoder revision and locked downstream policy. Route-specific observation geometry is expected and must be recorded in `uni2_feature_definition`.
- Compare the same processing stage for both routes, normally `medsam_refined` or a prespecified pre-MedSAM sensitivity stage.
- Do not remap arbitrary cluster IDs to reference classes. The benchmark uses label-invariant partition metrics.
- Keep abstentions as label `0` inside reference support. Report accepted-only agreement together with coverage and agreement including abstention; accepted-only performance alone can be inflated by selective omission.
- A forced cluster count is a sensitivity analysis unless the count was prespecified from independent biological knowledge.

## Prespecified endpoints

Recommended primary endpoint: patient-macro-averaged adjusted Rand index including abstention, with the route difference and bootstrap confidence interval computed on paired patients.

Secondary endpoints:

- accepted-pixel adjusted Rand index;
- normalized mutual information, homogeneity, completeness and variation of information;
- coverage and abstention fraction;
- reference-domain and predicted-cluster count;
- largest predicted-cluster fraction;
- internal-boundary precision, recall and F1 within a physical tolerance;
- mean symmetric internal-boundary error in micrometres;
- performance by site, scanner, tissue, compartment and image-quality stratum.

The manifest must predeclare `primary_metric`, `comparison_direction`, `acceptance_margin` and `decision_rule=bootstrap_ci_lower_bound`. Development results cannot produce a confirmatory pass decision.

## Execution

Use `resources/uni2_route_validation_manifest.template.csv`. Every ROI/condition requires exactly one `cells` row and one `grid` row.

```bash
python bin/benchmark_uni2_routes.py \
  --manifest uni2_route_validation_manifest.csv \
  --outdir uni2_route_validation_results \
  --bootstrap-replicates 2000 \
  --boundary-tolerance-um 8 \
  --seed 2026
```

Validation ROIs are deliberately bounded. The default refuses masks above 16 million pixels and evaluates at most one million identically sampled reference-support pixels per route pair. This controls memory and does not convert pixels into independent replicates.

## Outputs

- `uni2_route_metrics_per_roi.csv`: abstention-aware partition and physical-boundary metrics for each route and ROI.
- `uni2_route_metrics_per_patient.csv`: ROI-macro-averaged patient results.
- `uni2_route_metrics_aggregate.csv`: route estimates with patient bootstrap confidence intervals.
- `uni2_route_paired_differences.csv`: paired route differences and the prespecified primary decision.
- `uni2_route_metrics_by_stratum.csv`: descriptive site, scanner, tissue, compartment and quality results.
- `route_qc/*.png`: reference, cell-centred and grid masks displayed without label remapping.
- `route_boundary_disagreements.geojson`: largest unsupported or missed internal boundaries for blinded review, never automatic exclusion.
- `uni2_route_benchmark_provenance.csv`: full input paths and SHA-256 hashes plus immutable route definitions.
- `uni2_route_benchmark_summary.json` and `uni2_route_benchmark_report.html`: evidence contract, decision and claim ceiling.

## Interpretation

A route may be preferred only for the prespecified intended use and endpoint. A grid-route advantage for tissue-domain ARI cannot establish better single-cell phenotyping. A cell-route advantage on a selected ROI cannot establish generalization. Report failures, exclusions, abstentions, every tested condition and all secondary endpoints regardless of direction.
