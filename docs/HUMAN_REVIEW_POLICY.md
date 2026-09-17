# Human review policy

Revision: 2026-09-03

CellPhenotyper is a research pipeline. Human review is not a final decorative check and must not be used to select the most attractive result after seeing all alternatives. It is a documented decision about whether the input, quality gates and outputs are usable for the analysis intent declared before execution.

## Roles

| Role | Minimum responsibility |
|---|---|
| Pipeline operator | Verifies run identity, input/ROI provenance, process completion, hardware plan, failed gates and output integrity. |
| Pathology reviewer | Evaluates tissue, artifacts, nuclear instances, biological plausibility and spatial boundaries within their area of expertise. |
| Imaging or assay reviewer | Evaluates registration and measured-reference quality for virtual-marker validation. |
| Statistical reviewer | Confirms patient-level splits, statistical unit, endpoint, uncertainty method and absence of test-set tuning. |
| Adjudicator | Resolves prespecified disagreements without access to protected outcomes when blinding requires it. |

One person may hold multiple roles in exploratory work, but the report must state this. Accuracy or validation claims require review independent of pipeline development wherever feasible.

## Permitted decisions

| Decision | Meaning |
|---|---|
| `pending` | Review is incomplete; biological interpretation is not authorized. |
| `accepted_for_declared_research_use` | Required checks passed for the declared research question and evidence ceiling. This is not clinical approval. |
| `accepted_with_limitations` | The run may be used only with the recorded exclusions and claim restrictions. |
| `rejected` | At least one unacceptable failure invalidates the requested interpretation. Rerun, correct the source data, or exclude the sample under a prespecified rule. |
| `not_applicable` | The stage is outside the selected route; this must not be used to conceal a missing required output. |

Copy `resources/human_review.template.json` to `00_execution/human_review.json`, complete it without patient identifiers, and retain the reviewed evidence files. The final report and landing page surface this record when present.

## Review sequence

| Order | Evidence | Unacceptable examples |
|---|---|---|
| 1. Study declaration | `analysis_contract.json`, `validation_readiness.json`, locked protocol and cohort split | Analysis intent changed after examining results; cells or tiles treated as independent patients; test set used for tuning |
| 2. Input identity | Source/converted SHA-256, dimensions, channels, MPP sources, conversion relationship | Wrong specimen, composite image, unresolved/conflicting MPP, unexplained color conversion, missing pyramid for production WSI |
| 3. ROI | ROI QC, source-image association, native overlay, holes and bounds | Misregistration, wrong coordinate space, silent clipping/repair, relevant tissue unintentionally excluded |
| 4. GrandQC | Tissue/artifact overlays, class scores, excluded fractions and largest excluded regions | Systematic loss of interpretable tissue; artifact retained where it materially affects the endpoint; fallback used without declaration |
| 5. Cell instances | Per-detector overlays, count/scope report, disagreement cases, split/merge examples and edge cases | Coordinate shift, systematic missed compartment, severe merge/split behavior, consensus interpreted as ground truth |
| 6. TMA | TMA decision evidence, core polygons and cell-to-core assignments | Non-TMA called as TMA, merged/split cores, missing-core geometry propagated as a valid patient core |
| 7. Virtual markers | OME channels/MPP, seam QC, H&E/prediction patches and measured-reference registration | Failed seam gate, channel concatenation, wrong MPP, saturation/zero fields, biological claim without registered measured reference |
| 8. UNI-2 observations | Route metadata, calibrated tile geometry, tile audit sample, coverage and embedding completion | Grid described as cells, cell sampling described as unbiased tissue, substantial padding/blank/artifact tiles, incomplete shards |
| 9. KODAMA and clustering | Repeated-seed metrics, uncertainty, abstentions, spatial coherence, blinded packet and prespecified parameters | Cluster count chosen from appearance, instability hidden, abstentions rendered as confident labels, biological names assigned without evidence |
| 10. Spatial reconstruction | Pre-refinement mask, growth applicability, MedSAM edits, GrandQC leakage, random/high-change native crops and GeoJSON alignment | Grid mask passed through cell growth, leakage into excluded support, loss of thin/isolated tissue, topology failure relevant to endpoint |
| 11. TITAN/PathoFMPred | Selected section, cell evidence, cancer code, model provenance and research report | Wrong section or cancer registry; output reported as diagnosis, prognosis, treatment recommendation or calibrated probability |

## Sampling rules

1. Review every failed gate and every sample proposed for exclusion.
2. Review the largest and highest-change regions, not only random regions.
3. Add a reproducible random sample using a recorded seed.
4. Include low-change and apparently normal regions to estimate false-positive edits.
5. Include detector disagreements, rare clusters, boundaries, sparse tissue and artifact-adjacent regions.
6. Define counts per sample or cluster before review; do not stop once convincing examples are found.
7. Keep the blinded review packet separate from its answer key until decisions are locked.

## Route-specific sign-off

| Analysis intent | Required sign-off before biological use |
|---|---|
| `cell_segmentation` | Pathology review plus an expert-labelled held-out benchmark for any accuracy claim |
| `cell_phenotyping` | Instance QC, stable/uncertain cluster report and blinded biological interpretation at patient or slide level |
| `tissue_domain_discovery` | Grid coverage, spatial stability and expert region review; cell-mask growth must be absent |
| `virtual_staining` | Technical continuity plus registered measured multiplex validation for marker claims |
| `outcome_prediction` | Correct section/cancer configuration plus locked external patient-level evaluation |
| `exploratory` | All applicable technical QC, explicit limitations and no confirmatory or clinical language |

## Handling unacceptable failures

Do not manually edit a failed output and continue as if it were generated by the recorded pipeline. Preserve the failed result, reason, reviewer, timestamp and affected sample. Correct the input or configuration, start from the earliest invalidated stage, and retain both execution traces. A post hoc exclusion may be reported only as exploratory unless the exclusion rule was prespecified.

Warnings may be accepted only when the reviewer records why they do not affect the declared endpoint. Hard failures from input identity, MPP, ROI validity, GigaTIME seam QC, GrandQC support leakage or output completeness must not be downgraded for a confirmatory run.

## Audit and privacy

The review record should contain stable sample codes, output IDs and checksums, not names, accession numbers or other direct identifiers. Store reviewer identity according to the approved study protocol. Any exported screenshot or review packet must follow the same access and retention controls as the source slide.
