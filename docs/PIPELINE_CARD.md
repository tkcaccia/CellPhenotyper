# CellPhenotyper pipeline card

Revision: 2026-09-03

## Summary

CellPhenotyper is a research-use Nextflow workflow for explicitly selected analyses of brightfield H&E tissue images. It integrates image/artifact QC, nuclear-instance analysis, virtual marker prediction, pathology foundation-model representations, unsupervised clustering and spatial reconstruction. Its breadth does not make it a universal or clinically validated system.

The primary contribution is controlled orchestration: physical calibration, shared tissue/ROI support, route-specific observation units, model provenance, uncertainty, restartability and review artifacts. Upstream models retain their own training domains, licenses and limitations.

## Intended users and uses

| Item | Current contract |
|---|---|
| Users | Computational pathology researchers working with pathology and statistical collaborators |
| Input | Two-dimensional three-channel RGB-compatible brightfield H&E tissue image with trustworthy physical calibration |
| Primary use | Research-only exploratory multimodal characterization |
| Selectable intents | Nuclear instance segmentation, contextual cell phenotyping, grid tissue-domain discovery, virtual staining, or section-level outcome research |
| Required review | Technical operator review for every run; pathology and study-design review appropriate to the intended claim |
| Clinical status | Not validated for diagnosis, prognosis, treatment selection, screening or patient management |

See `docs/ANALYSIS_ROUTE_GUIDE.md`, `docs/COMPATIBILITY_MATRIX.md` and `docs/HUMAN_REVIEW_POLICY.md` before selecting a route.

## Components and model roles

| Component | Configured identity or scope | Pipeline role | Principal limitation | Access/provenance status |
|---|---|---|---|---|
| GrandQC | Official tissue model plus 1.0/1.5/2.0-MPP artifact checkpoints | Defines normal-tissue support and artifact exclusions | CellPhenotyper has not completed a multi-cohort expert artifact benchmark; fallback behavior requires review | Checkpoint paths and selected MPP are recorded; upstream license is CC BY-NC-SA 4.0 and restricts commercial use |
| StarDist | `2D_versatile_he` | Broad-scope H&E nuclear candidate detection | Generic pretrained scope is not specimen-specific accuracy | Model name and cache are recorded; release inventory still needs immutable model-file hash coverage |
| HoVer-Net | Official fast MoNuSAC checkpoint | Scoped typed-nucleus support | MoNuSAC positive classes are not an exhaustive nucleus ontology and must not define total cellularity | Checkpoint SHA-256 and upstream Git revision are recorded per run |
| CellViT++ | `cellvit==1.0.9`, HIPT model, PanNuke taxonomy | Broad-scope typed nuclear candidate detection | PanNuke classes are not interchangeable with MoNuSAC and are not a reference standard | Runtime is pinned; selected checkpoint readability is checked; complete release license inventory remains required |
| Role-aware fusion | CellPhenotyper custom | Canonical instance geometry from StarDist/CellViT++ agreement; MoNuSAC remains scoped evidence | Agreement is not accuracy and unmatched cells are abstentions, not proven false detections | Full component/alignment audit and detector-specific labels are published |
| GigaTIME | `prov-gigatime/GigaTIME` | Virtual mIF channel prediction and region quantification | Scores are uncalibrated predictions from H&E, not measured protein abundance | Resolved snapshot and checkpoint hash are recorded; the conflicting upstream license file/model-card metadata remains a release blocker |
| UNI-2 | `MahmoodLab/UNI2-h` | Cell-centred or spatial-grid morphology representation | Route, MPP, context and pooling definition change the scientific observation; route preference is unvalidated | Gated model cache, resolved snapshot, checkpoint hash and exact run configuration are recorded |
| KODAMA | Installed `tkcaccia/KODAMA` R package | Nonlinear representation after PCA | Latent axes are not biological endpoints and warnings/errors invalidate interpretation | Package version and remote SHA are logged when available |
| Leiden | R/igraph graph clustering with inverse-distance landmarks | Unsupervised partition of KODAMA coordinates | Cluster number and resolution must not be selected visually on a test cohort | Seeds, graph parameters, landmark strategy, stability and assignments are recorded; forced counts require acknowledgement and are stamped sensitivity-only |
| MedSAM | ViT-B `medsam_vit_b.pth` | External tissue-border refinement within a constrained editable band | Not a first-pass cluster model or ground truth; may change thin/isolated tissue | Git revision and checkpoint hash are recorded; exact checkpoint licensing still requires verification |
| Image-guided watershed | CellPhenotyper custom deterministic step | Internal inter-cluster boundary alignment to native H&E gradients | H&E gradients do not prove biological domain boundaries | Parameters and before/after metrics are recorded |
| TITAN/CONCH | Gated `MahmoodLab/TITAN`, CONCH v1.5 encoder | One 768-dimensional representation of a selected section | Section selection and model domain constrain interpretation | The requested revision and resolved immutable snapshot plus all snapshot weight hashes are recorded |
| PathoFMPred | Access-controlled private cancer-specific registry | Research endpoint estimates from the TITAN vector | TCGA-derived estimates are not calibrated clinical probabilities and require external cancer-specific validation | Installed package/model databases are hashed; absent private source/license metadata fails the release gate |

The authoritative description of wrapper modifications is `UPSTREAM_DIFFS.md`. Model terms must be checked against the exact checkpoint and revision used by a release; a repository URL alone is not sufficient license provenance.

## Observation units

| Route | Observation | Inferential warning |
|---|---|---|
| Cell segmentation | Nuclear instance | Multiple nuclei from one slide are not independent patients. |
| Cell phenotyping | Nucleus-centred context tile and central inner-square representation | The representation includes local context and is not a purified molecular cell state. |
| Tissue domains | Regular tissue-grid core with overlapping context | A grid observation is not a cell and is not conditioned on cell detection. |
| Virtual staining | Image pixel plus nuclear/cytoplasmic measurement region | Values remain model predictions even after region aggregation. |
| Outcome research | Selected connected tissue section | Patient, not section patch, is the primary inferential unit for clinical association. |

## Current evidence

| Evidence type | Status |
|---|---|
| Unit and integration tests | Passing suites cover route contracts, image/ROI QC, detector fusion, uncertainty, model provenance, GPU policy, restart boundaries and report generation |
| Input/ROI technical checks | Real and synthetic checks verify hashes, image layout, MPP conflicts, pyramids and immutable polygon geometry |
| Detector-fusion behavior | Role-aware policy audited on one archived Visium breast WSI; this verifies implementation behavior, not true-cell accuracy |
| Detector validation tooling | A standalone reference-standard protocol and benchmark implement adjudicated instance matching, patient/slide bootstrap intervals, paired detector comparisons, subgroup summaries and reviewable error polygons; no reference cohort has yet been evaluated |
| GigaTIME technical QC | Corrected five-channel GPU run on one WSI passed execution-block, archived-boundary and full patch-grid seam tests; new runs also emit sampled all-channel score distributions, tissue/background contrast and correlation without treating these as biological validation |
| GigaTIME validation tooling | A standalone registered-marker protocol and benchmark enforce patient-level inference, registration/matching gates, cellularity-adjusted correlations, spatial endpoints, locked binary thresholds, input hashes and discordance review; no measured-marker cohort has yet been evaluated |
| Clustering uncertainty | Repeated seeds, ARI, assignment margins, abstention, spatial coherence and blinded review packets are implemented |
| UNI-2 route validation tooling | A paired reference-domain protocol and benchmark implement label-invariant partition metrics, abstention-aware coverage, physical boundary error, patient bootstrap differences, prespecified decisions and blinded QC; no independent route-comparison cohort has yet been evaluated |
| UNI-2 dual-route execution | `both` now executes independent grid and `<sample>__cells` branches through clustering, route-appropriate mask handling, MedSAM and GeoJSON; a full downstream stub DAG passes, but this is orchestration evidence rather than biological validation |
| Biological accuracy | Not established for any complete route by an independent reference-standard cohort |
| External generalization | No complete multi-site/scanner validation |
| Clinical utility | Not evaluated and not claimed |

## Known failure modes

1. Incorrect or conflicting MPP changes physical context for every learned stage.
2. GrandQC may exclude biologically relevant low-cellularity, necrotic, hemorrhagic, folded or unusual tissue.
3. Detector scope and taxonomy differences can create large count and label discrepancies.
4. Dense overlap can duplicate morphology information and inflate the apparent sample size.
5. GigaTIME can generate technically smooth but biologically inaccurate virtual-marker fields.
6. Unsupervised clusters can follow stain, scanner, tissue preparation, artifact or site effects rather than biology.
7. Landmark sampling and KNN transfer can underrepresent rare states or blur boundaries.
8. MedSAM and watershed refinement can alter thin structures, isolated components and topology.
9. TMA heuristics may fail on irregular, fragmented, tilted or missing-core arrays and non-TMA mimics.
10. Section selection or an incorrect cancer code can make TITAN/PathoFMPred output scientifically irrelevant despite successful execution.
11. Auto-selected hardware parameters may expose untested numerical or throughput differences across GPU classes.
12. A green execution status can coexist with missing biological evidence; the validation-readiness claim ceiling remains authoritative.

## Uncertainty and abstention

The pipeline distinguishes hard failures, review warnings, descriptive agreement, uncalibrated scores and abstentions. Canonical cell fusion excludes components without required broad-detector support. Clustering retains raw assignments while permitting `interpretable_cluster` to abstain for weak or unstable observations. `uncertainty_reason` separates assignment ambiguity from seed instability, while KODAMA figures, tissue-space figures, observation-level GeoJSON, categorical rasters and H&E overlays expose the excluded observations. `00_execution/uncertainty_register.{json,tsv}` also records where uncertainty is not quantified, preventing omission from being read as confidence. `00_execution/specimen_atlas.{html,json}` groups bounded review evidence by specimen while preserving route, observation unit, output ID and interpretation semantics; absent layers are explicit. Route comparison is descriptive and does not select a preferred biology. Human review can accept, limit or reject a run but cannot convert missing reference-standard evidence into validation.

## Fairness and subgroup limitations

No fairness claim is supported. Relevant subgroup axes may include patient demographics, geography, organ, disease subtype, specimen preparation, stain batch, laboratory, site, scanner, compression and image quality. These metadata must be collected under an approved protocol and evaluated at patient or slide level without exposing protected information.

## Release and revalidation triggers

Revalidation is required after changes to a learned model or checkpoint, physical scale, tile/stride geometry, image normalization, detector threshold, fusion rule, feature pooling, KODAMA inputs, clustering/landmark policy, MedSAM edit envelope, section-selection logic, runtime library affecting inference, container architecture or supported specimen/scanner claim.

A manuscript or archived study must use an immutable pipeline commit, container digest, model revisions/checkpoint hashes, parameter file and study manifest. `00_execution/model_inventory.{json,tsv}` is the run-level gate: a mutable requested revision is acceptable only when the exact resolved snapshot commit is recorded, but unresolved model-license terms remain blocking. Any mutable model reference that cannot be resolved and recorded immutably must be replaced before a formal release claim.

## Required next evidence

1. Expert-labelled GrandQC artifact benchmark.
2. Expert-labelled nuclear instance and boundary benchmark across compartments.
3. Registered H&E/multiplex GigaTIME validation.
4. Prespecified cell-versus-grid UNI-2 biological comparison.
5. Training-only cluster-parameter selection followed by locked held-out evaluation.
6. Expert MedSAM outer- and internal-boundary benchmark with baseline ablations.
7. External-site, scanner and subgroup evaluation.
8. Multi-GPU-class performance and output-equivalence benchmark.
