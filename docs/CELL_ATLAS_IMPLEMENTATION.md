# Cell profiles and spatial atlas implementation

This is the implementation and verification ledger for the active spatial-atlas
goal. Existing research outputs and any remote experiment are inputs
for verification, not evidence that the additions below are complete.

## Requirements and acceptance evidence

| Requirement | Acceptance evidence | State |
|---|---|---|
| Canonical cell profiles | Every canonical cell retained; duplicate/foreign IDs rejected; explicit missing modalities; calibrated coordinates; separate feature blocks | Implemented; 20,563 real cells and latest mask-lineage profile/export read-back verified |
| Nuclear morphology and compartments | Physical raster geometry/appearance proxies, nucleus/ring/whole-cell distinction, tissue clipping and contamination-risk QC | Source-bound normalized-intensity/physical-lag morphology 2.0.0 implemented; 75-cell native probe verifies encoding equivalence; independent morphology/compartment accuracy pending |
| CellViT features | Actual upstream token export, source/canonical identity mapping, portable arrays/checkpoint metadata | Source-image/physical-preprocessing/population binding and actual configured-runtime identity implemented; synthetic producer/consumer contract tests pass; actual GPU feature extraction and independent validation pending |
| Spatial neighbourhoods | Per-specimen sparse radius graphs; no edges across represented support gaps; density, composition, mixing, alignment, boundary distances | Bounded native/source-raster graph backend verified; opt-in image-derived native bright-background support integrated and tested on real crop with all canonical measurements retained; default GrandQC remains low-resolution and biological gap precision/recall remains unverified |
| Cellular niches | Own-cell plus neighbourhood clustering, shared recurring niche identities, scale/seed diagnostics, isolated cells | Representation 2.0.0/shared fit and source-bound attachments implemented; opt-in array-backed own/neighbour groups and streamed specimen/cohort fitting verified against dense fixtures and actual two-specimen CPU export/resume; old real crop's one-niche result predates this representation; real cohort biology and large-scale acceptance pending |
| Hierarchical tissue domains | Immutable broad parents, optional subdomains, true wider-field/local UNI-2, scale stability, sparse tissue | Optional workflow now uses source-bound within-parent native KODAMA graphs with Leiden and graph-specific uncertainty; retained runner/consumer synthetic tests; real encoder inference, large-slide/container acceptance and histological validation pending |
| Reference atlas | Versioned cell/region references, representatives/distributions/descriptions, unmatched mapping, discovery separate | Cell/region mapping integrated with portable source-bound receipts and exact SpatialData attachments; real same-slide and independent synthetic-region round trips pass; multi-slide biological validation pending |
| Cell inspection and similarity search | Native H&E/boundary, markers/features/detectors/domains, similar cells/regions, review queue | Cell/region inspection, strict raster lineage and earlier headless browser interaction tested; shared cohort detail/source monitoring and actual HTTP/CLI checks added; new cohort visual rendering and real hierarchy review pending |
| Standard spatial export | Readable package of image, labels, polygons, profiles and transforms | Real 20,563-cell package written/read; exact original scalars/arrays/graphs/windows and cell-reference interpretations verified; actual synthetic two-specimen cell/region and shared-cohort attachment workflows passed; no new real multi-slide learned export claimed |
| Cell-to-hierarchy linkage | Exact nuclear/ring overlap fractions, uncertainty and connected-region foreign keys; preserve original profiles | Implemented with immutable stage-24 derivative, SpatialData region/relationship tables and inspector display; synthetic producer/consumer and Nextflow validation; real learned hierarchy pending |
| Measured-reference integration | Explicit assay/registration provenance, measured/predicted separation, correct cell/region unit | Import/export supports separately linked cell and registered-region assays; actual synthetic SpatialData/Nextflow round trips pass; real matched-assay validation pending |
| Clustering representation comparison | UMAP baseline versus higher-dimensional/graph clustering; distinct ncomp=50 | Real 16,898-observation representation and fixed-native-graph spatial comparisons executed, including mass-matched control and independent cardinal-edge oracle; paired native reconstruction differs despite identical recorded seed/runtime; independent accuracy, graph variability and cohort comparisons pending |
| Refinement uncertainty | Original unknown/inferred provenance through growth/refinement/vectorization; physical/multiclass options | Rasters/legend and native-label-linked GeoJSON implemented/tested; real GPU/held-out validation pending |
| Marker reproducibility | Full-precision ordered schema, background exclusion, incompatible-restart rejection | Float TIFF and actual Zarr-format-2/3 writer/requantification tests passed; native store completion/chunk inventory and mask lineage enforced; full new GPU run pending |
| Runtime improvements | Effective cores; fixed-input batch/binary/cache benchmarks with quality checks | Explicit policy inputs integrated for GrandQC/StarDist/HoVer-Net/UNI2/CellViT/KODAMA/GigaTIME/MedSAM/TITAN and seven atlas stages; actual commands/resources and representative stubs checked; remaining legacy CPU-stage propagation, real Linux admission and GPU/batching equivalence benchmarks pending |
| Pipeline integration | Modules/schema/docs/reports, real-artifact and representative full Nextflow run | Full stub includes hierarchy/region mapping/linkage; real artifact profile/export and synthetic measured/region mapping runs pass; optional isolated atlas packaging implemented, actual container build and full real inference pending |
| Independent accuracy evaluation | Held-out expert cell/tissue annotations and registered measured-marker evaluation; development crop excluded from claims of generalization | Pending |

## Constraints

- No expert GeoJSON is used at inference to construct the automatic result.
- The two-domain example remains available as the parent partition.
- No cells or tiles are treated as independent patients in comparative statistics.
- No right-side application panels are opened without Stefano's request.
- Existing uncommitted changes and local previous results are preserved.
- Goal completion requires evidence for the full scope, not just unit tests or
  successful creation of an export manifest.

### Own-cell, cohort-niche and morphology continuation

The [cohort/morphology audit](../audits/spatial_atlas_20260904/cohort_niches_and_morphology.md)
records new source-verified morphology, separately weighted own-cell/neighbour
features, missingness preservation, an optional stage-25 shared niche fit,
verified detector-taxonomy gating and actual per-specimen graph/array checks.
Existing results, individual discovery labels and specimen graphs remain
unchanged. All 41 module cache hooks have actual stub execution/unchanged-resume
coverage. The real 75-cell native probe verifies encoding equivalence, not new
cell detection or biological accuracy. Shared cohort outputs now have source-bound
SpatialData/inspector attachments. Opt-in array-backed storage/streamed fitting
now has the continuation evidence below; real large-cohort/independent validation
remains required. The full objective remains active.

### Array-backed neighbourhood continuation

The [array-backed audit](../audits/spatial_atlas_20260904/array_backed_neighborhoods.md)
records complete logical feature retention outside the scalar table, explicit
bounded aggregation/fitting, exact source receipts, immutable hierarchy copying,
and separate chunked SpatialData groups. The full two-specimen CPU workflow
assembles, fits and exports array profiles; unchanged resume reuses all five
tasks. A 3,000-cell/129-axis synthetic sparse aggregation checks all 387 retained
own-plus-neighbour axes and bounded gathers. This is not a large-cohort or
learned-feature biological benchmark. Intermediate NPY files are memory-mapped,
not compressed/chunked Zarr. The table default remains unchanged, and float64
reduction-order differences are disclosed rather than called bitwise equivalent.

The same audit records a separate actual execution-cache defect: after a
same-size/same-mtime helper edit, a new Nextflow task could still import stale
timestamp bytecode. Eight atlas stages now use an empty task-private no-write
cache prefix; the isolated launchers explicitly forward it despite `-E`.
Permanent tests verify changed helper stdout, unchanged stale shared bytecode,
real export readback and unchanged-run cache reuse. Local launcher tests also
cover the Bash 3.2 legacy-dispatch case; no built-container claim is made.

### Shared niche attachment continuation

The [attachment audit](../audits/spatial_atlas_20260904/cohort_niche_attachments.md)
records the portable non-fitting reader, five additive cohort fields, exact
model/source/row/geometry/status checks, actual two-specimen CPU fitting/export,
explicit existing-bundle reuse, inspector detail/source monitoring and all-41-
module cache acceptance. The inspector does not infer a producing run from a
leftover stage-25 directory. New visual rendering QA remains unverified after
a denied browser launch; HTTP/CLI checks do not close that gap. No pretrained
image-model inference or expert annotation was used in these tests. Existing
real results, per-slide discovery, frozen references and physical graphs are
unchanged.

### Default input and marker-search identity continuation

The default UNI2 CSV loader and downstream RData consumer now reject silent ID
deduplication/intersection and invalid source features, including columns omitted
by variance selection. Literal IDs, complete selected populations, source hashes
and row mappings are retained through PCA/KODAMA. A real saved-feature comparison
preserved all 16,898 observations, coordinates and 100 selected features per family
exactly. Marker input verification separately preserved all 20,563 canonical cells
and 21 biological mean scores per nucleus/whole-cell compartment.

Reference marker-discordance filtering now checks the marker's own verified
model/compartment/precision definition even when morphology alone defines distance.
Old/unverified references remain usable for unfiltered similarity, not unverified
cross-slide marker subtraction. Region-inspector search responses distinguish
overlap-weighted versus centroid-member means. Details, tests and evidence paths
are in `audits/spatial_atlas_20260904/input_identity_and_marker_search.md`.
These changes do not close the learned-inference or independent-validation rows.

### Portable reference/export continuation

Cell and region mappings now produce immutable source-bound three-file bundles,
and SpatialData attaches their interpretations without changing original
measurements or discovery labels. Actual two-specimen cell/region workflows pass.
A fresh existing-artifact crop run retained all 20,563 cells, 307 scalar columns,
four matrices, three graphs and six native windows, with exact 19,699 assigned
and 864 outside-reference results in the export. See
`audits/spatial_atlas_20260904/reference_mapping_export.md` for paths, tests,
provenance and scope. This is a same-slide engineering round trip, not new
inference or independent reference accuracy. That audit recorded
CellViT source-image/preprocessing and executable-runtime identity gaps; the
later source-bound implementation addresses their engineering contract,
but the CellViT row is still not acceptance-complete without real inference.

Actual resume tests also exposed a broader code-cache defect: computing an
external Python code hash only inside a process script did not invalidate
cached output after code edits in Nextflow 25.10.4. The reference mappers and
SpatialData now receive explicit upstream code-and-data fingerprint values.
All 40 modules now also use directly referenced `task.ext` code/directory
fingerprints. Full-main stub selective-resume tests and four direct probes cover
all 40 hooks. Static presence checks alone remain insufficient. See the
[pipeline cache acceptance audit](../audits/spatial_atlas_20260904/pipeline_cache_acceptance.md)
for actual traces, transitive-source tests and the runtime/model limits of this
source-code cache contract.

## Verification record (2026-09-04)

- Real immutable source crop:
  `/Users/stefano/Documents/CellPhenotyper/results_full_crop_ncomp50_k2_20260904`.
  Native dimensions 13,172 x 13,098; authoritative MPP
  0.273774374855905. Historical crop TIFF tags and some UNI-2 metadata disagree
  with that report; no existing embeddings are silently relabelled as calibrated.
- Real Nextflow profile/reference/export run:
  `/tmp/cellphenotyper_nextflow_profiles_20260904.aLKwD9/real_integrated_results`.
  Trace records profiles 1m33s, reference mapping 19s, SpatialData export 30s,
  all exit 0. Models were not rerun. All 20,563 canonical cells and compartment
  QC rows are present; 17,573 crowding flags describe possible contact risk,
  not measured contamination.
- Subsequent mask-bound run:
  `/tmp/cellphenotyper_nextflow_profiles_20260904.aLKwD9/real_bound_results`.
  Its trace has all three tasks COMPLETED/exit 0 (profiles 1m12s, cell reference
  6.9s, export 24.7s). A fresh read-only audit verified all 20,563 IDs in order,
  four exact feature matrices, all three exact sparse graphs and six exact
  native image/label windows. Compartment mask lineage is verified and cell UID
  versioning includes both object-table and label-raster hashes. These timings
  exclude model inference and are not a whole-pipeline speed claim.
- The three graphs contain 57,873 / 208,933 / 725,884 undirected edges at
  25 / 50 / 100 micrometres. Two cells lack a 100-micrometre neighbourhood.
  The exploratory selector found no supported subdivision of the remaining
  neighbourhoods. Tissue K=2 was not reused as niche K.
- Same-slide morphology reference mapping retained all rows and yielded 19,699
  assignments plus 864 outside-reference cases. This is an engineering test,
  not cross-slide accuracy or independent validation.
- Real direct SpatialData package:
  `/tmp/cellphenotyper_spatialdata_real_20260904.q0xTHU/crop_roi.zarr` (323 MiB).
  Verified all IDs, 2 tissue polygons, 4 feature arrays, all 3 graphs, physical
  transforms and 6 exact native image/label windows. Historical UNI-2/CellViT
  blocks are absent, explicitly not fabricated.
- Current-profile tests cover numerical missingness, compartment lineage,
  marker row/summary authority conflicts, calibration conflicts, leading-zero
  source IDs and label-raster versioning. Exact UNI2 receipts now reject changed
  sources, shard bytes, extra shards and malformed completion fingerprints;
  legacy receipts remain explicitly unverified for reference use. Marker tests
  verify all 23 channels and three compartments through float TIFF restart.
- Full-grid stubs including added cell UNI-2, refinement uncertainty and
  SpatialData succeeded for one and two samples. Their zero-content model
  outputs prove wiring only. The latest full stub, including hierarchy and
  region mapping, is under
  `/tmp/cellphenotyper_atlas_continuation_20260904.1r8S26/stub_results`.
  Grid construction and all primary/auxiliary feature routes now consume the
  passed resolution report; calibration guards are exercised under Python -O.
- Hierarchy/reference suite: 49 tests, including actual Nextflow region mapping
  against a frozen independent synthetic atlas with outside-reference and
  missing-feature unknown cases. This does not establish biological accuracy.
- Latest broad CPU regression: 295 passed, 2 optional-Torch tests skipped;
  those loader tests then passed in the Torch runtime (12-test feature suite).
  inspector/specimen atlas: 73 passed with loopback access and headless Chrome,
  including region selection, panning, similarity and constituent-cell navigation.
  The sandbox initially denied loopback bind; the permissioned rerun passed.
- A subsequent full two-specimen stub passed after the grid-resolution handoff
  fix, at `two_sample_stub_results` alongside the latest single-specimen stub.
  Both retain explicit stub-only model artifacts, not learned predictions.
- Measured integration: 81 importer/SpatialData tests; four actual export-only
  Nextflow tests verify separate packages, exact float64/NaN values, immutable
  existing profiles and rejection of silently ignored attachments. Region
  geometry is explicitly registered and linked to region shapes, never cells.
- Capacity preflight now models full-channel float32 storage, inference scale,
  pyramids, scratch, physical compartments and optional profile/export copies.
  Hierarchy accounting now includes separate field arrays, native rasters,
  region fragmentation/aggregation buffers and selected local-checkpoint staging;
  measured export includes separately registered packages and geometry bundles.
  CPU/MPS hierarchy execution does not acquire a CUDA lease. The focused
  storage/resource suite passed 53 tests with one platform-specific flock skip.
  Local free space is approximately 10 GiB: no full-image inference is launched
  locally and no old results have been deleted.

Reproduction commands and a read-only real-export auditor are in
`audits/spatial_atlas_20260904/`. Independent validation data locations have been
requested; no external benchmark outcome is inferred from their absence.

## Continuation verification (2026-09-05, local date)

- Cell-to-hierarchy linkage now distinguishes parent-zero unresolved tissue,
  parent-map background and unavailable uncertainty in tables **and** inspector
  picking. Exact producer/link/export tests and all 80 inspector/atlas tests
  passed. Full one/two-specimen stubs include stage 24; see
  `audits/spatial_atlas_20260904/hierarchy_linkage.md` for paths and scope.
- Native vector provenance is bound to exact finalized producer TIFFs, not just
  matching count summaries. Full/stream producer-vector suites passed 61 tests.
  Execution reports expose categorical evidence and explicitly do not claim to
  reverify source hashes or provide calibrated biological confidence.
- The latest focused broad CPU selection passed 211 tests with two optional
  Torch skips and a host-core-discovery warning. It used `.venv-spatial`;
  `.venv-spatialdata` alone lacks scikit-learn and is not the combined atlas
  runtime. A separate report/documentation selection passed 18 tests, including
  published-directory symlink traversal for native vector provenance.
- Read-only Chiamaka inspection verified existing Nextflow PID 3878221 live,
  retaining K=2 and KODAMA internal ncomp=50, at GigaTIME tile 1,012/1,840.
  This is the existing run, not deployment of the new atlas code. The cached
  container imports Torch 2.11.0+cu128 and timm 1.0.24 on Python 3.11.
- A four-location, native-pixel H&E runtime probe was prepared locally under
  `/tmp/cellphenotyper_real_uni2_20260905.tHJe2d/`, using the authoritative
  0.273774374855905 MPP report. No expert annotation was read. Transfer to the
  separately created Chiamaka directory was blocked pending explicit approval
  for that sensitive image payload; no image/code transfer or model inference
  occurred. The helper is
  `audits/spatial_atlas_20260904/prepare_real_uni2_probe.py`.
- GigaTIME Zarr now contains native completion/schema metadata and an exact
  inventory of encoded chunks and metadata. Actual storage-format-2/3 writers
  preserve all 23 float32 channels; nucleus, ring and whole-cell requantification
  agree with integrated analytic-field fixtures within 1e-12 absolute tolerance.
  A copied store is self-contained. Corruption, stale metadata and interrupted
  writes fail equivalent-restart checks; existing stores are preserved. The
  targeted writer/schema selection passed 28 tests in `.venv-marker-tests`.
  The larger marker selection passed 49 tests with one CUDA-only skip. No
  learned model was used; full inventory checks add stored-byte I/O.
- Runtime hardware-policy mutation after Nextflow module inclusion was proven
  ineffective for the GigaTIME profile. The module now receives the computed
  profile as an explicit value input. Six actual no-model process executions
  verify conservative/aggressive inheritance and stage overrides; the related
  hardware/resource/fingerprint/restart suite passed 30 tests.
- A separate actual KODAMA process-command probe verified requested internal
  ncomp=50 and 8 cores produce task allocation 8 and `--n-cores 8`. Its R
  executable was replaced with `/usr/bin/true`; this verifies command wiring,
  not R computation. Evidence is under
  `/private/tmp/cellphenotyper_kodama_scope.tLA4c6/`. Workflow-time policy
  reductions remain ineffective for KODAMA; regular/shared UNI2 and CellViT
  also read late resolved-device/resource parameters. Their auto-device and
  resource handoffs must be passed explicitly before claiming unified hardware
  autotuning. New atlas modules use independent config-time parameters/caps;
  adding them to the common policy is a separate outstanding integration.
- Optional atlas Docker/Singularity overlays retain the existing ML environment
  and install a separate Python-3.12 CPU atlas environment. Offline tests of
  interpreter routing, native numerical/morphology operations and real
  SpatialData readback passed 11 tests using existing local packages. This is
  not a built Linux image or strict-isolated environment; build verification
  requires exact pins, package origins, isolation and separate ML imports.
  Build/use commands and limitations are in `ATLAS_CONTAINER_RUNTIME.md`.
- The combined runtime/hardware/report selection passed 42 tests. A full
  two-specimen stub after the GigaTIME input-contract change completed at
  `/tmp/cellphenotyper_atlas_runtime_20260905.lnLZbU/full_stub_results`, including
  hierarchy, region mapping, cell-to-hierarchy linking and SpatialData. This
  stub did not build or exercise the optional container overlay.
- A later read-only check confirmed the same Chiamaka Nextflow process still
  live at GigaTIME tile 1,196/1,840. This is progress of the existing run only;
  no new code or image payload was transferred to it.

### Explicit runtime-map continuation

The previously outstanding explicit-map handoffs are now wired through main,
primary/auxiliary feature subworkflows, hierarchy, cell linkage and the standalone
artifact entry point. Tests verify actual task resources and command arguments,
including sub-GB caps and CellViT/Ray limits. A correctly configured K=2,
internal-ncomp=50 two-specimen stub completed. A fresh non-stub artifact run
preserved and exported all 20,563 real cells; the exact profile/export auditor
verified four feature matrices, three spatial graphs, six native windows and
mask lineage. This did not rerun learned models or validate biological accuracy.

Separate merged-config and generated-wrapper tests now cover the restored
GPU-admission hook, stage-specific device overrides, aliases, container options
and forwarding of the selected UUID, including unset-versus-empty visibility.
The whole-GPU helper preserves inherited allocation restrictions. Linux
Bash/flock admission, actual GPU/container execution, other legacy resource
handoffs and native GrandQC auto-device behavior remain outside this verified
scope. Full grid/both two-specimen stubs also verify exact published-plan
allocations and the corrected grid/cell KODAMA comparison staging.

Later legacy-stage integration passes the same explicit map to GrandQC,
StarDist, HoVer-Net, MedSAM and TITAN, including auxiliary MedSAM and post-cluster
calls. GrandQC CPU auto-selection and StarDist's CPU/Metal exclusion are now
explicit; container/admission and module arm64 defaults agree. Actual recorder
commands and a two-specimen consensus/TITAN/PathoFMPred stub extend the previous
grid/both coverage. These tests remain model-free; real Linux/GPU behavior is
not established by the stub.

The support-mask producer's source-slice stretching and ROI rounding were
corrected with crop-aligned pixel-centre classification. This is still a
low-resolution categorical approximation, not native tissue-gap detection.
The crop writer also has a tested bounded TIFF-window fallback for incompatible
TIFF/Zarr adapter versions. Existing source images/results were not rewritten.

The raster-backed neighbourhood implementation subsequently reproduced all
20,563 real cells, 307 profile columns, feature matrices and three sparse graphs
exactly, with a 63-second neighbourhood-only rebuild. All inputs remained
unchanged. The broad cell/hierarchy/measured/neighbourhood regression passed
188 tests after fixing large-origin tolerance and extreme-aspect-ratio grid
limits. This uses the same historical low-resolution support, not new native
tissue inference; it does not close the native-gap or independent-validation
requirements.

The consolidated runtime evidence, remaining container/runtime work and
pending transfer authority are recorded in
`audits/spatial_atlas_20260904/runtime_continuation.md`.

## Completion boundary

### Real clustering-representation continuation

The annotation-free saved-feature comparison now has real execution evidence,
not only synthetic R fixtures. Under matched random landmark selections, PCA50
had mean seed ARI 0.942769 versus 0.793277 for the 2-D KODAMA route on 16,898
tissue-grid observations. This changes the representation/transformation route,
not just a plotting dimension, and does not prove histological accuracy.

An inspected native KODAMA materialization API enabled opt-in portable graph
export and direct graph Leiden without reconstructing edges from UMAP. A real
all-row reconstruction preserved ncomp=50 and the existing kNN classifier, then
exported 1,689,800 directed edges. The graph consumer kept every observation,
performed only supported affinity merges for the requested K2 sensitivity, and
preserved explicit uncertainty. A real-scale dense-temporary bug was corrected;
all affinity values match an independent sparse oracle and every clustering
table value is identical before/after the sparse repair. Default UMAP selection
and historical outputs were unchanged.

Native raw-data kNN does not fit a 50-component PLS model. The actual classifier
and ncomp applicability are now explicit; no silent PLS-LDA switch was made.
Graph export is opt-in, retains native landmark algorithm semantics, and rejects
the outer >200,000-row subset/projection path. It is a feature graph, distinct
from tissue-gap-constrained physical neighbourhood graphs. Full evidence,
reproduction commands and limitations are in
`audits/spatial_atlas_20260904/clustering_representations.md`.

Root regression after the graph/candidate integration passed 126 tests with
one existing optional-runtime skip. Three representative full-workflow stubs
plus schema/fingerprint checks passed 39 tests. These stubs remain model-free
and do not establish current learned feature extraction or Linux/GPU accuracy.

### Binary UNI2 continuation

Original-specification section 7 now has an opt-in end-to-end binary storage
route: single/shared extraction, strict shard/grid/root receipts, R rawdata
loading, canonical cell-profile arrays and raw-grid hierarchy ingestion.
Feature families are explicit; a swapped paired cache cannot be accepted merely
because both sides share one extraction contract. CSV remains the default and
historical results are unchanged. The separate hierarchy NPY format is retained.

Real immutable tile and inner-square CSV inputs each contain 16,898 observations,
1,536 dimensions and 93 shards. Float64 storage/read-back exactly reproduced
their parsed doubles; float32 reproduced the explicit float32 cast with maximum
absolute differences 1.04e-7/1.96e-7 from historical decimal doubles. Current
production-reader verification also passed every row, metadata field and feature
against those original CSVs, with all source hashes unchanged. Converted storage
does not upgrade legacy encoder or calibration provenance.

The actual R loader retained every ID and the same variance-selected 100 features
per family. After explicit ID alignment, float64 values were bit-exact versus
CSV; float32 stayed within its precision tolerance. Canonical analysis order is
the same lexical ID order as the legacy analysis, although binary rawdata applies
that ordering earlier. Single-run loader times were CSV 49.03s, binary32 17.03s,
binary64 12.90s; these are local, non-isolated warm-cache observations, not a full
pipeline speed claim. Three repeated storage-read measurements and byte counts
are in `docs/UNI2_BINARY_STORAGE.md`.

Evidence is under
`/tmp/cellphenotyper_uni2_binary_20260905.5pWDEo/real_comparison/`, including the
initial storage report, current-reader verification and frozen-code R-loader
verification. Reproduction scripts are in `audits/spatial_atlas_20260904/`.
The model-free full two-specimen workflow stubs passed for grid, both and
consensus/research routes; the both fixture explicitly selects binary storage.
The small native CPU KODAMA handoff test uses synthetic features, not learned
UNI2 extraction. It does not replace actual updated GPU/container execution.

Final independent review tightened matching round-trip physical metadata across
CSV/binary (including literal string IDs) and validation of provided R root/grid
receipt chains. The final R helper hash is
`12d81d3cb6d0acb04784eda17dbaa37e355acf259140af3172fb8afe9caf49ea`.
Its repeated real-data check is in `r_loader_final_verification/verification.json`
under the same audit directory and reproduces the exact ID/feature/value findings.
The later loading run took CSV 26.53s, binary32 8.69s, binary64 7.57s; variation
from the first run reinforces that these are not isolated benchmarks.

Final root regression selections: 178 Python/profile/hierarchy/schema/fingerprint
tests passed with two optional-runtime skips and one existing host-core-discovery
warning; 52 Python-to-R binary/receipt/native-KODAMA tests passed; 53 scientific
documentation/execution-report/calibration/grid tests passed. Seven actual
model-free encoder command probes and three full workflow stubs passed, followed
by another three-stub run with the both-mode fixture selecting binary storage.
Compilation and diff whitespace checks passed. These are scoped engineering
checks, not independent cell/tissue/marker accuracy validation.

### Rejected-candidate review verification

Excluded detector observations remain separate from canonical cells. The unchanged
real review layer contains 22,792 candidates (8,827 valid original polygons,
1,914 invalid original outlines, 12,051 centroid-only observations) from 66,947
source predictions; all 20,563 canonical cells remain unchanged. Four explicit
detector/alignment source files enabled complete population, geometry, centroid,
decision and evidence re-verification. The inspector reports `source_verified`
only for that stronger path. Without the source files it explicitly reports
unverified candidate content despite matching raster hashes. Source consistency
is not detection accuracy. Details and arguments are in `CELL_ATLAS_USAGE.md`.

### Existing Chiamaka run terminal verification

A later read-only check found PID 3878221 absent and independently read the
successful `lethal_gutenberg` completion receipt (`convert` through
`cluster_geojson`). The final trace records MedSAM refinement COMPLETED/exit 0
(56m50s) and GeoJSON conversion COMPLETED/exit 0 (2.1s), with final outputs at the
remote trace's 2026-09-05 02:37 timestamp. This is the previously launched
full-image K2/ncomp50 run with older staged code, not the new atlas, 23-channel
storage contract or binary UNI2 deployment. No outputs were copied or altered.

### Native image-derived support and exact traversal continuation

The optional `cell_neighborhood_support_mode=brightfield_native` now constructs
a native image-only neighbourhood support derivative with conservative
background-colour adaptation, protected bright/structured edges, source/output
hashes and separate categorical reasons. The normal `provided` mode is unchanged.
No expert GeoJSON, nuclear labels or domain labels drive this subtraction.
Nextflow executes the actual helper and carries a portable bundle inside the
profile; SpatialData stores both categorical rasters and their exact transforms,
receipt and element mappings, distinct from canonical cell instances.

On the real 20,563-cell crop, the rule excluded 2,210,532 pixels (1.69235% of
upstream support) and preserved 1,318,970 ambiguous bright pixels. Three-condition
verification separates coarse support, a native resampling-only control and
image filtering. All original cell IDs/measurements/features remain unchanged;
no cell became unsupported, and all conditions retained one exploratory niche.
Relative to the native control, the image rule removed 241 / 6,167 / 65,508
undirected edges at 25 / 50 / 100 micrometres, without adding any. These removed
edges are review candidates, not independently proven errors.

Exact native traversal initially made profile assembly slow. Replacing per-pixel
NumPy operations with scalar arithmetic preserved 45,640 boundary-case decisions
and subsequently every full real profile value/dtype and all three CSR arrays
byte-for-byte. That assembly run decreased from 1,052.88s to 158.71s (6.63×
observed ratio on a shared warm-cache laptop; not a whole-pipeline benchmark).
Actual two-specimen Nextflow tests preserve all 10 synthetic cells and isolate
gap-centred cells. Native SpatialData tests use actual API write/readback, not
mock manifests. Full audit paths, hashes, tests and limitations are in
`audits/spatial_atlas_20260904/native_brightfield_support.md`.

Biological gap precision/recall and pale-tissue retention remain unverified;
this is not a MedSAM model replacement and cannot recover excluded GrandQC
tissue. A bounded local runtime/cache inspection found no usable UNI2/CellViT
checkpoint, so no new learned-feature inference occurred. No remote transfer,
deployment, old-result deletion or UI-panel opening occurred.

### Slide-consistent streamed MedSAM correction

Large-slide inspection identified processing-window-shaped tissue assignments in
the post-KODAMA refinement output. The input H&E and the pre-MedSAM grid mask did
not contain those rectangles. The cause was downstream: MedSAM normalized each
4096-pixel window by its own intensity maximum, while the pre-MedSAM Wald
competition fitted Lab/optical-density scaling and label prototypes separately
inside every overlapping window. Both choices made an identical pixel depend on
the surrounding processing window.

Streamed refinement now uses the fixed RGB uint8/255 MedSAM input contract and
fits one deterministic, bounded slide-level Lab/optical-density calibration and
set of trusted-label prototypes, which are reused for every window. The default
grid route still bypasses tissue growth. Grid observations that abstain under the
stability policy retain their raw KODAMA class in the categorical baseline while
their abstention remains explicit in the uncertainty raster. This prevents large
seedless blocks without representing an uncertain assignment as confident.
Model-free regression tests verify window-independent RGB and Wald features and
the separate categorical/uncertainty contracts. These checks establish the
engineering correction; biological boundary accuracy still requires held-out
expert evaluation.

### CellViT source binding and full-module cache continuation

Source-bound CellViT bundles now verify exact pixels/calibration, raw/retained
population, graph rows, detector-source coordinates, physical preprocessing and
the actual configured Python CLI/runtime. Consensus preserves detector coordinates
separately from its fused centroid; all canonical rows and explicit missing
vectors survive attachment. Legacy bundles remain unverified, and external
classifier taxonomies cannot claim a bound reference definition. The unchanged
UID policy gives newly generated additive object tables their own version;
historical outputs are untouched.

The focused producer/consumer/consensus/reference/cache selection passed 311
tests plus 24 subtests, with one rasterio-dependent skip. Real restricted PyTorch
serialization/export tests passed in the existing compatible runtime. A tiny
two-reference-specimen, three-query-cell profile/mapping/SpatialData write/readback
preserves exact vectors, NaNs, coordinates and metadata. These are synthetic
engineering inputs, not learned embeddings. Full evidence and runtime limitations
are in [the CellViT source-binding audit](../audits/spatial_atlas_20260904/cellvit_source_binding.md).

All 40 Nextflow modules now carry early code/directory cache fingerprints.
Actual copied full-main stubs plus four direct probes verify unchanged cache
reuse for every module and selective reruns after same-stat Python/R helper
edits. The final full-main acceptance rerun passed in 97.99 seconds; 23 actual
reference/cache/SpatialData workflow tests also passed against current code.
See [the pipeline cache audit](../audits/spatial_atlas_20260904/pipeline_cache_acceptance.md).
Source hashing is not a substitute for immutable model/container/runtime evidence
or real model execution. No external transfer or deployment was attempted.

### KODAMA-first clarification and optional encoder investigation

Stefano explicitly retained KODAMA as fundamental to tissue segmentation and
authorised investigation of alternative foundation encoders, optionally alongside
UNI2. The [encoder comparison contract](FOUNDATION_ENCODER_COMPARISON.md) records
the source-backed candidate shortlist, access requirements, model-specific
preprocessing/identity gaps and matched KODAMA evaluation design. It does not
claim that historical Python presets are supported end-to-end alternatives.
No alternative weights, credentials, model inference or default change occurred.
The earlier within-parent prototype used k-means. The continuation below replaces
the optional workflow's discovery with verified native KODAMA graphs; the legacy
implementation is now only an explicitly labelled standalone comparison. The
hierarchy remains disabled by default pending learned-feature/biological acceptance.

An experimental standalone spatial-comparison implementation now retains the
native KODAMA graph and adds separately audited, cardinal physical-grid edges.
Image-OD boundary penalties use a shared ungated feature-kernel normalizer, so
the penalties cannot be undone by renormalization. Fresh reruns in the existing
`.venv-spatial` runtime passed 61 adjacency tests and 49 R regularizer tests,
with zero failures, errors or skips; the coordinating process verified the
retained JUnit counts. Evidence is now stored in
[`tissue_spatial_adjacency_recheck_Ee7aHK.junit.xml`](../audits/foundation_encoder_review_20260905/tissue_spatial_adjacency_recheck_Ee7aHK.junit.xml)
and [`kodama_spatial_regularization_recheck_Ee7aHK.junit.xml`](../audits/foundation_encoder_review_20260905/kodama_spatial_regularization_recheck_Ee7aHK.junit.xml).
These replace earlier unavailable temporary receipts.
The subsequent real comparison and new full-CLI acceptance are recorded in
[the spatial KODAMA audit](../audits/spatial_atlas_20260904/spatial_kodama_comparison.md).
The helpers alone do not establish real boundary accuracy or a preferred configuration.

Read-only input checks found all 16,898 saved grid IDs/coordinates match after
literal-ID alignment (grid CSV order differs from saved PCA order), and 700 grid
centres lie outside the coarse support. Those observations must remain present
without added local edges. New physical geometry must use the report-bound
0.273774374855905 MPP, not the historical grid's 0.25; this cannot repair the
legacy feature-extraction provenance. The experiment's CLI additionally requires
full bounded crop/source pixel equality, now verified on all 172,526,856 pixels.
Original nonlocal feature edges are retained and are not relabelled as
gap-constrained physical adjacency.

The full interface passed 13 synthetic tests including a consumption-time
intermediate mutation. Actual 16,898-observation runs retained all 700 unsupported
centres and produced 31,083 added local edges, independently checked against every
cardinal candidate. On one frozen graph, 10% local added mass increased mean
clustering-seed ARI from 0.8255 to 0.9755; gated and mass-matched controls did not
establish a boundary-accuracy benefit. No production default changed.
Two fresh native graph reconstructions differed despite identical recorded
inputs/seed/runtime hashes; this repeatability issue is separate from conditional
clustering-seed stability. The new graph bundles and complete evidence are retained
inside the repository audit directory, not only in temporary storage.

### Native within-parent hierarchy continuation

The [native hierarchy audit](../audits/spatial_atlas_20260904/hierarchy_kodama_20260905/README.md)
records the new optional KODAMA/Leiden discovery, separate local/context/combined
graphs, all-observation lineage, native affinity uncertainty and realized graph
identity. The original broad parents are unchanged. K-means remains only an
explicit standalone legacy comparison; no CellCharter method replaces KODAMA.
The source-bound runner passed 41 real-R/negative-contract tests, and the final
consumer selection/runtime-isolation suite passed 108 tests. Actual two-specimen
synthetic native CLI and Nextflow runs exercise raster/profile exports, not
learned encoder inference or biological accuracy.

The first real CLI acceptance found rounded jsonlite metadata incompatible with
exact Python echo comparison. The fix preserves exact source-manifest checksums
and values while treating the rounded float echo as descriptive; IDs, hashes,
keys, integer fields and ordering remain exact. Independent readback verifies
actual float64 native inputs and reconstructs affinity evidence from graph slots.

Native interpreter/package bytes are not yet a scheduler pre-cache identity.
This opt-in CPU discovery stage therefore has caching disabled; expensive
upstream encoder features retain deep caching, and dependent derivatives may
rerun. The real full-main stub/cache-policy selection passed 129 tests after
this change. Earlier all-41-module cached-resume results above are historical,
not a claim that native hierarchy discovery now reuses cached graphs safely.
Container source routing/dependency/preflight work is separate from a built-image
or actual full-slide native-runtime acceptance claim. Hierarchy remains disabled
by default. Independent tissue/assay validation and the full goal remain open.

### Durable real-profile continuation

The [2026-09-05 real-artifact audit](../audits/spatial_atlas_20260905/README.md)
supersedes reliance on missing temporary profile bundles for current acceptance.
It retains all 20,563 cells, richer morphology, native-support neighbourhoods,
current-representation exploratory niche fitting, an exhaustively pixel-checked
SpatialData package and a pending human review queue. The current niche fit
also found no subdivision satisfying its heuristic gate; it is not evidence
of a single biological niche. Separate fresh 3 µm physical compartment masks
passed full-raster identity/support/count checks, but historical marker scores
were not reassigned to those new compartments. Learned inference, real
cross-slide references, human review and measured-assay validation remain open.

### First actual local UNI2 probe

After Stefano completed local authentication and authorised the download, the
exact UNI2-h revision `d517a8dd47902dd7c308b3c36f63bce47e7b9a43` passed checksum
verification and actual strict checkpoint loading in the existing Miniforge CPU
runtime. Four real tissue locations at two independently cropped physical field
sizes yielded two finite 4×1536 feature arrays; exact completed reuse passed
without loading the model again. Evidence is retained in
`audits/spatial_atlas_20260905/real_uni2_probe/verification.json` and described in
the current real-artifact audit. This closes the tiny local UNI2-inference
availability question, not full-slide inference, CellViT, hierarchy accuracy,
cross-slide reference validation or measured-assay validation.

### Full-goal completion remains unproven (current)

This goal remains active. Pending rows above are required work, not optional
future aspirations. Passing tests on mocks or one development crop cannot prove
the requested whole pipeline, independent accuracy, cross-slide atlas quality,
or performance gains. See `CELL_ATLAS_USAGE.md` for the currently implemented
interfaces and explicit migration constraints.
