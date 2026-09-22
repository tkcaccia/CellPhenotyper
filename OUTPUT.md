# Output

With `--outdir_base results_full`, CellPhenotyper writes the following stage directories:

- `00_execution/`: scientific analysis contract, validation-readiness/claim ceiling, stage-complete uncertainty register, learned-model inventory/release gate, Nextflow trace, timeline, DAG, report, runtime summaries, and the project output manifest with a stable `output_id` for every published file.
- `01_input/`: normalized pyramidal OME-TIFF input plus source/converted image QC reports containing SHA-256, dimensions, axes, dtype, bit depth, channel/color interpretation, compression, tiling, pyramid levels and MPP-source consistency.
- `02_grandqc/`: mandatory full-image GrandQC tissue/artifact masks, GeoJSON, summaries, and previews.
- `03_stardist/`: the shared `GrandQC tissue support intersection ROI` crop, StarDist labels, object table, crop GeoJSON, and coordinate shift. `prepared_crop/crop_roi.tif` is a validated tiled pyramidal RGB OME-TIFF.
- `03b_hovernet_monusac/`: official HoVer-Net fast MoNuSAC inference normalized to the StarDist crop frame.
- `03c_cellvitpp/`: official CellViT++ PanNuke inference normalized to the StarDist crop frame.
- `03d_cell_consensus/`: role-aware multi-detector instance fusion. Canonical instances require broad-scope StarDist/CellViT++ agreement by default; HoVer-Net MoNuSAC remains scoped supporting evidence.
- `04_TMA/`: TMA decision, core polygons, and cells assigned to cores when applicable.
- `04_tissue_mask/`: crop-aligned GrandQC tissue-support mask plus a separate `*_grandqc_artifact_candidates.tif` annotation mask.
- `05_gigatime/`: GigaTIME virtual-marker image plus nucleus/whole-cell/perinuclear-ring quantification, versioned marker schema and restart provenance.
- `06_roi/`: immutable ROI GeoJSON, ROI SHA-256/geometry/image-association QC, crop-aligned labeled mask, class map, and preview.
- `07_cell_assignments/`: cells assigned to ROI polygons.
- `08_cytoplasm/`: physical tissue-constrained nucleus/ring/whole-cell approximation masks and per-cell compartment QC; `cyto` is a legacy whole-cell filename.
- `09_grid_tiles/`: grid observation coordinates, calibrated tile geometry, GrandQC tissue occupancy, artifact-candidate fraction/flag, and a grid QC preview when `uni2_sampling_mode=grid` or `both`.
- `09_embeddings/`: cell-centred or spatial-grid UNI2-h tile and inner-square embeddings; `both` writes the cell route under `<sample>__cells` and leaves the primary grid route unsuffixed.
- `10_kodama/`: UNI-2 KODAMA coordinates, plots, and logs, plus optional GigaTIME-marker KODAMA output.
- `11_clustering/`: assignments, plots, logs, per-observation vote/stability status, and per-seed adjusted Rand index tables for the standard and any configured secondary clustering variant. `cluster` preserves the raw algorithmic result; `interpretable_cluster` may abstain. The standard KODAMA membership figure renders abstentions in neutral gray, and `_cluster_kodama_uncertainty.{png,pdf}` separates ambiguous landmark assignment from seed instability. Each variant also has a `_cluster_interpretation/` directory containing descriptive spatial coherence, GigaTIME marker enrichment when available, `cluster_spatial_uncertainty.png`, observation-level `cluster_abstentions.geojson`, a blinded region-review form and GeoJSON, a separate answer key, and `cluster_interpretation_summary.json`. These outputs support review but do not assign biological identities automatically.

When `cluster_target_clusters` is nonzero, every Stage 11 assignment and summary records `cluster_analysis_role=sensitivity_forced_cluster_count`, the membership plot is watermarked `SENSITIVITY ONLY`, and the execution landing page emits a dedicated review signal. Downstream masks retain the selected partition for comparison, but it must not be reported as the graph-derived primary result.
- `12_cluster_mask/`: canonical raster cluster masks plus categorical uncertainty masks, H&E uncertainty overlays, and summaries. Value `0` in the canonical mask means background or abstention; the separate uncertainty mask distinguishes ambiguous assignment, seed instability, both causes, and other abstention. Raw assignments remain in the Stage 11 CSV.
- `13_grown_tissue/`: sparse cell-centred cluster labels grown to tissue; absent for pure grid runs, but present under `<sample>__cells` in `both` runs.
- `14_medsam_refine_tissue/`: pre-MedSAM annealed multiclass boundary competition, GPU MedSAM tissue-border refinement, optional residual full-resolution internal-boundary refinement, separate change metrics, and native-resolution QC crops.
- `15_cluster_geojson/`: final connected tissue-section polygons.
- `16_neoplastic_section/`: per-section cell counts and the section with the most CellViT++ neoplastic cells.
- `17_titan/`: CONCH v1.5 patch features and one 768-dimensional TITAN section embedding.
- `18_pathofmpred/`: cancer-specific PathoFMPred research predictions and reports.
- `19_cell_profiles/`: optional canonical CSV/Parquet profiles, separate numerical feature blocks, calibrated sparse neighbourhood graphs, niche assignments and morphology. With `cell_neighborhood_feature_storage=arrays`, `neighborhood_features/feature_store.json` binds complete own-cell/neighbour axes to unchanged source arrays and per-radius float64 NPY payloads; the scalar table stays narrow. Keep the complete profile directory together. SpatialData places these groups in separate chunked `cells.obsm` arrays with exact axis/aggregation metadata, not scalar `obs` columns.
- `20_spatialdata/`: optional readable linked image/label/polygon/profile package with verified physical transforms. Available cell/region reference mappings are attached as separate `reference_*` observation fields with exact atlas metadata and source-bound receipts; original measurements and discovery labels remain unchanged.
- `22_tissue_hierarchy/`: optional separately encoded local/wider physical fields, unchanged broad parent domains and uncertainty, subdomains, connected region masks, region profiles and checksum-bound manifests.
- `23_region_reference_mapping/`: optional reference assignments/unknowns for discovered tissue regions; separate from both exploratory domain labels and cell-reference assignments. Each mapping contains sibling `.csv`, `.atlas.json`, and completion-last `.mapping.json` files.
- `21_reference_mapping/`: optional assignments to an immutable cell reference atlas in the same three-file bundle; unknown and discovery labels remain separate. Keep the files together when copying results.
- `25_cohort_niches/cohort_niches/`: optional shared exploratory niche model and `cohort_niche_assignments.parquet`, keyed by canonical sample/cell IDs, with summary and completion-last receipt. Keep all four files together. New SpatialData exports attach five separate `cohort_niche_*` fields and full shared-model provenance after exact source verification; original per-slide niche labels, profiles and graphs remain unchanged. The inspector accepts the bundle explicitly with `--cohort-niches`. This is not reference assignment or independently validated biology.

See [cell-atlas usage and interpretation limits](docs/CELL_ATLAS_USAGE.md) for
schemas, standalone artifact rebuilding, required runtimes and manual inspector
launch. New stage-14 uncertainty/provenance TIFFs and their JSON code legend
preserve original abstentions and distinguish newly inferred labels. Old
refinement outputs must be regenerated before strict sidecar-dependent restarts.

## Named Cell Types

`03b_hovernet_monusac/<sample>/hovernet_<sample>/hovernet_cells.json.gz` records both `type_id` and `type` in gzip-compressed JSON. The MoNuSAC map is:

- `0`: `background`
- `1`: `epithelial`
- `2`: `lymphocyte`
- `3`: `macrophage`
- `4`: `neutrophil`

`03c_cellvitpp/<sample>/cellvit_<sample>/cellvit_cells.json` also records both fields. The PanNuke map is:

- `1`: `neoplastic`
- `2`: `inflammatory`
- `3`: `connective`
- `4`: `dead`
- `5`: `epithelial`

`03d_cell_consensus/<sample>/consensus_<sample>/objects.csv` retains each detector's named class and numeric ID in separate columns. `broad_instance_support`, `scoped_support_count`, `instance_evidence_tier`, `geometry_source`, centroid-displacement summaries, `agreement_score`, and `agreement_tier` describe role-aware instance-fusion evidence. The score uses only broad-detector support and proximity; MoNuSAC support is reported separately. `phenotype_evidence_methods` and `phenotype_status` remain separate because MoNuSAC and PanNuke taxonomies are not equivalent; no synthetic consensus cell type is created. `mask_seed_x` and `mask_seed_y` record the unique raster seed that guarantees every canonical ID survives overlapping detector contours. `alignment.csv` includes accepted and abstained components with explicit decisions, including missing broad consensus and excessive broad-detector distance. `detector_agreement_benchmark.csv` and `.json` report pairwise detection match fractions, centroid displacement, polygon IoU, Hausdorff distance, scope-aware count ratios, and spatial coverage. These are inter-detector agreement measurements, not reference-standard accuracy or calibrated confidence. `consensus_summary.json` additionally records the acceptance policy, detector roles, combinations, evidence tiers, decisions, and acceptance counts. `consensus_cells.geojson` and `consensus_preview.png` provide geometry and visual QC.

Every detector output is GrandQC-gated before consensus. StarDist writes `grandqc_cell_filter_summary.json`; HoVer-Net and CellViT++ record equivalent input, retained, and removed counts in their metadata JSON files.

Optional CellViT token export adds `cellvit_embeddings.npy`,
`cellvit_embedding_ids.csv`, `cellvit_embeddings_metadata.json`,
`cellvit_raw_population.csv`, and completion-last
`cellvit_embeddings_completion.json` alongside `cellvit_cells.json` in stage
03c. These six files form one portable source-bound bundle. Stage 03d retains
the original detector coordinates and normalized-population SHA-256 in
`cellvitpp_x_px`, `cellvitpp_y_px`, and `cellvitpp_source_sha256`; these do not
replace fused canonical coordinates. See the
[source-bound feature contract](docs/CELL_ATLAS_USAGE.md#source-bound-cellvit-features)
for validation, migration and interpretation limits.

HoVer-Net MoNuSAC is deliberately reported as a non-exhaustive detector because the upstream training labels omit classes such as fibroblasts. Its raw count should not be interpreted as total cellularity or expected to match StarDist and CellViT++. `hovernet_metadata.json` records this scope, the exact checkpoint SHA-256, the upstream revision, and whether the shared GrandQC inference mask was used.

GrandQC background is hard negative support during MedSAM. Artifact candidates remain included through UNI-2 and KODAMA; only candidates confirmed as cluster-conditioned KODAMA-display outliers are excluded, recorded in the clustering CSV, rasterized with uncertainty code 5, and kept outside the final refinement support. Stage 14 publishes the tissue-support and exclusion QC maps plus support, exclusion and zero-leakage metrics.

`<sample>_<variant>_medsam_precompetition_labels.png` shows the label state supplied to MedSAM after bounded annealed competition. On streaming WSI runs this is explicitly a diagnostic-scale reconstruction; exact non-overlapping tile-commit change counts and the temperature/energy settings are recorded in `<sample>_<variant>_medsam_summary.json`.

For `uni2_sampling_mode=grid`, Stage 12 already rasterizes adjacent grid inner cores into a dense spatial phenotype mask. Stage 13 is therefore skipped and Stage 14 refines the Stage 12 mask directly. For `uni2_sampling_mode=cells`, Stage 12 contains sparse cell-supported labels and Stage 13 remains required before MedSAM. In `both` mode, the unsuffixed grid route follows the first behavior while the `<sample>__cells` route follows the second; stages 11, 12, 14 and 15 contain both namespaces, and stage 13 contains only `<sample>__cells`.

Each `09_embeddings/<sample>/embeddings_<sample>_<mode>/` directory contains a hidden `.*_embedding_complete.json` manifest. It records the observation type and expected/written observations and is emitted only after every non-empty processing partition has valid, complete shards; KODAMA is not allowed to receive a partially resumed UNI2 result.

`10_kodama/<sample>/gigatime/gigatime_kodama_output/` contains the independent KODAMA representation calculated from all matched GigaTIME nuclei and cytoplasm marker means. With `uni2_sampling_mode=both`, `10_kodama/<sample>__cells/kodama_output/` contains the secondary cell-centred UNI-2 representation; primary grid outputs remain under the unsuffixed sample ID. The `<sample>__cells` namespace continues independently through stages 11-15. `10_kodama/<sample>/uni2_route_comparison_<sample>/` aggregates cell-centred coordinates into retained grid cores and publishes coverage, Procrustes, distance, neighborhood, spatial-coherence and optional virtual-marker proxy endpoints. It does not automatically choose a preferred route.

Each persisted GigaTIME marker store includes `gigatime_seam_qc.json` and `gigatime_seam_qc.png`. The metric compares each block-boundary gradient with nearby within-block gradients and reports the zero-fill transition fraction. Default runs disable block skipping and fail when a systematic seam exceeds both configured magnitude and affected-fraction thresholds.

`gigatime_marker_score_qc.{json,tsv,png}` reports deterministic sampled distributions, saturation and near-zero fractions, tissue/background contrast, and inter-channel correlation for all model channels. The outputs and quantification tables explicitly identify values as `uncalibrated_virtual_marker_score`; the historical `gigatime_probs.*` and `*intensity*` filenames are compatibility names and must not be interpreted as calibrated probabilities or measured protein abundance.

New `gigatime_probs.zarr` stores also contain native marker/completion metadata
and `cellphenotyper_storage_manifest.json`, binding encoded chunks and metadata
by SHA256. They can be copied without adjacent sidecars and still support
verified all-channel float32 requantification. Missing/changed chunks, incomplete
stores and conflicting external metadata are rejected. Receipt-free historical
Zarr is explicitly non-equivalent; existing stores are not replaced silently.
Integrity verification reads all stored bytes and should be included in runtime
and I/O estimates.

`00_execution/project_outputs.json` and `project_outputs.tsv` assign a stable `output_id` to every published artifact. `absolute_path` always points into the durable output tree; when relative-link publishing is used, `resolved_target_path` separately records the underlying Nextflow cache target for provenance.

The default `publish_dir_mode: copy` makes published files independent of the Nextflow work cache, so `work/` can be removed after a successful run. Selecting `rellink` reduces temporary storage and publication I/O, but its published links break when the corresponding work cache is deleted.

## Neoplastic Section

For each connected polygon from `15_cluster_geojson`, stage 16 writes:

- `section_neoplastic_counts.csv`: section ID, neoplastic-cell count, total consensus-cell count, area, and selection status.
- `selected_section.geojson`: selected polygon in the level-0 StarDist ROI-crop coordinate system.
- `selected_section_crop.geojson`: selected polygon in the exported crop frame.
- `selected_section.ome.tif`: masked, padded, pyramidal section image.
- `selected_section_mask.tif`: tiled binary section mask.
- `selected_section_shift.json`: crop origin and source MPP.
- `selected_section_summary.json`: deterministic selection metadata.
- `selected_section_preview.png`: section QC preview.

Selection is deterministic: highest named CellViT++ `neoplastic` count, then highest total consensus-cell count, polygon area, and stable section ID. A requested TITAN run fails rather than silently selecting an unpopulated section when `neoplastic_section_require_cells=true`.

## TITAN

`17_titan/<sample>/titan_<sample>_<variant>/` contains:

- `titan_patch_features.h5`: 768-dimensional CONCH v1.5 patch vectors, regularized level-0 coordinates, and tissue coverage.
- `titan_embedding.csv`: one row with identifiers followed by exactly `titan_000` through `titan_767`.
- `titan_embedding.npy`: the same section embedding as float32.
- `titan_metadata.json`: pinned model revision, MPP, patch geometry, batch size, patch count, CUDA device, and GPU model.

Patches correspond to 512 pixels at 0.5 microns per pixel. The source crop size is rescaled from the image MPP, and the official gated TITAN implementation aggregates the CONCH v1.5 patch features.

## PathoFMPred

`18_pathofmpred/<sample>/pathofmpred_<sample>_<variant>/` contains:

- `pathofmpred_predictions.csv`
- `pathofmpred_continuous_radar.png`
- `pathofmpred_binary_predictions.png`
- `pathofmpred_research_report.html` when HTML reporting is enabled
- `pathofmpred_runtime.txt`
- `pathofmpred_model_provenance.json`: installed package version, source/license fields and SHA-256 records for serialized package/model databases.

These are TCGA-derived research estimates. They are not externally validated clinical assays, and binary scores must not be interpreted as calibrated probabilities.

## Execution Reports

When profiles and hierarchy are both enabled,
`24_cell_tissue_links/<sample>/cell_profiles/` contains the linked canonical table,
exact nuclear/ring overlap relations, retained feature blocks and spatial graphs,
and the immutable source registry. This is separate from the unchanged stage-19
profile. Stage-20 SpatialData includes registered hierarchy labels, actual region
feature blocks and the many-to-many relationship table. See
[the cell-to-tissue contract](docs/CELL_TISSUE_LINKS.md).

Final cluster GeoJSON files have a sibling `.geojson.provenance.json`. Features
link its SHA256 and their native-label summary. Counts and fractions describe
the source raster label, including disconnected or vector-filtered components;
they are not per-polygon confidence after smoothing or hole filling. Unresolved
label-zero pixels remain visible in the summary. Missing historical provenance
is explicit, not a confidence value. New refinement metadata binds the exact
finalized TIFFs by SHA256 only after successful label pyramidization. Vector
export verifies those bindings; legacy unbound metadata is labelled explicitly.
Execution reports expose these producer records without claiming to rerun the
full source-raster hash audit.

Important files under `00_execution/` include:

- `index.html`: review-first landing page with claim ceiling, non-passing QC signals, bounded preview gallery, stage storage and slowest processes.
- `model_inventory.json` and `model_inventory.tsv`: learned-model source, immutable revision, checkpoint hashes, cache paths, licenses, training domains and explicit release blockers.
- `specimen_atlas.html` and `specimen_atlas.json`: portable specimen-level review atlas grouping bounded previews, quantitative summaries, route and observation-unit labels, uncertainty coverage and stable source-output IDs. Predicted marker scores and measured quantities are explicitly separated; missing layers are shown as not produced rather than interpreted as negative results.
- `analysis_contract.json`
- `validation_readiness.json`
- `uncertainty_register.json` and `uncertainty_register.tsv`: every stage's uncertainty implementation, abstention behavior, calibration status, interpretation limit, and evidence paths; missing uncertainty is explicit.
- `hardware_plan.json`
- `storage_preflight.json`: per-input active-pixel estimate, active-stage coefficients, output/work/cache demand, filesystem aggregation, expected headroom and restart worst-case headroom.
- `trace.tsv`
- `timeline.html`
- `dag.html`
- `report.html`
- `outputs_manifest.txt`
- `project_outputs.tsv`
- `project_outputs.json`
- `final_report.md`
- `final_report.json`

Open `00_execution/index.html` first. It deliberately places failures, review-required signals and the evidence ceiling before attractive result images. Then use `00_execution/specimen_atlas.html` to review one specimen at a time without mixing grid-domain and cell-centred observation units. Both pages load only bounded PNG/JPEG QC assets; pyramidal WSI outputs remain links and are not decoded into browser memory.

`hardware_plan.json` records detected CPUs, effective RAM, GPU inventory, host reserves, the selected hardware profile, runtime worker settings, and CPU/RAM allocations for every pipeline stage. `storage_preflight.json` separately prevents CPU/GPU tuning from hiding disk risk: it accounts for durable outputs, Nextflow work, relevant caches and restart duplication on each filesystem. The project output tables list the absolute or resolved address, stage, size, modification time, and unique `output_id` for every relevant published output. The final report includes elapsed time per Nextflow process.
