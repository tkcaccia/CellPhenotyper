# Output

With `--outdir_base results_full`, CellPhenotyper writes the following stage directories:

- `00_execution/`: Nextflow trace, timeline, DAG, report, runtime summaries, and the project output manifest with a stable `output_id` for every published file.
- `01_input/`: normalized pyramidal OME-TIFF input.
- `02_grandqc/`: GrandQC tissue/artifact masks, GeoJSON, summaries, and previews.
- `03_stardist/`: StarDist crop, nuclei labels, object table, crop GeoJSON, and coordinate shift.
- `03b_hovernet_monusac/`: official HoVer-Net fast MoNuSAC inference normalized to the StarDist crop frame.
- `03c_cellvitpp/`: official CellViT++ PanNuke inference normalized to the StarDist crop frame.
- `03d_cell_consensus/`: aligned two-of-three StarDist, HoVer-Net, and CellViT++ consensus.
- `04_TMA/`: TMA decision, core polygons, and cells assigned to cores when applicable.
- `05_gigatime/`: GigaTIME virtual mIHC image plus nucleus/cytoplasm marker quantification.
- `06_roi/`: crop-aligned ROI GeoJSON, labeled mask, class map, and preview.
- `07_cell_assignments/`: cells assigned to ROI polygons.
- `08_cytoplasm/`: expanded cytoplasm labels.
- `09_embeddings/`: cell-centered UNI2-h tile and inner-square embeddings.
- `10_kodama/`: KODAMA coordinates, plots, and logs.
- `11_clustering/`: standard/fine cluster assignments, plots, and logs.
- `12_cluster_mask/`: raster cluster masks.
- `13_grown_tissue/`: cluster labels grown to tissue.
- `14_medsam_refine_tissue/`: GPU MedSAM border refinement and full-resolution QC crops.
- `15_cluster_geojson/`: final connected tissue-section polygons.
- `16_neoplastic_section/`: per-section cell counts and the section with the most CellViT++ neoplastic cells.
- `17_titan/`: CONCH v1.5 patch features and one 768-dimensional TITAN section embedding.
- `18_pathofmpred/`: cancer-specific PathoFMPred research predictions and reports.

## Named Cell Types

`03b_hovernet_monusac/<sample>/hovernet_<sample>/hovernet_cells.json` records both `type_id` and `type`. The MoNuSAC map is:

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

`03d_cell_consensus/<sample>/consensus_<sample>/objects.csv` retains each detector's named class and numeric ID in separate columns. `alignment.csv` includes accepted and rejected predictions, while `consensus_summary.json`, `consensus_cells.geojson`, and `consensus_preview.png` provide QC and provenance.

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
- runtime and provenance text files

These are TCGA-derived research estimates. They are not externally validated clinical assays, and binary scores must not be interpreted as calibrated probabilities.

## Execution Reports

Important files under `00_execution/` include:

- `trace.tsv`
- `timeline.html`
- `dag.html`
- `report.html`
- `outputs_manifest.txt`
- `project_outputs.tsv`
- `project_outputs.json`
- `final_report.md`
- `final_report.json`

The project output tables list the absolute or resolved address, stage, size, modification time, and unique `output_id` for every relevant published output. The final report includes elapsed time per Nextflow process.
