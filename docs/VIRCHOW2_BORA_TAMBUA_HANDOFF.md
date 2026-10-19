# CellPhenotyper feature and marker handoff

CellPhenotyper has two distinct observation routes on an H&E image. The tissue route creates non-overlapping Virchow2 tile embeddings for KODAMA, followed by a coarse mask. The Bora bridge then computes a full 1280-channel spatial-token field on calibrated, bounded WSI windows from the same pinned Virchow2 checkpoint. The cell route creates instance masks and quantifies GigaTIME's 21 biological/nuclear **virtual-marker scores** per cell. [Tambua](https://github.com/tkcaccia/Tambua) uses those per-cell tables with separately supplied cell-type labels and the GitHub fastPLS 0.4 classifier.

## Tissue route to Bora

| Producer | Output | Consumer |
| --- | --- | --- |
| Calibrated crop and resolution validation | uint8 RGB crop, physical-resolution sidecar | Virchow2 tile stage and Bora bridge |
| GrandQC | aligned tissue-support mask | tile selection and Bora background exclusion |
| Virchow2 tile stage | one 2560-value pooled embedding per disjoint 224-pixel tile at 0.5 µm/pixel | KODAMA clustering |
| KODAMA and cluster-mask stage | complete anonymous coarse tile mask at native pixel coordinates | Bora bridge |
| Bora bridge's pinned Virchow2 pass | complete 1280-channel spatial-token field per bounded window | Bora RGB evidence, refinement and Neighbor context classification |
| Bora | native classified OME-TIFF, compact GeoJSON and provenance | downstream spatial analysis |

Pooled 2560-value embeddings are **not** Bora's spatial-token input. The bridge recomputes the unselected spatial tokens within bounded windows; current software does not persist a whole-slide token tensor. Each window checks source MPP, grid origin, zero overlap, native mask shape and the audited checkpoint revision/hash before classification. Non-overlapping encoder tiles are padded white only at outer edges. The bridge does not read the test image's expert annotation.

## Cell route to Tambua

CellPhenotyper's GigaTIME stage writes `quantification_<sample>/<sample>_nuclei_gigatime_mean_intensity.csv`, the analogous `cyto` table, and/or `cell` when a whole-cell instance mask is supplied with `--cell-mask`. Each table contains a positive integer `label_id`, `mask_name`, `value_semantics=uncalibrated_virtual_marker_score`, per-cell area and the named channel means. `label_id` must refer to the same cell in the chosen compartments. Tambua joins on `(sample_id, label_id)`; it refuses duplicate IDs, missing marker panels, non-finite scores and patient overlap between train and test. TRITC and Cy5 are background channels and are excluded. DAPI remains a nuclear morphology control. A separately curated `label_id,cell_type` file supplies ground truth; classes must not be inferred from GigaTIME scores used as predictors.

Standalone and integrated GigaTIME quantifiers now use compact indices over **observed** cell IDs rather than arrays sized by the largest numeric ID. Singleton mask channel axes stay lazy, and mask-to-image resampling uses absolute pixel-centre coordinates independent of block boundaries. This protects sparse WSI cell IDs and reproducible cell locations. The tests use sparse IDs including 100000001 and a real TIFF/Zarr singleton plane.

MoNuSAC expert nuclei provide an initial controlled cell-classification reference; using those expert masks to quantify virtual markers tests the marker/classifier step, not CellPhenotyper's detector. End-to-end evaluation must match pipeline-detected cells to reference instances at a locked distance/IoU gate and account for unmatched cells. Same-section paired H&E/protein cohorts such as ORION are needed to evaluate whether GigaTIME marker signals agree with measured biology. Nominal magnification alone is not a verified micrometres-per-pixel calibration.

The current GPU runtime image must be rebuilt and tested with `timm`, `safetensors`, `pyvips`, the installed Bora wheel and its corrected native fastPLS extension before claiming a deployable full-slide run. CPU interface tests do not validate GPU throughput or complete-WSI operation. UCT HPC GPU work remains paused by the project policy; new Chiamaka GPU runs require their own bounded execution scope.
