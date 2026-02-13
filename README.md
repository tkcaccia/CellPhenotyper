# CellPhenotyper

Nextflow DSL2 pipeline for:

- input OME-TIFF preparation
- ROI StarDist segmentation
- tissue mask extraction
- tissue mask to GeoJSON conversion
- optional downstream assignment, cytoplasm expansion, UNI2 embeddings, and KODAMA analysis

## Key points

- Local profiles only: `singularity` and `docker`
- No SLURM profile
- Resource controls: `--max_cpus`, `--max_memory_gb`, `--compute_device`
- GPU path supported via `--compute_device gpu` and singularity `--nv`

## Main files

- `main.nf`
- `nextflow.config`
- `modules/prepare_input_ometiff.nf`
- `modules/run_stardist_roi_segmentation.nf`
- `modules/build_tissue_mask.nf`
- `modules/convert_tissue_mask_to_geojson.nf`
- `modules/map_cells_to_roi_polygons.nf`
- `modules/expand_labels_to_cytoplasm.nf`
- `modules/extract_uni2_embeddings.nf`
- `modules/run_kodama_analysis.nf`

## Setup and usage

- Installation guide: `INSTALL.md`
- Step-by-step tutorial: `TUTORIAL.md`

## Verified local result

A successful local singularity run produced:

- `results_singularity_local/04_tissue_geojson/ROI_tissue_mask.geojson`
- `examples/validated_outputs/ROI_tissue_mask.geojson` (copied, GitHub-safe artifact)

and completed with:

- `PIPELINE COMPLETED SUCCESSFULLY`
