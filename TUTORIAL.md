# How To Use The Pipeline

This page shows practical usage of CellPhenotyper with Nextflow.

Before running, ensure the selected container image already exists.
Do not rebuild `.sif` on every test run; rebuild only after changing the related `.def` file.
Default Singularity image in `pipeline_paramers.yml`: `singularity-cellphenotyper-0.1.0.sif`.

## Starting point

- Input image: `image_input` in `pipeline_paramers.yml` (example: `Data/ROI.ome.tif`)
- ROI polygons: `roi_geojson` in `pipeline_paramers.yml` (example: `Data/ROI.geojson`)

## Endpoint

- Main output base folder: `outdir_base` in `pipeline_paramers.yml` (example: `results_full`)
- Tissue segmentation endpoint: `results_full/04_tissue_geojson/*_tissue_mask.geojson`

## Stage window (where to start/stop)

Use `start_point` and `end_point` in `pipeline_paramers.yml` to select the execution window.
Allowed stages: `convert`, `stardist`, `tissue_mask`, `tissue_geojson`, `cell_assignment`, `cytoplasm`, `uni2`, `kodama`.

Example:

```yaml
start_point: stardist
end_point: uni2
```

This runs from StarDist and stops after UNI-2 embeddings.

## 1) Run the complete pipeline (Singularity)

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

## 2) Run the complete pipeline (Docker)

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

## 3) GPU execution

For GPU-capable hosts:

- set `compute_device: gpu` in `pipeline_paramers.yml`
- use a GPU image in `pipeline_paramers.yml`:
  - Singularity: `singularity_image: singularity/cellphenotyper_full_gpu.sif`
  - Docker: `docker_image: cellphenotyper:full-gpu`

## 4) Resume and rerun

To continue from previous successful steps, keep `-resume`.

To force a clean run with a new output folder:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --outdir_base results_full_fresh
```
