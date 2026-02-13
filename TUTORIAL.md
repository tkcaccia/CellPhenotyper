# How To Use The Pipeline

This page shows practical usage of CellPhenotyper with Nextflow.

Before running, ensure the selected container image already exists.
Do not rebuild `.sif` on every test run; rebuild only after changing the related `.def` file.

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

## 1) Run the complete pipeline (Singularity + UNI-2)

UNI-2 requires a valid Hugging Face token with access to `MahmoodLab/UNI2-h`.

```bash
export HF_TOKEN="<your_hf_token_with_uni2_access>"

nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

## 2) Run the complete pipeline (Docker + UNI-2)

```bash
export HF_TOKEN="<your_hf_token_with_uni2_access>"

nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

## 3) Run only up to tissue GeoJSON (fast validation)

This is the validated local test path for `Data/ROI.ome.tif` and `Data/ROI.geojson`.

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --start_point convert \
  --end_point tissue_geojson \
  -resume
```

Expected output:

- `results_full/04_tissue_geojson/ROI_tissue_mask.geojson`

## 4) GPU execution

For GPU-capable hosts:

- set `compute_device: gpu` in `pipeline_paramers.yml`
- use a GPU image in `pipeline_paramers.yml`:
  - Singularity: `singularity_image: singularity/cellphenotyper_full_gpu.sif`
  - Docker: `docker_image: cellphenotyper:full-gpu`

## 5) macOS M1 note (if StarDist fails with missing TensorFlow)

If your ARM `.sif` does not include TensorFlow, install it once into a writable host path and pass that path via `stardist_pythonpath`.

```bash
apptainer exec singularity-stardist_UNI-2-m1.sif \
  python -m pip install --no-cache-dir --target /var/tmp/tfdeps tensorflow==2.16.1
```

Then run:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --stardist_pythonpath /var/tmp/tfdeps \
  -resume
```

## 6) Resume and rerun

To continue from previous successful steps, keep `-resume`.

To force a clean run with a new output folder:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --outdir_base results_full_fresh
```
