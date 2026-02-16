# How To Use The Pipeline

This page shows practical usage of CellPhenotyper with Nextflow.

For `-profile docker`, you can pre-pull image from GHCR.
For `-profile singularity`, Nextflow pulls automatically from `singularity_image`.
Do not rebuild `.sif` on every test run.

## Easy example from repository data

Use the input files already present in this repo:

- `Data/ROI.ome.tif`
- `Data/ROI.geojson`

Clone and enter repository:

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
```

Prepare CPU runtime:

```bash
# Singularity/Apptainer:
# no manual pull command is required.
# Nextflow pulls `singularity_image` automatically on first run.

# Docker (optional pre-pull)
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0
```

Create UNI-2 token file (required for full pipeline):

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
```

Run full example:

```bash
# Singularity
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example

# Docker
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example
```

Final result:

- `results_example/12_cluster_geojson/ROI_grown_mask.geojson`

## Starting point

- Input image: `image_input` in `pipeline_paramers.yml` (example: `Data/ROI.ome.tif`)
- ROI polygons: `roi_geojson` in `pipeline_paramers.yml` (example: `Data/ROI.geojson`)

## Endpoint

- Main output base folder: `outdir_base` in `pipeline_paramers.yml` (example: `results_full`)
- Final segmentation endpoint: `results_full/12_cluster_geojson/*_grown_mask.geojson`

## Stage window (where to start/stop)

Use `start_point` and `end_point` in `pipeline_paramers.yml` to select the execution window.
Allowed stages:
`convert`, `stardist`, `tissue_mask`, `cell_assignment`, `cytoplasm`, `uni2`, `kodama`, `clustering`, `cluster_mask`, `grow_tissue`, `cluster_geojson`.

Example:

```yaml
start_point: stardist
end_point: uni2
```

This runs from StarDist and stops after UNI-2 embeddings.

## 1) Run the complete pipeline (Singularity + UNI-2)

UNI-2 requires a valid Hugging Face token with access to `MahmoodLab/UNI2-h`.
By default the pipeline loads the token from `tokens.env` using `hf_token_env_file` + `hf_token_env_var_name` in `pipeline_paramers.yml`.
Default runtime image in `pipeline_paramers.yml` is:
`singularity_image: docker://ghcr.io/tkcaccia/cellphenotyper:0.2.0`.

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

## 2) Run the complete pipeline (Docker + UNI-2)

Default runtime image in `pipeline_paramers.yml` is:
`docker_image: ghcr.io/tkcaccia/cellphenotyper:0.2.0`.

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

## 3) Runtime image behavior

- `-profile singularity`: Nextflow pulls `singularity_image` automatically.
- `-profile docker`: pull image with Docker if needed:

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu
```

## 4) Run only tissue mask stage (fast validation)

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --start_point convert \
  --end_point tissue_mask \
  -resume
```

Expected output:

- `results_full/03_tissue_mask/ROI_tissue_mask.tif`

## 5) GPU execution

For GPU-capable hosts:

- set `compute_device: gpu` in `pipeline_paramers.yml`
- use a GPU image in `pipeline_paramers.yml`:
  - Singularity: `singularity_image: docker://ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`
  - Docker: `docker_image: ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`

## 6) macOS M1 note (if StarDist fails with missing TensorFlow)

If your ARM `.sif` does not include TensorFlow, install it once into a writable host path and pass that path via `stardist_pythonpath`.

```bash
apptainer exec singularity/cellphenotyper_0.2.0_cpu.sif \
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

## 7) Resume and rerun

To continue from previous successful steps, keep `-resume`.

To force a clean run with a new output folder:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --outdir_base results_full_fresh
```
