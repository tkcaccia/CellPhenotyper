# How To Use The Pipeline

This page shows practical usage of CellPhenotyper with Nextflow.

For `-profile docker`, you can pre-pull image from GHCR.
For `-profile singularity`, Nextflow pulls automatically from `singularity_image`.
Do not rebuild `.sif` on every test run.

## Easy example from repository data

Use the input files already present in this repo:

- `Data/ROI_A.ome.tif`
- `Data/ROI_A.geojson`
- `Data/ROI_B.ome.tif`
- `Data/ROI_B.geojson`

Clone and enter repository:

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
```

Prepare runtime:

```bash
# Singularity/Apptainer:
# no manual pull command is required.
# Nextflow pulls `singularity_image` automatically on first run.

# Docker (optional pre-pull, amd64 CPU example)
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64
```

Create UNI-2 token file (required for full pipeline):

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
source tokens.env
export HF_TOKEN="${HF_UNI2}"
```

Run full example:

```bash
# Singularity
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example

# Docker
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Final results:

- `results_example/12_cluster_geojson/ROI_A/ROI_A_grown_mask_smooth_class.geojson`
- `results_example/12_cluster_geojson/ROI_B/ROI_B_grown_mask_smooth_class.geojson`

## Lima users: run location matters

If your project is under `/Users/...` inside Lima and that mount is read-only, Nextflow may fail creating
`.nextflow/history.lock`.

Use a writable Linux path:

```bash
cd ~
rsync -a --delete /Users/<your-user>/Documents/test/CellPhenotyper/ ~/CellPhenotyper/
cd ~/CellPhenotyper
```

Then run the pipeline from `~/CellPhenotyper`.

## Check run progress

```bash
ps aux | grep -E 'nextflow|java' | grep -v grep
tail -n 50 -f .nextflow.log
```

Quick recent activity:

```bash
ls -lt work | head
```

## Copy outputs back to macOS

From macOS terminal:

```bash
limactl copy default:/home/<lima-user>/CellPhenotyper/results_example \
  /Users/<your-user>/Documents/test/CellPhenotyper/
```

## Starting point

- Multi-image input folder: `folder_input` in `pipeline_paramers.yml` (example: `Data`)
- Single-image mode (optional): `image_input` + optional `roi_geojson`
- If a `<sample>.geojson` file is missing, the full image is used as ROI.

## Endpoint

- Main output base folder: `outdir_base` in `pipeline_paramers.yml` (example: `results_full`)
- Final segmentation endpoint: `results_full/12_cluster_geojson/<sample_root>/<sample_root>_grown_mask_smooth_class.geojson`

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
Default runtime image mode in `pipeline_paramers.yml` is automatic (`runtime_image_mode: auto`), so `singularity_image` stays empty unless you want manual override.

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

Default image selection is automatic (`runtime_image_mode: auto`), so no manual `docker_image` is required.

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

- `-profile singularity`: Nextflow resolves image automatically (local `.sif` -> release asset -> `docker://` fallback).
- `-profile docker`: pull image with Docker if needed:

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu-arm64
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

- `results_full/03_tissue_mask/ROI_A/ROI_A_tissue_mask.tif`
- `results_full/03_tissue_mask/ROI_B/ROI_B_tissue_mask.tif`

## 5) GPU execution

For GPU-capable Linux hosts (amd64/arm64 + NVIDIA), run:

```bash
# Docker profile
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_gpu \
  --compute_device gpu \
  --host_arch amd64

# Singularity profile
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_gpu \
  --compute_device gpu \
  --host_arch amd64
```

For Linux arm64/aarch64 Spark, use `--host_arch arm64` (same commands).

## 6) macOS M1 note (if StarDist fails with missing TensorFlow)

If your ARM `.sif` does not include TensorFlow, install it once into a writable host path and pass that path via `stardist_pythonpath`.

```bash
apptainer exec /path/to/cellphenotyper-0.2.0-arm64.sif \
  python -m pip install --no-cache-dir --target /var/tmp/tfdeps tensorflow==2.16.2
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
