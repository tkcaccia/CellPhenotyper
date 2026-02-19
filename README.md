# CellPhenotyper

CellPhenotyper is a Nextflow DSL2 pipeline for H&E tissue image analysis. It runs StarDist segmentation, extracts UNI-2 embeddings, performs KODAMA-based clustering, and generates a final tissue cluster GeoJSON.

Main command:

```bash
nextflow run main.nf
```

## Documentation

- [Installation](INSTALL.md)
- [How to run](TUTORIAL.md)
- [Parameters](PARAMETERS.md)
- [Output](OUTPUT.md)
- [Release](RELEASE.md)
- [Linux update playbook](LINUX_UPDATE.md)
- [Singularity maintainer guide](singularity/README.md)

## Example input in this repository

- `Data/ROI_A.ome.tif`
- `Data/ROI_A.geojson`
- `Data/ROI_B.ome.tif`
- `Data/ROI_B.geojson`

If `Data/<sample>.geojson` is missing, CellPhenotyper automatically uses the full image as ROI for that sample.

## Runtime behavior (automatic container selection)

Use one profile per run:

- `-profile docker`
- `-profile singularity`

Do not use both profiles in the same run.

Default image selection is automatic (`runtime_image_mode: auto`):

- Docker profile uses GHCR images.
- Singularity profile auto-resolves architecture-specific `.sif` assets when available.
- In GPU mode, only GPU-capable steps (StarDist, UNI-2 embeddings) use the GPU container; other steps stay on the CPU container.
- On arm64 GPU runs, missing GPU assets fall back to CPU containers (no amd64 GPU image fallback).

Current tags/assets (`v0.2.0`):

- Docker CPU amd64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64`
- Docker CPU arm64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- Docker GPU amd64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`
- Singularity CPU amd64: `cellphenotyper-0.2.0-amd64.sif`
- Singularity CPU arm64: `cellphenotyper-0.2.0-arm64.sif`
- Singularity GPU amd64: `cellphenotyper-0.2.0-gpu-amd64.sif`
- Singularity GPU arm64 (optional): `cellphenotyper-0.2.0-gpu-arm64.sif`

## UNI-2 token setup (required)

1. Create/sign in at [Hugging Face](https://huggingface.co).
2. Request access to [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h).
3. Create a read token at [Hugging Face tokens](https://huggingface.co/settings/tokens).
4. In the project root:

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
source tokens.env
export HF_TOKEN="${HF_UNI2}"
```

Run these `source/export` commands in every new shell before starting Nextflow.

If you get `401 Unauthorized` during UNI-2 download, check token validity and model access approval.

## Linux quick run

Before every run, sync and verify you are on the latest `main`:

```bash
git fetch origin
git checkout main
git pull --ff-only
git rev-parse --short HEAD
git rev-parse --short origin/main
git status --short
```

`HEAD` and `origin/main` must match, and `git status --short` should be empty.

Linux runtime precheck:

```bash
java -version
nextflow -version
```

If `nextflow` is not in PATH in your shell, use `/home/<user>/.local/bin/nextflow`.

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
source tokens.env
export HF_TOKEN="${HF_UNI2}"
```

Docker:

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Docker (GPU, Linux amd64 + NVIDIA):

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example_gpu \
  --compute_device gpu \
  --host_arch amd64
```

Singularity/Apptainer:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Singularity/Apptainer (GPU, Linux amd64 + NVIDIA):

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example_gpu \
  --compute_device gpu \
  --host_arch amd64
```

Singularity/Apptainer (GPU, Linux arm64 + NVIDIA):

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example_gpu_arm64 \
  --compute_device gpu \
  --host_arch arm64 \
  --enable_gpu_on_arm64 true
```

Rerun only `10_cluster_mask` and `11_grown_tissue`:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example \
  --start_point cluster_mask \
  --end_point grow_tissue
```

## macOS quick run

Docker (native with Docker Desktop):

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
source tokens.env
export HF_TOKEN="${HF_UNI2}"
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Singularity (via Lima Linux VM):

```bash
limactl shell default
mkdir -p ~/CellPhenotyper
rsync -a --delete /Users/<your-user>/Documents/CellPhenotyper/ ~/CellPhenotyper/
cd ~/CellPhenotyper
source tokens.env
export HF_TOKEN="${HF_UNI2}"
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Important for Lima: run from a writable Linux path (for example `~/CellPhenotyper`), not from `/Users/...` mount paths.
On Apple Silicon/Linux arm64, GPU mode requires an arm64-compatible GPU container asset (`singularity_gpu_asset_arm64` or `gpu_container_image`).

## Check status and outputs

Run status:

```bash
ps aux | grep -E 'nextflow|java' | grep -v grep
tail -n 50 -f .nextflow.log
```

Final output:

- `results_example/12_cluster_geojson/ROI_A/ROI_A_grown_mask_smooth_class.geojson`
- `results_example/12_cluster_geojson/ROI_B/ROI_B_grown_mask_smooth_class.geojson`

Execution report:

- `results_example/00_execution/final_report.md`
- `results_example/00_execution/final_report.json`

Copy results from Lima to macOS host:

```bash
limactl copy default:/home/<lima-user>/CellPhenotyper/results_example \
  /Users/<your-user>/Documents/CellPhenotyper/
```

## Maintainer: publish updated containers

Docker and Singularity build/publish workflows are documented here:

- [Release](RELEASE.md)
- [Singularity maintainer guide](singularity/README.md)
- [Linux update playbook](LINUX_UPDATE.md)

Minimal Docker publish example:

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env
export TAG="0.2.0"
export IMAGE="ghcr.io/${GHCR_USER}/cellphenotyper:${TAG}"
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker build -f docker/Dockerfile.full.cpu -t "${IMAGE}" .
docker push "${IMAGE}"
```
