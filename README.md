# CellPhenotyper

CellPhenotyper is a fully reproducible, modular computational pathology workflow that takes raw H&E whole-slide images to quantitative cell-level phenotypes.  
The pipeline performs image conversion/ROI handling, StarDist-based ROI-informed segmentation, UNI-2 foundation-model embeddings, unsupervised KODAMA analysis, and post-KODAMA spatial region construction (`Rcode_Clustering`, `labels_to_cluster_mask`, `grow_to_tissue`, `mask_to_geojson`) to produce final cluster-level GeoJSON.

The workflow is implemented in Nextflow DSL2 with containerized execution for portability, scalability, and auditability across datasets and compute environments.  
Container image selection is automatic by default (`runtime_image_mode: auto`): the pipeline detects host architecture/device and resolves the correct GHCR image.
For full UNI-2 execution, the Hugging Face token must have access to `MahmoodLab/UNI2-h`.
Token loading is configured via `hf_token_env_file` and `hf_token_env_var_name` in `pipeline_paramers.yml`.

The main entrypoint is always:

```bash
nextflow run main.nf
```

## Start here: clone, prepare runtime, run first example

Use this first if you want to run immediately with the repository example input files:

- `Data/ROI.ome.tif`
- `Data/ROI.geojson`

Current OCI image tags (Docker profile, or Singularity fallback):

- CPU amd64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64`
- CPU arm64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- GPU: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`

Current Singularity release assets (auto-selected in `-profile singularity`):

- `cellphenotyper-0.2.0-amd64.sif`
- `cellphenotyper-0.2.0-arm64.sif`
- `cellphenotyper-0.2.0-gpu-amd64.sif`

1. Clone repository:

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
```

2. Prepare runtime (choose one):

```bash
# Singularity/Apptainer:
# no manual pull command is needed.
# Nextflow auto-selects the release-hosted .sif by architecture/device.
# In default `singularity_image_source: auto`, Nextflow tries release `.sif` first
# and automatically falls back to docker:// if the release asset is missing.
# If a matching local `.sif` exists in repo root or `singularity/`, it is used first.

# Docker (optional pre-pull)
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0
```

3. Get and configure UNI-2 token:

- Create/sign in Hugging Face account: `https://huggingface.co`
- Request access to model: `https://huggingface.co/MahmoodLab/UNI2-h`
- Create a read token: `https://huggingface.co/settings/tokens`
- Save token in `tokens.env`:

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
source tokens.env
export HF_TOKEN="$HF_UNI2"
```

4. Run first full example (choose one profile):

Use only one profile per run. Do not run both `-profile singularity` and `-profile docker` for the same output folder.

If you use Lima on macOS, run from a writable Linux path (for example `~/CellPhenotyper`) instead of `/Users/...` mounts to avoid:
`ERROR ~ .nextflow/history.lock (No such file or directory)`.

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

If you need to force a specific image manually, switch to manual mode:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --runtime_image_mode manual \
  --singularity_image https://github.com/tkcaccia/CellPhenotyper/releases/download/v0.2.0/cellphenotyper-0.2.0-amd64.sif \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example
```

5. Final result:

- `results_example/12_cluster_geojson/ROI_grown_mask.geojson`

## Lima Note: writable run directory

Inside `limactl shell default`:

```bash
mkdir -p ~/CellPhenotyper
rsync -a --delete /Users/<your-user>/Documents/test/CellPhenotyper/ ~/CellPhenotyper/
cd ~/CellPhenotyper
```

Run Nextflow from this Linux-home copy.

## Monitor status and copy results back

Check if run is still active:

```bash
ps aux | grep -E 'nextflow|java' | grep -v grep
```

Watch live logs:

```bash
tail -n 50 -f .nextflow.log
```

Copy results to macOS host (from macOS terminal):

```bash
limactl copy default:/home/<lima-user>/CellPhenotyper/results_example \
  /Users/<your-user>/Documents/test/CellPhenotyper/
```

## Documentation

- [Installation](INSTALL.md)
- [How to use](TUTORIAL.md)
- [Parameters](PARAMETERS.md)
- [Output](OUTPUT.md)
- [Release](RELEASE.md)
- [Linux update playbook](LINUX_UPDATE.md)

## Quick start

After completing installation and UNI-2 token setup, edit `pipeline_paramers.yml` (`start_point` / `end_point` included) and run:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  -resume
```

## Runtime images (GHCR + release assets)

Default OCI runtime tags hosted on GHCR:

- CPU amd64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64`
- CPU arm64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- GPU: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`

Default Singularity assets hosted on GitHub Releases (`v0.2.0`):

- `cellphenotyper-0.2.0-amd64.sif`
- `cellphenotyper-0.2.0-arm64.sif`
- `cellphenotyper-0.2.0-gpu-amd64.sif`

Runtime behavior:

```bash
# Docker profile: optional pre-pull
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu
```

For Singularity/Apptainer profile, no manual pull script is required:
`runtime_image_mode: auto` + `singularity_image_source: auto` resolves the correct image by host architecture/device:
release `.sif` when available, otherwise `docker://` fallback.

Use only one runtime profile per execution: either `singularity` or `docker`.

Container publishing is documented in `RELEASE.md`.

## Maintainer: push Docker image to GHCR

Use this when `docker/Dockerfile.full.cpu` or `docker/Dockerfile.full.gpu` changes and you need to publish a new image tag.

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env
export TAG="0.2.0"
export IMAGE="ghcr.io/${GHCR_USER}/cellphenotyper:${TAG}"
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker build -f docker/Dockerfile.full.cpu -t "${IMAGE}" .
docker push "${IMAGE}"
```

GPU tag:

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env
export TAG="0.2.0-gpu"
export IMAGE="ghcr.io/${GHCR_USER}/cellphenotyper:${TAG}"
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker build -f docker/Dockerfile.full.gpu -t "${IMAGE}" .
docker push "${IMAGE}"
```

If you publish architecture-specific CPU tags:

```bash
docker buildx build --platform linux/amd64 -f docker/Dockerfile.full.cpu -t ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64 --push .
docker buildx build --platform linux/arm64 -f docker/Dockerfile.full.cpu -t ghcr.io/tkcaccia/cellphenotyper:0.2.0 --push .
```

Linux machine update workflow after image/definition changes is documented in `LINUX_UPDATE.md`.

Validated tissue GeoJSON example committed in this repository:

- `examples/validated_outputs/ROI_tissue_mask.geojson`
