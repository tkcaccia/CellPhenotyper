# Installation (Ubuntu, macOS, Windows)

## 1) Java + Nextflow

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless curl
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
nextflow -version
```

macOS / Windows equivalents are supported as long as Java 17 and Nextflow are available.

## 2) Choose runtime profile

Use one profile per run:

- `-profile singularity`
- `-profile docker`

## 3) Runtime tools

Docker path:

```bash
docker --version
```

Singularity/Apptainer path:

```bash
apptainer --version || singularity --version
```

On macOS for Singularity, use Lima (`limactl shell default`).
On Windows for Singularity, use WSL2 Ubuntu.

## 4) Clone repository

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
```

## 5) Configure UNI-2 token

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
```

## 6) Architecture-aware runtime config (recommended)

`pipeline_paramers.yml` already defaults to automatic selection:

- `compute_device: auto`
- `host_arch: auto`
- `singularity_image: ""`
- `docker_image: ""`

Auto tags:

- CPU arm64: `0.2.0`
- CPU amd64: `0.2.0`
- GPU amd64: `0.2.0-gpu`

Force architecture only if detection is wrong:

```bash
--host_arch amd64
```

## 7) Run first example

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example
```

## 8) Maintainer: container publish

Definition files in repo:

- `singularity/cellphenotyper_full_cpu.def`
- `singularity/cellphenotyper_full_gpu.def`

Publish GHCR tags used by runtime auto-selection:

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env

echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin

# CPU arm64 default tag
export IMAGE_ARM64="ghcr.io/${GHCR_USER}/cellphenotyper:0.2.0"
docker buildx build --platform linux/arm64 -f docker/Dockerfile.full.cpu -t "$IMAGE_ARM64" --push .

# CPU amd64 tag
export IMAGE_AMD64="ghcr.io/${GHCR_USER}/cellphenotyper:0.2.0"
docker buildx build --platform linux/amd64 -f docker/Dockerfile.full.cpu -t "$IMAGE_AMD64" --push .

# GPU amd64 tag
export IMAGE_GPU="ghcr.io/${GHCR_USER}/cellphenotyper:0.2.0-gpu"
docker buildx build --platform linux/amd64 -f docker/Dockerfile.full.gpu -t "$IMAGE_GPU" --push .
```
