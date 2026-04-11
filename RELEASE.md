# Release

## Latest state

Track official releases here:

- [GitHub Releases](https://github.com/tkcaccia/CellPhenotyper/releases)

Current default branch:

- `main`

Target runtime tags for `v2.2`:

- CPU amd64: `ghcr.io/tkcaccia/cellphenotyper:2.2-amd64`
- CPU arm64: `ghcr.io/tkcaccia/cellphenotyper:2.2-arm64`
- GPU amd64: `ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64`
- GPU arm64: `ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-arm64`

Runtime dependency note:

- Official runtime images are built with the slide-conversion and virtual mIF stack preinstalled, including `tensorflow`, `imagecodecs`, `pyvips`, `openslide`, `rasterio`, `huggingface_hub`, and `timm`.
- Official runtime images now also copy the pipeline code itself into `/opt/cellphenotyper`, so published Docker/SIF artifacts and repository code stay aligned for offline or standalone inspection.

Target Singularity assets for release `v2.2`:

- `cellphenotyper-2.2-amd64.sif`
- `cellphenotyper-2.2-arm64.sif`
- `cellphenotyper-2.2-gpu-amd64.sif`
- `cellphenotyper-2.2-gpu-arm64.sif`

Operational notes:

- GPU SIF files can be large; if GitHub Release asset upload limits are hit, keep Docker images on GHCR as authoritative source and document manual pre-pull + `runtime_image_mode: manual`.
- For user-side run failures and remediation commands, see [Troubleshooting](TROUBLESHOOTING.md).
- Verify GHCR/ORAS availability before telling users to pull a `2.2` asset. Do not assume the tag exists until `docker buildx imagetools inspect ...` or `oras manifest fetch ...` succeeds.

## Versioning policy

- Release tags should follow semantic versioning (`vMAJOR.MINOR.PATCH`).
- Update `INSTALL.md`, `TUTORIAL.md`, `PARAMETERS.md`, and `OUTPUT.md` whenever behavior or defaults change.
- Update `LINUX_UPDATE.md` whenever container tags, Singularity assets, or runtime defaults change.

## Container release workflow (GHCR + Singularity assets)

1. Set credentials:

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env
```

2. Publish Docker images to GHCR with explicit tags:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker buildx create --name cellphenotyper-builder --use --bootstrap 2>/dev/null || docker buildx use cellphenotyper-builder

# Recommended host split:
# - build amd64 images on a native Linux amd64/x86_64 host
# - build arm64 images on a native arm64 host (Apple Silicon or Linux arm64)
# QEMU-emulated amd64 validation of TensorFlow/PyTorch can fail with exit 132 on Apple Silicon.

# amd64 GPU build prerequisite:
# upload the custom TensorFlow wheel asset
# tensorflow-2.22.0.dev0+selfbuilt-cp311-cp311-linux_x86_64.whl
# to the GitHub release v2.2 before building docker/Dockerfile.full.gpu.

docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.full.cpu \
  -t ghcr.io/tkcaccia/cellphenotyper:2.2-amd64 \
  --push .

docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.full.cpu \
  -t ghcr.io/tkcaccia/cellphenotyper:2.2-arm64 \
  --push .

docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.full.gpu \
  -t ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64 \
  --push .

docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.full.gpu \
  -t ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-arm64 \
  --push .

docker buildx imagetools create \
  -t ghcr.io/tkcaccia/cellphenotyper:2.2 \
  ghcr.io/tkcaccia/cellphenotyper:2.2-amd64 \
  ghcr.io/tkcaccia/cellphenotyper:2.2-arm64

docker buildx imagetools create \
  -t ghcr.io/tkcaccia/cellphenotyper:2.2-gpu \
  ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64 \
  ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-arm64
```

3. Verify pushed Docker tags (explicit inspect commands):

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-arm64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-arm64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu

docker buildx imagetools inspect ghcr.io/tkcaccia/cellphenotyper:2.2-amd64
docker buildx imagetools inspect ghcr.io/tkcaccia/cellphenotyper:2.2-arm64
docker buildx imagetools inspect ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
docker buildx imagetools inspect ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-arm64
docker buildx imagetools inspect ghcr.io/tkcaccia/cellphenotyper:2.2
docker buildx imagetools inspect ghcr.io/tkcaccia/cellphenotyper:2.2-gpu
```

4. Build/publish Singularity assets (run on each architecture host):

```bash
# On arm64 host (Mac M1/M2 or Linux arm64)
singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device cpu \
  --upload \
  --upload-mode auto

# On amd64 host (Linux x86_64)
singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device cpu \
  --upload \
  --upload-mode auto
```

Optional GPU Singularity asset (amd64):

```bash
singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device gpu \
  --upload \
  --upload-mode auto
```

Optional GPU Singularity asset (arm64/aarch64 Spark):

```bash
singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device gpu \
  --source docker \
  --docker-tag 2.2-gpu-arm64 \
  --upload \
  --upload-mode auto
```

Important behavior:

- `--source docker` in `publish_sif_release_asset.sh` pulls an existing OCI image and converts it to `.sif`, then publishes the SIF to GHCR over `oras://`.
- Small SIFs can still be mirrored to GitHub Release assets, but large SIFs exceed GitHub's 2 GiB upload limit and therefore must use `oras://`.
- It does **not** build/push Docker images to GHCR.
- Docker publish is a separate step (`docker build` + `docker push`).

## After publishing: Linux user update steps

After a new Docker/Singularity release is published, Linux users should:

1. Pull latest repository changes (`git pull`).
2. Pull the updated Docker image (or clear singularity cache and rerun).
3. Run pipeline with one profile only (`docker` or `singularity`).
4. Verify `results_example/12_cluster_geojson/ROI_grown_mask.geojson`.

Use the complete command list in `LINUX_UPDATE.md`.

For Singularity users, no manual pull command is needed.
Nextflow auto-selects the local/ORAS-hosted `.sif` by architecture/device when using `-profile singularity`.

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --start_point convert \
  --end_point tissue_mask
```

## Changelog source

Use Git history for full change details:

```bash
git log --oneline --decorate --graph
```
