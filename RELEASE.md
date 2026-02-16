# Release

## Latest state

Track official releases here:

- [GitHub Releases](https://github.com/tkcaccia/CellPhenotyper/releases)

Current default branch:

- `main`

Current published runtime tags:

- CPU amd64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64`
- CPU arm64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- GPU: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`

Current published Singularity assets on release `v0.2.0`:

- `cellphenotyper-0.2.0-amd64.sif`
- `cellphenotyper-0.2.0-arm64.sif`
- `cellphenotyper-0.2.0-gpu-amd64.sif`

## Versioning policy

- Release tags should follow semantic versioning (`vMAJOR.MINOR.PATCH`).
- Update `INSTALL.md`, `TUTORIAL.md`, `PARAMETERS.md`, and `OUTPUT.md` whenever behavior or defaults change.

## Container release workflow (GHCR + Singularity assets)

1. Set credentials:

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env
```

2. Publish CPU image for a new version:

```bash
export TAG="0.2.0"
export IMAGE="ghcr.io/${GHCR_USER}/cellphenotyper:${TAG}"
docker build -f docker/Dockerfile.full.cpu -t "${IMAGE}" .
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker push "${IMAGE}"
```

3. Publish GPU image (Linux x86_64 + NVIDIA only):

```bash
export TAG="0.2.0-gpu"
export IMAGE="ghcr.io/${GHCR_USER}/cellphenotyper:${TAG}"
docker build -f docker/Dockerfile.full.gpu -t "${IMAGE}" .
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker push "${IMAGE}"
```

4. Verify pull:

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu
```

5. Build/upload Singularity release assets (run on each architecture host):

```bash
# On arm64 host (Mac M1/M2 or Linux arm64)
singularity/publish_sif_release_asset.sh \
  --version 0.2.0 \
  --device cpu \
  --upload

# On amd64 host (Linux x86_64)
singularity/publish_sif_release_asset.sh \
  --version 0.2.0 \
  --device cpu \
  --upload
```

Optional GPU Singularity asset (amd64 only):

```bash
singularity/publish_sif_release_asset.sh \
  --version 0.2.0 \
  --device gpu \
  --upload
```

Important behavior:

- `--source docker` in `publish_sif_release_asset.sh` pulls an existing OCI image and converts it to `.sif`, then uploads the `.sif` release asset.
- It does **not** build/push Docker images to GHCR.
- Docker publish is a separate step (`docker build` + `docker push`).

For Singularity users, no manual pull command is needed.
Nextflow auto-selects the release-hosted `.sif` by architecture/device when using `-profile singularity`.

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
