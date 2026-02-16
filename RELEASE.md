# Release

## Latest state

Track official releases here:

- [GitHub Releases](https://github.com/tkcaccia/CellPhenotyper/releases)

Current default branch:

- `main`

Current published runtime tags:

- CPU default: `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- GPU: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`

## Versioning policy

- Release tags should follow semantic versioning (`vMAJOR.MINOR.PATCH`).
- Update `INSTALL.md`, `TUTORIAL.md`, `PARAMETERS.md`, and `OUTPUT.md` whenever behavior or defaults change.

## Container release workflow (GHCR)

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
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu
```

For Singularity users, no manual pull command is needed.
Nextflow pulls `singularity_image` automatically when using `-profile singularity`.

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
