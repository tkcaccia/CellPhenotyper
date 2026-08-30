# Release

## Published runtimes

Track source releases on [GitHub Releases](https://github.com/tkcaccia/CellPhenotyper/releases). Runtime images and SIF artifacts are published through GHCR.

The pipeline defaults point only to tags verified to exist:

| Runtime | Published image or artifact |
| --- | --- |
| CPU amd64 | `ghcr.io/tkcaccia/cellphenotyper:2.2-amd64` |
| CPU arm64 | `ghcr.io/tkcaccia/cellphenotyper:0.2.0` |
| GPU amd64 | `ghcr.io/tkcaccia/cellphenotyper-runtime:2.7-gpu-amd64` |
| CPU SIF amd64 | `oras://ghcr.io/tkcaccia/cellphenotyper:2.2-sif-amd64` |
| CPU SIF arm64 | `oras://ghcr.io/tkcaccia/cellphenotyper:2.2-sif-arm64` |
| GPU SIF amd64 | `oras://ghcr.io/tkcaccia/cellphenotyper:2.2-sif-gpu-amd64` |
| GPU SIF arm64 | `oras://ghcr.io/tkcaccia/cellphenotyper:2.2-sif-gpu-arm64` |

Do not document or configure a tag until its registry manifest has been checked. In particular, the former `2.3` and `2.6` examples were unpublished build targets, not pullable releases.

The GPU amd64 update is produced by `.github/workflows/publish-runtime-release.yml`. A tag such as `runtime-v2.7-buildN` publishes `2.7-gpu-amd64` to `cellphenotyper-runtime` and validates the bundled R, Python, model, and pipeline test stacks.

## Runtime requirements

- Official images include slide conversion, virtual mIF, UNI-2, StarDist, MedSAM, TITAN/CONCH, KODAMA, and the required R libraries.
- Runtime images copy the pipeline into `/opt/cellphenotyper` so code and dependencies can be validated together.
- Normal runs must not depend on host `R_LIBS_USER`, host Python packages, or task-local package installation.
- Nextflow launches on the host and requires Java 17 or newer even when all processes run in Docker or Singularity.
- Internal tissue masks must remain ordinary compressed TIFFs because downstream `tifffile` consumers reopen them during `GROW_TO_TISSUE`.
- Large SIF files should be distributed through GHCR/ORAS. GitHub Release assets have a 2 GiB limit.

## Automated publication

The preferred release path is the GitHub Actions workflow:

1. Use `workflow_dispatch` with `gpu-amd64-update` to refresh only the current amd64 GPU runtime.
2. Use `workflow_dispatch` with `all` to build the CPU/GPU architecture matrix, multi-architecture manifests, and SIF artifacts.
3. Use a `runtime-vX.Y-buildN` tag for the amd64 GPU update path.

The workflow runs the complete `pytest` suite inside the published amd64 GPU runtime. The full matrix also builds architecture-specific images and publishes SIF artifacts to GHCR over ORAS.

## Manual container publication

Use a new version rather than overwriting an existing release:

```bash
export VERSION=X.Y.Z
export IMAGE_REPO=ghcr.io/tkcaccia/cellphenotyper

echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker buildx create --name cellphenotyper-builder --use --bootstrap 2>/dev/null || \
  docker buildx use cellphenotyper-builder

docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.full.cpu \
  -t "${IMAGE_REPO}:${VERSION}-amd64" --push .

docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.full.cpu \
  -t "${IMAGE_REPO}:${VERSION}-arm64" --push .

docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.full.gpu \
  -t "${IMAGE_REPO}:${VERSION}-gpu-amd64" --push .

docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.full.gpu \
  -t "${IMAGE_REPO}:${VERSION}-gpu-arm64" --push .
```

Build natively on each target architecture where possible. QEMU-emulated TensorFlow or PyTorch validation can fail even when a native runtime is valid.

After all architecture-specific images pass validation, create the aggregate tags:

```bash
docker buildx imagetools create \
  -t "${IMAGE_REPO}:${VERSION}" \
  "${IMAGE_REPO}:${VERSION}-amd64" \
  "${IMAGE_REPO}:${VERSION}-arm64"

docker buildx imagetools create \
  -t "${IMAGE_REPO}:${VERSION}-gpu" \
  "${IMAGE_REPO}:${VERSION}-gpu-amd64" \
  "${IMAGE_REPO}:${VERSION}-gpu-arm64"
```

## Release validation

Check every manifest before updating defaults or user documentation:

```bash
docker buildx imagetools inspect "${IMAGE_REPO}:${VERSION}-amd64"
docker buildx imagetools inspect "${IMAGE_REPO}:${VERSION}-arm64"
docker buildx imagetools inspect "${IMAGE_REPO}:${VERSION}-gpu-amd64"
docker buildx imagetools inspect "${IMAGE_REPO}:${VERSION}-gpu-arm64"
```

Validate the package stacks and repository tests in the built GPU image:

```bash
docker run --rm "${IMAGE_REPO}:${VERSION}-gpu-amd64" \
  Rscript -e 'library(KODAMA); library(KODAMAextra); library(SPARK); library(umap); cat("R packages OK\n")'

docker run --rm --gpus all "${IMAGE_REPO}:${VERSION}-gpu-amd64" \
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)'

docker run --rm "${IMAGE_REPO}:${VERSION}-gpu-amd64" bash -lc \
  'cd /opt/cellphenotyper && python -m pytest -q tests'
```

## Singularity publication

Build and publish an architecture-specific SIF from the already validated Docker image:

```bash
./singularity/publish_sif_release_asset.sh \
  --version "${VERSION}" \
  --source docker \
  --device cpu \
  --docker-repo "${IMAGE_REPO}" \
  --docker-tag "${VERSION}-amd64" \
  --oras-repo "${IMAGE_REPO}" \
  --asset-arch amd64 \
  --upload-mode auto \
  --upload
```

Repeat with the matching device, architecture, and Docker tag. `--source docker` converts an existing OCI image; it does not build or push the Docker image.

## Updating defaults

After all artifacts are published and verified, update these files together:

- `nextflow.config`
- `pipeline_paramers.yml`
- `README.md`
- `INSTALL.md`
- `PARAMETERS.md`
- `LINUX_UPDATE.md`
- `singularity/README.md`

Then run the repository test suite and a minimal Nextflow smoke test for each affected runtime path. Verify final spatial output under `results/15_cluster_geojson/<sample>/`.

For user-side failures and remediation commands, see [Troubleshooting](TROUBLESHOOTING.md).
