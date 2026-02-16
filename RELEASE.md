# Release

## Runtime tags expected by automatic architecture selection

- CPU arm64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- CPU amd64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64`
- GPU amd64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`

## Singularity definition files in repository

- `singularity/cellphenotyper_full_cpu.def`
- `singularity/cellphenotyper_full_gpu.def`

## Publish workflow (GHCR)

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
```

### CPU arm64

```bash
export IMAGE_ARM64="ghcr.io/${GHCR_USER}/cellphenotyper:0.2.0"
docker buildx build --platform linux/arm64 -f docker/Dockerfile.full.cpu -t "$IMAGE_ARM64" --push .
```

### CPU amd64

```bash
export IMAGE_AMD64="ghcr.io/${GHCR_USER}/cellphenotyper:0.2.0-amd64"
docker buildx build --platform linux/amd64 -f docker/Dockerfile.full.cpu -t "$IMAGE_AMD64" --push .
```

### GPU amd64

```bash
export IMAGE_GPU="ghcr.io/${GHCR_USER}/cellphenotyper:0.2.0-gpu"
docker buildx build --platform linux/amd64 -f docker/Dockerfile.full.gpu -t "$IMAGE_GPU" --push .
```

## Verify pulls

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu
```
