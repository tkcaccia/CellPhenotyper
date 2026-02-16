#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Build and optionally upload a Singularity/Apptainer SIF asset for this host architecture.

Usage:
  singularity/publish_sif_release_asset.sh [options]

Options:
  --version <semver>             Version string used in asset name (default: 0.2.0)
  --device <cpu|gpu>             Runtime target (default: cpu)
  --source <def|docker>          Build from definition file or pull from docker:// (default: def)
  --outdir <path>                Output directory for .sif (default: .)
  --repo <owner/repo>            GitHub repository for release upload (default: tkcaccia/CellPhenotyper)
  --release-tag <tag>            GitHub release tag (default: v<version>)
  --docker-repo <image_repo>     OCI image repo used when --source docker (default: ghcr.io/tkcaccia/cellphenotyper)
  --docker-tag <tag>             OCI image tag override when --source docker
  --cpu-def <path>               CPU Singularity definition file (default: singularity/cellphenotyper_full_cpu.def)
  --gpu-def <path>               GPU Singularity definition file (default: singularity/cellphenotyper_full_gpu.def)
  --upload                        Upload to GitHub release after build
  --fakeroot                      Use --fakeroot for definition builds (requires host support)
  --force                         Overwrite existing output file
  -h, --help                      Show this message

Examples:
  # Build arm64 CPU SIF on Apple Silicon/Linux ARM and upload it
  singularity/publish_sif_release_asset.sh --version 0.2.0 --device cpu --upload

  # Build amd64 CPU SIF on Linux amd64 and upload it
  singularity/publish_sif_release_asset.sh --version 0.2.0 --device cpu --upload

  # Build from docker:// image instead of definition file
  singularity/publish_sif_release_asset.sh --source docker --device cpu --version 0.2.0 --upload
USAGE
}

normalize_arch() {
  local raw
  raw="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    x86_64|amd64|x64|x86-64)
      echo "amd64"
      ;;
    aarch64|arm64|arm64v8|arm64/v8|armv8|armv8l)
      echo "arm64"
      ;;
    *)
      echo "$raw"
      ;;
  esac
}

VERSION="0.2.0"
DEVICE="cpu"
SOURCE="def"
OUTDIR="."
REPO="tkcaccia/CellPhenotyper"
RELEASE_TAG=""
DOCKER_REPO="ghcr.io/tkcaccia/cellphenotyper"
DOCKER_TAG=""
CPU_DEF="singularity/cellphenotyper_full_cpu.def"
GPU_DEF="singularity/cellphenotyper_full_gpu.def"
UPLOAD="false"
USE_FAKEROOT="false"
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --device)
      DEVICE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
      shift 2
      ;;
    --source)
      SOURCE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --repo)
      REPO="$2"
      shift 2
      ;;
    --release-tag)
      RELEASE_TAG="$2"
      shift 2
      ;;
    --docker-repo)
      DOCKER_REPO="$2"
      shift 2
      ;;
    --docker-tag)
      DOCKER_TAG="$2"
      shift 2
      ;;
    --cpu-def)
      CPU_DEF="$2"
      shift 2
      ;;
    --gpu-def)
      GPU_DEF="$2"
      shift 2
      ;;
    --upload)
      UPLOAD="true"
      shift
      ;;
    --fakeroot)
      USE_FAKEROOT="true"
      shift
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
  echo "Invalid --device '$DEVICE'. Use cpu or gpu." >&2
  exit 1
fi

if [[ "$SOURCE" != "def" && "$SOURCE" != "docker" ]]; then
  echo "Invalid --source '$SOURCE'. Use def or docker." >&2
  exit 1
fi

if [[ -z "$RELEASE_TAG" ]]; then
  RELEASE_TAG="v${VERSION}"
fi

SING_BIN="$(command -v singularity || command -v apptainer || true)"
if [[ -z "$SING_BIN" ]]; then
  echo "singularity/apptainer was not found in PATH." >&2
  exit 1
fi

HOST_ARCH="$(normalize_arch "$(uname -m)")"
if [[ "$DEVICE" == "gpu" && "$HOST_ARCH" != "amd64" ]]; then
  echo "GPU SIF publishing is supported only on amd64 hosts. Detected: $HOST_ARCH" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

name_suffix=""
if [[ "$DEVICE" == "gpu" ]]; then
  name_suffix="-gpu"
fi
ASSET_NAME="cellphenotyper-${VERSION}${name_suffix}-${HOST_ARCH}.sif"
OUT_SIF="${OUTDIR%/}/${ASSET_NAME}"

if [[ -e "$OUT_SIF" && "$FORCE" != "true" ]]; then
  echo "Output already exists: $OUT_SIF (use --force to overwrite)" >&2
  exit 1
fi

if [[ "$SOURCE" == "def" ]]; then
  DEF_FILE="$CPU_DEF"
  if [[ "$DEVICE" == "gpu" ]]; then
    DEF_FILE="$GPU_DEF"
  fi
  if [[ ! -f "$DEF_FILE" ]]; then
    echo "Definition file not found: $DEF_FILE" >&2
    exit 1
  fi

  build_cmd=("$SING_BIN" build)
  if [[ "$USE_FAKEROOT" == "true" ]]; then
    build_cmd+=(--fakeroot)
  fi
  build_cmd+=("$OUT_SIF" "$DEF_FILE")

  if [[ "$USE_FAKEROOT" == "true" || "$EUID" -eq 0 ]]; then
    "${build_cmd[@]}"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "${build_cmd[@]}"
  else
    echo "Definition build requires root/sudo or --fakeroot support." >&2
    exit 1
  fi
else
  if [[ -z "$DOCKER_TAG" ]]; then
    if [[ "$DEVICE" == "gpu" ]]; then
      DOCKER_TAG="${VERSION}-gpu"
    elif [[ "$HOST_ARCH" == "amd64" ]]; then
      DOCKER_TAG="${VERSION}-amd64"
    else
      DOCKER_TAG="${VERSION}"
    fi
  fi
  OCI_REF="docker://${DOCKER_REPO}:${DOCKER_TAG}"
  "$SING_BIN" pull --force "$OUT_SIF" "$OCI_REF"
fi

echo "Built: $OUT_SIF"

if [[ "$UPLOAD" == "true" ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI (gh) is required for --upload." >&2
    exit 1
  fi

  if ! gh release view "$RELEASE_TAG" --repo "$REPO" >/dev/null 2>&1; then
    gh release create "$RELEASE_TAG" --repo "$REPO" --title "$RELEASE_TAG" --notes "Release assets for CellPhenotyper ${VERSION}."
  fi

  gh release upload "$RELEASE_TAG" "$OUT_SIF" --repo "$REPO" --clobber
  echo "Uploaded asset '$ASSET_NAME' to $REPO release $RELEASE_TAG"
fi
