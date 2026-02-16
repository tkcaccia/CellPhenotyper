#!/usr/bin/env bash
set -euo pipefail

# Pull a published CellPhenotyper image from GHCR and optionally materialize a
# Singularity/Apptainer SIF from the same image tag.
#
# Examples:
#   ./docker/pull_runtime_image.sh --tag 0.2.0 --variant cpu
#   ./docker/pull_runtime_image.sh --tag 0.2.0 --variant gpu --no-singularity

usage() {
  cat <<'USAGE'
Usage:
  ./docker/pull_runtime_image.sh --tag <version> [--variant cpu|gpu] [--no-docker] [--no-singularity]

Options:
  --tag             Required. Image tag version (example: 0.2.0)
  --variant         cpu (default) or gpu
  --no-docker       Skip docker pull
  --no-singularity  Skip SIF pull

Environment:
  CONTAINER_CLI Optional override: docker or nerdctl
USAGE
}

TAG=""
VARIANT="cpu"
DO_PULL_DOCKER="true"
DO_PULL_SIF="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --variant)
      VARIANT="${2:-}"
      shift 2
      ;;
    --no-docker)
      DO_PULL_DOCKER="false"
      shift
      ;;
    --no-singularity)
      DO_PULL_SIF="false"
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

if [[ -z "${TAG}" ]]; then
  echo "Missing required --tag" >&2
  usage
  exit 1
fi

if [[ "${VARIANT}" != "cpu" && "${VARIANT}" != "gpu" ]]; then
  echo "--variant must be cpu or gpu" >&2
  exit 1
fi

IMAGE_BASE="ghcr.io/tkcaccia/cellphenotyper"
IMAGE_TAG="${TAG}"
if [[ "${VARIANT}" == "gpu" ]]; then
  IMAGE_TAG="${TAG}-gpu"
fi
IMAGE="${IMAGE_BASE}:${IMAGE_TAG}"

if [[ "${DO_PULL_DOCKER}" == "true" ]]; then
  CONTAINER_CLI="${CONTAINER_CLI:-}"
  if [[ -z "${CONTAINER_CLI}" ]]; then
    if command -v docker >/dev/null 2>&1; then
      CONTAINER_CLI="docker"
    elif command -v nerdctl >/dev/null 2>&1; then
      CONTAINER_CLI="nerdctl"
    else
      echo "Neither docker nor nerdctl found in PATH" >&2
      exit 1
    fi
  fi
  echo "Pulling Docker image: ${IMAGE}"
  "${CONTAINER_CLI}" pull "${IMAGE}"
fi

if [[ "${DO_PULL_SIF}" == "true" ]]; then
  PULL_TOOL=""
  if command -v apptainer >/dev/null 2>&1; then
    PULL_TOOL="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    PULL_TOOL="singularity"
  else
    echo "Apptainer/Singularity not found in PATH; skipping SIF pull."
    echo "You can still run with Docker image: ${IMAGE}"
    exit 0
  fi

  mkdir -p singularity
  SIF="singularity/cellphenotyper_${TAG}_${VARIANT}.sif"
  echo "Pulling Singularity image: ${SIF} from docker://${IMAGE}"
  "${PULL_TOOL}" pull "${SIF}" "docker://${IMAGE}"
fi

echo "Done."
