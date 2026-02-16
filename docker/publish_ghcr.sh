#!/usr/bin/env bash
set -euo pipefail

# Publish CellPhenotyper runtime images to GHCR.
# Examples:
#   ./docker/publish_ghcr.sh --tag 0.2.0 --variant cpu --latest
#   ./docker/publish_ghcr.sh --tag 0.2.0 --variant gpu

usage() {
  cat <<'USAGE'
Usage:
  ./docker/publish_ghcr.sh --tag <version> [--variant cpu|gpu] [--latest]

Options:
  --tag       Required. Image tag version (example: 0.2.0)
  --variant   cpu (default) or gpu
  --latest    Also publish latest aliases

Environment:
  GHCR_USER   GHCR namespace user/org (default: tkcaccia)
  GHCR_TOKEN  GitHub token with write:packages
  CONTAINER_CLI Optional override: docker or nerdctl

Auth file:
  If GHCR_TOKEN is unset and GHCRtoken.env exists in repo root, it is sourced.
USAGE
}

TAG=""
VARIANT="cpu"
PUSH_LATEST="false"

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
    --latest)
      PUSH_LATEST="true"
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${GHCR_TOKEN:-}" && -f "${REPO_ROOT}/GHCRtoken.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/GHCRtoken.env"
fi

export GHCR_USER="${GHCR_USER:-tkcaccia}"
if [[ -z "${GHCR_TOKEN:-}" ]]; then
  echo "GHCR_TOKEN is not set. Export it or put it in GHCRtoken.env." >&2
  exit 1
fi

IMAGE_BASE="ghcr.io/${GHCR_USER}/cellphenotyper"

CONTAINER_CLI="${CONTAINER_CLI:-}"
if [[ -z "${CONTAINER_CLI}" ]]; then
  if command -v docker >/dev/null 2>&1; then
    CONTAINER_CLI="docker"
  elif command -v nerdctl >/dev/null 2>&1; then
    CONTAINER_CLI="nerdctl"
  else
    echo "Neither docker nor nerdctl found in PATH." >&2
    exit 1
  fi
fi

if [[ "${VARIANT}" == "cpu" ]]; then
  DOCKERFILE="docker/Dockerfile.full.cpu"
  IMAGE_TAGS=("${TAG}" "${TAG}-cpu")
  if [[ "${PUSH_LATEST}" == "true" ]]; then
    IMAGE_TAGS+=("latest" "latest-cpu")
  fi
else
  DOCKERFILE="docker/Dockerfile.full.gpu"
  IMAGE_TAGS=("${TAG}-gpu")
  if [[ "${PUSH_LATEST}" == "true" ]]; then
    IMAGE_TAGS+=("latest-gpu")
  fi
fi

echo "${GHCR_TOKEN}" | "${CONTAINER_CLI}" login ghcr.io -u "${GHCR_USER}" --password-stdin

BUILD_ARGS=()
for t in "${IMAGE_TAGS[@]}"; do
  BUILD_ARGS+=(-t "${IMAGE_BASE}:${t}")
done

echo "Building ${DOCKERFILE} -> ${IMAGE_BASE} tags: ${IMAGE_TAGS[*]}"
if [[ "${CONTAINER_CLI}" == "nerdctl" ]] && command -v buildctl >/dev/null 2>&1; then
  # Rootless nerdctl can fail at final "unpack" on small Lima disks.
  # Direct BuildKit export with push=true and unpack=false avoids that.
  BUILDKIT_ADDR="${BUILDKIT_ADDR:-unix:///run/user/$(id -u)/buildkit-default/buildkitd.sock}"
  BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-auto}"
  DOCKERFILE_DIR="$(dirname "${DOCKERFILE}")"
  DOCKERFILE_NAME="$(basename "${DOCKERFILE}")"

  for t in "${IMAGE_TAGS[@]}"; do
    echo "BuildKit push ${IMAGE_BASE}:${t}"
    buildctl \
      --addr "${BUILDKIT_ADDR}" \
      build \
      --progress "${BUILDKIT_PROGRESS}" \
      --frontend dockerfile.v0 \
      --local context=. \
      --local dockerfile="${DOCKERFILE_DIR}" \
      --opt "filename=${DOCKERFILE_NAME}" \
      --output "type=image,name=${IMAGE_BASE}:${t},push=true,unpack=false"
  done
else
  "${CONTAINER_CLI}" build -f "${DOCKERFILE}" "${BUILD_ARGS[@]}" .

  for t in "${IMAGE_TAGS[@]}"; do
    echo "Pushing ${IMAGE_BASE}:${t}"
    "${CONTAINER_CLI}" push "${IMAGE_BASE}:${t}"
  done
fi

echo "Done."
