#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_SIF="${1:-singularity/cellphenotyper_full_gpu.sif}"

sudo singularity build --force "$OUT_SIF" singularity/cellphenotyper_full_gpu.def
