#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 3 ]]; then
  cat <<USAGE
Usage: $0 <image_input> <roi_geojson> <singularity_sif> [outdir]
Example:
  $0 Data/ROI.ome.tif Data/ROI.geojson singularity/cellphenotyper_tissue.sif results_singularity_local
USAGE
  exit 1
fi

IMAGE_INPUT="$1"
ROI_GEOJSON="$2"
SINGULARITY_SIF="$3"
OUTDIR="${4:-results_singularity_local}"

MAX_CPUS="${MAX_CPUS:-2}"
MAX_MEM_GB="${MAX_MEM_GB:-6}"
TISSUE_WORK_DOWNSAMPLE="${TISSUE_WORK_DOWNSAMPLE:-8}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work_singularity_local}"

nextflow run main.nf \
  -profile singularity \
  --image_input "$IMAGE_INPUT" \
  --roi_geojson "$ROI_GEOJSON" \
  --singularity_image "$SINGULARITY_SIF" \
  --run_full_pipeline false \
  --tissue_mask_from_input true \
  --outdir_base "$OUTDIR" \
  --max_cpus "$MAX_CPUS" \
  --max_memory_gb "$MAX_MEM_GB" \
  --tissue_work_downsample "$TISSUE_WORK_DOWNSAMPLE" \
  -work-dir "$WORK_DIR" \
  -with-report "$OUTDIR/report.html" \
  -with-trace "$OUTDIR/trace.txt" \
  -with-timeline "$OUTDIR/timeline.html" \
  -resume
