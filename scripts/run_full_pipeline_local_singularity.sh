#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 3 ]]; then
  cat <<USAGE
Usage: $0 <image_input> <roi_geojson> <singularity_sif> [outdir]
Example:
  $0 Data/ROI.ome.tif Data/ROI.geojson singularity/cellphenotyper_full_cpu.sif results_full
USAGE
  exit 1
fi

IMAGE_INPUT="$1"
ROI_GEOJSON="$2"
SINGULARITY_SIF="$3"
OUTDIR="${4:-results_full}"

MAX_CPUS="${MAX_CPUS:-8}"
MAX_MEM_GB="${MAX_MEM_GB:-32}"
TISSUE_WORK_DOWNSAMPLE="${TISSUE_WORK_DOWNSAMPLE:-4}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-cpu}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work_full_singularity_local}"

nextflow run main.nf \
  -profile singularity \
  --image_input "$IMAGE_INPUT" \
  --roi_geojson "$ROI_GEOJSON" \
  --singularity_image "$SINGULARITY_SIF" \
  --run_full_pipeline true \
  --tissue_mask_from_input false \
  --compute_device "$COMPUTE_DEVICE" \
  --outdir_base "$OUTDIR" \
  --max_cpus "$MAX_CPUS" \
  --max_memory_gb "$MAX_MEM_GB" \
  --tissue_work_downsample "$TISSUE_WORK_DOWNSAMPLE" \
  -work-dir "$WORK_DIR" \
  -with-report "$OUTDIR/report.html" \
  -with-trace "$OUTDIR/trace.txt" \
  -with-timeline "$OUTDIR/timeline.html" \
  -resume
