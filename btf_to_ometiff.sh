#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./btf_to_ometiff.sh --in INPUT.btf --out OUTPUT.ome.tif [options]

Options:
  --compression  UNCOMPRESSED|LZW|JPEG|JPEG_2000|JPEG_2000_LOSSY   (default: LZW)
  --max-workers  N                                                 (default: 16)
  --downsample   SIMPLE|GAUSSIAN|AREA|LINEAR|CUBIC|LANCZOS          (default: GAUSSIAN)

  --rgb          Store as chunky RGB (recommended for brightfield WSI)
  --no-rgb       Do not RGB-convert (default)

  --legacy       Write a Bio-Formats 5.9.x-compatible pyramid
  --overwrite    Remove existing output before writing
  --keep-tmp     Keep the intermediate .rawdir (default: delete)

Examples:
  ./btf_to_ometiff.sh --in slide.btf --out slide.ome.tif --overwrite
  ./btf_to_ometiff.sh --in slide.btf --out slide_rgb.ome.tif --rgb --overwrite
  ./btf_to_ometiff.sh --in slide.btf --out slide_legacy.ome.tif --legacy --overwrite
EOF
}

# Defaults
COMP="LZW"
MAXW="16"
DOWNSAMPLE="GAUSSIAN"
RGB=0
LEGACY=0
OVERWRITE=0
KEEP_TMP=0
IN=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in) IN="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --compression) COMP="${2:-}"; shift 2 ;;
    --max-workers) MAXW="${2:-}"; shift 2 ;;
    --downsample) DOWNSAMPLE="${2:-}"; shift 2 ;;
    --rgb) RGB=1; shift ;;
    --no-rgb) RGB=0; shift ;;
    --legacy) LEGACY=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --keep-tmp) KEEP_TMP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1"; usage; exit 2 ;;
  esac
done

[[ -n "$IN" && -n "$OUT" ]] || { echo "ERROR: --in and --out are required"; usage; exit 2; }

command -v bioformats2raw >/dev/null 2>&1 || { echo "ERROR: bioformats2raw not found on PATH"; exit 1; }
command -v raw2ometiff   >/dev/null 2>&1 || { echo "ERROR: raw2ometiff not found on PATH"; exit 1; }

# Overwrite output if requested
if [[ -e "$OUT" ]]; then
  if [[ "$OVERWRITE" -eq 1 ]]; then
    rm -f "$OUT"
  else
    echo "ERROR: output exists (use --overwrite): $OUT"
    exit 1
  fi
fi

# UNIQUE tmpdir every run (prevents 'already exists' permanently)
OUTDIR="$(dirname "$OUT")"
BASE="$(basename "$OUT")"
STAMP="$(date +%Y%m%d_%H%M%S)"
RAND="$(printf "%06d" $((RANDOM % 1000000)))"
TMPDIR="${OUTDIR}/${BASE}.${STAMP}.${RAND}.rawdir"

cleanup() {
  if [[ "$KEEP_TMP" -eq 0 ]]; then
    rm -rf "$TMPDIR" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "bioformats2raw -> $TMPDIR"
bioformats2raw --downsample-type "$DOWNSAMPLE" "$IN" "$TMPDIR"

R2O_ARGS=( "--compression=$COMP" "--max_workers=$MAXW" "-p" )
if [[ "$RGB" -eq 1 ]]; then
  R2O_ARGS+=( "--rgb" )
fi
if [[ "$LEGACY" -eq 1 ]]; then
  R2O_ARGS+=( "--legacy" )
fi

echo "raw2ometiff -> $OUT"
raw2ometiff "${R2O_ARGS[@]}" "$TMPDIR" "$OUT"

echo "Done: $OUT"
