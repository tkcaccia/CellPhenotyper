process RUN_STARDIST_ROI_SEGMENTATION {
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/02_stardist/${sample_id}", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.stardist_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.stardist_memory_gb as int))} GB" }
    time { params.stardist_time as String }

    input:
    tuple val(sample_id), path(ome_tif), path(roi_geojson)

    output:
    tuple val(sample_id), path("stardist_out/crop_roi.tif"), emit: crop_roi
    tuple val(sample_id), path("stardist_out/labels.tif"), emit: labels_tif
    tuple val(sample_id), path("stardist_out/labels_full.tif"), optional: true, emit: labels_full_tif
    tuple val(sample_id), path("stardist_out/objects.csv"), emit: objects_csv
    tuple val(sample_id), path("stardist_out/shift.json"), emit: shift_json
    tuple val(sample_id), path("stardist_out"), emit: stardist_dir

    script:
    def write_full_flag = params.write_full_labels ? '--write-full-labels' : ''
    def allow_huge_flag = params.allow_huge_tif ? '--allow-huge-tif' : ''
    def precomputed_flag = params.stardist_precomputed_labels_full ? "--precomputed-labels-full \"${params.stardist_precomputed_labels_full}\"" : ''
    def stardist_script = "${projectDir}/${params.stardist_script}"
    """
    set -euo pipefail

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1
    export LOKY_MAX_CPU_COUNT=${task.cpus}
    export MPLCONFIGDIR="\$PWD/.mplconfig"
    export XDG_CACHE_HOME="\$PWD/.cache"
    mkdir -p "\$MPLCONFIGDIR" "\$XDG_CACHE_HOME"
    if [[ "${params.stardist_autoinstall_runtime}" == "true" ]]; then
      export STARDIST_PYDEPS="\$PWD/.pydeps"
      export STARDIST_TF_VERSION="${params.stardist_tensorflow_version}"
      mkdir -p "\$STARDIST_PYDEPS"
      python - <<'PY'
import importlib
import os
import subprocess
import sys

mods = ["tensorflow", "stardist", "csbdeep", "imagecodecs"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

if missing:
    target = os.environ.get("STARDIST_PYDEPS", ".pydeps")
    tf_version = os.environ.get("STARDIST_TF_VERSION", "").strip()
    req = []
    if "tensorflow" in missing:
        req.append(f"tensorflow=={tf_version}" if tf_version else "tensorflow")
    if "stardist" in missing:
        req.append("stardist")
    if "csbdeep" in missing:
        req.append("csbdeep")
    if "imagecodecs" in missing:
        req.append("imagecodecs")
    print(f"[INFO] Installing missing StarDist runtime packages: {', '.join(req)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", target, *req])
PY
      export PYTHONPATH="\$STARDIST_PYDEPS:\${PYTHONPATH:-}"
    fi
    if [[ -n "${params.stardist_pythonpath}" ]]; then
      export PYTHONPATH="${params.stardist_pythonpath}:\${PYTHONPATH:-}"
    fi

    mkdir -p stardist_out

    if [[ "${params.gpu_debug_diagnostics}" == "true" && "${params.compute_device}" == "gpu" ]]; then
      echo "[DEBUG] StarDist GPU diagnostics"
      nvidia-smi -L || true
      python - <<'PY'
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[DEBUG] tensorflow={tf.__version__} built_with_cuda={tf.test.is_built_with_cuda()} gpus={len(gpus)}")
except Exception as exc:
    print(f"[DEBUG] TensorFlow diagnostics unavailable: {exc}")
PY
    fi

    MIN_AREA_FLAG=""
    if python "${stardist_script}" --help 2>&1 | grep -q -- "--min-area"; then
      MIN_AREA_FLAG="--min-area ${params.stardist_min_area}"
    else
      echo "[WARN] StarDist script does not support --min-area; skipping area filter at segmentation step."
    fi

    python "${stardist_script}" \\
      --in "${ome_tif}" \\
      --roi "${roi_geojson}" \\
      --outdir stardist_out \\
      --model "${params.stardist_model}" \\
      --prob ${params.stardist_prob} \\
      --nms ${params.stardist_nms} \\
      \${MIN_AREA_FLAG} \\
      --tiles ${params.stardist_tiles_y} ${params.stardist_tiles_x} \\
      --pad ${params.stardist_crop_pad} \\
      --full-format "${params.full_format}" \\
      --full-out "stardist_out/labels_full.tif" \\
      ${precomputed_flag} \\
      ${write_full_flag} \\
      ${allow_huge_flag}
    """

    stub:
    """
    mkdir -p stardist_out
    touch stardist_out/crop_roi.tif
    touch stardist_out/labels.tif
    touch stardist_out/objects.csv
    touch stardist_out/shift.json
    if [[ "${params.write_full_labels}" == "true" ]]; then
      touch stardist_out/labels_full.tif
    fi
    """
}
