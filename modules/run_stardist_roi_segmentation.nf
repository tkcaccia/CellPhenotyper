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
    tuple val(sample_id), path("stardist_out/roi_all_crop.geojson"), emit: roi_crop_geojson
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
    STARDIST_KERAS_HOME="${params.stardist_keras_home}"
    if [[ -n "\$STARDIST_KERAS_HOME" ]]; then
      export KERAS_HOME="\$STARDIST_KERAS_HOME"
    fi
    if [[ -z "\${KERAS_HOME:-}" ]]; then
      export KERAS_HOME="\$PWD/.keras"
    fi
    export XDG_CACHE_HOME="\${KERAS_HOME}"
    mkdir -p "\$MPLCONFIGDIR" "\$XDG_CACHE_HOME" "\$KERAS_HOME/models" "\$KERAS_HOME/models/StarDist2D"

    MODEL_CACHE_ID="${params.stardist_model}"
    if [[ "\$MODEL_CACHE_ID" == "auto" ]]; then
      MODEL_CACHE_ID="2D_versatile_he"
    fi
    PY_MODEL_CACHE_ID="\$MODEL_CACHE_ID"
    if [[ "\$PY_MODEL_CACHE_ID" != python_* ]]; then
      PY_MODEL_CACHE_ID="python_\${MODEL_CACHE_ID}"
    fi

    copy_if_needed() {
      local src="\$1"
      local dst="\$2"
      local src_real dst_real
      src_real="\$(readlink -f "\$src" 2>/dev/null || echo "\$src")"
      dst_real="\$(readlink -f "\$dst" 2>/dev/null || echo "\$dst")"
      if [[ "\$src_real" == "\$dst_real" ]]; then
        return 0
      fi
      cp -f "\$src" "\$dst"
    }

    if [[ -n "${params.stardist_pretrained_zip}" ]]; then
      if [[ -f "${params.stardist_pretrained_zip}" ]]; then
        copy_if_needed "${params.stardist_pretrained_zip}" "\$KERAS_HOME/models/\${PY_MODEL_CACHE_ID}.zip"
        copy_if_needed "${params.stardist_pretrained_zip}" "\$KERAS_HOME/models/\${MODEL_CACHE_ID}.zip"
        copy_if_needed "${params.stardist_pretrained_zip}" "\$KERAS_HOME/models/StarDist2D/\${MODEL_CACHE_ID}.zip"
        copy_if_needed "${params.stardist_pretrained_zip}" "\$KERAS_HOME/models/StarDist2D/\${PY_MODEL_CACHE_ID}.zip"
      else
        echo "[WARN] stardist_pretrained_zip not found: ${params.stardist_pretrained_zip}"
      fi
    fi

    # Backward-compatible fallback: normalize known cache filename variants.
    STARDIST_CACHE_SRC=""
    for CAND in \
      "\$KERAS_HOME/models/\${PY_MODEL_CACHE_ID}.zip" \
      "\$KERAS_HOME/models/\${MODEL_CACHE_ID}.zip" \
      "\$KERAS_HOME/models/StarDist2D/\${MODEL_CACHE_ID}.zip" \
      "\$KERAS_HOME/models/StarDist2D/\${PY_MODEL_CACHE_ID}.zip"; do
      if [[ -f "\$CAND" ]]; then
        STARDIST_CACHE_SRC="\$CAND"
        break
      fi
    done
    if [[ -n "\$STARDIST_CACHE_SRC" ]]; then
      for DST in \
        "\$KERAS_HOME/models/\${PY_MODEL_CACHE_ID}.zip" \
        "\$KERAS_HOME/models/\${MODEL_CACHE_ID}.zip" \
        "\$KERAS_HOME/models/StarDist2D/\${MODEL_CACHE_ID}.zip" \
        "\$KERAS_HOME/models/StarDist2D/\${PY_MODEL_CACHE_ID}.zip"; do
        if [[ ! -f "\$DST" ]]; then
          cp -f "\$STARDIST_CACHE_SRC" "\$DST"
        fi
      done

      python - "\$STARDIST_CACHE_SRC" "\$KERAS_HOME/models/StarDist2D/\${MODEL_CACHE_ID}" "\$KERAS_HOME/models/StarDist2D/\${PY_MODEL_CACHE_ID}" <<'PY'
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

zip_path = Path(sys.argv[1])
target_dirs = [Path(p) for p in sys.argv[2:] if p]
required = {"config.json", "thresholds.json", "weights_best.h5"}

def is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    return required.issubset({p.name for p in path.iterdir() if p.is_file()})

def extract_model(target_dir: Path) -> None:
    if is_complete(target_dir):
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="stardist_extract_", dir=str(target_dir.parent)))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_root)

        candidates = [tmp_root] + [p for p in tmp_root.iterdir() if p.is_dir()]
        extracted = None
        for cand in candidates:
            if is_complete(cand):
                extracted = cand
                break

        if extracted is None:
            raise RuntimeError(f"Could not find StarDist model files inside {zip_path}")

        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(extracted, target_dir)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

for target in target_dirs:
    extract_model(target)
PY
    fi
    echo "[INFO] StarDist cache env: KERAS_HOME=\$KERAS_HOME XDG_CACHE_HOME=\$XDG_CACHE_HOME model_cache_id=\$MODEL_CACHE_ID py_model_cache_id=\$PY_MODEL_CACHE_ID"
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
    # Keep TensorFlow-compatible NumPy for task-local fallback installs.
    if any(m in missing for m in ("tensorflow", "stardist", "csbdeep")):
        req.append("numpy<2")
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
else:
    print("[INFO] StarDist runtime packages already available; skipping auto-install.")
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
