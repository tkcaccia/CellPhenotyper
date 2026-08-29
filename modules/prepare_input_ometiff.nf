process PREPARE_INPUT_OMETIFF {
    tag "${sample_id}"
    label 'io_heavy'

    publishDir "${params.outdir_base}/01_input/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.convert_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.convert_memory_gb as int))} GB" }
    time { params.convert_time as String }

    input:
    tuple val(sample_id), path(image_file), val(input_region)

    output:
    tuple val(sample_id), path("${sample_id}.ome.tif"), emit: ome_tif
    tuple val(sample_id), path("${sample_id}.source_resolution.json"), emit: source_resolution_report
    tuple val(sample_id), path("${sample_id}.converted_resolution.json"), emit: converted_resolution_report

    script:
    def image_name = image_file.getName().toLowerCase()
    def staged_image_name = image_file.getName()
    def is_ome = image_name.endsWith('.ome.tif') || image_name.endsWith('.ome.tiff')
    def is_btf = image_name.endsWith('.btf')
    def is_tiff_like = image_name.endsWith('.tif') || image_name.endsWith('.tiff')
    def rgb_flag = params.convert_rgb ? '--rgb' : ''
    def overwrite_flag = params.convert_overwrite ? '--overwrite' : ''
    def btf_converter_script = "${projectDir}/${params.btf_converter_script}"
    def generic_converter_script = "${projectDir}/bin/convert_image_to_tiff.py"
    def resolution_validator_script = "${projectDir}/${params.input_resolution_validator_script}"
    def resolution_strict_flag = params.input_resolution_strict ? '--strict' : ''
    """
    set -euo pipefail

    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1

    if [[ "${image_name}" == *.czi ]]; then
      export CONVERT_PYDEPS="\$PWD/.pydeps"
      mkdir -p "\$CONVERT_PYDEPS"
      python - <<'PY'
import importlib.util
import os
import subprocess
import sys

mods = ['aicsimageio', 'aicspylibczi']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    target = os.environ['CONVERT_PYDEPS']
    print(f"[INFO] Installing missing CZI conversion packages: {', '.join(missing)}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--target', target, *missing])
PY
      export PYTHONPATH="\$CONVERT_PYDEPS:\${PYTHONPATH:-}"
    fi

    if [[ "${params.input_resolution_check}" == "true" ]]; then
      python "${resolution_validator_script}" \
        --image "${staged_image_name}" \
        --report "${sample_id}.source_resolution.json" \
        --min-mpp ${params.input_resolution_min_mpp} \
        --max-mpp ${params.input_resolution_max_mpp} \
        --cell-target-mpp ${params.input_resolution_cell_target_mpp} \
        --max-anisotropy-fraction ${params.input_resolution_max_anisotropy_fraction} \
        --max-conversion-drift-fraction ${params.input_resolution_max_conversion_drift_fraction} \
        --override-mpp ${params.input_resolution_override_mpp} \
        ${resolution_strict_flag}
    else
      printf '{"schema_version":1,"image":"%s","status":"skipped","strict":false}\n' \
        "${staged_image_name}" > "${sample_id}.source_resolution.json"
    fi

    if [[ -s "${sample_id}.ome.tif" ]]; then
      echo "[SKIP] Existing non-empty output: ${sample_id}.ome.tif"
    elif [[ "${is_ome}" == "true" ]]; then
      [[ -s "${staged_image_name}" ]] || { echo "Input OME-TIFF missing or empty: ${staged_image_name}" >&2; exit 1; }
      cp -f "${staged_image_name}" "${sample_id}.ome.tif"
    elif [[ "${is_btf}" == "true" ]]; then
      bash "${btf_converter_script}" \
        --in "${staged_image_name}" \
        --out "${sample_id}.ome.tif" \
        --compression "${params.convert_compression}" \
        --downsample "${params.convert_downsample}" \
        --max-workers ${task.cpus} \
        ${rgb_flag} \
        ${overwrite_flag}
      rm -rf "${sample_id}.ome.tif.rawdir"
    elif [[ "${is_tiff_like}" == "true" ]]; then
      [[ -s "${staged_image_name}" ]] || { echo "Input TIFF missing or empty: ${staged_image_name}" >&2; exit 1; }
      cp -f "${staged_image_name}" "${sample_id}.ome.tif"
    else
      python "${generic_converter_script}" \
        --input "${staged_image_name}" \
        --output "${sample_id}.ome.tif" \
        --input-region "${input_region}" \
        --compression "${params.convert_compression}" \
        --quality ${params.convert_jpeg_quality} \
        ${params.convert_pyramid ? '--pyramid' : ''} \
        --tile 512
    fi

    [[ -s "${sample_id}.ome.tif" ]] || { echo "Converted OME-TIFF missing or empty: ${sample_id}.ome.tif" >&2; exit 1; }

    if [[ "${params.input_resolution_check}" == "true" ]]; then
      python "${resolution_validator_script}" \
        --image "${sample_id}.ome.tif" \
        --report "${sample_id}.converted_resolution.json" \
        --reference-report "${sample_id}.source_resolution.json" \
        --min-mpp ${params.input_resolution_min_mpp} \
        --max-mpp ${params.input_resolution_max_mpp} \
        --cell-target-mpp ${params.input_resolution_cell_target_mpp} \
        --max-anisotropy-fraction ${params.input_resolution_max_anisotropy_fraction} \
        --max-conversion-drift-fraction ${params.input_resolution_max_conversion_drift_fraction} \
        --override-mpp ${params.input_resolution_override_mpp} \
        ${resolution_strict_flag}
    else
      printf '{"schema_version":1,"image":"%s","status":"skipped","strict":false}\n' \
        "${sample_id}.ome.tif" > "${sample_id}.converted_resolution.json"
    fi
    """

    stub:
    """
    touch "${sample_id}.ome.tif"
    printf '{"status":"stub"}\n' > "${sample_id}.source_resolution.json"
    printf '{"status":"stub"}\n' > "${sample_id}.converted_resolution.json"
    """
}
