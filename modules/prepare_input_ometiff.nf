process PREPARE_INPUT_OMETIFF {
    tag "${sample_id}"
    label 'io_heavy'

    publishDir "${params.outdir_base}/01_input", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.convert_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.convert_memory_gb as int))} GB" }
    time { params.convert_time as String }

    input:
    tuple val(sample_id), path(image_file)

    output:
    tuple val(sample_id), path("${sample_id}.ome.tif"), emit: ome_tif

    script:
    def image_name = image_file.getName().toLowerCase()
    def is_ome = image_name.endsWith('.ome.tif')
    def rgb_flag = params.convert_rgb ? '--rgb' : ''
    def overwrite_flag = params.convert_overwrite ? '--overwrite' : ''
    """
    set -euo pipefail

    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1

    if [[ "${is_ome}" == "true" ]]; then
      [[ -s "${image_file}" ]] || { echo "Input OME-TIFF missing or empty: ${image_file}" >&2; exit 1; }
      exit 0
    fi

    if [[ -s "${sample_id}.ome.tif" ]]; then
      echo "[SKIP] Existing non-empty output: ${sample_id}.ome.tif"
      exit 0
    fi

    bash "${params.btf_converter_script}" \\
      --in "${image_file}" \\
      --out "${sample_id}.ome.tif" \\
      --compression "${params.convert_compression}" \\
      --downsample "${params.convert_downsample}" \\
      --max-workers ${task.cpus} \\
      ${rgb_flag} \\
      ${overwrite_flag}

    rm -rf "${sample_id}.ome.tif.rawdir"
    """

    stub:
    """
    touch "${sample_id}.ome.tif"
    """
}
