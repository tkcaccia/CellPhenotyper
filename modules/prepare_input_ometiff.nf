process PREPARE_INPUT_OMETIFF {
    tag "${sample_id}"
    label 'io_heavy'

    publishDir "${params.outdir_base}/01_input/${sample_id}", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.convert_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.convert_memory_gb as int))} GB" }
    time { params.convert_time as String }

    input:
    tuple val(sample_id), path(image_file)

    output:
    tuple val(sample_id), path("${sample_id}.ome.tif"), emit: ome_tif

    script:
    def image_name = image_file.getName().toLowerCase()
    def is_ome = image_name.endsWith('.ome.tif') || image_name.endsWith('.ome.tiff')
    def is_btf = image_name.endsWith('.btf')
    def is_tiff_like = image_name.endsWith('.tif') || image_name.endsWith('.tiff')
    def rgb_flag = params.convert_rgb ? '--rgb' : ''
    def overwrite_flag = params.convert_overwrite ? '--overwrite' : ''
    def btf_converter_script = "${projectDir}/${params.btf_converter_script}"
    def generic_converter_script = "${projectDir}/bin/convert_image_to_tiff.py"
    """
    set -euo pipefail

    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1

    if [[ -s "${sample_id}.ome.tif" ]]; then
      echo "[SKIP] Existing non-empty output: ${sample_id}.ome.tif"
      exit 0
    fi

    if [[ "${is_ome}" == "true" ]]; then
      [[ -s "${image_file}" ]] || { echo "Input OME-TIFF missing or empty: ${image_file}" >&2; exit 1; }
      cp -f "${image_file}" "${sample_id}.ome.tif"
      exit 0
    fi

    if [[ "${is_btf}" == "true" ]]; then
      bash "${btf_converter_script}" \\
        --in "${image_file}" \\
        --out "${sample_id}.ome.tif" \\
        --compression "${params.convert_compression}" \\
        --downsample "${params.convert_downsample}" \\
        --max-workers ${task.cpus} \\
        ${rgb_flag} \\
        ${overwrite_flag}

      rm -rf "${sample_id}.ome.tif.rawdir"
      exit 0
    fi

    if [[ "${is_tiff_like}" == "true" ]]; then
      [[ -s "${image_file}" ]] || { echo "Input TIFF missing or empty: ${image_file}" >&2; exit 1; }
      cp -f "${image_file}" "${sample_id}.ome.tif"
      exit 0
    fi

    python "${generic_converter_script}" \
      --input "${image_file}" \
      --output "${sample_id}.ome.tif" \
      --compression "${params.convert_compression}" \
      --quality ${params.convert_jpeg_quality} \
      ${params.convert_pyramid ? '--pyramid' : ''} \
      --tile 512
    exit 0

    echo "Unsupported input image format: ${image_file}" >&2
    echo "Supported extensions: .ome.tif, .ome.tiff, .btf, .svs, .ndpi, .scn, .mrxs, .vms, .vmu, .tif, .tiff, .png, .jpg, .jpeg" >&2
    exit 2
    """

    stub:
    """
    touch "${sample_id}.ome.tif"
    """
}
