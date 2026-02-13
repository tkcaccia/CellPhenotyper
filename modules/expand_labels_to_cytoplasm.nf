process EXPAND_LABELS_TO_CYTOPLASM {
    tag "${sample_id}:${label_kind}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/06_cytoplasm", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.expand_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.expand_memory_gb as int))} GB" }
    time { params.expand_time as String }

    input:
    tuple val(sample_id), path(labels_tif), val(label_kind)

    output:
    tuple val(sample_id), path("${sample_id}_${label_kind}.tif"), val(label_kind), emit: expanded_labels

    script:
    def compression_flag = params.expand_compression ? "--compression ${params.expand_compression}" : ''
    def expand_script = "${projectDir}/${params.expand_script}"
    """
    set -euo pipefail

    python "${expand_script}" \\
      --labels "${labels_tif}" \\
      --out "${sample_id}_${label_kind}.tif" \\
      --expand-px ${params.expand_px} \\
      ${compression_flag}
    """

    stub:
    """
    touch "${sample_id}_${label_kind}.tif"
    """
}
