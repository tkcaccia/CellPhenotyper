process EXPAND_LABELS_TO_CYTOPLASM {
    tag "${sample_id}:${label_kind}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/06_cytoplasm/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.expand_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.expand_memory_gb as int))} GB" }
    time { params.expand_time as String }

    input:
    tuple val(sample_id), path(labels_tif), val(label_kind), val(preview_background_path)

    output:
    tuple val(sample_id), path("${sample_id}_${label_kind}.tif"), val(label_kind), emit: expanded_labels
    tuple val(sample_id), path("${sample_id}_${label_kind}_preview.png"), val(label_kind), emit: expanded_preview

    script:
    def compression_flag = params.expand_compression ? "--compression ${params.expand_compression}" : ''
    def expand_script = "${projectDir}/${params.expand_script}"
    """
    set -euo pipefail

    python "${expand_script}" \\
      --labels "${labels_tif}" \\
      --out "${sample_id}_${label_kind}.tif" \\
      --expand-px ${params.expand_px} \\
      --mode "${params.expand_mode}" \\
      --tile-size ${params.expand_tile_size} \\
      --auto-threshold-mpix ${params.expand_auto_threshold_mpix} \\
      --preview "${sample_id}_${label_kind}_preview.png" \\
      --preview-background "${preview_background_path}" \\
      --preview-factor ${params.expand_preview_factor} \\
      --preview-threshold-mb ${params.expand_preview_threshold_mb} \\
      --preview-alpha ${params.expand_preview_alpha} \\
      ${compression_flag}
    """

    stub:
    """
    touch "${sample_id}_${label_kind}.tif"
    touch "${sample_id}_${label_kind}_preview.png"
    """
}
