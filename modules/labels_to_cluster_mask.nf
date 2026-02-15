process LABELS_TO_CLUSTER_MASK {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/10_cluster_mask", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.cluster_mask_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.cluster_mask_memory_gb as int))} GB" }
    time { params.cluster_mask_time as String }

    input:
    tuple val(sample_id), path(labels_tif), path(cluster_csv)

    output:
    tuple val(sample_id), path("${sample_id}_cluster_mask.tif"), emit: cluster_mask
    tuple val(sample_id), path("${sample_id}_cluster_mask_preview.png"), emit: cluster_mask_preview

    script:
    def cluster_mask_script = "${projectDir}/${params.cluster_mask_script}"
    """
    set -euo pipefail

    python "${cluster_mask_script}" \
      --mask "${labels_tif}" \
      --map "${cluster_csv}" \
      --out "${sample_id}_cluster_mask.tif" \
      --default ${params.cluster_mask_default_value} \
      --compress "${params.cluster_mask_compression}" \
      --preview "${sample_id}_cluster_mask_preview.png" \
      --preview-factor ${params.cluster_mask_preview_factor}
    """

    stub:
    """
    touch "${sample_id}_cluster_mask.tif"
    touch "${sample_id}_cluster_mask_preview.png"
    """
}
