process LABELS_TO_CLUSTER_MASK {
    tag "${sample_id}:${cluster_variant}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/12_cluster_mask/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.cluster_mask_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.cluster_mask_memory_gb as int))} GB" }
    time { params.cluster_mask_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(labels_tif), path(cluster_csv), path(preview_image_tif)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_mask.tif"), emit: cluster_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_mask_preview.png"), emit: cluster_mask_preview

    script:
    def cluster_mask_script = "${projectDir}/${params.cluster_mask_script}"
    """
    set -euo pipefail

    python "${cluster_mask_script}" \
      --mask "${labels_tif}" \
      --map "${cluster_csv}" \
      --out "${sample_id}_${cluster_variant}_cluster_mask.tif" \
      --default ${params.cluster_mask_default_value} \
      --compress "${params.cluster_mask_compression}" \
      --block-rows ${params.cluster_mask_block_rows} \
      --preview "${sample_id}_${cluster_variant}_cluster_mask_preview.png" \
      --preview-factor ${params.cluster_mask_preview_factor} \
      --preview-threshold-mb ${params.cluster_mask_preview_threshold_mb} \
      --preview-alpha ${params.cluster_mask_preview_alpha} \
      --preview-background "${preview_image_tif}"
    """

    stub:
    """
    touch "${sample_id}_${cluster_variant}_cluster_mask.tif"
    touch "${sample_id}_${cluster_variant}_cluster_mask_preview.png"
    """
}
