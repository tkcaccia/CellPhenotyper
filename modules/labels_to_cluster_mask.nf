process LABELS_TO_CLUSTER_MASK {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'labels_to_cluster_mask', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([labels_tif, cluster_csv, preview_image_tif]) }
    tag "${sample_id}:${cluster_variant}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/12_cluster_mask/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.cluster_mask_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.cluster_mask_memory_gb as int))} GB" }
    time { params.cluster_mask_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(labels_tif), path(cluster_csv), path(preview_image_tif)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_mask.tif"), emit: cluster_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_mask_preview.png"), emit: cluster_mask_preview
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_uncertainty_mask.tif"), emit: uncertainty_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_uncertainty_preview.png"), emit: uncertainty_preview
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_mask_summary.json"), emit: cluster_mask_summary

    script:
    def cluster_mask_script = "${projectDir}/${params.cluster_mask_script}"
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"

    python "${cluster_mask_script}" \
      --mask "${labels_tif}" \
      --map "${cluster_csv}" \
      --out "${sample_id}_${cluster_variant}_cluster_mask.tif" \
      --uncertainty-out "${sample_id}_${cluster_variant}_cluster_uncertainty_mask.tif" \
      --summary "${sample_id}_${cluster_variant}_cluster_mask_summary.json" \
      --default ${params.cluster_mask_default_value} \
      --compress "${params.cluster_mask_compression}" \
      --block-rows ${params.cluster_mask_block_rows} \
      --preview "${sample_id}_${cluster_variant}_cluster_mask_preview.png" \
      --uncertainty-preview "${sample_id}_${cluster_variant}_cluster_uncertainty_preview.png" \
      --preview-factor ${params.cluster_mask_preview_factor} \
      --preview-threshold-mb ${params.cluster_mask_preview_threshold_mb} \
      --preview-alpha ${params.cluster_mask_preview_alpha} \
      --preview-background "${preview_image_tif}"
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    touch "${sample_id}_${cluster_variant}_cluster_mask.tif"
    touch "${sample_id}_${cluster_variant}_cluster_mask_preview.png"
    touch "${sample_id}_${cluster_variant}_cluster_uncertainty_mask.tif"
    touch "${sample_id}_${cluster_variant}_cluster_uncertainty_preview.png"
    printf '{"observation_type":"contextual_cell","abstained_observations":0}\n' > "${sample_id}_${cluster_variant}_cluster_mask_summary.json"
    """
}
