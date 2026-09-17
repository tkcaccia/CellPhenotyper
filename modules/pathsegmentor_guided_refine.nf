process PATHSEGMENTOR_GUIDED_REFINE {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'pathsegmentor_guided_refine', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([baseline_mask_tif, tissue_mask_tif, probabilities_tif, pathsegmentor_manifest]) }
    tag "${sample_id}:${cluster_variant}:pathsegmentor_sensitivity"
    label 'compute_medium'

    publishDir "${params.outdir_base}/14b_pathsegmentor_refine/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.pathsegmentor_refine_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.pathsegmentor_refine_memory_gb as int))} GB" }
    time { params.pathsegmentor_refine_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(baseline_mask_tif), path(tissue_mask_tif), path(probabilities_tif), path(pathsegmentor_manifest)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_pathsegmentor_refined.tif"), emit: refined_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_pathsegmentor_change_mask.tif"), emit: change_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_pathsegmentor_refined.provenance.json"), emit: provenance

    script:
    def scriptPath = "${projectDir}/${params.pathsegmentor_refine_script}"
    def fillFlag = (params.pathsegmentor_assign_unlabeled_within_tissue as boolean) ? '--assign-unlabeled-within-tissue' : ''
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    python "${scriptPath}" \
      --cluster-mask "${baseline_mask_tif}" \
      --tissue-mask "${tissue_mask_tif}" \
      --probabilities "${probabilities_tif}" \
      --pathsegmentor-manifest "${pathsegmentor_manifest}" \
      --out "${sample_id}_${cluster_variant}_pathsegmentor_refined.tif" \
      --change-mask "${sample_id}_${cluster_variant}_pathsegmentor_change_mask.tif" \
      --provenance "${sample_id}_${cluster_variant}_pathsegmentor_refined.provenance.json" \
      --boundary-band-um ${params.pathsegmentor_boundary_band_um} \
      --core-erosion-um ${params.pathsegmentor_core_erosion_um} \
      --min-distance-improvement ${params.pathsegmentor_min_distance_improvement} \
      --block-rows ${params.pathsegmentor_refine_block_rows} \
      ${fillFlag}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    touch "${sample_id}_${cluster_variant}_pathsegmentor_refined.tif"
    touch "${sample_id}_${cluster_variant}_pathsegmentor_change_mask.tif"
    printf '{"schema_version":"1.0.0","stub":true,"scientific_role":"experimental_supervised_boundary_refinement_preserving_raw_kodama_output"}\n' > "${sample_id}_${cluster_variant}_pathsegmentor_refined.provenance.json"
    """
}
