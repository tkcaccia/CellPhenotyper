process ASSESS_CLUSTER_INTERPRETATION {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'assess_cluster_interpretation', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([cluster_csv, objects_csv, marker_quant_dir]) }
    tag "${sample_id}:${cluster_variant}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/11_clustering/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.cluster_assessment_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.cluster_assessment_memory_gb as int))} GB" }
    time { params.cluster_assessment_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(cluster_csv), path(objects_csv), path(marker_quant_dir)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_interpretation"), emit: assessment_dir
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_interpretation/cluster_interpretation_summary.json"), emit: summary_json

    script:
    def scriptPath = "${projectDir}/${params.cluster_assessment_script}"
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    codeDigest.update(new File(scriptPath).bytes)
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    export HOME="\$PWD/.runtime_home"
    export XDG_CACHE_HOME="\$PWD/.runtime_cache"
    export MPLCONFIGDIR="\$XDG_CACHE_HOME/matplotlib"
    mkdir -p "\$HOME" "\$XDG_CACHE_HOME/fontconfig" "\$MPLCONFIGDIR"
    echo "[INFO] Cluster interpretation assessment code fingerprint: ${codeFingerprint}"
    python "${scriptPath}" \
      --sample-id "${sample_id}" \
      --variant "${cluster_variant}" \
      --clusters "${cluster_csv}" \
      --objects "${objects_csv}" \
      --marker-quant-dir "${marker_quant_dir}" \
      --outdir "${sample_id}_${cluster_variant}_cluster_interpretation" \
      --spatial-k ${params.cluster_spatial_k} \
      --spatial-max-observations ${params.cluster_spatial_max_observations} \
      --review-per-cluster ${params.cluster_review_per_cluster} \
      --review-radius-px ${params.cluster_review_radius_px} \
      --seed ${params.cluster_assessment_seed}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p "${sample_id}_${cluster_variant}_cluster_interpretation"
    printf 'cluster,sampled_observations,mean_same_cluster_fraction,median_same_cluster_fraction\n' > "${sample_id}_${cluster_variant}_cluster_interpretation/cluster_spatial_coherence.csv"
    printf 'cluster,marker,observations,mean,median,overall_mean,standardized_mean_difference\n' > "${sample_id}_${cluster_variant}_cluster_interpretation/cluster_marker_enrichment.csv"
    printf 'review_id,tissue_interpretation,reviewer_confidence,artifact_present,notes\n' > "${sample_id}_${cluster_variant}_cluster_interpretation/blinded_review_form.csv"
    printf 'review_id,label_key,cluster,x,y\n' > "${sample_id}_${cluster_variant}_cluster_interpretation/blinded_review_key.csv"
    printf '{"type":"FeatureCollection","features":[]}' > "${sample_id}_${cluster_variant}_cluster_interpretation/blinded_review_regions.geojson"
    printf '{"type":"FeatureCollection","features":[]}' > "${sample_id}_${cluster_variant}_cluster_interpretation/cluster_abstentions.geojson"
    touch "${sample_id}_${cluster_variant}_cluster_interpretation/cluster_spatial_uncertainty.png"
    printf '{"status":"independent_review_required"}' > "${sample_id}_${cluster_variant}_cluster_interpretation/cluster_interpretation_summary.json"
    """
}
