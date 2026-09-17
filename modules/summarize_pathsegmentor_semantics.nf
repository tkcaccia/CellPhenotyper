process SUMMARIZE_PATHSEGMENTOR_SEMANTICS {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'summarize_pathsegmentor_semantics', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([observations_csv, clusters_csv, probabilities_tif, pathsegmentor_manifest]) }
    tag "${sample_id}:${observation_role}:${cluster_variant}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/09c_pathsegmentor_annotations/${sample_id}/${observation_role}/${cluster_variant}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.pathsegmentor_annotation_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.pathsegmentor_annotation_memory_gb as int))} GB" }
    time { params.pathsegmentor_annotation_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), val(observation_role), path(observations_csv), path(clusters_csv), val(has_clusters), path(probabilities_tif), path(pathsegmentor_manifest)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), val(observation_role), path("pathsegmentor_${sample_id}_${observation_role}_${cluster_variant}"), emit: annotations

    script:
    def scriptPath = "${projectDir}/${params.pathsegmentor_annotation_script}"
    def clusterFlag = has_clusters ? "--clusters \"${clusters_csv}\"" : ''
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    python "${scriptPath}" \
      --probabilities "${probabilities_tif}" \
      --manifest "${pathsegmentor_manifest}" \
      --observations "${observations_csv}" \
      ${clusterFlag} \
      --outdir "pathsegmentor_${sample_id}_${observation_role}_${cluster_variant}" \
      --cell-window-um ${params.pathsegmentor_cell_window_um} \
      --observation-role "${observation_role}"
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p "pathsegmentor_${sample_id}_${observation_role}_${cluster_variant}"
    touch "pathsegmentor_${sample_id}_${observation_role}_${cluster_variant}/pathsegmentor_${observation_role}_semantics.csv.gz"
    if [[ "${has_clusters}" == "true" ]]; then
      printf 'cluster,observations,semantic_role\n' > "pathsegmentor_${sample_id}_${observation_role}_${cluster_variant}/pathsegmentor_cluster_semantics.csv"
    fi
    printf '{"schema_version":"1.0.0","stub":true}\n' > "pathsegmentor_${sample_id}_${observation_role}_${cluster_variant}/pathsegmentor_annotation_manifest.json"
    """
}
