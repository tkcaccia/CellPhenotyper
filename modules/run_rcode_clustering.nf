process RUN_RCODE_CLUSTERING {
    tag "${sample_id}:${cluster_variant}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/11_clustering/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    publishDir "${params.outdir_base}/11_clustering/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true, pattern: '*.Rout'

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.cluster_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.cluster_memory_gb as int))} GB" }
    time { params.cluster_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), val(cluster_profile), val(cluster_resolution), path(kodama_dir), path(objects_assigned_csv)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster.csv"), emit: cluster_csv
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_kodama_membership.pdf"), emit: membership_pdf
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_kodama_membership.png"), emit: membership_png
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_summary.csv"), emit: cluster_summary
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("Rcode_Clustering_${sample_id}_${cluster_variant}.Rout"), emit: clustering_log

    script:
    def cluster_script = "${projectDir}/${params.cluster_r_script}"
    def clusterRLibrary = (params.cluster_r_library_dir ?: '/opt/micromamba/envs/stardist/lib/R/library').toString()
    """
    set -euo pipefail
    export R_LIBS_USER="${clusterRLibrary}"
    export R_ENVIRON_USER=/dev/null
    export R_PROFILE_USER=/dev/null

    Rscript "${cluster_script}" \
      "${kodama_dir}" \
      "${sample_id}_${cluster_variant}_cluster.csv" \
      --dim ${params.cluster_kodama_dim} \
      --k ${params.cluster_snn_k} \
      --algorithm ${params.cluster_algorithm} \
      --walktrap-clusters ${params.cluster_walktrap_clusters} \
      --walktrap-max-cells ${params.cluster_walktrap_max_cells} \
      --walktrap-assign-k ${params.cluster_walktrap_assign_k} \
      --landmark-cells ${params.cluster_landmark_cells} \
      --landmark-assign-k ${params.cluster_landmark_assign_k} \
      --landmark-sample-strategy ${params.cluster_landmark_sample_strategy} \
      --landmark-density-knn-k ${params.cluster_landmark_density_knn_k} \
      --landmark-density-power ${params.cluster_landmark_density_power} \
      --landmark-grid-bins ${params.cluster_landmark_grid_bins} \
      --landmark-grid-max-per-bin ${params.cluster_landmark_grid_max_per_bin} \
      --resolution ${cluster_resolution} \
      --profile ${cluster_profile} \
      --fine-multiplier ${params.cluster_fine_resolution_multiplier} \
      --fine-score-margin ${params.cluster_fine_score_margin} \
      --fine-resolution-max ${params.cluster_fine_resolution_max} \
      --fine-min-cluster-increase ${params.cluster_fine_min_cluster_increase} \
      > "Rcode_Clustering_${sample_id}_${cluster_variant}.Rout" 2>&1
    """

    stub:
    """
    touch "${sample_id}_${cluster_variant}_cluster.csv"
    touch "${sample_id}_${cluster_variant}_cluster_kodama_membership.pdf"
    touch "${sample_id}_${cluster_variant}_cluster_kodama_membership.png"
    touch "${sample_id}_${cluster_variant}_cluster_summary.csv"
    touch "Rcode_Clustering_${sample_id}_${cluster_variant}.Rout"
    """
}
