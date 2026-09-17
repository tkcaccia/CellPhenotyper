process RUN_RCODE_CLUSTERING {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'run_rcode_clustering', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([kodama_dir, objects_assigned_csv]) }
    tag "${sample_id}:${cluster_variant}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/11_clustering/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    publishDir "${params.outdir_base}/11_clustering/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true, pattern: '*.Rout'

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.cluster_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.cluster_memory_gb as int))} GB" }
    time { params.cluster_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), val(cluster_profile), val(cluster_resolution), path(kodama_dir), path(objects_assigned_csv)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster.csv"), emit: cluster_csv
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_kodama_membership.pdf"), emit: membership_pdf
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_kodama_membership.png"), emit: membership_png
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_kodama_uncertainty.pdf"), emit: uncertainty_pdf
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_kodama_uncertainty.png"), emit: uncertainty_png
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_summary.csv"), emit: cluster_summary
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_cluster_stability.csv"), emit: cluster_stability
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("Rcode_Clustering_${sample_id}_${cluster_variant}.Rout"), emit: clustering_log

    script:
    def forcedTarget = params.cluster_target_clusters as int
    def forcedTargetAcknowledged = (params.cluster_forced_count_sensitivity_acknowledged ?: false) as boolean
    if (forcedTarget > 0 && !forcedTargetAcknowledged) {
        error "cluster_target_clusters=${forcedTarget} forces a sensitivity partition. Set --cluster_forced_count_sensitivity_acknowledged true to confirm that it will not be treated as the graph-derived primary result."
    }
    def cluster_script = "${projectDir}/${params.cluster_r_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([cluster_script,
        "${projectDir}/bin/kodama_graph_export.R", "${projectDir}/bin/kodama_graph_clustering.R"])
    def clusterRLibrary = (params.cluster_r_library_dir ?: '/opt/micromamba/envs/stardist/lib/R/library').toString()
    def nativeGraphMode = params.cluster_representation == 'kodama_graph'
    if (nativeGraphMode && (params.cluster_algorithm != 'leiden' || cluster_profile != 'standard' || cluster_resolution.toString() == 'auto')) {
        error 'kodama_graph requires Leiden, a fixed resolution and the standard profile'
    }
    if (nativeGraphMode && ((params.cluster_landmark_cells as int) != 0 || (params.cluster_representation_dimensions as int) != 0)) {
        error 'kodama_graph requires cluster_landmark_cells=0 and cluster_representation_dimensions=0; every exported vertex is retained'
    }
    // Native graphs already contain their neighbors: do not pass coordinate
    // graph construction, landmark assignment, or fine-profile tuning options.
    def coordinateOptions = nativeGraphMode ? '' : [
        "--k ${params.cluster_snn_k}",
        "--walktrap-clusters ${params.cluster_walktrap_clusters}",
        "--walktrap-max-cells ${params.cluster_walktrap_max_cells}",
        "--walktrap-assign-k ${params.cluster_walktrap_assign_k}",
        "--landmark-cells ${params.cluster_landmark_cells}",
        "--landmark-assign-k ${params.cluster_landmark_assign_k}",
        "--landmark-sample-strategy ${params.cluster_landmark_sample_strategy}",
        "--landmark-density-knn-k ${params.cluster_landmark_density_knn_k}",
        "--landmark-density-power ${params.cluster_landmark_density_power}",
        "--landmark-grid-bins ${params.cluster_landmark_grid_bins}",
        "--landmark-grid-max-per-bin ${params.cluster_landmark_grid_max_per_bin}",
        "--fine-multiplier ${params.cluster_fine_resolution_multiplier}",
        "--fine-score-margin ${params.cluster_fine_score_margin}",
        "--fine-resolution-max ${params.cluster_fine_resolution_max}",
        "--fine-min-cluster-increase ${params.cluster_fine_min_cluster_increase}"
    ].join(' ')
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Clustering code fingerprint: ${codeFingerprint}; representation=${params.cluster_representation}"
    export R_LIBS_USER="${clusterRLibrary}"
    export R_ENVIRON_USER=/dev/null
    export R_PROFILE_USER=/dev/null
    export HOME="\$PWD/.runtime_home"
    export XDG_CACHE_HOME="\$PWD/.runtime_cache"
    export MPLCONFIGDIR="\$XDG_CACHE_HOME/matplotlib"
    mkdir -p "\$HOME" "\$XDG_CACHE_HOME/fontconfig" "\$MPLCONFIGDIR"

    Rscript "${cluster_script}" \
      "${kodama_dir}" \
      "${sample_id}_${cluster_variant}_cluster.csv" \
      --dim ${params.cluster_kodama_dim} \
      --cluster-representation ${params.cluster_representation} \
      --cluster-dimensions ${params.cluster_representation_dimensions} \
      --algorithm ${params.cluster_algorithm} \
      --target-clusters ${params.cluster_target_clusters} \
      ${coordinateOptions} \
      --resolution ${cluster_resolution} \
      --profile ${cluster_profile} \
      --seed ${params.cluster_seed} \
      --stability-runs ${params.cluster_stability_runs} \
      --assignment-min-vote-margin ${params.cluster_assignment_min_vote_margin} \
      --stability-min-fraction ${params.cluster_stability_min_fraction} \
      --abstain-uncertain ${params.cluster_abstain_uncertain} \
      > "Rcode_Clustering_${sample_id}_${cluster_variant}.Rout" 2>&1
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    touch "${sample_id}_${cluster_variant}_cluster.csv"
    touch "${sample_id}_${cluster_variant}_cluster_kodama_membership.pdf"
    touch "${sample_id}_${cluster_variant}_cluster_kodama_membership.png"
    touch "${sample_id}_${cluster_variant}_cluster_kodama_uncertainty.pdf"
    touch "${sample_id}_${cluster_variant}_cluster_kodama_uncertainty.png"
    touch "${sample_id}_${cluster_variant}_cluster_summary.csv"
    touch "${sample_id}_${cluster_variant}_cluster_stability.csv"
    touch "Rcode_Clustering_${sample_id}_${cluster_variant}.Rout"
    """
}
