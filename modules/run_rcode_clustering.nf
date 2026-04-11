process RUN_RCODE_CLUSTERING {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/09_clustering/${sample_id}", mode: 'copy', overwrite: true
    publishDir "${params.outdir_base}/09_clustering_logs/${sample_id}", mode: 'copy', overwrite: true, pattern: '*.Rout'

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.cluster_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.cluster_memory_gb as int))} GB" }
    time { params.cluster_time as String }

    input:
    tuple val(sample_id), path(kodama_dir), path(objects_assigned_csv)

    output:
    tuple val(sample_id), path("${sample_id}_cluster.csv"), emit: cluster_csv
    tuple val(sample_id), path("${sample_id}_cluster_kodama_membership.pdf"), emit: membership_pdf
    tuple val(sample_id), path("${sample_id}_cluster_summary.csv"), emit: cluster_summary
    tuple val(sample_id), path("Rcode_Clustering_${sample_id}.Rout"), emit: clustering_log

    script:
    def cluster_script = "${projectDir}/${params.cluster_r_script}"
    """
    set -euo pipefail

    Rscript "${cluster_script}" \
      "${kodama_dir}" \
      "${sample_id}_cluster.csv" \
      --dim ${params.cluster_kodama_dim} \
      --k ${params.cluster_snn_k} \
      --resolution ${params.cluster_resolution} \
      > "Rcode_Clustering_${sample_id}.Rout" 2>&1
    """

    stub:
    """
    touch "${sample_id}_cluster.csv"
    touch "${sample_id}_cluster_kodama_membership.pdf"
    touch "${sample_id}_cluster_summary.csv"
    touch "Rcode_Clustering_${sample_id}.Rout"
    """
}
