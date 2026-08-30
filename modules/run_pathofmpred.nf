process RUN_PATHOFMPRED {
    tag "${sample_id}:${cluster_variant}"
    label 'compute_medium'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/18_pathofmpred/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params.max_cpus as int, params.pathofmpred_cpus as int)) }
    memory { "${Math.max(4, Math.min(params.max_memory_gb as int, params.pathofmpred_memory_gb as int))} GB" }
    time { params.pathofmpred_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(titan_embedding)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("pathofmpred_${sample_id}_${cluster_variant}/pathofmpred_predictions.csv"), emit: predictions
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("pathofmpred_${sample_id}_${cluster_variant}"), emit: pathofmpred_dir

    script:
    def scriptPath = "${projectDir}/${params.pathofmpred_script}"
    def rscriptPath = params.pathofmpred_rscript as String
    """
    set -euo pipefail
    test -d "${params.pathofmpred_library_dir}" || { echo "Protected PathoFMPred R library is missing: ${params.pathofmpred_library_dir}" >&2; exit 2; }
    export R_LIBS_USER="${params.pathofmpred_library_dir}"
    export R_ENVIRON_USER=/dev/null
    export R_PROFILE_USER=/dev/null
    export HOME="\$PWD/.runtime_home"
    export XDG_CACHE_HOME="\$PWD/.runtime_cache"
    export MPLCONFIGDIR="\$XDG_CACHE_HOME/matplotlib"
    mkdir -p "\$HOME" "\$XDG_CACHE_HOME/fontconfig" "\$MPLCONFIGDIR"
    "${rscriptPath}" "${scriptPath}" \
      --features "${titan_embedding}" --cancer "${params.pathofmpred_cancer}" \
      --patient-id "${sample_id}" --outdir "pathofmpred_${sample_id}_${cluster_variant}" \
      --report-format "${params.pathofmpred_report_format}" \
      --include-limited-evidence "${params.pathofmpred_include_limited_evidence}"
    """

    stub:
    """
    mkdir -p "pathofmpred_${sample_id}_${cluster_variant}"
    touch "pathofmpred_${sample_id}_${cluster_variant}/pathofmpred_predictions.csv"
    """
}
