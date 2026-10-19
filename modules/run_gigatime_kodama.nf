process RUN_GIGATIME_KODAMA {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/10_kodama/${sample_id}/gigatime", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.r_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.r_memory_gb as int))} GB" }
    time { params.r_time as String }

    input:
    tuple val(sample_id), path(quant_dir)

    output:
    tuple val(sample_id), path("gigatime_kodama_output"), emit: kodama_dir
    tuple val(sample_id), path("KODAMA_GIGATIME_${sample_id}.Rout"), emit: kodama_log

    script:
    def loader = "${projectDir}/${params.gigatime_kodama_loader_script}"
    def analysis = "${projectDir}/${params.r_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([loader, analysis])
    def nCores = Math.max(1, Math.min(task.cpus as int, params.kodama_n_cores as int))
    """
    set -euo pipefail
    echo "[INFO] GigaTIME KODAMA code fingerprint: ${codeFingerprint}"
    export R_LIBS_USER="${params.kodama_r_library_dir}"
    export R_ENVIRON_USER=/dev/null
    export R_PROFILE_USER=/dev/null
    export HOME="\$PWD/.runtime_home"
    mkdir -p "\$HOME" gigatime_kodama_output
    {
      "${params.kodama_rscript}" "${loader}" "${quant_dir}" gigatime_kodama_output
      "${params.kodama_rscript}" "${analysis}" \
        gigatime_kodama_output/rawdata.RData gigatime_kodama_output \
        --embedding-mode tile \
        --dims-to-run ${params.kodama_dims_to_run} \
        --spark-top ${params.kodama_spark_top_features} \
        --landmarks ${params.kodama_landmarks} \
        --kodama-ncomp ${params.kodama_ncomp} \
        --n-cores ${nCores} \
        --backend "${params.kodama_backend}" \
        --gpu-device ${params.kodama_gpu_device}
    } > "KODAMA_GIGATIME_${sample_id}.Rout" 2>&1
    """

    stub:
    """
    mkdir -p gigatime_kodama_output
    touch gigatime_kodama_output/kodama_stub.txt "KODAMA_GIGATIME_${sample_id}.Rout"
    """
}
