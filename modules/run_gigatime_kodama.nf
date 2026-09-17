process RUN_GIGATIME_KODAMA {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'run_gigatime_kodama', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([quant_dir]) }
    tag "${sample_id}"
    label 'compute_medium'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/10_kodama/${sample_id}/gigatime", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'kodama') }
    memory { TaskRuntime.memory(runtime_plan, 'kodama') }
    time { params.r_time as String }

    input:
    tuple val(sample_id), path(quant_dir)
    val(runtime_plan)

    output:
    tuple val(sample_id), path("gigatime_kodama_output"), emit: kodama_dir
    tuple val(sample_id), path("KODAMA_GIGATIME_${sample_id}.Rout"), emit: kodama_log

    script:
    def loader = "${projectDir}/${params.gigatime_kodama_loader_script}"
    def analysis = "${projectDir}/${params.r_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([loader, analysis, "${projectDir}/bin/kodama_graph_export.R"])
    def exportGraph = (params.cluster_representation ?: 'umap2d').toString() == 'kodama_graph' || (params.kodama_export_native_graph ?: false)
    def requestedCores = TaskRuntime.setting(runtime_plan, 'kodama_n_cores', params.kodama_n_cores) as int
    def nCores = requestedCores > 0 ? Math.max(1, Math.min(task.cpus as int, requestedCores)) : (task.cpus as int)
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] GigaTIME KODAMA code fingerprint: ${codeFingerprint}; effective CPU cores=${nCores}; task allocation=${task.cpus}"
    export R_LIBS_USER="${params.kodama_r_library_dir}"
    export R_ENVIRON_USER=/dev/null
    export R_PROFILE_USER=/dev/null
    export HOME="\$PWD/.runtime_home"
    mkdir -p "\$HOME" gigatime_kodama_output
    "${params.kodama_rscript}" - <<'RS'
required_pkgs <- c("KODAMA", "fastEmbedR", "data.table"${exportGraph ? ', "Matrix", "digest", "jsonlite"' : ''})
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing_pkgs)) stop(sprintf("Container is missing required R packages: %s", paste(missing_pkgs, collapse = ", ")))
RS
    {
      "${params.kodama_rscript}" "${loader}" "${quant_dir}" gigatime_kodama_output
      "${params.kodama_rscript}" "${analysis}" \
        gigatime_kodama_output/rawdata.RData gigatime_kodama_output \
        --embedding-mode tile \
        --dims-to-run ${params.kodama_dims_to_run} \
        --spark-top ${params.kodama_spark_top_features} \
        --landmarks ${params.kodama_landmarks} \
        --kodama-ncomp ${params.kodama_ncomp} \
        --save-selected-features ${params.kodama_save_selected_features} \
        --export-native-graph ${exportGraph} \
        --n-cores ${nCores} \
        --backend "${params.kodama_backend}" \
        --gpu-device ${params.kodama_gpu_device}
    } > "KODAMA_GIGATIME_${sample_id}.Rout" 2>&1
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p gigatime_kodama_output
    touch gigatime_kodama_output/kodama_stub.txt "KODAMA_GIGATIME_${sample_id}.Rout"
    """
}
