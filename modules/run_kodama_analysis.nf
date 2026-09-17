process RUN_KODAMA_ANALYSIS {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'run_kodama_analysis', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([tile_embeddings_dir, cyto_embeddings_dir, inner_square_embeddings_dir, nuclei_embeddings_dir, objects_assigned_csv]) }
    tag "${sample_id}"
    label 'compute_medium'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/10_kodama/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    publishDir "${params.outdir_base}/10_kodama/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true, pattern: "*.Rout"

    cpus { TaskRuntime.cpus(runtime_plan, 'kodama') }
    memory { TaskRuntime.memory(runtime_plan, 'kodama') }
    time { params.r_time as String }

    input:
    tuple val(sample_id), path(tile_embeddings_dir, stageAs: 'tile_embeddings_dir'), path(cyto_embeddings_dir, stageAs: 'cyto_embeddings_dir'), path(inner_square_embeddings_dir, stageAs: 'inner_square_embeddings_dir'), path(nuclei_embeddings_dir, stageAs: 'nuclei_embeddings_dir'), path(objects_assigned_csv)
    val(runtime_plan)

    output:
    tuple val(sample_id), path("kodama_output"), emit: kodama_dir
    tuple val(sample_id), path("KODAMA_${sample_id}.Rout"), emit: kodama_log

    script:
    def r_loader_script = "${projectDir}/${params.r_data_loader_script}"
    def r_script = "${projectDir}/${params.r_script}"
    def requestedCores = TaskRuntime.setting(runtime_plan, 'kodama_n_cores', params.kodama_n_cores) as int
    def nCores = requestedCores > 0 ? Math.max(1, Math.min(task.cpus as int, requestedCores)) : (task.cpus as int)
    def codeFingerprint = PipelineHelpers.codeFingerprint([r_loader_script, r_script, "${projectDir}/bin/kodama_graph_export.R", "${projectDir}/bin/uni2_embedding_io.R"])
    def exportGraph = (params.cluster_representation ?: 'umap2d').toString() == 'kodama_graph' || (params.kodama_export_native_graph ?: false)
    def kodamaBackend = (params.kodama_backend ?: 'cpu').toString().toLowerCase()
    def kodamaGpuDevice = Math.max(0, params.kodama_gpu_device as int)
    def kodamaRscript = (params.kodama_rscript ?: 'Rscript').toString()
    def kodamaRLibrary = (params.kodama_r_library_dir ?: '/opt/micromamba/envs/kodama-r/lib/R/library').toString()
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] KODAMA code fingerprint: ${codeFingerprint}; effective CPU cores=${nCores}; task allocation=${task.cpus}"
    export R_LIBS_USER="${kodamaRLibrary}"
    export R_ENVIRON_USER=/dev/null
    export R_PROFILE_USER=/dev/null
    export HOME="\$PWD/.runtime_home"
    export XDG_CACHE_HOME="\$PWD/.runtime_cache"
    export MPLCONFIGDIR="\$XDG_CACHE_HOME/matplotlib"
    mkdir -p "\$HOME" "\$XDG_CACHE_HOME/fontconfig" "\$MPLCONFIGDIR"

    mkdir -p kodama_output

    "${kodamaRscript}" - <<'RS'
required_pkgs <- c("KODAMA", "fastEmbedR", "data.table", "digest", "jsonlite"${exportGraph ? ', "Matrix"' : ''})
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing_pkgs) > 0L) {
  stop(sprintf("Container is missing required R packages: %s", paste(missing_pkgs, collapse = ", ")))
}
cat(sprintf("[INFO] KODAMA runtime packages OK: %s\n", paste(required_pkgs, collapse = ",")))
cat(sprintf("[INFO] R library paths: %s\n", paste(.libPaths(), collapse = " | ")))
RS

    {
      "${kodamaRscript}" "${r_loader_script}" \
        "${tile_embeddings_dir}" \
        "${cyto_embeddings_dir}" \
        "${inner_square_embeddings_dir}" \
        "${nuclei_embeddings_dir}" \
        "${objects_assigned_csv}" \
        "kodama_output" \
        "${params.kodama_embedding_mode}" \
        "${params.kodama_spark_top_features}"

      "${kodamaRscript}" "${r_script}" \
        "kodama_output/rawdata.RData" \
        "kodama_output" \
        --embedding-mode "${params.kodama_embedding_mode}" \
        --dims-to-run ${params.kodama_dims_to_run} \
        --spark-top ${params.kodama_spark_top_features} \
        --landmarks ${params.kodama_landmarks} \
        --kodama-ncomp ${params.kodama_ncomp} \
        --save-selected-features ${params.kodama_save_selected_features} \
        --export-native-graph ${exportGraph} \
        --n-cores ${nCores} \
        --backend "${kodamaBackend}" \
        --gpu-device ${kodamaGpuDevice}
    } > "KODAMA_${sample_id}.Rout" 2>&1
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p kodama_output
    touch "kodama_output/kodama_stub.txt"
    touch "KODAMA_${sample_id}.Rout"
    """
}
