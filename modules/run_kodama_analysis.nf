process RUN_KODAMA_ANALYSIS {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/11_kodama/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    publishDir "${params.outdir_base}/12_kodama_logs/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true, pattern: "*.Rout"

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.r_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.r_memory_gb as int))} GB" }
    time { params.r_time as String }

    input:
    tuple val(sample_id), path(tile_embeddings_dir), path(cyto_embeddings_dir), path(inner_square_embeddings_dir), path(nuclei_embeddings_dir), path(objects_assigned_csv)

    output:
    tuple val(sample_id), path("kodama_output"), emit: kodama_dir
    tuple val(sample_id), path("KODAMA_${sample_id}.Rout"), emit: kodama_log

    script:
    def r_loader_script = "${projectDir}/${params.r_data_loader_script}"
    def r_script = "${projectDir}/${params.r_script}"
    def nCores = Math.max(1, Math.min(task.cpus as int, params.kodama_n_cores as int))
    """
    set -euo pipefail

    mkdir -p kodama_output

    Rscript - <<'RS'
required_pkgs <- c("KODAMA", "KODAMAextra", "SPARK", "data.table", "irlba", "umap", "bluster")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing_pkgs) > 0L) {
  stop(sprintf("Container is missing required R packages: %s", paste(missing_pkgs, collapse = ", ")))
}
cat(sprintf("[INFO] KODAMA runtime packages OK: %s\n", paste(required_pkgs, collapse = ",")))
cat(sprintf("[INFO] R library paths: %s\n", paste(.libPaths(), collapse = " | ")))
RS

    {
      Rscript "${r_loader_script}" \
        "${tile_embeddings_dir}" \
        "${cyto_embeddings_dir}" \
        "${inner_square_embeddings_dir}" \
        "${nuclei_embeddings_dir}" \
        "${objects_assigned_csv}" \
        "kodama_output"

      Rscript "${r_script}" \
        "kodama_output/rawdata.RData" \
        "kodama_output" \
        --embedding-mode "${params.kodama_embedding_mode}" \
        --dims-to-run ${params.kodama_dims_to_run} \
        --spark-top ${params.kodama_spark_top_features} \
        --n-cores ${nCores}
    } > "KODAMA_${sample_id}.Rout" 2>&1
    """

    stub:
    """
    mkdir -p kodama_output
    touch "kodama_output/kodama_stub.txt"
    touch "KODAMA_${sample_id}.Rout"
    """
}
