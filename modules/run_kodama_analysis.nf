process RUN_KODAMA_ANALYSIS {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/08_kodama", mode: 'copy', overwrite: true
    publishDir "${params.outdir_base}/08_kodama_logs", mode: 'copy', overwrite: true, pattern: "*.Rout"

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.r_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.r_memory_gb as int))} GB" }
    time { params.r_time as String }

    input:
    tuple val(sample_id), path(embeddings_dir), path(objects_assigned_csv)

    output:
    tuple val(sample_id), path("kodama_output"), emit: kodama_dir
    tuple val(sample_id), path("KODAMA_${sample_id}.Rout"), emit: kodama_log

    script:
    def r_script = "${projectDir}/${params.r_script}"
    """
    set -euo pipefail

    mkdir -p kodama_output

    Rscript "${r_script}" \\
      "${embeddings_dir}" \\
      "${objects_assigned_csv}" \\
      "kodama_output" \\
      > "KODAMA_${sample_id}.Rout" 2>&1
    """

    stub:
    """
    mkdir -p kodama_output
    touch "kodama_output/kodama_stub.txt"
    touch "KODAMA_${sample_id}.Rout"
    """
}
