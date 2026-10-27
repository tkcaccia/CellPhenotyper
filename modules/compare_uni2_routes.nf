process COMPARE_UNI2_ROUTES {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/10_kodama/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.uni2_route_compare_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.uni2_route_compare_memory_gb as int))} GB" }
    time { params.uni2_route_compare_time as String }

    input:
    tuple val(sample_id), path(grid_kodama_dir), path(cell_kodama_dir), path(grid_objects_csv), path(cell_objects_csv), path(marker_quant_dir)

    output:
    tuple val(sample_id), path("uni2_route_comparison_${sample_id}"), emit: comparison_dir

    script:
    def scriptPath = "${projectDir}/${params.uni2_route_compare_script}"
    def rscript = (params.kodama_rscript ?: 'Rscript').toString()
    def rLibrary = (params.kodama_r_library_dir ?: '/opt/micromamba/envs/kodama-r/lib/R/library').toString()
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    codeDigest.update(new File(scriptPath).bytes)
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    """
    set -euo pipefail
    echo "[INFO] UNI-2 route comparison code fingerprint: ${codeFingerprint}"
    export R_LIBS_USER="${rLibrary}"
    export R_ENVIRON_USER=/dev/null
    export R_PROFILE_USER=/dev/null
    export HOME="\$PWD/.runtime_home"
    mkdir -p "\$HOME" "uni2_route_comparison_${sample_id}"

    "${rscript}" "${scriptPath}" \
      "${sample_id}" \
      "${grid_kodama_dir}" \
      "${cell_kodama_dir}" \
      "${grid_objects_csv}" \
      "${cell_objects_csv}" \
      "${marker_quant_dir}" \
      "uni2_route_comparison_${sample_id}"
    """

    stub:
    """
    mkdir -p "uni2_route_comparison_${sample_id}"
    touch "uni2_route_comparison_${sample_id}/${sample_id}_uni2_route_comparison.csv"
    touch "uni2_route_comparison_${sample_id}/${sample_id}_uni2_route_marker_endpoints.csv"
    touch "uni2_route_comparison_${sample_id}/${sample_id}_uni2_route_comparison.md"
    touch "uni2_route_comparison_${sample_id}/${sample_id}_uni2_route_comparison.png"
    """
}
