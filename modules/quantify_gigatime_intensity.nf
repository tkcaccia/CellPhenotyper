process QUANTIFY_GIGATIME_INTENSITY {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'quantify_gigatime_intensity', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([gigatime_input, mask_tif]) }
    tag "${sample_id}:${mask_name}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/05_gigatime/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.marker_quantification_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.marker_quantification_memory_gb as int))} GB" }
    time { params.marker_quantification_time as String }

    input:
    tuple val(sample_id), path(gigatime_input), path(mask_tif), val(mask_name)

    output:
    tuple val(sample_id), val(mask_name), path("${sample_id}_${mask_name}_gigatime_quantification.csv"), emit: quant_csv
    tuple val(sample_id), val(mask_name), path("${sample_id}_${mask_name}_gigatime_mean_intensity.csv"), emit: mean_csv
    tuple val(sample_id), val(mask_name), path("${sample_id}_${mask_name}_gigatime_intensity_stats.csv"), emit: stats_csv
    tuple val(sample_id), val(mask_name), path("${sample_id}_${mask_name}_gigatime_intensity_summary.json"), emit: summary_json
    tuple val(sample_id), val(mask_name), path("${sample_id}_${mask_name}_gigatime_quantification.csv"), path("${sample_id}_${mask_name}_gigatime_intensity_summary.json"), emit: profile_bundle

    script:
    def quantify_script = "${projectDir}/${params.marker_quantification_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([quantify_script, "${projectDir}/bin/profile_cell_morphology.py", "${projectDir}/bin/model_provenance.py"])
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Marker quantification code fingerprint: ${codeFingerprint}"

    python "${quantify_script}" \\
      --image "${gigatime_input}" \\
      --mask "${mask_tif}" \\
      --mask-name "${mask_name}" \\
      --out-quant-csv "${sample_id}_${mask_name}_gigatime_quantification.csv" \\
      --out-mean-csv "${sample_id}_${mask_name}_gigatime_mean_intensity.csv" \\
      --out-stats-csv "${sample_id}_${mask_name}_gigatime_intensity_stats.csv" \\
      --out-summary-json "${sample_id}_${mask_name}_gigatime_intensity_summary.json"
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    touch "${sample_id}_${mask_name}_gigatime_quantification.csv"
    touch "${sample_id}_${mask_name}_gigatime_mean_intensity.csv"
    touch "${sample_id}_${mask_name}_gigatime_intensity_stats.csv"
    touch "${sample_id}_${mask_name}_gigatime_intensity_summary.json"
    """
}
