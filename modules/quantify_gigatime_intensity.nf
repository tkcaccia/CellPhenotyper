process QUANTIFY_GIGATIME_INTENSITY {
    tag "${sample_id}:${mask_name}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/06_marker_quantification/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.marker_quantification_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.marker_quantification_memory_gb as int))} GB" }
    time { params.marker_quantification_time as String }

    input:
    tuple val(sample_id), path(gigatime_input), path(mask_tif), val(mask_name)

    output:
    tuple val(sample_id), val(mask_name), path("${sample_id}_${mask_name}_gigatime_quantification.csv"), emit: quant_csv
    tuple val(sample_id), val(mask_name), path("${sample_id}_${mask_name}_gigatime_mean_intensity.csv"), emit: mean_csv
    tuple val(sample_id), val(mask_name), path("${sample_id}_${mask_name}_gigatime_intensity_stats.csv"), emit: stats_csv
    tuple val(sample_id), val(mask_name), path("${sample_id}_${mask_name}_gigatime_intensity_summary.json"), emit: summary_json

    script:
    def quantify_script = "${projectDir}/${params.marker_quantification_script}"
    """
    set -euo pipefail

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
    touch "${sample_id}_${mask_name}_gigatime_quantification.csv"
    touch "${sample_id}_${mask_name}_gigatime_mean_intensity.csv"
    touch "${sample_id}_${mask_name}_gigatime_intensity_stats.csv"
    touch "${sample_id}_${mask_name}_gigatime_intensity_summary.json"
    """
}
