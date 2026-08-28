process EXPORT_GIGATIME_OMETIFF {
    tag "${sample_id}"
    label 'compute_medium'

    // The primary GigaTIME directory may itself be a read-only publish symlink.
    publishDir "${params.outdir_base}/05_gigatime/${sample_id}/gigatime_${sample_id}_ometiff", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true, pattern: "gigatime_probs.ome.tif*"

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.gigatime_export_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.gigatime_export_memory_gb as int))} GB" }
    time { params.gigatime_export_time as String }

    input:
    tuple val(sample_id), path(gigatime_dir)

    output:
    tuple val(sample_id), path("gigatime_probs.ome.tif"), emit: ome_tiff
    tuple val(sample_id), path("gigatime_probs.ome.tif.verify.json"), emit: verification

    script:
    def export_script = "${projectDir}/${params.gigatime_export_script}"
    def pyramid_flag = params.gigatime_output_pyramid ? '' : '--no-pyramid'
    def output_channels = params.gigatime_export_channels ?: params.gigatime_output_channels ?: ''
    def output_channels_flag = output_channels ? "--output-channels \"${output_channels}\"" : ''
    """
    set -euo pipefail

    python "${export_script}" \\
      --input "${gigatime_dir}" \\
      --output "gigatime_probs.ome.tif" \\
      --tile-size ${params.gigatime_export_tile_size} \\
      --compression "${params.gigatime_export_compression}" \\
      --output-dtype "${params.gigatime_export_output_dtype}" \\
      --verification-json "gigatime_probs.ome.tif.verify.json" \\
      ${output_channels_flag} \\
      ${pyramid_flag}
    """

    stub:
    """
    touch gigatime_probs.ome.tif
    echo '{}' > gigatime_probs.ome.tif.verify.json
    """
}
