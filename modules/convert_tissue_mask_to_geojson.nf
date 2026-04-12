process CONVERT_TISSUE_MASK_TO_GEOJSON {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/04_tissue_geojson", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.tissue_geojson_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.tissue_geojson_memory_gb as int))} GB" }
    time { params.tissue_geojson_time as String }

    input:
    tuple val(sample_id), path(tissue_mask_tif)

    output:
    tuple val(sample_id), path("${sample_id}_tissue_mask.geojson"), emit: tissue_geojson

    script:
    def binary_flag = params.tissue_geojson_binary ? '--binary' : ''
    def dissolve_flag = params.tissue_geojson_dissolve ? '--dissolve' : ''
    def dissolve_by_value_flag = params.tissue_geojson_dissolve_by_value ? '--dissolve-by-value' : ''
    def preserve_topology_flag = params.tissue_geojson_preserve_topology ? '--preserve-topology' : ''
    def fill_holes_flag = params.tissue_geojson_fill_holes ? '--fill-holes' : ''
    def tissue_geojson_script = "${projectDir}/${params.tissue_geojson_script}"
    """
    set -euo pipefail

    python "${tissue_geojson_script}" \\
      --mask "${tissue_mask_tif}" \\
      --page ${params.tissue_geojson_page} \\
      --out "${sample_id}_tissue_mask.geojson" \\
      --scale-x ${params.tissue_work_downsample} \\
      --scale-y ${params.tissue_work_downsample} \\
      ${binary_flag} \\
      ${dissolve_flag} \\
      ${dissolve_by_value_flag} \\
      --min-area ${params.tissue_geojson_min_area} \\
      --smooth-buffer ${params.tissue_geojson_smooth_buffer} \\
      --smooth-passes ${params.tissue_geojson_smooth_passes} \\
      --simplify ${params.tissue_geojson_simplify} \\
      ${preserve_topology_flag} \\
      ${fill_holes_flag}
    """

    stub:
    """
    printf '{"type":"FeatureCollection","features":[]}\n' > "${sample_id}_tissue_mask.geojson"
    """
}
