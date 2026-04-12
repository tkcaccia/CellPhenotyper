process MASK_TO_GEOJSON {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/12_cluster_geojson/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.cluster_geojson_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.cluster_geojson_memory_gb as int))} GB" }
    time { params.cluster_geojson_time as String }

    input:
    tuple val(sample_id), path(mask_tif)

    output:
    tuple val(sample_id), path("${sample_id}_grown_mask_smooth_class.geojson"), emit: cluster_geojson

    script:
    def geojson_script = "${projectDir}/${params.cluster_geojson_script}"
    def dissolve_by_value_flag = (params.cluster_geojson_dissolve_by_value as boolean) ? '--dissolve-by-value' : ''
    def fill_holes_flag = (params.cluster_geojson_fill_holes as boolean) ? '--fill-holes' : ''
    def preserve_flag = (params.cluster_geojson_preserve_topology as boolean) ? '--preserve-topology' : ''
    def group_map_flag = params.cluster_geojson_group_map ? "--group-map \"${params.cluster_geojson_group_map}\"" : ''
    def tail_flags = [dissolve_by_value_flag, fill_holes_flag, preserve_flag, group_map_flag].findAll { it?.trim() }.join(' ')
    """
    set -euo pipefail

    python "${geojson_script}" \
      --mask "${mask_tif}" \
      --page ${params.cluster_geojson_page} \
      --out "${sample_id}_grown_mask_smooth_class.geojson" \
      --min-area ${params.cluster_geojson_min_area} \
      --smooth-buffer ${params.cluster_geojson_smooth_buffer} \
      --smooth-passes ${params.cluster_geojson_smooth_passes} \
      --simplify ${params.cluster_geojson_simplify} \
      --group-prefix "${params.cluster_geojson_group_prefix}" \
      ${tail_flags}
    """

    stub:
    """
    touch "${sample_id}_grown_mask_smooth_class.geojson"
    """
}
