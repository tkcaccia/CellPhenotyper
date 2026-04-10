process FINAL_GEOJSON_TO_MASK {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/13_cluster_geojson_mask/${sample_id}", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.final_geojson_mask_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.final_geojson_mask_memory_gb as int))} GB" }
    time { params.final_geojson_mask_time as String }

    input:
    tuple val(sample_id), path(cluster_geojson), path(reference_tif)

    output:
    tuple val(sample_id), path("${sample_id}_grown_mask_smooth_class.tif"), emit: cluster_mask

    script:
    def rasterize_script = "${projectDir}/${params.geojson_to_mask_script}"
    """
    set -euo pipefail

    python "${rasterize_script}" \
      --geojson "${cluster_geojson}" \
      --reference "${reference_tif}" \
      --reference-page 0 \
      --out "${sample_id}_grown_mask_smooth_class.tif" \
      --value-prop "${params.final_geojson_mask_value_prop}" \
      --default-value ${params.final_geojson_mask_default_value} \
      --fill-value 0 \
      --compression "${params.final_geojson_mask_compression}"
    """

    stub:
    """
    touch "${sample_id}_grown_mask_smooth_class.tif"
    """
}
