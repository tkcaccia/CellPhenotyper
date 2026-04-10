process ROI_GEOJSON_TO_MASK {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/04_roi_mask/${sample_id}", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.input_roi_mask_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.input_roi_mask_memory_gb as int))} GB" }
    time { params.input_roi_mask_time as String }

    input:
    tuple val(sample_id), path(roi_geojson), path(reference_tif)

    output:
    tuple val(sample_id), path("${sample_id}_input_roi_mask.tif"), emit: roi_mask

    script:
    def rasterize_script = "${projectDir}/${params.geojson_to_mask_script}"
    """
    set -euo pipefail

    python "${rasterize_script}" \
      --geojson "${roi_geojson}" \
      --reference "${reference_tif}" \
      --reference-page 0 \
      --out "${sample_id}_input_roi_mask.tif" \
      --binary \
      --default-value 1 \
      --fill-value 0 \
      --compression "${params.input_roi_mask_compression}"
    """

    stub:
    """
    touch "${sample_id}_input_roi_mask.tif"
    """
}
