process BUILD_TISSUE_MASK {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/03_tissue_mask/${sample_id}", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.tissue_mask_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.tissue_mask_memory_gb as int))} GB" }
    time { params.tissue_mask_time as String }

    input:
    tuple val(sample_id), path(image_for_mask)

    output:
    tuple val(sample_id), path("${sample_id}_tissue_mask.tif"), emit: tissue_mask
    tuple val(sample_id), path("${sample_id}_tissue_mask_preview.png"), emit: tissue_preview

    script:
    def keep_largest_flag = params.tissue_keep_largest ? '--keep-largest' : ''
    def tissue_mask_script = "${projectDir}/${params.tissue_mask_script}"
    """
    set -euo pipefail

    python "${tissue_mask_script}" \\
      --image "${image_for_mask}" \\
      --out-mask "${sample_id}_tissue_mask.tif" \\
      --preview "${sample_id}_tissue_mask_preview.png" \\
      --preview-factor ${params.tissue_preview_factor} \\
      --work-downsample ${params.tissue_work_downsample} \\
      --auto-no-downsample-max-side ${params.tissue_auto_no_downsample_max_side} \\
      --close-radius ${params.tissue_close_radius} \\
      --min-obj-area ${params.tissue_min_obj_area} \\
      --hole-area ${params.tissue_hole_area} \\
      --tile ${params.tissue_tile} \\
      --compression ${params.tissue_compression} \\
      ${params.tissue_bigtiff ? '--bigtiff' : ''} \\
      ${keep_largest_flag}
    """

    stub:
    """
    touch "${sample_id}_tissue_mask.tif"
    touch "${sample_id}_tissue_mask_preview.png"
    """
}
