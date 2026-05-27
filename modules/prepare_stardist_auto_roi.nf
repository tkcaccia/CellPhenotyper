process PREPARE_STARDIST_AUTO_ROI {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/02_stardist/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.stardist_auto_roi_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.stardist_auto_roi_memory_gb as int))} GB" }
    time { params.stardist_auto_roi_time as String }

    input:
    tuple val(sample_id), path(image_path)

    output:
    tuple val(sample_id), path("${sample_id}.stardist_auto_roi.geojson"), emit: roi_geojson
    tuple val(sample_id), path("${sample_id}.stardist_auto_roi_preview.png"), optional: true, emit: preview_png

    script:
    def auto_roi_script = "${projectDir}/${params.stardist_auto_roi_script}"
    def keep_largest_flag = params.stardist_auto_roi_keep_largest ? '--keep-largest' : ''
    def fill_holes_flag = params.stardist_auto_roi_fill_holes ? '--fill-holes' : ''
    """
    set -euo pipefail

      python "${auto_roi_script}" \\
      --image "${image_path}" \\
      --out "${sample_id}.stardist_auto_roi.geojson" \\
      --preview "${sample_id}.stardist_auto_roi_preview.png" \\
      --downsample ${params.stardist_auto_roi_downsample} \\
      --close-radius ${params.stardist_auto_roi_close_radius} \\
      --min-obj-area ${params.stardist_auto_roi_min_obj_area} \\
      --hole-area ${params.stardist_auto_roi_hole_area} \\
      --smooth-buffer ${params.stardist_auto_roi_smooth_buffer} \\
      --smooth-passes ${params.stardist_auto_roi_smooth_passes} \\
      --simplify ${params.stardist_auto_roi_simplify} \\
      ${keep_largest_flag} \\
      ${fill_holes_flag}
    """

    stub:
    """
    printf '{"type":"FeatureCollection","features":[]}\n' > "${sample_id}.stardist_auto_roi.geojson"
    touch "${sample_id}.stardist_auto_roi_preview.png"
    """
}
