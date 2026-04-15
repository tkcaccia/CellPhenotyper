process REFINE_GROWN_TISSUE_MEDSAM {
    tag "${sample_id}"
    label 'gpu_capable'

    publishDir "${params.outdir_base}/17_medsam_refined_tissue/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.medsam_refine_cpus as int)) }
    memory { "${Math.max(4, Math.min(params.max_memory_gb as int, params.medsam_refine_memory_gb as int))} GB" }
    time { params.medsam_refine_time as String }

    input:
    tuple val(sample_id), path(image_tif), path(cluster_mask_tif), path(grown_mask_tif)

    output:
    tuple val(sample_id), path("${sample_id}_grown_mask_refined.ome.tif"), emit: refined_mask
    tuple val(sample_id), path("${sample_id}_grown_refined_qc_preview.png"), emit: refined_preview
    path("${sample_id}_medsam_*.png"), emit: medsam_pngs
    path("${sample_id}_medsam_summary.json"), emit: medsam_summary
    path("${sample_id}_medsam_debug"), optional: true, emit: medsam_debug

    script:
    def refine_script = "${projectDir}/${params.grown_tissue_refine_script}"
    def overwrite_flag = (params.grow_overwrite as boolean) ? '--overwrite' : ''
    def keep_tmp_flag = (params.grow_keep_tmp as boolean) ? '--keep-tmp' : ''
    def legacy_flag = (params.grow_legacy as boolean) ? '--legacy' : ''
    def medsamCheckpoint = params.medsam_checkpoint ? "--medsam-checkpoint \"${params.medsam_checkpoint}\"" : ''
    def medsamDevice = "--medsam-device ${(params.medsam_device ?: (params.compute_device == 'gpu' ? 'cuda' : 'cpu')).toString()}"
    def medsamBBoxMargin = "--medsam-bbox-margin ${params.medsam_bbox_margin}"
    def medsamComponentMinArea = "--medsam-component-min-area ${params.medsam_component_min_area}"
    def medsamComponentMergeDistance = "--medsam-component-merge-distance ${params.medsam_component_merge_distance}"
    def medsamSeedDilationRadius = "--medsam-seed-dilation-radius ${params.medsam_seed_dilation_radius}"
    def medsamCoreErosionRadius = "--medsam-core-erosion-radius ${params.medsam_core_erosion_radius}"
    def medsamOuterDilationRadius = "--medsam-outer-dilation-radius ${params.medsam_outer_dilation_radius}"
    def medsamMinObjectSize = "--medsam-min-object-size ${params.medsam_min_object_size}"
    def medsamSmoothRadius = "--medsam-smooth-radius ${params.medsam_smooth_radius}"
    def medsamCorePreservation = (params.medsam_force_core_preservation as boolean) ? '--medsam-force-core-preservation' : '--no-medsam-force-core-preservation'
    def medsamSaveDebug = (params.medsam_save_debug as boolean) ? '--medsam-save-debug' : '--no-medsam-save-debug'
    def tail_flags = [overwrite_flag, keep_tmp_flag, legacy_flag].findAll { it?.trim() }.join(' ')
    """
    set -euo pipefail

    python "${refine_script}" \
      --sample-id "${sample_id}" \
      --image "${image_tif}" \
      --seed-mask "${cluster_mask_tif}" \
      --grown-mask "${grown_mask_tif}" \
      --out "${sample_id}_grown_mask_refined.ome.tif" \
      --preview "${sample_id}_grown_refined_qc_preview.png" \
      --preview-factor ${params.grow_preview_factor} \
      --preview-threshold-mb ${params.grow_preview_threshold_mb} \
      --preview-alpha ${params.grow_preview_alpha} \
      --pyr-compression ${params.grow_pyr_compression} \
      --max-workers ${Math.max(1, Math.min(task.cpus as int, params.grow_max_workers as int))} \
      --downsample ${params.grow_downsample} \
      ${medsamCheckpoint} \
      ${medsamDevice} \
      ${medsamBBoxMargin} \
      ${medsamComponentMinArea} \
      ${medsamComponentMergeDistance} \
      ${medsamSeedDilationRadius} \
      ${medsamCoreErosionRadius} \
      ${medsamOuterDilationRadius} \
      ${medsamMinObjectSize} \
      ${medsamSmoothRadius} \
      ${medsamCorePreservation} \
      ${medsamSaveDebug} \
      ${tail_flags}
    """

    stub:
    """
    touch "${sample_id}_grown_mask_refined.ome.tif"
    touch "${sample_id}_grown_refined_qc_preview.png"
    touch "${sample_id}_medsam_editable_band.png"
    touch "${sample_id}_medsam_summary.json"
    mkdir -p "${sample_id}_medsam_debug"
    """
}
