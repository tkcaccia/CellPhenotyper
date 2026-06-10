process REFINE_GROWN_TISSUE_MEDSAM {
    tag "${sample_id}:${cluster_variant}"
    label 'gpu_capable'
    maxForks 1

    publishDir "${params.outdir_base}/14_medsam_refine_tissue/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.medsam_refine_cpus as int)) }
    memory { "${Math.max(4, Math.min(params.max_memory_gb as int, params.medsam_refine_memory_gb as int))} GB" }
    time { params.medsam_refine_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(image_tif), path(cluster_mask_tif), path(grown_mask_tif), path(kodama_membership_png)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif"), emit: refined_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_refined_qc_preview.png"), emit: refined_preview
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_medsam_kodama_membership.png"), emit: kodama_plot_png
    path("${sample_id}_${cluster_variant}_medsam_*.png"), emit: medsam_pngs
    path("${sample_id}_${cluster_variant}_medsam_summary.json"), emit: medsam_summary
    path("${sample_id}_${cluster_variant}_medsam_debug"), optional: true, emit: medsam_debug

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
    def medsamClusterTileSize = "--medsam-cluster-tile-size ${params.medsam_cluster_tile_size}"
    def medsamClusterTileOverlap = "--medsam-cluster-tile-overlap ${params.medsam_cluster_tile_overlap}"
    def medsamLargeImageMode = "--large-image-mode ${(params.medsam_large_image_mode ?: 'auto').toString()}"
    def medsamLargeImageMaxPixels = "--large-image-max-pixels ${params.medsam_large_image_max_pixels}"
    def medsamStreamBlockRows = "--stream-block-rows ${params.medsam_stream_block_rows}"
    def medsamStreamResumeTmp = params.medsam_stream_resume_tmp ? "--stream-resume-tmp \"${params.medsam_stream_resume_tmp}\"" : ''
    def medsamStreamResumeTiles = (params.medsam_stream_resume_tiles as int) > 0 ? "--stream-resume-tiles ${params.medsam_stream_resume_tiles}" : ''
    def medsamStreamResumeGridY = (params.medsam_stream_resume_grid_y as int) > 0 ? "--stream-resume-grid-y ${params.medsam_stream_resume_grid_y}" : ''
    def medsamStreamResumeGridX = (params.medsam_stream_resume_grid_x as int) > 0 ? "--stream-resume-grid-x ${params.medsam_stream_resume_grid_x}" : ''
    def medsamStreamProgressJson = params.medsam_stream_progress_json ? "--stream-progress-json \"${params.medsam_stream_progress_json}\"" : ''
    def medsamQcCropSize = "--medsam-qc-crop-size ${params.medsam_qc_crop_size}"
    def medsamQcRandomSeed = "--medsam-qc-random-seed ${params.medsam_qc_random_seed}"
    def medsamCorePreservation = (params.medsam_force_core_preservation as boolean) ? '--medsam-force-core-preservation' : '--no-medsam-force-core-preservation'
    def medsamSaveDebug = (params.medsam_save_debug as boolean) ? '--medsam-save-debug' : '--no-medsam-save-debug'
    def tail_flags = [overwrite_flag, keep_tmp_flag, legacy_flag].findAll { it?.trim() }.join(' ')
    """
    set -euo pipefail

    python "${refine_script}" \
      --sample-id "${sample_id}_${cluster_variant}" \
      --image "${image_tif}" \
      --seed-mask "${cluster_mask_tif}" \
      --grown-mask "${grown_mask_tif}" \
      --out "${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif" \
      --preview "${sample_id}_${cluster_variant}_grown_refined_qc_preview.png" \
      --preview-factor ${params.grow_preview_factor} \
      --preview-threshold-mb ${params.grow_preview_threshold_mb} \
      --preview-alpha ${params.grow_preview_alpha} \
      --pyr-compression ${params.grow_pyr_compression} \
      --max-workers ${Math.max(1, Math.min(task.cpus as int, params.medsam_refine_max_workers as int))} \
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
      ${medsamClusterTileSize} \
      ${medsamClusterTileOverlap} \
      ${medsamLargeImageMode} \
      ${medsamLargeImageMaxPixels} \
      ${medsamStreamBlockRows} \
      ${medsamStreamResumeTmp} \
      ${medsamStreamResumeTiles} \
      ${medsamStreamResumeGridY} \
      ${medsamStreamResumeGridX} \
      ${medsamStreamProgressJson} \
      ${medsamQcCropSize} \
      ${medsamQcRandomSeed} \
      ${medsamCorePreservation} \
      ${medsamSaveDebug} \
      ${tail_flags}

    cp "${kodama_membership_png}" "${sample_id}_${cluster_variant}_medsam_kodama_membership.png"
    """

    stub:
    """
    touch "${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif"
    touch "${sample_id}_${cluster_variant}_grown_refined_qc_preview.png"
    touch "${sample_id}_${cluster_variant}_medsam_kodama_membership.png"
    touch "${sample_id}_${cluster_variant}_medsam_editable_band.png"
    touch "${sample_id}_${cluster_variant}_medsam_summary.json"
    mkdir -p "${sample_id}_${cluster_variant}_medsam_debug"
    """
}
