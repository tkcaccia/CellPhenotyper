process REFINE_GROWN_TISSUE_MEDSAM {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'refine_grown_tissue_medsam', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([image_tif, cluster_mask_tif, grown_mask_tif, tissue_mask_tif, kodama_membership_png, resolution_json, clustering_uncertainty_tif]) }
    tag "${sample_id}:${cluster_variant}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/14_medsam_refine_tissue/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { TaskRuntime.cpus(runtime_plan, 'medsam_refine') }
    memory { TaskRuntime.memory(runtime_plan, 'medsam_refine') }
    time { params.medsam_refine_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(image_tif), path(cluster_mask_tif, stageAs: 'seed_cluster_mask.tif'), path(grown_mask_tif, stageAs: 'baseline_grown_mask.ome.tif'), path(tissue_mask_tif), path(kodama_membership_png), path(resolution_json), path(clustering_uncertainty_tif, stageAs: 'input_clustering_uncertainty.tif')
    val(runtime_plan)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif"), emit: refined_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_refined_uncertainty.tif"), emit: refined_uncertainty
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_refined_provenance.tif"), emit: refined_provenance
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif.provenance.json"), emit: provenance_metadata
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_refined_qc_preview.png"), emit: refined_preview
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_medsam_kodama_membership.png"), emit: kodama_plot_png
    path("${sample_id}_${cluster_variant}_medsam_*.png"), emit: medsam_pngs
    path("${sample_id}_${cluster_variant}_medsam_summary.json"), emit: medsam_summary
    path("${sample_id}_${cluster_variant}_medsam_debug"), optional: true, emit: medsam_debug

    script:
    def refine_script = "${projectDir}/${params.grown_tissue_refine_script}"
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    [
      refine_script,
      "${projectDir}/bin/grow_to_tissue.py",
      "${projectDir}/bin/grow_to_tissue_core.py",
      "${projectDir}/bin/medsam_border_refine.py",
      "${projectDir}/bin/annealed_wand_boundary.py",
      "${projectDir}/bin/tissue_appearance_refine.py",
      "${projectDir}/bin/ome_tiff_metadata.py",
    ].each { codeDigest.update(new File(it).bytes) }
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    def overwrite_flag = (params.grow_overwrite as boolean) ? '--overwrite' : ''
    def keep_tmp_flag = (params.grow_keep_tmp as boolean) ? '--keep-tmp' : ''
    def legacy_flag = (params.grow_legacy as boolean) ? '--legacy' : ''
    def medsamCheckpoint = params.medsam_checkpoint ? "--medsam-checkpoint \"${params.medsam_checkpoint}\"" : ''
    def resolvedComputeDevice = TaskRuntime.device(runtime_plan)
    def requestedMedsamDevice = (params.medsam_device ?: 'auto').toString().trim().toLowerCase()
    def resolvedMedsamDevice = requestedMedsamDevice == 'auto' ? (resolvedComputeDevice == 'gpu' ? 'cuda' : 'cpu') : requestedMedsamDevice
    def medsamDevice = "--medsam-device ${resolvedMedsamDevice}"
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
    def preBoundaryCompetition = (params.medsam_pre_boundary_competition as boolean) ? '--pre-boundary-competition' : '--no-pre-boundary-competition'
    def preBoundaryFlags = "--pre-boundary-radius ${params.medsam_pre_boundary_radius} --pre-boundary-downsample ${params.medsam_pre_boundary_downsample} --pre-boundary-iterations ${params.medsam_pre_boundary_iterations} --pre-boundary-initial-temperature ${params.medsam_pre_boundary_initial_temperature} --pre-boundary-final-temperature ${params.medsam_pre_boundary_final_temperature} --pre-boundary-data-weight ${params.medsam_pre_boundary_data_weight} --pre-boundary-smoothness-weight ${params.medsam_pre_boundary_smoothness_weight} --pre-boundary-edge-beta ${params.medsam_pre_boundary_edge_beta} --pre-boundary-connectivity ${params.medsam_pre_boundary_connectivity ?: 8}"
    def imageGuidedRefine = (params.medsam_image_guided_internal_refine as boolean) ? '--image-guided-internal-refine' : '--no-image-guided-internal-refine'
    def internalBoundaryRadius = "--internal-boundary-radius ${params.medsam_internal_boundary_radius}"
    def internalGradientFlags = "--internal-gradient-space ${params.medsam_internal_gradient_space ?: 'luminance'} --internal-gradient-sigma-px ${params.medsam_internal_gradient_sigma_px ?: 0.0} --internal-watershed-compactness ${params.medsam_internal_watershed_compactness ?: 0.0}"
    def appearanceRefine = (params.medsam_appearance_refine as boolean) ? '--appearance-refine' : '--no-appearance-refine'
    def appearanceFlags = "--appearance-downsample ${params.medsam_appearance_downsample} --appearance-clusters ${params.medsam_appearance_clusters} --appearance-core-erosion-px ${params.medsam_appearance_core_erosion_px} --appearance-smooth-sigma-px ${params.medsam_appearance_smooth_sigma_px} --appearance-min-region-area-px ${params.medsam_appearance_min_region_area_px} --appearance-sample-pixels ${params.medsam_appearance_sample_pixels} --appearance-random-seed ${params.medsam_appearance_random_seed}"
    def appearanceMulticlass = (params.medsam_appearance_multiclass as boolean) ? '--appearance-multiclass' : '--no-appearance-multiclass'
    def appearanceVoteFlags = "--appearance-min-vote-fraction ${params.medsam_appearance_min_vote_fraction ?: 0.0} --appearance-min-vote-margin ${params.medsam_appearance_min_vote_margin ?: 0.0}"
    def physicalOptions = [
      'medsam-core-erosion-um': params.medsam_core_erosion_um,
      'medsam-outer-dilation-um': params.medsam_outer_dilation_um,
      'medsam-smooth-um': params.medsam_smooth_um,
      'pre-boundary-radius-um': params.medsam_pre_boundary_radius_um,
      'internal-boundary-radius-um': params.medsam_internal_boundary_radius_um,
      'internal-gradient-sigma-um': (params.medsam_internal_gradient_sigma_um ?: null),
      'appearance-core-erosion-um': params.medsam_appearance_core_erosion_um,
      'appearance-smooth-sigma-um': params.medsam_appearance_smooth_sigma_um,
      'appearance-min-region-area-um2': params.medsam_appearance_min_region_area_um2,
    ]
    def physicalFlags = physicalOptions.findAll { key, value -> value != null && value.toString().trim() != '' }
        .collect { key, value -> "--${key} ${value as double}" }.join(' ')
    def authoritativeMpp = (params.input_resolution_override_mpp as double) > 0.0 ? (params.input_resolution_override_mpp as double) : 0.0
    def tail_flags = [overwrite_flag, keep_tmp_flag, legacy_flag].findAll { it?.trim() }.join(' ')
    def memoryBudgetGb = task.memory.toBytes() / (1024.0d * 1024.0d * 1024.0d)
    def hardwareProfile = TaskRuntime.profile(runtime_plan)
    def medsamAutoHardware = (TaskRuntime.setting(runtime_plan, 'hardware_auto', params.hardware_auto) as boolean) &&
      (TaskRuntime.setting(runtime_plan, 'medsam_refine_auto_hardware', params.medsam_refine_auto_hardware) as boolean) &&
      resolvedMedsamDevice.startsWith('cuda')
    // HardwarePolicy already resolves automatic workers. Do not re-expand an
    // explicit/planned worker cap with a second task-local profile heuristic.
    def requestedMaxWorkers = TaskRuntime.setting(runtime_plan, 'medsam_refine_max_workers', params.medsam_refine_max_workers) as int
    def resolvedMaxWorkers = Math.max(1, Math.min(task.cpus as int, requestedMaxWorkers))
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] MedSAM refinement code fingerprint: ${codeFingerprint}"
    echo "[INFO] MedSAM runtime tune: profile=${hardwareProfile}, auto_hardware=${medsamAutoHardware}, memory_budget_gb=${memoryBudgetGb}, max_workers=${resolvedMaxWorkers}"
    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=${task.cpus}
    export OPENBLAS_NUM_THREADS=${task.cpus}
    export NUMEXPR_NUM_THREADS=${task.cpus}

    python "${refine_script}" \
      --sample-id "${sample_id}_${cluster_variant}" \
      --image "${image_tif}" \
      --seed-mask "${cluster_mask_tif}" \
      --grown-mask "${grown_mask_tif}" \
      --tissue-mask "${tissue_mask_tif}" \
      --resolution-json "${resolution_json}" \
      --clustering-uncertainty "${clustering_uncertainty_tif}" \
      --default-mpp ${authoritativeMpp} \
      --out "${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif" \
      --uncertainty-out "${sample_id}_${cluster_variant}_grown_mask_refined_uncertainty.tif" \
      --provenance-out "${sample_id}_${cluster_variant}_grown_mask_refined_provenance.tif" \
      --preview "${sample_id}_${cluster_variant}_grown_refined_qc_preview.png" \
      --preview-factor ${params.grow_preview_factor} \
      --preview-threshold-mb ${params.grow_preview_threshold_mb} \
      --preview-alpha ${params.grow_preview_alpha} \
      --pyr-compression ${params.grow_pyr_compression} \
      --max-workers ${resolvedMaxWorkers} \
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
      ${preBoundaryCompetition} \
      ${preBoundaryFlags} \
      ${imageGuidedRefine} \
      ${internalBoundaryRadius} \
      ${internalGradientFlags} \
      ${appearanceRefine} \
      ${appearanceFlags} \
      ${appearanceMulticlass} \
      ${appearanceVoteFlags} \
      ${physicalFlags} \
      ${tail_flags}

    cp "${kodama_membership_png}" "${sample_id}_${cluster_variant}_medsam_kodama_membership.png"
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    touch "${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif"
    touch "${sample_id}_${cluster_variant}_grown_mask_refined_uncertainty.tif"
    touch "${sample_id}_${cluster_variant}_grown_mask_refined_provenance.tif"
    printf '{"schema_version":"1.1.0","stub":true,"claim":"stub contains no uncertainty evidence"}\\n' > "${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif.provenance.json"
    touch "${sample_id}_${cluster_variant}_grown_refined_qc_preview.png"
    touch "${sample_id}_${cluster_variant}_medsam_kodama_membership.png"
    touch "${sample_id}_${cluster_variant}_medsam_editable_band.png"
    printf '{"model_provenance":{"used_model":false}}\n' > "${sample_id}_${cluster_variant}_medsam_summary.json"
    mkdir -p "${sample_id}_${cluster_variant}_medsam_debug"
    """
}
