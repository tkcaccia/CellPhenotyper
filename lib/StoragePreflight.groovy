import groovy.json.JsonSlurper


class StoragePreflight {
    private static boolean truthy(def value, boolean fallback = false) {
        if (value == null) return fallback
        value.toString().trim().toLowerCase() in ['true', '1', 'yes', 'y', 'on']
    }

    private static void addArg(List<String> command, String flag, def value) {
        command << flag << (value == null ? '' : value.toString())
    }

    private static void addCache(List<String> command, String label, def rawPath) {
        def value = rawPath == null ? '' : rawPath.toString().trim()
        if (value) command << '--cache' << "${label}=${value}".toString()
    }

    static Map run(def params, List<String> imagePaths, def baseDirValue, def workDirValue, def profileValue) {
        def rawMode = params.storage_preflight_mode
        def mode = rawMode == null
            ? 'fail'
            : (rawMode instanceof Boolean && !rawMode ? 'off' : rawMode.toString().trim().toLowerCase())
        if (mode == 'off') return [schema_version: 1, status: 'off', policy_mode: 'off']
        if (!(mode in ['warn', 'fail'])) {
            throw new IllegalArgumentException("storage_preflight_mode must be off, warn, or fail")
        }

        def baseDir = new File(baseDirValue.toString()).canonicalFile
        def script = new File(
            baseDir,
            (params.storage_preflight_script ?: 'bin/storage_preflight.py').toString()
        ).canonicalFile
        if (!script.isFile()) {
            throw new IllegalStateException("Storage preflight script not found: ${script}")
        }
        def outdir = new File(params.outdir_base.toString()).canonicalFile
        def executionDir = new File(outdir, '00_execution')
        executionDir.mkdirs()
        def output = new File(executionDir, 'storage_preflight.json')
        def python = (params.storage_preflight_python ?: 'python3').toString().trim()
        def command = [python, script.absolutePath]
        imagePaths.each { path -> addArg(command, '--input', path) }
        def roi = params.roi_geojson == null ? '' : params.roi_geojson.toString().trim()
        if (roi) addArg(command, '--roi-geojson', roi)
        def inputMetadata = params.storage_preflight_input_metadata == null ? '' : params.storage_preflight_input_metadata.toString().trim()
        if (inputMetadata) addArg(command, '--input-metadata-json', new File(inputMetadata).canonicalPath)
        addArg(command, '--outdir', outdir.absolutePath)
        addArg(command, '--workdir', workDirValue)
        addArg(command, '--output-json', output.absolutePath)
        addArg(command, '--start-point', params._resolved_start_point)
        addArg(command, '--end-point', params._resolved_end_point)
        addArg(command, '--publish-dir-mode', params.publish_dir_mode ?: 'copy')
        addArg(command, '--cell-detection-mode', params._resolved_cell_detection_mode)
        addArg(command, '--hovernet-execution-mode', params.hovernet_execution_mode ?: 'streaming_tiles')
        addArg(command, '--hovernet-target-mpp', params.hovernet_target_mpp == null ? 0.25 : params.hovernet_target_mpp)
        addArg(command, '--hovernet-stream-core-size', params.hovernet_stream_core_size ?: 4096)
        addArg(command, '--hovernet-stream-halo', params.hovernet_stream_halo ?: 256)
        addArg(command, '--hovernet-stream-batch-tiles', params.hovernet_stream_batch_tiles ?: 64)
        addArg(command, '--hovernet-export-contours', truthy(params.hovernet_export_contours, false))
        addArg(command, '--uni2-sampling-mode', params._resolved_uni2_sampling_mode)
        addArg(command, '--gigatime-enabled', truthy(params.gigatime_enable, true))
        addArg(command, '--marker-quantification-enabled', truthy(params.marker_quantification_enable, true))
        // Empty channel selection means the complete 23-channel model panel.
        // Pass the selection unchanged so Python owns this shared interpretation.
        addArg(command, '--gigatime-output-channels', params.gigatime_output_channels ?: '')
        addArg(command, '--gigatime-output-dtype', params.gigatime_output_dtype ?: 'float32')
        addArg(command, '--gigatime-output-format', params.gigatime_output_format ?: 'zarr')
        addArg(command, '--gigatime-output-pyramid', truthy(params.gigatime_output_pyramid, true))
        addArg(command, '--gigatime-export-ometiff', truthy(params.gigatime_export_ometiff, true))
        addArg(command, '--gigatime-export-channels', params.gigatime_export_channels ?: '')
        addArg(command, '--gigatime-export-output-dtype', params.gigatime_export_output_dtype ?: 'auto')
        addArg(command, '--gigatime-blockwise', truthy(params.gigatime_blockwise, true))
        addArg(command, '--gigatime-target-mpp', params.gigatime_target_mpp == null ? 0.25 : params.gigatime_target_mpp)
        addArg(command, '--source-mpp', params.input_resolution_override_mpp ?: 0.0)
        addArg(command, '--physical-compartments', params.expand_um == null || (params.expand_um as double) >= 0)
        addArg(command, '--cell-profiles-enabled', truthy(params.cell_profiles_enable, false))
        addArg(command, '--cohort-niches-enabled', truthy(params.cohort_niches_enable, false))
        addArg(command, '--cohort-niches-max-k', params.cohort_niches_max_k ?: 8)
        addArg(command, '--cell-neighborhood-support-mode', params.cell_neighborhood_support_mode ?: 'provided')
        addArg(command, '--cell-neighborhood-feature-storage', params.cell_neighborhood_feature_storage ?: 'table')
        addArg(command, '--cell-profiles-spatialdata', truthy(params.cell_profiles_spatialdata, false))
        for (kind in ['cell', 'region']) {
            def atlas = params["${kind}_reference_atlas"]
            if (atlas) addArg(command, "--${kind}-reference-atlas", new File(atlas.toString()).canonicalPath)
        }
        addArg(command, '--spatialdata-pyramid-levels', params.spatialdata_pyramid_levels ?: 0)
        addArg(command, '--cell-profiles-uni2-enabled', truthy(params.cell_profiles_uni2_enable, true))
        addArg(command, '--cell-profiles-markers-enabled', truthy(params.cell_profiles_markers_enable, true))
        addArg(command, '--cellvit-embeddings-enabled', truthy(params.cellvit_export_embeddings, false))
        addArg(command, '--cell-neighborhood-radii-um', params.cell_neighborhood_radii_um ?: '25,50,100')
        addArg(command, '--tissue-hierarchy-enabled', truthy(params.tissue_hierarchy_enable, false))
        addArg(command, '--hierarchy-grid-model-tile-size', params.uni2_tile_size ?: 224)
        addArg(command, '--hierarchy-grid-inner-size', params.uni2_inner_square_fixed_px ?: 90)
        addArg(command, '--hierarchy-grid-target-mpp', params.uni2_target_mpp ?: 0.25)
        addArg(command, '--hierarchy-max-components', params.tissue_hierarchy_max_components ?: 1000000)
        addArg(command, '--hierarchy-device', params.tissue_hierarchy_device ?: 'cuda')
        addArg(command, '--hierarchy-feature-batch', params.tissue_hierarchy_feature_batch ?: 16)
        addArg(command, '--hierarchy-max-window-pixels', params.tissue_hierarchy_max_window_pixels ?: 16777216)
        addArg(command, '--hierarchy-fit-limit', params.tissue_hierarchy_fit_limit ?: 5000)
        addArg(command, '--hierarchy-components-per-block', params.tissue_hierarchy_components_per_block ?: 64)
        // Resolve relative paths in the launch directory before ProcessBuilder
        // changes its working directory to the repository. Never download models.
        def snapshot = params.tissue_hierarchy_model_snapshot == null ? '' : params.tissue_hierarchy_model_snapshot.toString().trim()
        if (snapshot) addArg(command, '--hierarchy-model-snapshot', new File(snapshot).canonicalPath)
        addArg(command, '--hierarchy-weights-filename', params.tissue_hierarchy_weights_filename ?: '')
        def measured = params.cell_measured_assays == null ? '' : params.cell_measured_assays.toString().trim()
        if (measured) addArg(command, '--cell-measured-assays', new File(measured).canonicalPath)
        addArg(command, '--uni2-save-tiles', truthy(params.uni2_save_tiles, false))
        addArg(command, '--mode', mode)
        addArg(command, '--min-free-gib', params.storage_min_free_gib ?: 20.0)
        addArg(command, '--safety-factor', params.storage_safety_factor ?: 1.25)
        addArg(command, '--restart-duplication-factor', params.storage_restart_duplication_factor ?: 1.0)
        addArg(command, '--source-expansion-factor', params.storage_source_expansion_factor ?: 8.0)

        addCache(command, 'stardist', params.stardist_keras_home)
        addCache(command, 'grandqc', params.grandqc_cache_dir)
        addCache(command, 'hf', params.hf_home)
        addCache(command, 'titan', params.titan_cache_dir)
        addCache(command, 'pathofmpred', params.pathofmpred_library_dir)
        def profiles = (profileValue ?: '').toString().split(',').collect { it.trim().toLowerCase() }
        if (profiles.contains('singularity')) {
            addCache(
                command,
                'singularity',
                params.singularity_cache_dir ?: new File(baseDir, '.apptainer_cache').absolutePath
            )
        }

        def process = new ProcessBuilder(command.collect { it.toString() })
            .directory(baseDir)
            .redirectErrorStream(true)
            .start()
        def processOutput = process.inputStream.getText('UTF-8')
        def exitCode = process.waitFor()
        processOutput.readLines().each { line -> println line }
        if (exitCode != 0) {
            throw new IllegalStateException(
                "Storage preflight rejected the run (exit ${exitCode}). " +
                "Review ${output}; use --storage_preflight_mode warn only after accepting the capacity risk."
            )
        }
        if (!output.isFile()) {
            throw new IllegalStateException("Storage preflight did not write ${output}")
        }
        params._storage_preflight_file = output.absolutePath
        def payload = new JsonSlurper().parse(output)
        payload instanceof Map ? payload as Map : [status: 'unknown']
    }
}
