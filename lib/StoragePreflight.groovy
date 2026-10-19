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
        addArg(command, '--outdir', outdir.absolutePath)
        addArg(command, '--workdir', workDirValue)
        addArg(command, '--output-json', output.absolutePath)
        addArg(command, '--start-point', params._resolved_start_point)
        addArg(command, '--end-point', params._resolved_end_point)
        addArg(command, '--publish-dir-mode', params.publish_dir_mode ?: 'copy')
        addArg(command, '--cell-detection-mode', params._resolved_cell_detection_mode)
        addArg(command, '--uni2-sampling-mode', params._resolved_uni2_sampling_mode)
        addArg(command, '--gigatime-enabled', truthy(params.gigatime_enable, true))
        addArg(command, '--marker-quantification-enabled', truthy(params.marker_quantification_enable, true))
        def outputChannels = (params.gigatime_output_channels ?: '').toString()
            .split(',').collect { it.trim() }.findAll { it }
        addArg(command, '--gigatime-channel-count', Math.max(1, outputChannels.size()))
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
