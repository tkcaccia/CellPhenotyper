class PipelineHelpers {
    static void validateCohortOptions(def parameters) {
        validateCohortBundleOptions(parameters)
        if (parameters.cohort_niches_enable && !parameters.cell_profiles_enable)
            throw new IllegalArgumentException('cohort_niches_enable requires cell_profiles_enable=true')
        if (parameters.cohort_niches_bundle && !parameters.cell_profiles_enable)
            throw new IllegalArgumentException('cohort_niches_bundle requires cell_profiles_enable=true')
    }

    static void validateCohortBundleOptions(def parameters) {
        if (parameters.cohort_niches_bundle && parameters.cohort_niches_enable)
            throw new IllegalArgumentException('Choose cohort_niches_enable or cohort_niches_bundle, not both.')
        if (parameters.cohort_niches_bundle && !parameters.cell_profiles_spatialdata)
            throw new IllegalArgumentException('cohort_niches_bundle requires cell_profiles_spatialdata=true; it must not be silently ignored.')
    }

    /** Bind external executable code before Nextflow considers a cached task.
     * A fingerprint computed only in a process script closure is too late.
     * Keep this scoped dependency declaration aligned with the atlas modules.
     */
    static String atlasTaskFingerprint(String stage, def projectDirectory, List dataPaths) {
        def dependencies = [
            cell_reference_mapping: ['cell_reference_atlas.py', 'cell_profile_io.py', 'reference_mapping_io.py'],
            region_reference_mapping: ['cell_reference_atlas.py', 'cell_profile_io.py', 'reference_mapping_io.py'],
            spatialdata_export: ['export_spatialdata.py', 'integrate_measured_assay.py', 'cell_profile_io.py',
                'profile_cell_morphology.py', 'link_cell_tissue_hierarchy.py', 'reference_mapping_io.py', 'cohort_niche_io.py']
        ]
        if (!dependencies.containsKey(stage))
            throw new IllegalArgumentException("Unknown atlas fingerprint stage: ${stage}")
        def directory = projectDirectory instanceof java.nio.file.Path ? projectDirectory :
            java.nio.file.Paths.get(projectDirectory.toString())
        directory = directory.toRealPath().toRealPath()
        def codePaths = dependencies[stage].collect { directory.resolve('bin').resolve(it) }
        codePaths.add(directory.resolve('lib/PipelineHelpers.groovy'))
        def code = contentFingerprint(codePaths)
        def data = contentFingerprint(dataPaths)
        def digest = java.security.MessageDigest.getInstance('SHA-256')
        digest.update("cellphenotyper-atlas-task-v1\u0000${stage}\u0000${code}\u0000${data}".getBytes('UTF-8'))
        digest.digest().encodeHex().toString()
    }

    /** Content cache key for directory inputs, whose children are not reliably
     * traversed by Nextflow's built-in deep cache. Uses public Path APIs; a
     * process TaskPath.toRealPath() resolves its source before task staging.
     * No absolute source paths, file metadata or whole-file buffers are hashed.
     */
    static String contentFingerprint(List paths, boolean allowExternalFiles = false) {
        def digest = java.security.MessageDigest.getInstance('SHA-256')
        digest.update('cellphenotyper-content-fingerprint-v1\u0000'.getBytes('UTF-8'))
        byte[] buffer = new byte[1024 * 1024]
        paths.flatten().eachWithIndex { value, index ->
            if (!(value instanceof java.nio.file.Path))
                throw new IllegalArgumentException('Content fingerprints require file/directory Paths')
            // The first call unwraps Nextflow TaskPath; a second normal Path
            // call resolves a caller-provided root symlink canonically.
            def root = value.toRealPath().toRealPath()
            digest.update("input:${index}\u0000".getBytes('UTF-8'))
            hashContentPath(root, root, '.', [] as Set, digest, buffer, allowExternalFiles)
        }
        digest.digest().encodeHex().toString()
    }

    private static void hashContentPath(java.nio.file.Path root, java.nio.file.Path path,
                                        String relative, Set ancestors, def digest, byte[] buffer, boolean allowExternalFiles) {
        def resolved = path.toRealPath()
        if (!resolved.startsWith(root) && !(allowExternalFiles && java.nio.file.Files.isRegularFile(resolved)))
            throw new IllegalArgumentException("Content fingerprint symlink escapes input directory: ${relative}")
        if (java.nio.file.Files.isDirectory(resolved)) {
            if (ancestors.contains(resolved))
                throw new IllegalArgumentException("Content fingerprint directory cycle: ${relative}")
            digest.update("directory:${relative}\u0000".getBytes('UTF-8'))
            def children = []
            java.nio.file.Files.newDirectoryStream(resolved).withCloseable { stream ->
                stream.each { child -> children.add(child) }
            }
            def nextAncestors = ancestors + [resolved]
            children.sort { it.fileName.toString() }.each { child ->
                hashContentPath(root, child, "${relative}/${child.fileName}", nextAncestors, digest, buffer, allowExternalFiles)
            }
        } else if (java.nio.file.Files.isRegularFile(resolved)) {
            digest.update("file:${relative}\u0000".getBytes('UTF-8'))
            def content = java.security.MessageDigest.getInstance('SHA-256')
            java.nio.file.Files.newInputStream(resolved).withCloseable { input ->
                int count
                while ((count = input.read(buffer)) != -1) {
                    if (count > 0) content.update(buffer, 0, count)
                }
            }
            digest.update(content.digest())
        } else {
            throw new IllegalArgumentException("Unsupported content fingerprint artifact: ${relative}")
        }
    }

    static String codeFingerprint(List paths) {
        def digest = java.security.MessageDigest.getInstance('SHA-256')
        paths.each { digest.update(new File(it.toString()).bytes) }
        digest.digest().encodeHex().toString()
    }

    static List<String> resolveKodamaModes(def rawValue) {
        def raw = (rawValue == null ? '' : rawValue.toString()).trim().toLowerCase()
        if (!raw || raw in ['all', 'default']) return ['tile', 'inner_square']
        if (raw in ['full', 'all4', 'all_four', 'full_stack']) return ['tile', 'nuclei', 'cyto', 'inner_square']
        def aliases = [
            tile: 'tile', full: 'tile', full_tile: 'tile', 'full-tile': 'tile',
            nuclei: 'nuclei', nucleus: 'nuclei', nuclear: 'nuclei', label: 'nuclei', labels: 'nuclei',
            cyto: 'cyto', cytoplasm: 'cyto', inner: 'inner_square', inner_square: 'inner_square',
            'inner-square': 'inner_square', square: 'inner_square'
        ]
        def modes = raw.replace('+', ',').split(',').collect { it.trim() }.findAll { it }.collect { token ->
            if (!aliases.containsKey(token)) throw new IllegalArgumentException("Unknown KODAMA embedding mode token: ${token}")
            aliases[token]
        }.unique()
        if (!modes) throw new IllegalArgumentException('KODAMA embedding mode resolved to an empty set.')
        modes
    }
}
