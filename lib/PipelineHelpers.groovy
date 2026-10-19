class PipelineHelpers {
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
