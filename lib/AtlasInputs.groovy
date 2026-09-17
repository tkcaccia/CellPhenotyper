import groovy.json.JsonSlurper

/** Read-only declarations for independent measured and reference products. */
class AtlasInputs {
    static List mappingBundle(def raw, File base, String field) {
        if (!(raw instanceof String) || !raw.trim())
            throw new IllegalArgumentException("${field} must be a mapping receipt path")
        def receipt = new File(raw)
        if (!receipt.isAbsolute()) receipt = new File(base, raw)
        receipt = receipt.canonicalFile
        if (!(receipt.name ==~ /[A-Za-z0-9_.-]+\.mapping\.json/))
            throw new IllegalArgumentException("${field} must name a safe *.mapping.json receipt")
        def stem = receipt.name.replaceFirst(/\.mapping\.json$/, '')
        def files = [new File(receipt.parentFile, "${stem}.csv"),
                     new File(receipt.parentFile, "${stem}.atlas.json"), receipt]
        if (files.any { !it.isFile() })
            throw new IllegalArgumentException("${field} requires CSV, frozen atlas manifest and completion receipt siblings")
        // This only resolves files for staging. Python independently verifies
        // bytes, exact source-profile identity, complete UIDs and atlas status.
        files.collect { it.toString() }
    }

    static List resolveDirectories(def values, File base, String field) {
        if (values == null) return []
        if (!(values instanceof List) || values.any { !(it instanceof String) || !it.trim() })
            throw new IllegalArgumentException("${field} must be a list of directory paths")
        def result = values.collect { raw ->
            def path = new File(raw)
            if (!path.isAbsolute()) path = new File(base, raw)
            path = path.canonicalFile
            if (!path.isDirectory()) throw new IllegalArgumentException("${field} directory does not exist: ${path}")
            path.toString()
        }
        if (result.unique(false).size() != result.size())
            throw new IllegalArgumentException("${field} contains duplicate directories")
        result
    }

    static Map measuredRecord(Map row, File base) {
        def packages = resolveDirectories(row.measured_assays, base, 'measured_assays')
        def shapes = resolveDirectories(row.measured_region_shapes, base, 'measured_region_shapes')
        if (shapes && !packages) throw new IllegalArgumentException('Measured region shapes require measured assay packages')
        packages.each { directory ->
            if (!new File(directory, 'measured_assay_manifest.json').isFile())
                throw new IllegalArgumentException("Missing measured_assay_manifest.json in ${directory}")
        }
        shapes.each { directory ->
            if (!new File(directory, 'link.json').isFile())
                throw new IllegalArgumentException("Measured region-shape bundles require link.json: ${directory}")
        }
        [packages: packages, shapes: shapes]
    }

    static Map measuredBySample(def manifestPath) {
        if (!manifestPath) return [:]
        def source = new File(manifestPath.toString()).canonicalFile
        def rows = new JsonSlurper().parse(source)
        if (!(rows instanceof List) || !rows)
            throw new IllegalArgumentException('cell_measured_assays must name a nonempty JSON list')
        def result = [:]
        rows.each { row ->
            def id = row instanceof Map ? row.sample_id : null
            if (!(id instanceof String) || id in ['.', '..'] || !(id ==~ /[A-Za-z0-9_.-]+/) || result.containsKey(id))
                throw new IllegalArgumentException('Measured-assay sample IDs must be unique, safe path segments')
            result[id] = measuredRecord(row, source.parentFile)
            if (!result[id].packages) throw new IllegalArgumentException("Sample ${id} has no measured assay packages")
        }
        result
    }
}
