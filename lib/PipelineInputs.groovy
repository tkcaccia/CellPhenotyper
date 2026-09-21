class PipelineInputs {
    static final List<String> STAGES = [
        'convert', 'grandqc', 'stardist', 'cell_consensus', 'tma', 'tissue_mask',
        'pathsegmentor', 'cell_assignment', 'cytoplasm', 'gigatime',
        'marker_quantification', 'grid_tiles', 'uni2', 'kodama', 'clustering',
        'cluster_mask', 'grow_tissue', 'medsam_refine', 'cluster_geojson',
        'neoplastic_section', 'titan', 'pathofmpred'
    ]
    static final Map<String, String> STAGE_ALIASES = [
        image_conversion: 'convert', prepare_input: 'convert', qc: 'grandqc',
        artifact: 'grandqc', artifacts: 'grandqc', startdist: 'stardist',
        hovernet: 'cell_consensus', hover_net: 'cell_consensus',
        hovernet_monusac: 'cell_consensus', cellvit: 'cell_consensus',
        cellvitpp: 'cell_consensus', 'cellvit++': 'cell_consensus',
        consensus: 'cell_consensus', tma_spots: 'tma',
        tissue_microarray: 'tma', virtual_mif: 'gigatime', mif: 'gigatime',
        mask_tissue: 'tissue_mask', tissue_geojson: 'tissue_mask',
        pathsegmentor_semantics: 'pathsegmentor', path_segmentor: 'pathsegmentor',
        geojson: 'cluster_geojson', assign: 'cell_assignment',
        quantification: 'marker_quantification', marker_intensity: 'marker_quantification',
        gigatime_quantification: 'marker_quantification', grid: 'grid_tiles',
        spatial_grid: 'grid_tiles', uni2_grid: 'grid_tiles', uni_2: 'uni2',
        'uni-2': 'uni2', embeddings: 'uni2', rcode_clustering: 'clustering',
        labels_to_cluster_mask: 'cluster_mask', grow_to_tissue: 'grow_tissue',
        medsam: 'medsam_refine', medsam_refine_tissue: 'medsam_refine',
        refine_tissue: 'medsam_refine', mask_to_geojson: 'cluster_geojson',
        final_geojson: 'cluster_geojson', tumor_section: 'neoplastic_section',
        tumour_section: 'neoplastic_section', titan_embedding: 'titan',
        pathofm: 'pathofmpred'
    ]
    static final List<Map<String, Object>> IMAGE_SUFFIXES = [
        [suffix: '.ome.tif', priority: 80], [suffix: '.ome.tiff', priority: 75],
        [suffix: '.btf', priority: 70], [suffix: '.czi', priority: 69],
        [suffix: '.vsi', priority: 68], [suffix: '.svs', priority: 67], [suffix: '.ndpi', priority: 66],
        [suffix: '.scn', priority: 65], [suffix: '.mrxs', priority: 64],
        [suffix: '.vms', priority: 63], [suffix: '.vmu', priority: 62],
        [suffix: '.tif', priority: 60], [suffix: '.tiff', priority: 55],
        [suffix: '.png', priority: 50], [suffix: '.jpg', priority: 45],
        [suffix: '.jpeg', priority: 40]
    ]

    static String normalizeStage(def raw, String fallback) {
        def key = (raw ?: fallback).toString().trim().toLowerCase()
        if (key == 'auto') key = fallback
        key = STAGE_ALIASES.getOrDefault(key, key)
        if (!STAGES.contains(key))
            throw new IllegalArgumentException("Invalid stage '${raw}'. Allowed stages: ${STAGES.join(', ')}")
        key
    }

    static String imageSuffix(String fileName) {
        def lower = (fileName ?: '').toLowerCase()
        def hit = IMAGE_SUFFIXES.find { lower.endsWith(it.suffix as String) }
        hit?.suffix ?: ''
    }

    static int imageSuffixPriority(String suffix) {
        def hit = IMAGE_SUFFIXES.find { it.suffix == suffix }
        (hit?.priority ?: 0) as int
    }

    static String sampleId(def imageFile) {
        def fileObj = imageFile instanceof File ? imageFile : new File(imageFile.toString())
        def suffix = imageSuffix(fileObj.name)
        if (suffix) return fileObj.name.substring(0, fileObj.name.length() - suffix.length())
        def dot = fileObj.name.lastIndexOf('.')
        dot > 0 ? fileObj.name.substring(0, dot) : fileObj.name
    }

    static File inputSupport(def imageFile, def baseDir) {
        def image = imageFile instanceof File ? imageFile : new File(imageFile.toString())
        if (imageSuffix(image.name) != '.vsi')
            return new File(baseDir.toString(), 'resources/no_input_companion.txt')
        def companion = new File(image.parentFile, "_${sampleId(image)}_")
        if (!companion.isDirectory())
            throw new IllegalArgumentException(
                "Olympus VSI companion directory is missing for ${image.name}. Expected: ${companion}"
            )
        companion
    }

    static String cziRegionLabel(String rawName) {
        def matcher = ((rawName ?: '') =~ /(?i)(ScanRegion\d+)/)
        matcher.find() ? matcher.group(1) : ''
    }

    static int cziRegionOrder(String rawName) {
        def matcher = ((rawName ?: '') =~ /(?i)ScanRegion(\d+)/)
        if (!matcher.find()) return Integer.MAX_VALUE
        try { return matcher.group(1).toInteger() } catch (Throwable ignored) { return Integer.MAX_VALUE }
    }

    static String buildSampleId(def imageFile, String regionLabel = '') {
        def baseId = sampleId(imageFile)
        def rawId = regionLabel ? "${baseId}__${regionLabel}" : baseId
        def safeId = rawId.replaceAll(/[^A-Za-z0-9._-]+/, '_').replaceAll(/^_+|_+$/, '')
        safeId ?: 'sample'
    }

    /**
     * Resolve folder or single-image inputs without constructing Nextflow values.
     * Keeping filesystem discovery outside main.nf avoids Groovy's 65,535-character
     * class-constant limit on large DSL2 workflow bodies.
     */
    static List<Map<String, Object>> resolveSamples(
        def folderInput,
        def imageInput,
        def roiGeojson,
        def requireMatchingRoi,
        def baseDir
    ) {
        def folder = (folderInput ?: '').toString().trim()
        def imagePath = (imageInput ?: '').toString().trim()
        def roiPath = (roiGeojson ?: '').toString().trim()
        def requireRoi = ((requireMatchingRoi ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on']
        def sampleRows = []

        if (folder) {
            if (imagePath || roiPath)
                println 'WARN: --folder_input is set; --image_input/--roi_geojson are ignored.'
            def inputDir = new File(folder).canonicalFile
            if (!inputDir.exists())
                throw new IllegalArgumentException("--folder_input does not exist: ${inputDir}")
            if (!inputDir.isDirectory())
                throw new IllegalArgumentException("--folder_input must be a directory. Got: ${inputDir}")

            def candidates = []
            inputDir.eachFile(groovy.io.FileType.FILES) { File candidate ->
                def suffix = imageSuffix(candidate.name)
                if (suffix) candidates << [file: candidate, suffix: suffix]
            }
            if (!candidates)
                throw new IllegalArgumentException("No supported images found in --folder_input ${inputDir}.")

            def sampleMap = [:]
            candidates.each { candidate ->
                def imageFile = candidate.file as File
                def id = sampleId(imageFile)
                def priority = imageSuffixPriority(candidate.suffix as String)
                def previous = sampleMap[id]
                if (previous == null || priority > previous.priority)
                    sampleMap[id] = [file: imageFile, priority: priority]
            }

            sampleMap.keySet().sort().each { id ->
                def imageFile = sampleMap[id].file as File
                def suffix = imageSuffix(imageFile.name)
                def support = inputSupport(imageFile, baseDir)
                if (suffix == '.czi') {
                    def cziGeojsons = []
                    inputDir.eachFile(groovy.io.FileType.FILES) { File roiFile ->
                        if (roiFile.name.toLowerCase().endsWith('.geojson') && roiFile.name.startsWith("${imageFile.name} - ")) {
                            def region = cziRegionLabel(roiFile.name)
                            if (region)
                                cziGeojsons << [file: roiFile, region: region, order: cziRegionOrder(roiFile.name)]
                        }
                    }
                    if (cziGeojsons) {
                        cziGeojsons
                            .sort { a, b -> (a.order as int) <=> (b.order as int) ?: (a.file.name as String) <=> (b.file.name as String) }
                            .each { entry ->
                                def roiFile = entry.file as File
                                def region = entry.region as String
                                sampleRows << [sampleId: buildSampleId(imageFile, region), image: imageFile,
                                               inputRegion: region, roiName: roiFile.name,
                                               roiBase64: roiFile.bytes.encodeBase64().toString(), inputSupport: support]
                            }
                        return
                    }
                    if (requireRoi) {
                        println "WARN: Skipping CZI input ${imageFile.name} because no region-specific ScanRegion GeoJSON files were found and --require_matching_roi is enabled."
                        return
                    }
                    println "WARN: No region-specific ScanRegion GeoJSON files found for CZI input ${imageFile.name}. The pipeline will treat it as a single sample."
                }

                def roiCandidate = new File(inputDir, "${id}.geojson")
                if (!roiCandidate.exists() && requireRoi) {
                    println "WARN: Skipping image ${imageFile.name} because matching ROI GeoJSON ${id}.geojson was not found and --require_matching_roi is enabled."
                    return
                }
                sampleRows << [sampleId: buildSampleId(imageFile), image: imageFile, inputRegion: '',
                               roiName: roiCandidate.exists() ? roiCandidate.name : '',
                               roiBase64: roiCandidate.exists() ? roiCandidate.bytes.encodeBase64().toString() : '',
                               inputSupport: support]
            }
        } else {
            if (!imagePath)
                throw new IllegalArgumentException('Set either --folder_input (directory with images) or --image_input (single image file).')
            def imageFile = new File(imagePath).canonicalFile
            if (!imageFile.exists())
                throw new IllegalArgumentException("--image_input does not exist: ${imageFile}")
            def suffix = imageSuffix(imageFile.name)
            if (!suffix)
                throw new IllegalArgumentException("Unsupported image extension for --image_input '${imageFile.name}'. Supported extensions: ${IMAGE_SUFFIXES.collect { it.suffix }.join(', ')}")
            def baseSampleId = sampleId(imageFile)
            def roiName = ''
            def roiBase64 = ''
            def inputRegion = ''
            if (roiPath) {
                def roiFile = new File(roiPath).canonicalFile
                if (!roiFile.exists())
                    throw new IllegalArgumentException("--roi_geojson does not exist: ${roiFile}")
                roiName = roiFile.name
                roiBase64 = roiFile.bytes.encodeBase64().toString()
                if (suffix == '.czi') {
                    inputRegion = cziRegionLabel(roiFile.name)
                    if (!inputRegion)
                        println "WARN: --roi_geojson ${roiFile.name} does not contain a ScanRegion selector for CZI input ${imageFile.name}."
                }
            } else {
                def roiCandidate = new File(imageFile.parentFile, "${baseSampleId}.geojson")
                if (roiCandidate.exists()) {
                    roiName = roiCandidate.name
                    roiBase64 = roiCandidate.bytes.encodeBase64().toString()
                    if (suffix == '.czi') inputRegion = cziRegionLabel(roiCandidate.name)
                } else if (requireRoi) {
                    throw new IllegalArgumentException("Matching ROI GeoJSON not found for ${imageFile.name} and --require_matching_roi is enabled. Expected: ${roiCandidate}")
                }
            }
            sampleRows << [sampleId: buildSampleId(imageFile, inputRegion), image: imageFile,
                           inputRegion: inputRegion, roiName: roiName, roiBase64: roiBase64,
                           inputSupport: inputSupport(imageFile, baseDir)]
        }

        if (!sampleRows)
            throw new IllegalArgumentException('No input samples were resolved.')
        sampleRows
    }
}
