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
}
