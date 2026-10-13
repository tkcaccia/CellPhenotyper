include { LABELS_TO_CLUSTER_MASK } from '../modules/labels_to_cluster_mask'
include { GRID_CLUSTERS_TO_MASK } from '../modules/grid_clusters_to_mask'

workflow BUILD_CLUSTER_MASK_OUTPUT {
    take:
    crop_roi_ch
    cluster_csv_ch
    labels_for_cluster_ch
    grid_objects_ch
    grid_metadata_ch
    uni2_grid_mode

    main:
    def clusterVariants = [(params.cluster_primary_variant ?: 'standard').toString().trim() ?: 'standard']
    def secondaryVariant = (params.cluster_secondary_variant ?: '').toString().trim()
    if (secondaryVariant && !(secondaryVariant.toLowerCase() in ['none', 'false', 'off', '0'])) {
        if (!clusterVariants.contains(secondaryVariant)) clusterVariants << secondaryVariant
    }
    def previewImageCh = crop_roi_ch
    def previewVariantCh = previewImageCh.flatMap { sample_id, preview_tif ->
        clusterVariants.collect { variant -> tuple("${sample_id}::${variant}", preview_tif) }
    }

    def resultCh
    def uncertaintyCh
    if (uni2_grid_mode as boolean) {
        def gridObjectsVariantCh = grid_objects_ch.flatMap { sample_id, grid_objects_csv ->
            clusterVariants.collect { variant -> tuple("${sample_id}::${variant}", grid_objects_csv) }
        }
        def gridMetadataVariantCh = grid_metadata_ch.flatMap { sample_id, grid_metadata_json ->
            clusterVariants.collect { variant -> tuple("${sample_id}::${variant}", grid_metadata_json) }
        }
        def inputCh = cluster_csv_ch
            .join(gridObjectsVariantCh)
            .join(gridMetadataVariantCh)
            .join(previewVariantCh)
            .map { sample_key, sample_id, variant, cluster_csv, objects_csv, metadata_json, preview_tif ->
                tuple(sample_key, sample_id, variant, objects_csv, metadata_json, cluster_csv, preview_tif)
            }
        GRID_CLUSTERS_TO_MASK(inputCh)
        resultCh = GRID_CLUSTERS_TO_MASK.out.cluster_mask
        uncertaintyCh = GRID_CLUSTERS_TO_MASK.out.uncertainty_mask
    } else {
        def labelsVariantCh = labels_for_cluster_ch.flatMap { sample_id, labels_tif ->
            clusterVariants.collect { variant -> tuple("${sample_id}::${variant}", labels_tif) }
        }
        def inputCh = cluster_csv_ch
            .join(labelsVariantCh)
            .join(previewVariantCh)
            .map { sample_key, sample_id, variant, cluster_csv, labels_tif, preview_tif ->
                tuple(sample_key, sample_id, variant, labels_tif, cluster_csv, preview_tif)
            }
        LABELS_TO_CLUSTER_MASK(inputCh)
        resultCh = LABELS_TO_CLUSTER_MASK.out.cluster_mask
        uncertaintyCh = LABELS_TO_CLUSTER_MASK.out.uncertainty_mask
    }

    emit:
    cluster_mask = resultCh
    uncertainty_mask = uncertaintyCh
}
