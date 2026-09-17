include { SUMMARIZE_PATHSEGMENTOR_SEMANTICS as SUMMARIZE_PATHSEGMENTOR_PRIMARY } from '../modules/summarize_pathsegmentor_semantics'
include { SUMMARIZE_PATHSEGMENTOR_SEMANTICS as SUMMARIZE_PATHSEGMENTOR_CELLS } from '../modules/summarize_pathsegmentor_semantics'

workflow ANNOTATE_PATHSEGMENTOR_EVIDENCE {
    take:
    pathsegmentor_bundle_ch
    primary_objects_ch
    cell_objects_ch
    cluster_csv_ch
    cluster_variant_defs
    grid_mode
    placeholder_csv

    main:
    def variants = cluster_variant_defs as List
    def gridMode = grid_mode as boolean
    def pathsegmentorVariantCh = pathsegmentor_bundle_ch.flatMap { sample_id, probabilities, manifest ->
        variants.collect { spec -> tuple("${sample_id}::${spec.variant}", probabilities, manifest) }
    }
    def observationsVariantCh = primary_objects_ch.flatMap { sample_id, observations ->
        variants.collect { spec -> tuple("${sample_id}::${spec.variant}", observations) }
    }
    def primaryRole = gridMode ? 'grid' : 'cell'
    def primaryInputCh = cluster_csv_ch
        .join(observationsVariantCh, failOnMismatch: true, failOnDuplicate: true)
        .join(pathsegmentorVariantCh, failOnMismatch: true, failOnDuplicate: true)
        .map { key, id, variant, clusters, observations, probabilities, manifest ->
            tuple(key, id, variant, primaryRole, observations, clusters, true, probabilities, manifest)
        }
    SUMMARIZE_PATHSEGMENTOR_PRIMARY(primaryInputCh)
    def annotationsCh = SUMMARIZE_PATHSEGMENTOR_PRIMARY.out.annotations
    if (gridMode) {
        def cellInputCh = cell_objects_ch
            .join(pathsegmentor_bundle_ch, failOnMismatch: true, failOnDuplicate: true)
            .map { id, observations, probabilities, manifest ->
                tuple("${id}::unclustered", id, 'unclustered', 'cell', observations,
                    placeholder_csv, false, probabilities, manifest)
            }
        SUMMARIZE_PATHSEGMENTOR_CELLS(cellInputCh)
        annotationsCh = annotationsCh.mix(SUMMARIZE_PATHSEGMENTOR_CELLS.out.annotations)
    }

    emit:
    annotations = annotationsCh
}
