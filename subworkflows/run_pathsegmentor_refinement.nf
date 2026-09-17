include { PATHSEGMENTOR_GUIDED_REFINE } from '../modules/pathsegmentor_guided_refine'

workflow RUN_PATHSEGMENTOR_REFINEMENT {
    take:
    pathsegmentor_bundle_ch
    baseline_mask_ch
    tissue_mask_variant_ch
    cluster_variant_defs

    main:
    def variants = cluster_variant_defs as List
    def pathsegmentorVariantCh = pathsegmentor_bundle_ch.flatMap { sample_id, probabilities, manifest ->
        variants.collect { spec -> tuple("${sample_id}::${spec.variant}", probabilities, manifest) }
    }
    def inputCh = baseline_mask_ch
        .join(tissue_mask_variant_ch, failOnMismatch: true, failOnDuplicate: true)
        .join(pathsegmentorVariantCh, failOnMismatch: true, failOnDuplicate: true)
        .map { key, id, variant, baseline, tissue, probabilities, manifest ->
            tuple(key, id, variant, baseline, tissue, probabilities, manifest)
        }
    PATHSEGMENTOR_GUIDED_REFINE(inputCh)

    emit:
    refined_mask = PATHSEGMENTOR_GUIDED_REFINE.out.refined_mask
    change_mask = PATHSEGMENTOR_GUIDED_REFINE.out.change_mask
    provenance = PATHSEGMENTOR_GUIDED_REFINE.out.provenance
}
