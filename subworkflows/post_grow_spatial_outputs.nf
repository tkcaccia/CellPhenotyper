include { REFINE_GROWN_TISSUE_MEDSAM } from '../modules/refine_grown_tissue_medsam'
include { MASK_TO_GEOJSON } from '../modules/mask_to_geojson'
include { REFINE_WITH_BORA } from '../modules/refine_with_bora'

workflow POST_GROW_SPATIAL_OUTPUTS {
    take:
    image_input_ch
    spatial_baseline_mask_ch
    image_for_growth_variant_ch
    cluster_mask_ch
    cluster_uncertainty_ch
    tissue_mask_variant_ch
    cluster_kodama_png_ch
    resolution_for_growth_variant_ch
    run_medsam_refine
    run_cluster_geojson
    runtime_plan

    main:
    def runMedsamRefine = run_medsam_refine as boolean
    def runClusterGeoJSON = run_cluster_geojson as boolean
    def grownRefineMethod = (params.grown_tissue_refine_method ?: 'bora').toString().trim().toLowerCase()

    def clusterVariants = [(params.cluster_primary_variant ?: 'standard').toString().trim() ?: 'standard']
    def secondaryVariant = (params.cluster_secondary_variant ?: '').toString().trim()
    if (secondaryVariant && !(secondaryVariant.toLowerCase() in ['none', 'false', 'off', '0'])) {
        if (!clusterVariants.contains(secondaryVariant)) clusterVariants << secondaryVariant
    }

    def refinedMaskCh = spatial_baseline_mask_ch
    // Without refinement this is source clustering uncertainty, not a claim
    // that subsequent grown pixels were directly supported observations.
    def finalUncertaintyCh = cluster_uncertainty_ch
    def finalProvenanceCh = Channel.empty()
    def provenanceMetadataCh = Channel.empty()
    def clusterGeojsonCh = Channel.empty()
    def vectorProvenanceSummaryCh = Channel.empty()
    if (runMedsamRefine && grownRefineMethod in ['bora', 'medsam_border_refine']) {
        def clusterMaskFileCh = cluster_mask_ch.map { sample_key, sample_id, cluster_variant, cluster_mask_tif ->
            tuple(sample_key, cluster_mask_tif)
        }
        def clusterKodamaPngFileCh = cluster_kodama_png_ch.map { sample_key, sample_id, cluster_variant, membership_png ->
            tuple(sample_key, membership_png)
        }
        def uncertaintyFileCh = cluster_uncertainty_ch.map { sample_key, sample_id, cluster_variant, uncertainty_tif ->
            tuple(sample_key, uncertainty_tif)
        }
        def refineInputCh = spatial_baseline_mask_ch
            .join(image_for_growth_variant_ch)
            .join(clusterMaskFileCh)
            .join(tissue_mask_variant_ch)
            .join(clusterKodamaPngFileCh)
            .join(resolution_for_growth_variant_ch)
            .join(uncertaintyFileCh, failOnMismatch: true, failOnDuplicate: true)
            .map { sample_key, sample_id, cluster_variant, baseline_mask_tif, image_tif, cluster_mask_tif, tissue_mask_tif, kodama_membership_png, resolution_json, uncertainty_tif ->
                tuple(sample_key, sample_id, cluster_variant, image_tif, cluster_mask_tif, baseline_mask_tif, tissue_mask_tif, kodama_membership_png, resolution_json, uncertainty_tif)
            }
        if (grownRefineMethod == 'bora') {
            REFINE_WITH_BORA(refineInputCh, runtime_plan)
            refinedMaskCh = REFINE_WITH_BORA.out.refined_mask
            finalUncertaintyCh = REFINE_WITH_BORA.out.refined_uncertainty
            finalProvenanceCh = REFINE_WITH_BORA.out.refined_provenance
            provenanceMetadataCh = REFINE_WITH_BORA.out.provenance_metadata
            clusterGeojsonCh = runClusterGeoJSON ? REFINE_WITH_BORA.out.cluster_geojson : Channel.empty()
            vectorProvenanceSummaryCh = runClusterGeoJSON ? REFINE_WITH_BORA.out.provenance_summary : Channel.empty()
        } else {
            REFINE_GROWN_TISSUE_MEDSAM(refineInputCh, runtime_plan)
            refinedMaskCh = REFINE_GROWN_TISSUE_MEDSAM.out.refined_mask
            finalUncertaintyCh = REFINE_GROWN_TISSUE_MEDSAM.out.refined_uncertainty
            finalProvenanceCh = REFINE_GROWN_TISSUE_MEDSAM.out.refined_provenance
            provenanceMetadataCh = REFINE_GROWN_TISSUE_MEDSAM.out.provenance_metadata
        }
    } else if (runMedsamRefine && !(grownRefineMethod in ['none', ''])) {
        error "Unsupported grown_tissue_refine_method: ${grownRefineMethod}"
    } else if (runClusterGeoJSON && grownRefineMethod in ['bora', 'medsam_border_refine']) {
        refinedMaskCh = image_input_ch.flatMap { sample_id, _image_input ->
            clusterVariants.collect { clusterVariant ->
                def sampleKey = "${sample_id}::${clusterVariant}"
                tuple(sampleKey, sample_id, clusterVariant, file("${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/${sample_id}_${clusterVariant}_grown_mask_refined.ome.tif", checkIfExists: true))
            }
        }
        finalUncertaintyCh = refinedMaskCh.map { sample_key, sample_id, variant, mask ->
            tuple(sample_key, sample_id, variant, file("${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/${sample_id}_${variant}_grown_mask_refined_uncertainty.tif", checkIfExists: true))
        }
        finalProvenanceCh = refinedMaskCh.map { sample_key, sample_id, variant, mask ->
            tuple(sample_key, sample_id, variant, file("${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/${sample_id}_${variant}_grown_mask_refined_provenance.tif", checkIfExists: true))
        }
        provenanceMetadataCh = refinedMaskCh.map { sample_key, sample_id, variant, mask ->
            tuple(sample_key, sample_id, variant, file("${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/${sample_id}_${variant}_grown_mask_refined.ome.tif.provenance.json", checkIfExists: true))
        }
        if (grownRefineMethod == 'bora') {
            clusterGeojsonCh = refinedMaskCh.map { sample_key, sample_id, variant, mask ->
                tuple(sample_key, sample_id, variant, file("${params.outdir_base}/15_cluster_geojson/${sample_id}/${sample_id}_${variant}_grown_mask_smooth_class.geojson", checkIfExists: true))
            }
            vectorProvenanceSummaryCh = refinedMaskCh.map { sample_key, sample_id, variant, mask ->
                tuple(sample_key, sample_id, variant, file("${params.outdir_base}/15_cluster_geojson/${sample_id}/${sample_id}_${variant}_grown_mask_smooth_class.geojson.provenance.json", checkIfExists: true))
            }
        }
    } else if (runClusterGeoJSON && !(grownRefineMethod in ['none', ''])) {
        error "Unsupported grown_tissue_refine_method: ${grownRefineMethod}"
    }

    if (runClusterGeoJSON && grownRefineMethod != 'bora') {
        def vectorInputCh
        if (grownRefineMethod == 'medsam_border_refine') {
            vectorInputCh = refinedMaskCh
                .join(finalUncertaintyCh, by: [0, 1, 2], failOnMismatch: true, failOnDuplicate: true)
                .join(finalProvenanceCh, by: [0, 1, 2], failOnMismatch: true, failOnDuplicate: true)
                .join(provenanceMetadataCh, by: [0, 1, 2], failOnMismatch: true, failOnDuplicate: true)
                .join(resolution_for_growth_variant_ch, by: 0, failOnMismatch: true, failOnDuplicate: true)
                .map { key, id, variant, mask, uncertainty, provenance, metadata, shift_json ->
                    tuple(key, id, variant, mask, uncertainty, provenance, metadata, shift_json, [uncertainty: true, refinement: true])
                }
        } else {
            // Optional source uncertainty can accompany an unrefined/grown mask.
            // Missing growth provenance remains unavailable; zero is not confidence.
            vectorInputCh = refinedMaskCh
                .join(finalUncertaintyCh, by: [0, 1, 2], remainder: true, failOnDuplicate: true)
                .join(resolution_for_growth_variant_ch, by: 0, failOnMismatch: true, failOnDuplicate: true)
                .map { key, id, variant, mask, uncertainty, shift_json ->
                    if (mask == null) error "Unmatched uncertainty sidecar for vectorization: ${key}"
                    tuple(key, id, variant, mask, uncertainty ?: mask, mask, mask, shift_json, [uncertainty: uncertainty != null, refinement: false])
                }
        }
        MASK_TO_GEOJSON(vectorInputCh)
        clusterGeojsonCh = MASK_TO_GEOJSON.out.cluster_geojson
        vectorProvenanceSummaryCh = MASK_TO_GEOJSON.out.provenance_summary
    }

    emit:
    final_masks = refinedMaskCh
    final_uncertainty = finalUncertaintyCh
    final_provenance = finalProvenanceCh
    provenance_metadata = provenanceMetadataCh
    refined_masks = runMedsamRefine && grownRefineMethod == 'bora' ? REFINE_WITH_BORA.out.refined_mask : (runMedsamRefine && grownRefineMethod == 'medsam_border_refine' ? REFINE_GROWN_TISSUE_MEDSAM.out.refined_mask : Channel.empty())
    cluster_geojson = clusterGeojsonCh
    vector_provenance_summary = vectorProvenanceSummaryCh
}
