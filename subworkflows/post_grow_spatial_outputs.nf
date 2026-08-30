include { REFINE_GROWN_TISSUE_MEDSAM } from '../modules/refine_grown_tissue_medsam'
include { MASK_TO_GEOJSON } from '../modules/mask_to_geojson'

workflow POST_GROW_SPATIAL_OUTPUTS {
    take:
    image_input_ch
    grown_mask_ch
    image_for_growth_variant_ch
    cluster_mask_ch
    cluster_kodama_png_ch
    resolution_for_growth_variant_ch
    run_medsam_refine
    run_cluster_geojson

    main:
    def runMedsamRefine = run_medsam_refine as boolean
    def runClusterGeoJSON = run_cluster_geojson as boolean
    def grownRefineMethod = (params.grown_tissue_refine_method ?: 'medsam_border_refine').toString().trim().toLowerCase()

    def clusterVariants = [(params.cluster_primary_variant ?: 'standard').toString().trim() ?: 'standard']
    def secondaryVariant = (params.cluster_secondary_variant ?: '').toString().trim()
    if (secondaryVariant && !(secondaryVariant.toLowerCase() in ['none', 'false', 'off', '0'])) {
        if (!clusterVariants.contains(secondaryVariant)) clusterVariants << secondaryVariant
    }

    def refinedMaskCh = grown_mask_ch
    if (runMedsamRefine && grownRefineMethod == 'medsam_border_refine') {
        def clusterMaskFileCh = cluster_mask_ch.map { sample_key, sample_id, cluster_variant, cluster_mask_tif ->
            tuple(sample_key, cluster_mask_tif)
        }
        def clusterKodamaPngFileCh = cluster_kodama_png_ch.map { sample_key, sample_id, cluster_variant, membership_png ->
            tuple(sample_key, membership_png)
        }
        def refineInputCh = grown_mask_ch
            .join(image_for_growth_variant_ch)
            .join(clusterMaskFileCh)
            .join(clusterKodamaPngFileCh)
            .join(resolution_for_growth_variant_ch)
            .map { sample_key, sample_id, cluster_variant, grown_mask_tif, image_tif, cluster_mask_tif, kodama_membership_png, resolution_json ->
                tuple(sample_key, sample_id, cluster_variant, image_tif, cluster_mask_tif, grown_mask_tif, kodama_membership_png, resolution_json)
            }
        REFINE_GROWN_TISSUE_MEDSAM(refineInputCh)
        refinedMaskCh = REFINE_GROWN_TISSUE_MEDSAM.out.refined_mask
    } else if (runMedsamRefine && !(grownRefineMethod in ['none', ''])) {
        error "Unsupported grown_tissue_refine_method: ${grownRefineMethod}"
    } else if (runClusterGeoJSON && grownRefineMethod == 'medsam_border_refine') {
        refinedMaskCh = image_input_ch.flatMap { sample_id, _image_input ->
            clusterVariants.collect { clusterVariant ->
                def sampleKey = "${sample_id}::${clusterVariant}"
                tuple(sampleKey, sample_id, clusterVariant, file("${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/${sample_id}_${clusterVariant}_grown_mask_refined.ome.tif", checkIfExists: true))
            }
        }
    } else if (runClusterGeoJSON && !(grownRefineMethod in ['none', ''])) {
        error "Unsupported grown_tissue_refine_method: ${grownRefineMethod}"
    }

    if (runClusterGeoJSON) MASK_TO_GEOJSON(refinedMaskCh)

    emit:
    refined_masks = runMedsamRefine && grownRefineMethod == 'medsam_border_refine' ? REFINE_GROWN_TISSUE_MEDSAM.out.refined_mask : Channel.empty()
    cluster_geojson = runClusterGeoJSON ? MASK_TO_GEOJSON.out.cluster_geojson : Channel.empty()
}
