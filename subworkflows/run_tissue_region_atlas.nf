include { RUN_TISSUE_HIERARCHY } from './run_tissue_hierarchy'
include { MAP_REGION_REFERENCE_ATLAS } from '../modules/map_region_reference_atlas'

workflow RUN_TISSUE_REGION_ATLAS {
    take:
    image_input_ch
    crop_ch
    grid_objects_ch
    grid_metadata_ch
    tissue_ch
    shift_ch
    resolution_ch
    final_domains_ch
    final_uncertainty_ch
    have_final_domains
    runtime_plan

    main:
    def variant = (params.tissue_hierarchy_parent_variant ?: params.cluster_primary_variant).toString()
    def parents = have_final_domains ? final_domains_ch : image_input_ch.map { id, _image ->
        tuple("${id}::${variant}", id, variant,
            file("${params.outdir_base}/14_medsam_refine_tissue/${id}/${id}_${variant}_grown_mask_refined.ome.tif", checkIfExists: true))
    }
    def uncertainty = have_final_domains ? final_uncertainty_ch : image_input_ch.map { id, _image ->
        tuple("${id}::${variant}", id, variant,
            file("${params.outdir_base}/14_medsam_refine_tissue/${id}/${id}_${variant}_grown_mask_refined_uncertainty.tif", checkIfExists: true))
    }
    RUN_TISSUE_HIERARCHY(crop_ch, grid_objects_ch, grid_metadata_ch, tissue_ch,
        shift_ch, resolution_ch, parents, uncertainty, true, runtime_plan)
    if (params.region_reference_atlas) {
        MAP_REGION_REFERENCE_ATLAS(RUN_TISSUE_HIERARCHY.out.region_profiles.map { key, id, v, profiles ->
            def reference = file(params.region_reference_atlas, checkIfExists: true)
            tuple(key, id, v, profiles, reference, PipelineHelpers.atlasTaskFingerprint('region_reference_mapping', projectDir, [profiles, reference]))
        }, runtime_plan)
    }

    emit:
    hierarchy = RUN_TISSUE_HIERARCHY.out.hierarchy
    region_profiles = RUN_TISSUE_HIERARCHY.out.region_profiles
    reference_assignments = params.region_reference_atlas ? MAP_REGION_REFERENCE_ATLAS.out.assignments : Channel.empty()
    reference_mapping_bundles = params.region_reference_atlas ? MAP_REGION_REFERENCE_ATLAS.out.mapping_bundle : Channel.empty()
}
