include { BUILD_SPATIAL_CELL_PROFILES } from '../modules/build_spatial_cell_profiles'
include { RUN_AUXILIARY_CELL_UNI2 as RUN_CELL_PROFILE_UNI2 } from './run_auxiliary_cell_uni2'
include { EXPORT_SPATIALDATA } from '../modules/export_spatialdata'
include { MAP_CELL_REFERENCE_ATLAS } from '../modules/map_cell_reference_atlas'
include { LINK_CELL_TISSUE_HIERARCHY } from '../modules/link_cell_tissue_hierarchy'
include { FIT_COHORT_NICHES } from '../modules/fit_cohort_niches'

workflow RUN_CELL_PROFILE_ATLAS {
    take:
    image_input_ch
    crop_ch
    labels_ch
    objects_ch
    shift_ch
    resolution_ch
    tissue_ch
    domains_ch
    cluster_masks_ch
    domain_uncertainty_ch
    cluster_uncertainty_ch
    compartments_ch
    geojson_ch
    marker_inputs_ch
    cellvit_ch
    primary_tile_ch
    primary_local_ch
    auxiliary_tile_ch
    auxiliary_local_ch
    hierarchy_ch
    region_reference_ch
    flags
    runtime_plan

    main:
    PipelineHelpers.validateCohortBundleOptions(params)
    def placeholder = file("${projectDir}/resources/empty_embeddings_placeholder", checkIfExists: true)
    def measuredBySample = AtlasInputs.measuredBySample(params.cell_measured_assays)
    if (measuredBySample && !params.cell_profiles_spatialdata)
        error 'cell_measured_assays requires cell_profiles_spatialdata=true'
    if (measuredBySample) {
        image_input_ch.map { id, _image -> id.toString() }.collect().subscribe { ids ->
            def foreign = measuredBySample.keySet() - ids
            if (foreign) error "Measured assays refer to unknown pipeline samples: ${foreign}"
        }
    }
    def emptyCh = image_input_ch.map { sample_id, _image -> tuple(sample_id, placeholder) }
    def tileCh = emptyCh
    def localCh = emptyCh
    if (flags.uni2) {
        if (flags.grid && !flags.both) {
            // Extract cell features without a second KODAMA/clustering/refinement route.
            RUN_CELL_PROFILE_UNI2(image_input_ch, crop_ch, labels_ch, objects_ch, primary_tile_ch, resolution_ch, flags.run_uni2, false, runtime_plan)
            tileCh = RUN_CELL_PROFILE_UNI2.out.tile_embeddings.map { id, directory -> tuple(id.replaceFirst(/__cells$/, ''), directory) }
            localCh = RUN_CELL_PROFILE_UNI2.out.inner_embeddings.map { id, directory -> tuple(id.replaceFirst(/__cells$/, ''), directory) }
        } else if (flags.both) {
            tileCh = flags.run_uni2 ? auxiliary_tile_ch.map { id, directory -> tuple(id.replaceFirst(/__cells$/, ''), directory) }
                : image_input_ch.map { id, _image -> tuple(id, file("${params.outdir_base}/09_embeddings/${id}__cells/embeddings_${id}__cells_tile", checkIfExists: true)) }
            localCh = flags.run_uni2 ? auxiliary_local_ch.map { id, directory -> tuple(id.replaceFirst(/__cells$/, ''), directory) }
                : image_input_ch.map { id, _image -> tuple(id, file("${params.outdir_base}/09_embeddings/${id}__cells/embeddings_${id}__cells_inner_square", checkIfExists: true)) }
        } else {
            tileCh = flags.run_uni2 ? primary_tile_ch : image_input_ch.map { id, _image -> tuple(id, file("${params.outdir_base}/09_embeddings/${id}/embeddings_${id}_tile", checkIfExists: true)) }
            localCh = flags.run_uni2 ? primary_local_ch : image_input_ch.map { id, _image -> tuple(id, file("${params.outdir_base}/09_embeddings/${id}/embeddings_${id}_inner_square", checkIfExists: true)) }
        }
        tileCh = tileCh.ifEmpty { error 'Cell profiles requested UNI-2 context features but none were emitted. Resume the UNI-2 stage or explicitly disable cell_profiles_uni2_enable.' }
        localCh = localCh.ifEmpty { error 'Cell profiles requested UNI-2 local features but none were emitted. Enable paired local features or explicitly disable cell_profiles_uni2_enable.' }
    }
    def markerCh = emptyCh
    if (flags.markers) {
        markerCh = flags.run_gigatime ? marker_inputs_ch : (flags.run_marker_quantification
            ? marker_inputs_ch.map { id, compartment, csv, summary -> tuple(id, [csv, summary]) }.groupTuple(size: ((params.expand_um as double) >= 0 ? 3 : 2)).map { id, bundles -> tuple(id, bundles.flatten()) }
            : image_input_ch.map { id, _image -> tuple(id, file("${params.outdir_base}/05_gigatime/${id}/quantification_${id}", checkIfExists: true)) })
    }
    if (flags.cellvit && !flags.consensus) error 'CellViT features require the CellViT/consensus detection route'
    def detectorCh = flags.cellvit ? (flags.run_consensus ? cellvit_ch : image_input_ch.map { id, _image -> tuple(id, file("${params.outdir_base}/03c_cellvitpp/${id}/cellvit_${id}", checkIfExists: true)) }) : emptyCh
    def domainSource = params.cell_profiles_domain_source.toString()
    if (!(domainSource in ['refined', 'cluster', 'none'])) error 'cell_profiles_domain_source must be refined, cluster or none'
    def domainCh = emptyCh
    def uncertaintyCh = emptyCh.map { id, path -> tuple(id, path, false) }
    if (domainSource != 'none') {
        def current = domainSource == 'refined' ? domains_ch : cluster_masks_ch
        def haveCurrent = domainSource == 'refined' ? flags.have_final_domains : flags.run_cluster_mask
        domainCh = haveCurrent ? current.filter { key, id, variant, mask -> variant == flags.primary_variant }.map { key, id, variant, mask -> tuple(id, mask) }
            : image_input_ch.map { id, _image ->
                def relative = domainSource == 'refined' ? "14_medsam_refine_tissue/${id}/${id}_${flags.primary_variant}_grown_mask_refined.ome.tif" : "12_cluster_mask/${id}/${id}_${flags.primary_variant}_cluster_mask.tif"
                tuple(id, file("${params.outdir_base}/${relative}", checkIfExists: true))
            }
        def currentUncertainty = domainSource == 'refined' ? domain_uncertainty_ch : cluster_uncertainty_ch
        uncertaintyCh = haveCurrent ? currentUncertainty.filter { key, id, variant, mask -> variant == flags.primary_variant }.map { key, id, variant, mask -> tuple(id, mask, true) }
            : image_input_ch.map { id, _image ->
                def relative = domainSource == 'refined' ? "14_medsam_refine_tissue/${id}/${id}_${flags.primary_variant}_grown_mask_refined_uncertainty.tif" : "12_cluster_mask/${id}/${id}_${flags.primary_variant}_cluster_uncertainty_mask.tif"
                def candidate = file("${params.outdir_base}/${relative}")
                tuple(id, candidate.exists() ? candidate : placeholder, candidate.exists())
            }
    }
    def compartmentCh = emptyCh.map { id, path -> tuple(id, path, false) }
    if ((params.expand_um as double) >= 0) {
        compartmentCh = flags.run_cytoplasm ? compartments_ch.map { id, directory, kind -> tuple(id, directory, true) }
            : image_input_ch.map { id, _image ->
                def candidate = file("${params.outdir_base}/08_cytoplasm/${id}/${id}_labels_cyto_compartments")
                tuple(id, candidate.exists() ? candidate : placeholder, candidate.exists())
            }
    }
    def strictJoin = [failOnDuplicate: true, failOnMismatch: true]
    def inputCh = crop_ch.join(strictJoin, labels_ch).join(strictJoin, objects_ch).join(strictJoin, shift_ch).join(strictJoin, resolution_ch)
        .join(strictJoin, tissue_ch).join(strictJoin, markerCh).join(strictJoin, tileCh).join(strictJoin, localCh).join(strictJoin, detectorCh).join(strictJoin, domainCh)
        .join(strictJoin, compartmentCh).join(strictJoin, uncertaintyCh)
        .map { sample_id, image, labels, objects, shift, resolution, tissue, markers, context, local, cellvit, domains, compartments, haveCompartments, uncertainty, haveUncertainty ->
            tuple(sample_id, image, labels, objects, shift, resolution, tissue, markers, context, local, cellvit, domains, compartments, uncertainty,
                [markers: flags.markers, uni2: flags.uni2, cellvit: flags.cellvit, domains: domainSource != 'none', compartments: haveCompartments, uncertainty: haveUncertainty])
        }
        .ifEmpty { error 'Cell-profile input join emitted no samples; inspect canonical objects, feature and domain channels.' }
    BUILD_SPATIAL_CELL_PROFILES(inputCh, runtime_plan)
    def profileCh = BUILD_SPATIAL_CELL_PROFILES.out.profiles
    def hierarchyInputs = emptyCh.map { id, path -> tuple(id, path, false) }
    if (flags.hierarchy) {
        def hierarchyVariant = (params.tissue_hierarchy_parent_variant ?: flags.primary_variant).toString()
        if (domainSource != 'refined' || hierarchyVariant != flags.primary_variant.toString())
            error 'Linked cell hierarchy requires refined cell-profile domains and the same primary parent variant.'
        hierarchyInputs = hierarchy_ch.filter { key, id, variant, directory -> variant.toString() == hierarchyVariant }
            .map { key, id, variant, directory -> tuple(id, directory, true) }
            .ifEmpty { error 'Cell profiles requested hierarchy linkage but no matching hierarchy was emitted.' }
        def linkInputs = profileCh.join(strictJoin, labels_ch).join(strictJoin, hierarchyInputs).join(strictJoin, compartmentCh)
            .map { id, profiles, labels, hierarchy, haveHierarchy, compartments, haveCompartments ->
                tuple(id, profiles, labels, hierarchy, compartments, haveCompartments)
            }
        LINK_CELL_TISSUE_HIERARCHY(linkInputs, runtime_plan)
        profileCh = LINK_CELL_TISSUE_HIERARCHY.out.profiles
    }
    def cohortExportCh = emptyCh.map { id, path -> tuple(id, path, false) }
    def cohortBundleCh = Channel.empty()
    if (params.cohort_niches_enable) {
        def cohortInputs = profileCh.collect(flat: false).map { samples ->
            def ordered = samples.sort { a, b -> a[0].toString() <=> b[0].toString() }
            def ids = ordered.collect { it[0].toString() }
            if (ids.size() < 2 || ids.unique(false).size() != ids.size())
                error 'Cohort niche discovery requires at least two distinct specimens; empty or duplicate specimen inputs are invalid.'
            tuple(ids, ordered.collect { it[1] })
        }
        FIT_COHORT_NICHES(cohortInputs, runtime_plan)
        cohortBundleCh = FIT_COHORT_NICHES.out.cohort
    } else if (params.cohort_niches_bundle) {
        cohortBundleCh = Channel.value(file(params.cohort_niches_bundle, checkIfExists: true))
    }
    if (params.cohort_niches_enable || params.cohort_niches_bundle) {
        // A queue channel containing one bundle must not feed only one export.
        // Assert one global result, then broadcast against the final registry.
        def cohortBundleValue = cohortBundleCh.collect(flat: false).map { bundles ->
            if (bundles.size() != 1) error 'Cohort export requires exactly one completed global bundle.'
            bundles[0]
        }
        cohortExportCh = profileCh.map { id, _profiles -> id }.combine(cohortBundleValue)
            .map { id, bundle -> tuple(id, bundle, true) }
    }
    def cellReferenceCh = emptyCh.map { id, path -> tuple(id, [path], false) }
    if (params.cell_reference_atlas) {
        MAP_CELL_REFERENCE_ATLAS(profileCh.map { id, profiles ->
            def reference = file(params.cell_reference_atlas, checkIfExists: true)
            tuple(id, profiles, reference, PipelineHelpers.atlasTaskFingerprint('cell_reference_mapping', projectDir, [profiles, reference]))
        }, runtime_plan)
        cellReferenceCh = MAP_CELL_REFERENCE_ATLAS.out.mapping_bundle.map { id, files -> tuple(id, files, true) }
    }
    def regionReferenceCh = emptyCh.map { id, path -> tuple(id, [path], false) }
    if (params.region_reference_atlas) {
        if (!flags.hierarchy) error 'Region reference mapping export requires tissue hierarchy.'
        def hierarchyVariant = (params.tissue_hierarchy_parent_variant ?: flags.primary_variant).toString()
        regionReferenceCh = region_reference_ch.filter { key, id, variant, files -> variant.toString() == hierarchyVariant }
            .map { key, id, variant, files -> tuple(id, files, true) }
            .ifEmpty { error 'Region reference atlas requested but no matching mapping bundle was emitted.' }
    }
    if (params.cell_profiles_spatialdata) {
        if (domainSource != 'refined') error 'Pipeline SpatialData export requires refined tissue polygons; use cell_profiles.nf for explicitly registered external polygons.'
        def polygons = flags.run_cluster_geojson ? geojson_ch.filter { key, id, variant, path -> variant == flags.primary_variant }.map { key, id, variant, path -> tuple(id, path) }
            : image_input_ch.map { id, _image -> tuple(id, file("${params.outdir_base}/15_cluster_geojson/${id}/${id}_${flags.primary_variant}_grown_mask_smooth_class.geojson", checkIfExists: true)) }
        def exportInputs = profileCh.join(strictJoin, crop_ch).join(strictJoin, labels_ch).join(strictJoin, shift_ch).join(strictJoin, resolution_ch).join(strictJoin, polygons).join(strictJoin, domainCh).join(strictJoin, hierarchyInputs)
            .join(strictJoin, cellReferenceCh).join(strictJoin, regionReferenceCh).join(strictJoin, cohortExportCh)
            .map { id, profiles, image, labels, shift, resolution, polygon, domains, hierarchy, haveHierarchy, cellReference, haveCellReference, regionReference, haveRegionReference, cohortBundle, haveCohort ->
                def measured = measuredBySample[id] ?: [packages: [], shapes: []]
                def packages = measured.packages ? measured.packages.collect { file(it, checkIfExists: true) } : [placeholder]
                def shapes = measured.shapes ? measured.shapes.collect { file(it, checkIfExists: true) } : [placeholder]
                tuple(id, profiles, image, labels, shift, resolution, polygon, domains, 'crop_pixels', true,
                    packages, shapes,
                    [packages: !!measured.packages, shapes: !!measured.shapes], hierarchy, haveHierarchy,
                    cellReference, regionReference, [cell: haveCellReference, region: haveRegionReference],
                    cohortBundle, haveCohort,
                    PipelineHelpers.atlasTaskFingerprint('spatialdata_export', projectDir, [profiles, packages, shapes, hierarchy, cohortBundle]))
            }
        EXPORT_SPATIALDATA(exportInputs, runtime_plan)
    }

    emit:
    profiles = profileCh
    morphology = BUILD_SPATIAL_CELL_PROFILES.out.morphology
    spatialdata = params.cell_profiles_spatialdata ? EXPORT_SPATIALDATA.out.spatialdata : Channel.empty()
    reference_assignments = params.cell_reference_atlas ? MAP_CELL_REFERENCE_ATLAS.out.assignments : Channel.empty()
    cohort_niches = params.cohort_niches_enable ? FIT_COHORT_NICHES.out.cohort : Channel.empty()
}
