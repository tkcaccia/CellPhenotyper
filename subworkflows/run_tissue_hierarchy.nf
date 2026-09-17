include { PREPARE_HIERARCHY_FEATURES } from '../modules/prepare_hierarchy_features'
include { DISCOVER_TISSUE_HIERARCHY } from '../modules/discover_tissue_hierarchy'

workflow RUN_TISSUE_HIERARCHY {
    take:
    crop_ch
    grid_objects_ch
    grid_metadata_ch
    tissue_ch
    shift_ch
    resolution_ch
    parent_masks_ch
    parent_uncertainty_ch
    enabled
    runtime_plan

    main:
    def featuresCh = Channel.empty()
    def hierarchyCh = Channel.empty()
    def regionsCh = Channel.empty()
    def parentsCh = Channel.empty()
    def uncertaintyCh = Channel.empty()
    def subdomainsCh = Channel.empty()
    def statusCh = Channel.empty()
    def regionMasksCh = Channel.empty()
    def mappingsCh = Channel.empty()
    def summariesCh = Channel.empty()
    if (enabled) {
        def variant = (params.tissue_hierarchy_parent_variant ?: params.cluster_primary_variant).toString()
        if (!(variant ==~ /[A-Za-z0-9][A-Za-z0-9_.-]*/) || variant == 'features')
            error 'Hierarchy parent variant must be a safe nonempty identifier other than the reserved features directory'
        def snapshotRaw = params.tissue_hierarchy_model_snapshot
        if (!snapshotRaw) error 'Enabled tissue hierarchy requires tissue_hierarchy_model_snapshot: an existing local UNI2 snapshot, never an automatic download'
        def snapshot = file(snapshotRaw, checkIfExists: true)
        if (!snapshot.isDirectory()) error 'tissue_hierarchy_model_snapshot must be a local directory'
        def config = file("${snapshot}/config.json", checkIfExists: true)
        def weightsName = params.tissue_hierarchy_weights_filename
        if (!weightsName) weightsName = ['model.safetensors', 'pytorch_model.bin'].find { file("${snapshot}/${it}").exists() }
        if (!weightsName || !(weightsName.toString() ==~ /[A-Za-z0-9][A-Za-z0-9_.-]*\.(safetensors|bin|pt|pth)/))
            error 'Hierarchy requires a safe local checkpoint filename (safetensors, bin, pt or pth)'
        def weights = file("${snapshot}/${weightsName}", checkIfExists: true)
        def strictJoin = [failOnDuplicate: true, failOnMismatch: true]
        def sources = crop_ch.ifEmpty { error 'Tissue hierarchy requires analysis crops' }
            .map { id, image ->
                if (!(id.toString() ==~ /[A-Za-z0-9][A-Za-z0-9_.-]*/)) error 'Hierarchy sample IDs must be safe path identifiers'
                tuple(id, image)
            }
            .join(strictJoin, grid_objects_ch).join(strictJoin, grid_metadata_ch)
            .join(strictJoin, shift_ch).join(strictJoin, resolution_ch).join(strictJoin, tissue_ch)
        def parents = parent_masks_ch.filter { key, id, v, mask -> v.toString() == variant }
            .map { key, id, v, mask -> tuple(id, key, v, mask) }
            .ifEmpty { error "No immutable parent masks available for hierarchy variant ${variant}" }
        def uncertain = parent_uncertainty_ch.filter { key, id, v, mask -> v.toString() == variant }
            .map { key, id, v, mask -> tuple(id, key, v, mask) }
            .ifEmpty { error "Hierarchy requires canonical parent uncertainty for variant ${variant}" }
        def inputs = sources.join(strictJoin, parents).join(strictJoin, uncertain)
            .map { id, image, objects, metadata, shift, resolution, support, key, v, parent, ukey, uv, uncertainty ->
                if (key != ukey || v != uv) error "Hierarchy parent and uncertainty keys differ for ${id}"
                tuple(id, image, objects, metadata, shift, resolution, support, key, v, parent, uncertainty)
            }
        PREPARE_HIERARCHY_FEATURES(inputs.map { id, image, objects, metadata, shift, resolution, support, key, v, parent, uncertainty ->
            // Stage the actual checkpoint file, not a snapshot-directory symlink
            // whose target may lie outside a container's mounted inputs.
            tuple(id, image, objects, metadata, shift, resolution, config, weights, weightsName.toString())
        }, runtime_plan)
        featuresCh = PREPARE_HIERARCHY_FEATURES.out.features
        def discoveryInputs = inputs.join(strictJoin, featuresCh)
            .map { id, image, objects, metadata, shift, resolution, support, key, v, parent, uncertainty, features ->
                tuple(key, id, v, image, objects, metadata, shift, resolution, support, parent, uncertainty, features)
            }
        DISCOVER_TISSUE_HIERARCHY(discoveryInputs, runtime_plan)
        hierarchyCh = DISCOVER_TISSUE_HIERARCHY.out.hierarchy
        regionsCh = DISCOVER_TISSUE_HIERARCHY.out.region_profiles
        parentsCh = DISCOVER_TISSUE_HIERARCHY.out.parent_masks
        uncertaintyCh = DISCOVER_TISSUE_HIERARCHY.out.parent_uncertainty
        subdomainsCh = DISCOVER_TISSUE_HIERARCHY.out.subdomain_masks
        statusCh = DISCOVER_TISSUE_HIERARCHY.out.status_masks
        regionMasksCh = DISCOVER_TISSUE_HIERARCHY.out.region_masks
        mappingsCh = DISCOVER_TISSUE_HIERARCHY.out.mappings
        summariesCh = DISCOVER_TISSUE_HIERARCHY.out.summaries
    }

    emit:
    features = featuresCh
    hierarchy = hierarchyCh
    region_profiles = regionsCh
    parent_masks = parentsCh
    parent_uncertainty = uncertaintyCh
    subdomain_masks = subdomainsCh
    status_masks = statusCh
    region_masks = regionMasksCh
    mappings = mappingsCh
    summaries = summariesCh
}
