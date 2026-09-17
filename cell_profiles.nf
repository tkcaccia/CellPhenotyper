// Rebuild linked profiles from immutable existing artifacts without rerunning models.
import groovy.json.JsonSlurper
nextflow.enable.dsl = 2
include { BUILD_SPATIAL_CELL_PROFILES } from './modules/build_spatial_cell_profiles'
include { EXPORT_SPATIALDATA } from './modules/export_spatialdata'
include { MAP_CELL_REFERENCE_ATLAS } from './modules/map_cell_reference_atlas'
include { MAP_REGION_REFERENCE_ATLAS } from './modules/map_region_reference_atlas'
include { LINK_CELL_TISSUE_HIERARCHY } from './modules/link_cell_tissue_hierarchy'
include { FIT_COHORT_NICHES } from './modules/fit_cohort_niches'

workflow {
    PipelineHelpers.validateCohortBundleOptions(params)
    if (!params.cell_profile_samples) error 'Supply --cell_profile_samples with a JSON list of sample artifact records.'
    def manifest = file(params.cell_profile_samples, checkIfExists: true)
    def records = new JsonSlurper().parse(manifest.toFile())
    if (!(records instanceof List) || !records) error 'Cell-profile sample manifest must be a nonempty JSON list.'
    def ids = records.collect { it.sample_id }
    if (ids.any { !it || it.toString() in ['.', '..'] || !(it.toString() ==~ /[A-Za-z0-9_.-]+/) } || ids.unique(false).size() != ids.size()) error 'Sample IDs must be unique, must not be dot path segments, and must use letters, numbers, dot, dash or underscore.'
    def runtime_plan = TaskRuntime.forArtifacts(params)
    def placeholder = file("${projectDir}/resources/empty_embeddings_placeholder", checkIfExists: true)
    def resolve = { raw -> file(manifest.parent.resolve(raw.toString()).normalize(), checkIfExists: true) }
    def rows = records.findAll { !it.existing_profiles }.collect { row ->
        def required = ['image', 'labels', 'objects', 'shift', 'resolution', 'support']
        if (required.any { !row[it] }) error "Sample ${row.sample_id} is missing a required artifact: ${required}"
        if (!!row.uni2_context != !!row.uni2_local) error 'Supply both local and contextual cell features, or neither.'
        def markers = row.markers ? (row.markers instanceof List ? row.markers.collect { resolve(it) } : resolve(row.markers)) : placeholder
        tuple(row.sample_id.toString(), resolve(row.image), resolve(row.labels), resolve(row.objects),
            resolve(row.shift), resolve(row.resolution), resolve(row.support), markers,
            row.uni2_context ? resolve(row.uni2_context) : placeholder,
            row.uni2_local ? resolve(row.uni2_local) : placeholder,
            row.cellvit ? resolve(row.cellvit) : placeholder,
            row.domains ? resolve(row.domains) : placeholder,
            row.compartments ? resolve(row.compartments) : placeholder,
            row.domain_uncertainty ? resolve(row.domain_uncertainty) : placeholder,
            [markers: !!row.markers, uni2: !!row.uni2_context, cellvit: !!row.cellvit, domains: !!row.domains,
             compartments: !!row.compartments, uncertainty: !!row.domain_uncertainty])
    }
    if (!params.cell_profiles_spatialdata && records.any { it.measured_assays || it.measured_region_shapes })
        error 'Measured assay attachments require cell_profiles_spatialdata=true; they must not be silently ignored.'
    if (!params.cell_profiles_spatialdata && records.any { it.cell_reference_mapping || it.region_reference_mapping })
        error 'Reference mapping attachments require cell_profiles_spatialdata=true; they must not be silently ignored.'
    if (params.cell_reference_atlas && records.any { it.cell_reference_mapping })
        error 'Choose cell_reference_atlas or per-record cell_reference_mapping, not both.'
    if (params.region_reference_atlas && records.any { it.region_reference_mapping })
        error 'Choose region_reference_atlas or per-record region_reference_mapping, not both.'
    if (records.any { it.region_reference_mapping && !it.hierarchy })
        error 'Region reference mapping requires the corresponding tissue hierarchy.'
    if (params.region_reference_atlas && !records.any { it.hierarchy })
        error 'region_reference_atlas requires at least one sample with tissue hierarchy.'
    if (params.cell_measured_assays)
        error 'cell_profiles.nf reads measured_assays from each sample record; cell_measured_assays is for main.nf.'
    BUILD_SPATIAL_CELL_PROFILES(Channel.fromList(rows), runtime_plan)
    def alreadyLinkedIds = []
    def existingRows = records.findAll { it.existing_profiles }.collect { row ->
        def directory = resolve(row.existing_profiles)
        if (!directory.resolve('cell_profiles_manifest.json').exists()) error "Sample ${row.sample_id}: existing_profiles has no immutable profile manifest."
        def profileManifest = new JsonSlurper().parse(directory.resolve('cell_profiles_manifest.json').toFile())
        if (profileManifest.sample_id != row.sample_id || profileManifest.observation_unit != 'cell')
            error "Sample ${row.sample_id}: existing_profiles has a different specimen or observation unit."
        if (profileManifest.hierarchy_links) {
            alreadyLinkedIds.add(row.sample_id.toString())
            if (row.compartments) error "Sample ${row.sample_id}: an already linked profile retains its ring source; do not replace compartments during re-export."
        }
        tuple(row.sample_id.toString(), directory)
    }
    def profileCh = BUILD_SPATIAL_CELL_PROFILES.out.profiles.mix(Channel.fromList(existingRows))
    def hierarchyIds = records.findAll { it.hierarchy && !(it.sample_id.toString() in alreadyLinkedIds) }.collect { it.sample_id.toString() }
    if (hierarchyIds) {
        def linkRows = records.findAll { it.sample_id.toString() in hierarchyIds }.collect { row ->
            if (!row.labels) error "Sample ${row.sample_id}: hierarchy linkage requires the canonical label raster."
            tuple(row.sample_id.toString(), resolve(row.labels), resolve(row.hierarchy),
                row.compartments ? resolve(row.compartments) : placeholder, !!row.compartments)
        }
        def linkedInputs = profileCh.filter { id, profiles -> id in hierarchyIds }
            .join(Channel.fromList(linkRows), failOnDuplicate: true, failOnMismatch: true)
        LINK_CELL_TISSUE_HIERARCHY(linkedInputs, runtime_plan)
        profileCh = profileCh.filter { id, profiles -> !(id in hierarchyIds) }.mix(LINK_CELL_TISSUE_HIERARCHY.out.profiles)
    }
    def cohortExportCh = Channel.fromList(records.collect { row -> tuple(row.sample_id.toString(), placeholder, false) })
    def cohortBundleCh = Channel.empty()
    if (params.cohort_niches_enable) {
        def cohortInputs = profileCh.collect(flat: false).map { samples ->
            def ordered = samples.sort { a, b -> a[0].toString() <=> b[0].toString() }
            def cohortIds = ordered.collect { it[0].toString() }
            if (cohortIds.size() < 2 || cohortIds.unique(false).size() != cohortIds.size())
                error 'Cohort niche discovery requires at least two distinct specimens; empty or duplicate specimen inputs are invalid.'
            tuple(cohortIds, ordered.collect { it[1] })
        }
        FIT_COHORT_NICHES(cohortInputs, runtime_plan)
        cohortBundleCh = FIT_COHORT_NICHES.out.cohort
    } else if (params.cohort_niches_bundle) {
        cohortBundleCh = Channel.value(file(params.cohort_niches_bundle, checkIfExists: true))
    }
    if (params.cohort_niches_enable || params.cohort_niches_bundle) {
        // Collecting makes this a single reusable value, not a queue item
        // consumed by only one specimen. Export joins remain specimen keyed.
        def cohortBundleValue = cohortBundleCh.collect(flat: false).map { bundles ->
            if (bundles.size() != 1) error 'Cohort export requires exactly one completed global bundle.'
            bundles[0]
        }
        cohortExportCh = profileCh.map { id, _profiles -> id }.combine(cohortBundleValue)
            .map { id, bundle -> tuple(id, bundle, true) }
    }
    def mappingFiles = { row, field -> row[field]
        ? AtlasInputs.mappingBundle(row[field], manifest.parent.toFile(), field).collect { file(it, checkIfExists: true) }
        : [placeholder] }
    def cellReferenceCh = Channel.fromList(records.collect { row ->
        tuple(row.sample_id.toString(), mappingFiles(row, 'cell_reference_mapping'), !!row.cell_reference_mapping)
    })
    if (params.cell_reference_atlas) {
        MAP_CELL_REFERENCE_ATLAS(profileCh.map { id, profiles ->
            def reference = file(params.cell_reference_atlas, checkIfExists: true)
            tuple(id, profiles, reference, PipelineHelpers.atlasTaskFingerprint('cell_reference_mapping', projectDir, [profiles, reference]))
        }, runtime_plan)
        cellReferenceCh = MAP_CELL_REFERENCE_ATLAS.out.mapping_bundle.map { id, files -> tuple(id, files, true) }
    }
    def regionReferenceCh = Channel.fromList(records.collect { row ->
        tuple(row.sample_id.toString(), mappingFiles(row, 'region_reference_mapping'), !!row.region_reference_mapping)
    })
    if (params.region_reference_atlas) {
        def regionRows = records.findAll { it.hierarchy }.collect { row ->
            def profiles = resolve(row.hierarchy).resolve('region_profiles')
            if (!profiles.resolve('region_profiles_manifest.json').exists())
                error "Sample ${row.sample_id}: hierarchy has no region profiles for reference mapping."
            def reference = file(params.region_reference_atlas, checkIfExists: true)
            tuple("${row.sample_id}::artifact_hierarchy", row.sample_id.toString(), 'artifact_hierarchy',
                profiles, reference, PipelineHelpers.atlasTaskFingerprint('region_reference_mapping', projectDir, [profiles, reference]))
        }
        MAP_REGION_REFERENCE_ATLAS(Channel.fromList(regionRows), runtime_plan)
        regionReferenceCh = MAP_REGION_REFERENCE_ATLAS.out.mapping_bundle.map { key, id, variant, files -> tuple(id, files, true) }
            .mix(Channel.fromList(records.findAll { !it.hierarchy }.collect { row -> tuple(row.sample_id.toString(), [placeholder], false) }))
    }
    if (params.cell_profiles_spatialdata) {
        def exportRows = records.collect { row ->
            if (['image', 'labels', 'shift', 'resolution'].any { !row[it] }) error "Sample ${row.sample_id}: SpatialData requires image, labels, shift and passed resolution."
            if (!row.tissue_geojson || !(row.tissue_coordinates in ['crop_pixels', 'original_pixels', 'original_um'])) error "Sample ${row.sample_id}: SpatialData requires tissue_geojson and an explicit tissue_coordinates frame."
            def measured = AtlasInputs.measuredRecord(row, manifest.parent.toFile())
            tuple(row.sample_id.toString(), resolve(row.image), resolve(row.labels), resolve(row.shift), resolve(row.resolution),
                resolve(row.tissue_geojson), placeholder, row.tissue_coordinates.toString(), false,
                measured.packages ? measured.packages.collect { file(it, checkIfExists: true) } : [placeholder],
                measured.shapes ? measured.shapes.collect { file(it, checkIfExists: true) } : [placeholder],
                [packages: !!measured.packages, shapes: !!measured.shapes],
                row.hierarchy ? resolve(row.hierarchy) : placeholder, !!row.hierarchy)
        }
        def exportInputs = profileCh.join(Channel.fromList(exportRows), failOnDuplicate: true, failOnMismatch: true)
            .join(cellReferenceCh, failOnDuplicate: true, failOnMismatch: true)
            .join(regionReferenceCh, failOnDuplicate: true, failOnMismatch: true)
            .join(cohortExportCh, failOnDuplicate: true, failOnMismatch: true)
            .map { id, profiles, image, labels, shift, resolution, polygon, domain, frame, checkDomains, packages, shapes, measuredFlags, hierarchy, haveHierarchy, cellReference, haveCellReference, regionReference, haveRegionReference, cohortBundle, haveCohort ->
                tuple(id, profiles, image, labels, shift, resolution, polygon, domain, frame, checkDomains, packages, shapes, measuredFlags,
                    hierarchy, haveHierarchy, cellReference, regionReference, [cell: haveCellReference, region: haveRegionReference],
                    cohortBundle, haveCohort,
                    PipelineHelpers.atlasTaskFingerprint('spatialdata_export', projectDir, [profiles, packages, shapes, hierarchy, cohortBundle]))
            }
        EXPORT_SPATIALDATA(exportInputs, runtime_plan)
    }
}
