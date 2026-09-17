process EXPORT_SPATIALDATA {
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'export_spatialdata', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([profiles, image_tif, labels_tif, shift_json, resolution_json, tissue_geojson, domain_mask, measured_packages, measured_region_bundles, hierarchy, cell_reference_files, region_reference_files, cohort_niches]) }
    tag "${sample_id}"
    label 'compute_medium'
    cache 'deep'
    publishDir "${params.outdir_base}/20_spatialdata/${sample_id}", mode: (params.publish_dir_mode ?: 'copy'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'spatialdata') }
    memory { TaskRuntime.memory(runtime_plan, 'spatialdata') }
    time '24h'

    input:
    tuple val(sample_id), path(profiles), path(image_tif), path(labels_tif), path(shift_json), path(resolution_json), path(tissue_geojson), path(domain_mask), val(tissue_coordinates), val(check_native_domains), path(measured_packages, stageAs: 'measured/package??'), path(measured_region_bundles, stageAs: 'measured_regions/bundle??'), val(measured_flags), path(hierarchy, stageAs: 'hierarchy'), val(have_hierarchy), path(cell_reference_files, stageAs: 'cell_reference/*'), path(region_reference_files, stageAs: 'region_reference/*'), val(reference_flags), path(cohort_niches, stageAs: 'cohort_niches_input'), val(have_cohort_niches), val(source_fingerprint)

    val(runtime_plan)

    output:
    tuple val(sample_id), path('spatialdata.zarr'), emit: spatialdata

    script:
    def codeFingerprint = PipelineHelpers.codeFingerprint(['export_spatialdata.py', 'integrate_measured_assay.py', 'cell_profile_io.py', 'profile_cell_morphology.py', 'link_cell_tissue_hierarchy.py', 'reference_mapping_io.py', 'cohort_niche_io.py'].collect { "${projectDir}/bin/${it}" })
    def gridCheck = check_native_domains ? "\"${params.spatialdata_python}\" -c \"import sys; sys.path.insert(0, '${projectDir}/bin'); from cell_profile_io import RasterReader; a=RasterReader('${domain_mask}'); b=RasterReader('${image_tif}'); valid=(a.width,a.height)==(b.width,b.height); a.close(); b.close(); sys.exit(0 if valid else 'Tissue polygons are not on the native crop grid')\"" : ''
    def packages = measured_packages instanceof List ? measured_packages : [measured_packages]
    def bundles = measured_region_bundles instanceof List ? measured_region_bundles : [measured_region_bundles]
    def measuredArgs = measured_flags.packages ? packages.collect { "--measured-assay '${it}'" }.join(' ') : ''
    def regionArgs = measured_flags.shapes ? bundles.collect { "--measured-region-shapes '${it}/link.json'" }.join(' ') : ''
    def hierarchyArg = have_hierarchy ? "--hierarchy-dir '${hierarchy}'" : ''
    def cohortArg = have_cohort_niches ? "--cohort-niches '${cohort_niches}'" : ''
    def mappingArg = { values, enabled, unit ->
        if (!enabled) return ''
        def files = values instanceof List ? values : [values]
        def receipts = files.findAll { it.toString().endsWith('.mapping.json') }
        if (receipts.size() != 1 || files.size() != 3)
            error "${unit} reference mapping requires one complete CSV/atlas/receipt bundle"
        "--${unit}-reference-mapping '${receipts[0]}'"
    }
    def cellReferenceArg = mappingArg(cell_reference_files, reference_flags.cell, 'cell')
    def regionReferenceArg = mappingArg(region_reference_files, reference_flags.region, 'region')
    """
    set -euo pipefail
    source "${projectDir}/bin/activate_source_python.sh"
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] SpatialData code fingerprint: ${codeFingerprint}"
    echo "[INFO] SpatialData code-and-directory-content fingerprint: ${source_fingerprint}"
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    ${gridCheck}
    "${params.spatialdata_python}" "${projectDir}/bin/export_spatialdata.py" \
      --sample-id '${sample_id}' --profile-dir '${profiles}' --image '${image_tif}' --labels '${labels_tif}' \
      --shift '${shift_json}' --resolution-json '${resolution_json}' \
      --tissue-geojson '${tissue_geojson}' --tissue-coordinates '${tissue_coordinates}' \
      --workers ${task.cpus} --tile-size ${params.spatialdata_tile_size} \
      --pyramid-levels ${params.spatialdata_pyramid_levels} ${measuredArgs} ${regionArgs} ${hierarchyArg} \
      ${cellReferenceArg} ${regionReferenceArg} ${cohortArg} --outdir spatialdata.zarr
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p spatialdata.zarr
    printf '{"stub":true}' > spatialdata.zarr/stub.json
    """
}
