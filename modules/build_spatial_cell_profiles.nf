process BUILD_SPATIAL_CELL_PROFILES {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'build_spatial_cell_profiles', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([image_tif, labels_tif, objects_csv, shift_json, resolution_json, tissue_mask, marker_inputs, context_dir, local_dir, cellvit_dir, domain_mask, compartment_dir, domain_uncertainty]) }
    tag "${sample_id}"
    label 'compute_medium'
    publishDir "${params.outdir_base}/19_cell_profiles/${sample_id}", mode: (params.publish_dir_mode ?: 'copy'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'cell_profiles') }
    memory { TaskRuntime.memory(runtime_plan, 'cell_profiles') }
    time '24h'

    input:
    tuple val(sample_id), path(image_tif), path(labels_tif), path(objects_csv), path(shift_json), path(resolution_json), path(tissue_mask), path(marker_inputs, stageAs: 'markers/*'), path(context_dir, stageAs: 'context/*'), path(local_dir, stageAs: 'local/*'), path(cellvit_dir, stageAs: 'cellvit/*'), path(domain_mask, stageAs: 'domains/*'), path(compartment_dir, stageAs: 'compartments/*'), path(domain_uncertainty, stageAs: 'uncertainty/*'), val(features)

    val(runtime_plan)

    output:
    tuple val(sample_id), path('cell_profiles'), emit: profiles
    tuple val(sample_id), path('morphology'), emit: morphology

    script:
    def scripts = ['profile_cell_morphology.py', 'cell_morphology_io.py', 'cell_phenotype_io.py', 'build_cell_profiles.py', 'cell_profile_io.py', 'assemble_spatial_cell_profiles.py', 'assemble_neighborhood_arrays.py', 'neighborhood_feature_io.py', 'niche_clustering.py', 'analyze_cell_neighborhoods.py', 'uni2_embedding_io.py', 'cellvit_embedding_io.py', 'build_native_tissue_support.py']
    def codeFingerprint = PipelineHelpers.codeFingerprint(scripts.collect { "${projectDir}/bin/${it}" })
    def featureStorage = String.valueOf(params.cell_neighborhood_feature_storage)
    if (!(featureStorage in ['table', 'arrays'])) error 'cell_neighborhood_feature_storage must be table or arrays'
    def arraySettings = [cell_neighborhood_row_batch_size: params.cell_neighborhood_row_batch_size,
        cell_neighborhood_column_batch_size: params.cell_neighborhood_column_batch_size,
        cell_niche_fit_limit: params.cell_niche_fit_limit]
    arraySettings.each { name, value ->
        if (!(String.valueOf(value) ==~ /[0-9]+/) || (value as BigInteger) < (name == 'cell_niche_fit_limit' ? 10 : 1))
            error "${name} must be a positive integer${name == 'cell_niche_fit_limit' ? ' >=10' : ''}"
    }
    def supportMode = String.valueOf(params.cell_neighborhood_support_mode)
    if (!(supportMode in ['provided', 'brightfield_native'])) {
        error "cell_neighborhood_support_mode must be provided or brightfield_native"
    }
    def nativeSupportCommand = supportMode == 'brightfield_native' ? """
    "${params.cell_atlas_python}" "${projectDir}/bin/build_native_tissue_support.py" --image '${image_tif}' --support-mask '${tissue_mask}' --shift '${shift_json}' --resolution-json '${resolution_json}' --outdir native_support
    """ : ''
    def nativeSupportFlags = supportMode == 'brightfield_native' ? '--native-support-mask native_support/support.tif --native-support-coordinates crop_pixels --native-support-manifest native_support/support_manifest.json' : ''
    def markerFlag = features.markers ? '--marker-quant-dir markers' : ''
    def contextFlag = features.uni2 ? "--uni2-context '${context_dir}' --uni2-local '${local_dir}'" : ''
    def cellvitFlag = features.cellvit ? "--cellvit-features '${cellvit_dir}'" : ''
    def domainFlag = features.domains ? "--domain-mask '${domain_mask}'" : ''
    def compartmentFlag = features.compartments ? "--compartment-qc '${compartment_dir}/compartment_qc.csv'" : ''
    def uncertaintyFlag = features.uncertainty ? "--domain-uncertainty '${domain_uncertainty}'" : ''
    def nicheK = (params.cell_niche_fixed_k as int) > 0 ? "--fixed-k ${params.cell_niche_fixed_k}" : ''
    def nicheWeights = String.valueOf(params.cell_neighborhood_feature_weights ?: '{}').replace("'", "'\\''")
    def nicheWorkingMb = Math.max(1L, (task.memory.toMega() / 2L) as long)
    """
    set -euo pipefail
    source "${projectDir}/bin/activate_source_python.sh"
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Cell profile/atlas code fingerprint: ${codeFingerprint}"
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus} LOKY_MAX_CPU_COUNT=${task.cpus}
    "${params.cell_atlas_python}" -c "import numpy, pandas, scipy, sklearn, pyarrow, tifffile; print('Cell profile runtime available')"
    "${params.cell_atlas_python}" "${projectDir}/bin/profile_cell_morphology.py" \
      --labels '${labels_tif}' --image '${image_tif}' --objects '${objects_csv}' \
      --shift '${shift_json}' --resolution-json '${resolution_json}' --outdir morphology
    "${params.cell_atlas_python}" "${projectDir}/bin/build_cell_profiles.py" \
      --objects '${objects_csv}' --sample-id '${sample_id}' --shift '${shift_json}' \
      --image '${image_tif}' --labels '${labels_tif}' --tissue-mask '${tissue_mask}' \
      --resolution-json '${resolution_json}' --morphology morphology/cell_morphology.csv \
      ${markerFlag} ${contextFlag} ${cellvitFlag} ${domainFlag} ${compartmentFlag} ${uncertaintyFlag} --outdir base_profiles
    ${nativeSupportCommand}
    "${params.cell_atlas_python}" "${projectDir}/bin/assemble_spatial_cell_profiles.py" \
      --profile-dir base_profiles --support-mask '${tissue_mask}' --shift '${shift_json}' \
      --resolution-json '${resolution_json}' ${domainFlag} \
      --radii-um '${params.cell_neighborhood_radii_um}' --feature-groups '${params.cell_neighborhood_feature_groups}' \
      --feature-weights '${nicheWeights}' --max-working-mb ${nicheWorkingMb} \
      --feature-storage '${params.cell_neighborhood_feature_storage}' \
      --row-batch-size ${params.cell_neighborhood_row_batch_size} --column-batch-size ${params.cell_neighborhood_column_batch_size} \
      --seed ${params.cell_niche_seed} --max-k ${params.cell_niche_max_k} ${nicheK} \
      --fit-limit ${params.cell_niche_fit_limit} \
      ${nativeSupportFlags} --outdir cell_profiles
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p cell_profiles morphology
    printf '{"stub":true,"cell_count":0}' > cell_profiles/cell_profiles_manifest.json
    printf 'sample_id,cell_id,cell_uid\n' > cell_profiles/cell_profiles.csv
    printf 'label,area_um2\n' > morphology/cell_morphology.csv
    """
}
