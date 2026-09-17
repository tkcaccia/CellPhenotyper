process DISCOVER_TISSUE_HIERARCHY {
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'discover_tissue_hierarchy', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([image_tif, grid_objects, grid_metadata, shift_json, resolution_json, tissue_mask, parent_mask, parent_uncertainty, feature_dir]) }
    tag "${sample_id}:${cluster_variant}:immutable_parents"
    label 'compute_medium'
    // Native interpreter/package bytes are verified inside the task, not in
    // the scheduler cache key. Until a pre-cache runtime identity is bound,
    // rerun this opt-in CPU discovery stage instead of reusing a stale graph.
    // Expensive upstream local/context encoder inference remains cached.
    cache false
    publishDir "${params.outdir_base}/22_tissue_hierarchy/${sample_id}", mode: (params.publish_dir_mode ?: 'copy'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'hierarchy_discovery') }
    memory { TaskRuntime.memory(runtime_plan, 'hierarchy_discovery') }
    time '24h'

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(image_tif, stageAs: 'input/image.tif'), path(grid_objects, stageAs: 'input/grid_objects.csv'), path(grid_metadata, stageAs: 'input/grid_metadata.json'), path(shift_json, stageAs: 'input/shift.json'), path(resolution_json, stageAs: 'input/resolution.json'), path(tissue_mask, stageAs: 'input/support.tif'), path(parent_mask, stageAs: 'input/parent.tif'), path(parent_uncertainty, stageAs: 'input/parent_uncertainty.tif'), path(feature_dir, stageAs: 'hierarchy_features')

    val(runtime_plan)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${cluster_variant}"), emit: hierarchy
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${cluster_variant}/region_profiles"), emit: region_profiles
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${cluster_variant}/parent_domains.ome.tif"), emit: parent_masks
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${cluster_variant}/parent_uncertainty.ome.tif"), emit: parent_uncertainty
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${cluster_variant}/subdomain_mask.ome.tif"), emit: subdomain_masks
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${cluster_variant}/hierarchy_status.ome.tif"), emit: status_masks
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${cluster_variant}/region_mask.ome.tif"), emit: region_masks
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${cluster_variant}/grid_subdomains.csv"), emit: mappings
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${cluster_variant}/hierarchy_summary.json"), emit: summaries

    script:
    def scripts = ['discover_tissue_hierarchy.py', 'hierarchy_kodama.py', 'run_hierarchy_kodama.R', 'kodama_graph_export.R', 'kodama_graph_clustering.R', 'cell_profile_io.py', 'profile_cell_morphology.py', 'ome_tiff_metadata.py', 'uni2_embedding_io.py']
    def codeFingerprint = PipelineHelpers.codeFingerprint(scripts.collect { "${projectDir}/bin/${it}" })
    [kodama_ncomp: params.kodama_ncomp, tissue_hierarchy_kodama_m: params.tissue_hierarchy_kodama_m,
     tissue_hierarchy_kodama_tcycle: params.tissue_hierarchy_kodama_tcycle,
     tissue_hierarchy_kodama_neighbors: params.tissue_hierarchy_kodama_neighbors].each { name, value ->
        if (!(String.valueOf(value) ==~ /[0-9]+/) || (value as BigInteger) < 1)
            error "${name} must be a positive integer"
    }
    def minAffinity
    try { minAffinity = new BigDecimal(String.valueOf(params.tissue_hierarchy_min_affinity_margin)) }
    catch (NumberFormatException ignored) { error 'tissue_hierarchy_min_affinity_margin must be finite in [0,1]' }
    if (minAffinity < 0 || minAffinity > 1) error 'tissue_hierarchy_min_affinity_margin must be finite in [0,1]'
    def rLibrary = params.tissue_hierarchy_kodama_r_library
    if (rLibrary != null && (!(rLibrary instanceof CharSequence) || !rLibrary.toString().trim() ||
        rLibrary.toString() != rLibrary.toString().trim() || (rLibrary.toString() =~ /[\x00-\x1f\x7f]/).find()))
        error 'tissue_hierarchy_kodama_r_library must be null or an explicit nonempty path without edge whitespace/control characters'
    def rLibraryFlag = rLibrary == null ? '' : "--kodama-r-library '" + rLibrary.toString().replace("'", "'\"'\"'") + "'"
    def fixedK = (params.tissue_hierarchy_fixed_k as int) > 0 ? "--fixed-k ${params.tissue_hierarchy_fixed_k}" : ''
    def python = params.tissue_hierarchy_python.toString().replace("'", "'\"'\"'")
    """
    set -euo pipefail
    source "${projectDir}/bin/activate_source_python.sh"
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Tissue hierarchy code fingerprint: ${codeFingerprint}"
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus} LOKY_MAX_CPU_COUNT=${task.cpus}
    '${python}' "${projectDir}/bin/discover_tissue_hierarchy.py" \
      --image '${image_tif}' --parent-mask '${parent_mask}' --parent-uncertainty '${parent_uncertainty}' --support-mask '${tissue_mask}' \
      --grid-objects '${grid_objects}' --grid-metadata '${grid_metadata}' --shift-json '${shift_json}' --resolution-json '${resolution_json}' \
      --local-embeddings '${feature_dir}/local' --context-embeddings '${feature_dir}/context' --embedding-metadata '${feature_dir}/embedding_metadata.json' \
      --sample-id '${sample_id}' --local-weight ${params.tissue_hierarchy_local_weight} --context-weight ${params.tissue_hierarchy_context_weight} \
      --discovery-method kodama_graph --kodama-ncomp ${params.kodama_ncomp} \
      --kodama-m ${params.tissue_hierarchy_kodama_m} --kodama-tcycle ${params.tissue_hierarchy_kodama_tcycle} \
      --kodama-neighbors ${params.tissue_hierarchy_kodama_neighbors} --kodama-cpus ${task.cpus} ${rLibraryFlag} \
      --max-k ${params.tissue_hierarchy_max_k} ${fixedK} --seed ${params.tissue_hierarchy_seed} --repeats ${params.tissue_hierarchy_repeats} \
      --fit-limit ${params.tissue_hierarchy_fit_limit} --components-per-block ${params.tissue_hierarchy_components_per_block} \
      --min-observations ${params.tissue_hierarchy_min_observations} --parent-purity ${params.tissue_hierarchy_parent_purity} \
      --min-seed-stability ${params.tissue_hierarchy_min_seed_stability} --min-scale-agreement ${params.tissue_hierarchy_min_scale_agreement} \
      --min-affinity-margin ${minAffinity.toPlainString()} --tile-size ${params.tissue_hierarchy_tile_size} \
      --max-components ${params.tissue_hierarchy_max_components} --outdir '${cluster_variant}'
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p '${cluster_variant}/region_profiles'
    cp '${parent_mask}' '${cluster_variant}/parent_domains.ome.tif'
    cp '${parent_uncertainty}' '${cluster_variant}/parent_uncertainty.ome.tif'
    touch '${cluster_variant}/subdomain_mask.ome.tif' '${cluster_variant}/hierarchy_status.ome.tif' '${cluster_variant}/region_mask.ome.tif'
    printf 'sample_id,label,parent_domain_id,subdomain_id,status_code\n' > '${cluster_variant}/grid_subdomains.csv'
    printf 'region_uid,region_id,sample_id,parent_domain_id,subdomain_id\n' > '${cluster_variant}/region_profiles/region_profiles.csv'
    printf 'region_uid,region_id,sample_id\n' > '${cluster_variant}/region_profiles/feature_rows.csv'
    printf '{"stub":true,"observation_unit":"tissue_region","region_count":0,"feature_blocks":{}}\n' > '${cluster_variant}/region_profiles/region_profiles_manifest.json'
    printf '{"stub":true,"sample_id":"${sample_id}","parent_labels_immutable":true,"scientific_claim":"No discovery or valid derived raster produced"}\n' > '${cluster_variant}/hierarchy_summary.json'
    """
}
