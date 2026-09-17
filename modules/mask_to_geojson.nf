process MASK_TO_GEOJSON {
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'mask_to_geojson', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([mask_tif, uncertainty_tif, provenance_tif, provenance_json]) }
    tag "${sample_id}:${cluster_variant}"
    label 'compute_medium'
    cache 'deep'

    publishDir "${params.outdir_base}/15_cluster_geojson/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.cluster_geojson_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.cluster_geojson_memory_gb as int))} GB" }
    time { params.cluster_geojson_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(mask_tif, stageAs: 'source_labels/*'), path(uncertainty_tif, stageAs: 'source_uncertainty/*'), path(provenance_tif, stageAs: 'source_provenance/*'), path(provenance_json, stageAs: 'source_metadata/*'), val(provenance_flags)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson"), emit: cluster_geojson
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson.provenance.json"), emit: provenance_summary

    script:
    def geojson_script = "${projectDir}/${params.cluster_geojson_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([geojson_script, "${projectDir}/bin/profile_cell_morphology.py", "${projectDir}/bin/shared_boundary_smoothing.py"])
    if (sample_key.toString() != "${sample_id}::${cluster_variant}") error 'Vectorization sample key conflicts with sample/variant lineage.'
    if (provenance_flags.refinement && !provenance_flags.uncertainty) error 'Refinement vectorization requires categorical uncertainty.'
    def uncertaintyFlag = provenance_flags.uncertainty ? "--uncertainty-mask \"${uncertainty_tif}\"" : ''
    def provenanceFlag = provenance_flags.refinement ? "--provenance-mask \"${provenance_tif}\" --provenance-metadata \"${provenance_json}\"" : ''
    def dissolve_by_value_flag = (params.cluster_geojson_dissolve_by_value as boolean) ? '--dissolve-by-value' : ''
    def fill_holes_flag = (params.cluster_geojson_fill_holes as boolean) ? '--fill-holes' : ''
    def preserve_flag = (params.cluster_geojson_preserve_topology as boolean) ? '--preserve-topology' : ''
    def shared_boundary_flag = (params.cluster_geojson_shared_boundary_simplify as boolean) ? '--shared-boundary-simplify' : ''
    def fidelity_flag = (params.cluster_geojson_shared_boundary_simplify as boolean) ? "--simplify-max-categorical-difference-fraction ${params.cluster_geojson_simplify_max_categorical_difference_fraction} --simplify-search-steps ${params.cluster_geojson_simplify_search_steps}" : ''
    def boundary_smoothing_flag = (params.cluster_geojson_shared_boundary_smooth as boolean) ? "--shared-boundary-smooth --shared-boundary-smoothing-coefficient ${params.cluster_geojson_shared_boundary_smoothing_coefficient} --shared-boundary-smoothing-passes ${params.cluster_geojson_shared_boundary_smoothing_passes} --shared-boundary-minimum-turn-degrees ${params.cluster_geojson_shared_boundary_minimum_turn_degrees} --shared-boundary-minimum-adjacent-length ${params.cluster_geojson_shared_boundary_minimum_adjacent_length}" : ''
    def group_map_flag = params.cluster_geojson_group_map ? "--group-map \"${params.cluster_geojson_group_map}\"" : ''
    def tail_flags = [dissolve_by_value_flag, fill_holes_flag, preserve_flag, shared_boundary_flag, fidelity_flag, boundary_smoothing_flag, group_map_flag].findAll { it?.trim() }.join(' ')
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Vectorization code fingerprint: ${codeFingerprint}"

    python "${geojson_script}" \
      --mask "${mask_tif}" \
      --sample-key "${sample_key}" --sample-id "${sample_id}" --cluster-variant "${cluster_variant}" \
      ${uncertaintyFlag} ${provenanceFlag} \
      --page ${params.cluster_geojson_page} \
      --max-page-side ${params.cluster_geojson_max_page_side} \
      --out "${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson" \
      --min-area ${params.cluster_geojson_min_area} \
      --polygon-backend ${params.cluster_geojson_polygon_backend} \
      --connectivity ${params.cluster_geojson_connectivity} \
      --smooth-buffer ${params.cluster_geojson_smooth_buffer} \
      --smooth-passes ${params.cluster_geojson_smooth_passes} \
      --simplify ${params.cluster_geojson_simplify} \
      --group-prefix "${params.cluster_geojson_group_prefix}" \
      ${tail_flags}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo '{"type":"FeatureCollection","features":[],"stub":true}' > "${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson"
    echo '{"schema_version":"cellphenotyper.vectorized_refinement_provenance.v1","status":"stub_only_not_computed"}' > "${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson.provenance.json"
    """
}
