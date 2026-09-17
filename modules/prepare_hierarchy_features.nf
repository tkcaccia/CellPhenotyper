process PREPARE_HIERARCHY_FEATURES {
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'prepare_hierarchy_features', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([image_tif, grid_objects, grid_metadata, shift_json, resolution_json, model_config, model_weights]) }
    tag "${sample_id}:independent_physical_fields"
    label 'compute_heavy'
    label 'gpu_capable'
    cache 'deep'
    publishDir "${params.outdir_base}/22_tissue_hierarchy/${sample_id}", mode: (params.publish_dir_mode ?: 'copy'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'hierarchy_features') }
    memory { TaskRuntime.memory(runtime_plan, 'hierarchy_features') }
    time '24h'

    input:
    tuple val(sample_id), path(image_tif, stageAs: 'input/image.tif'), path(grid_objects, stageAs: 'input/grid_objects.csv'), path(grid_metadata, stageAs: 'input/grid_metadata.json'), path(shift_json, stageAs: 'input/shift.json'), path(resolution_json, stageAs: 'input/resolution.json'), path(model_config, stageAs: 'uni2_snapshot/config.json'), path(model_weights, stageAs: 'uni2_snapshot/*'), val(weights_filename)

    val(runtime_plan)

    output:
    tuple val(sample_id), path('features'), emit: features

    script:
    def scripts = ['prepare_hierarchy_features.py', 'extract_uni2_embeddings.py', 'discover_tissue_hierarchy.py', 'cell_profile_io.py', 'profile_cell_morphology.py', 'model_provenance.py', 'ome_tiff_metadata.py', 'uni2_grid.py', 'uni2_embedding_io.py']
    def codeFingerprint = PipelineHelpers.codeFingerprint(scripts.collect { "${projectDir}/bin/${it}" })
    def python = params.tissue_hierarchy_python.toString().replace("'", "'\"'\"'")
    """
    set -euo pipefail
    source "${projectDir}/bin/activate_source_python.sh"
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Hierarchy feature code fingerprint: ${codeFingerprint}"
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    '${python}' "${projectDir}/bin/prepare_hierarchy_features.py" \
      --image '${image_tif}' --grid-objects '${grid_objects}' --grid-metadata '${grid_metadata}' \
      --shift-json '${shift_json}' --resolution-json '${resolution_json}' --sample-id '${sample_id}' \
      --model-snapshot uni2_snapshot --weights-filename '${weights_filename}' \
      --local-field-um ${params.tissue_hierarchy_local_field_um} --context-field-um ${params.tissue_hierarchy_context_field_um} \
      --minimum-field-coverage ${params.tissue_hierarchy_minimum_field_coverage} --max-window-pixels ${params.tissue_hierarchy_max_window_pixels} \
      --batch ${params.tissue_hierarchy_feature_batch} --device '${params.tissue_hierarchy_device}' \
      --torch-threads ${task.cpus} --seed ${params.tissue_hierarchy_seed} --outdir features
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p features/local features/context
    printf '{"stub":true,"sample_id":"${sample_id}","scientific_claim":"No model inference or valid feature vectors produced"}\n' > features/hierarchy_features_summary.json
    printf '{"stub":true}\n' > features/embedding_metadata.json
    printf '{"stub":true}\n' > features/hierarchy_feature_request.json
    for block in local context; do
      printf 'cell_id,cx,cy,observation_type,source_mpp,extraction_tile_size\n' > "features/\$block/feature_rows.csv"
      printf '{"stub":true,"observations_encoded":0}\n' > "features/\$block/embedding_manifest.json"
      touch "features/\$block/features.npy"
    done
    """
}
