process RUN_CELLVITPP {
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/03c_cellvitpp/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.cellvit_cpus as int)) }
    memory { "${Math.max(1, Math.min(params._executor_max_memory_gb as int, Math.max(8, params.cellvit_memory_gb as int)))} GB" }
    time { params.cellvit_time as String }
    errorStrategy 'terminate'
    maxRetries 0

    input:
    tuple val(sample_id), path(crop_tif), path(shift_json), path(clean_tissue_mask)
    val resolved_settings

    output:
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_cells.json"), emit: cells_json
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_metadata.json"), emit: metadata_json
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_cell_features.manifest.json"), emit: feature_manifest
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_cell_features.csv.gz"), emit: cell_features
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_feature_cells.json"), emit: feature_cells
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_cell_classification.json"), emit: classification_json
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_cell_classification.csv.gz"), emit: classification_csv
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_model_preflight.json"), emit: model_preflight
    tuple val(sample_id), path("cellvit_${sample_id}"), emit: cellvit_dir

    script:
    def shellQuote = { value -> "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    def scriptPath = "${projectDir}/${params.cellvit_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([scriptPath, "${projectDir}/bin/grandqc_mask.py", "${projectDir}/bin/cellvit_models.py", "${projectDir}/bin/hardware_runtime.py", "${projectDir}/bin/model_provenance.py", "${projectDir}/resources/cellvit_model_catalog.json"])
    def configuredExecutable = shellQuote((params.cellvit_executable ?: '').toString().trim())
    def configuredPython = shellQuote((params.cellvit_python ?: '').toString().trim())
    def ampFlag = params.cellvit_amp ? '--amp' : ''
    if (!(resolved_settings instanceof Map) || !(resolved_settings.enabled instanceof Boolean) ||
        !(resolved_settings.taxonomies instanceof List) || !resolved_settings.taxonomies ||
        resolved_settings.classification_device != 'cpu' ||
        !(resolved_settings.default_mpp instanceof Number) || !resolved_settings.compute_device) {
        error 'RUN_CELLVITPP requires explicit validated cell-identification settings; mutable internal params are not a settings input.'
    }
    def requestedTaxonomies = resolved_settings.enabled ? resolved_settings.taxonomies : ['pannuke']
    if (requestedTaxonomies == ['all']) {
        requestedTaxonomies = ['pannuke', 'consep', 'lizard', 'nucls_main', 'nucls_super', 'ocelot', 'midog', 'panoptils']
    }
    def taxonomyArgs = requestedTaxonomies.collect { "'" + it.toString().replace("'", "'\"'\"'") + "'" }.join(' ')
    def classifierCache = (params.cellvit_classifier_cache ?: '').toString().trim()
    def classifierCacheFlag = classifierCache ? "--classifier-cache '" + classifierCache.replace("'", "'\"'\"'") + "'" : ''
    def classificationDevice = resolved_settings.classification_device
    def defaultMpp = resolved_settings.default_mpp
    def memoryMb = task.memory ? Math.max(1024, task.memory.toMega() as int) : Math.max(1024, Math.min(params._executor_max_memory_gb as int, params.cellvit_memory_gb as int) * 1024)
    """
    set -euo pipefail
    echo "[INFO] CellViT++ wrapper code fingerprint: ${codeFingerprint}"
    test "${resolved_settings.compute_device}" = "gpu" || { echo "CellViT++ requires a resolved GPU runtime" >&2; exit 2; }
    # CELLVIT_RUNTIME_BINDING_BEGIN
    # The container's default python belongs to StarDist. Bind the wrapper to
    # the configured CellViT console script's own environment instead.
    cellvit_executable=${configuredExecutable}
    cellvit_python=${configuredPython}
    if [[ "\$cellvit_executable" != */* ]]; then
      cellvit_executable="\$(command -v -- "\$cellvit_executable")" || { echo "Configured CellViT executable was not found on PATH" >&2; exit 2; }
    fi
    test -x "\$cellvit_executable" || { echo "Configured CellViT executable is missing or not executable: \$cellvit_executable" >&2; exit 2; }
    if [[ -z "\$cellvit_python" ]]; then
      cellvit_python="\$(dirname -- "\$cellvit_executable")/python"
    elif [[ "\$cellvit_python" != */* ]]; then
      cellvit_python="\$(command -v -- "\$cellvit_python")" || { echo "Configured --cellvit_python was not found on PATH" >&2; exit 2; }
    fi
    test -x "\$cellvit_python" || { echo "CellViT wrapper Python is missing or not executable: \$cellvit_python. Set --cellvit_python to the Python from the CellViT environment." >&2; exit 2; }
    cellvit_runtime_bin="\$(dirname -- "\$cellvit_python")"
    export PATH="\$cellvit_runtime_bin:\$PATH"
    export LD_LIBRARY_PATH="\$cellvit_runtime_bin/../lib:\${LD_LIBRARY_PATH:-}"
    echo "[INFO] CellViT++ wrapper Python: \$cellvit_python"
    "\$cellvit_python" ${shellQuote(scriptPath)} --preflight-only --executable "\$cellvit_executable" ${ampFlag}
    # CELLVIT_RUNTIME_BINDING_END
    export CELLVIT_CACHE=${shellQuote(params.cellvit_cache_dir)}
    mkdir -p "\$CELLVIT_CACHE"
    "\$cellvit_python" ${shellQuote(scriptPath)} \
      --image ${shellQuote(crop_tif)} --shift ${shellQuote(shift_json)} --outdir ${shellQuote("cellvit_${sample_id}")} \
      --clean-tissue-mask ${shellQuote(clean_tissue_mask)} \
      --executable "\$cellvit_executable" --model ${shellQuote(params.cellvit_model)} \
      --taxonomy ${shellQuote(params.cellvit_taxonomy)} --gpu ${params.cellvit_gpu} \
      --taxonomies ${taxonomyArgs} ${classifierCacheFlag} --classification-device '${classificationDevice}' \
      --batch-size ${params.cellvit_batch_size} --cpus ${task.cpus} --memory-mb ${memoryMb} \
      --ray-workers ${params.cellvit_ray_workers} --ray-worker-cpus ${params.cellvit_ray_worker_cpus} \
      --default-mpp ${defaultMpp} ${ampFlag}
    """

    stub:
    """
    mkdir -p "cellvit_${sample_id}"
    printf '{"pipeline_metadata":{"model":"HIPT"},"cells":[]}' > "cellvit_${sample_id}/cellvit_cells.json"
    printf '{"model_provenance":{"used_model":false}}\n' > "cellvit_${sample_id}/cellvit_metadata.json"
    printf '{"stub_only":true}\n' > "cellvit_${sample_id}/cellvit_cell_features.manifest.json"
    printf 'cell_id,source_id,x,y\n' | gzip -n > "cellvit_${sample_id}/cellvit_cell_features.csv.gz"
    printf '{"stub_only":true,"cells":[]}\n' > "cellvit_${sample_id}/cellvit_feature_cells.json"
    printf '{"stub_only":true,"models":[],"cells":[]}\n' > "cellvit_${sample_id}/cellvit_cell_classification.json"
    printf 'cell_id,model_id,status\n' | gzip -n > "cellvit_${sample_id}/cellvit_cell_classification.csv.gz"
    printf '{"stub_only":true,"models":[]}\n' > "cellvit_${sample_id}/cellvit_model_preflight.json"
    """
}
