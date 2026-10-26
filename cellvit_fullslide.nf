/*
 * Explicit one-action CellViT++ full-slide entry point.
 *
 * The run action executes official CellViT++ WSI inference sequentially on
 * bounded cores, verifies each GPU lifecycle and checkpoint, then publishes
 * the full-slide cell project (and an accepted Bora annotation when supplied).
 */
nextflow.enable.dsl = 2

process PLAN_CELLVIT_FULLSLIDE {
    label 'compute_medium'
    container '' // host action launches the pinned SIF where required
    cpus 2
    memory '4 GB'
    time '2h'
    errorStrategy 'terminate'
    maxRetries 0
    input:
    val command
    output:
    path 'cellvit-action-receipt.json'
    script:
    """
    set -euo pipefail
    ${command} > cellvit-action-receipt.json
    """
}

process FINALIZE_CELLVIT_FULLSLIDE {
    label 'compute_medium'
    container '' // host action launches the pinned SIF where required
    cpus 2
    memory '8 GB'
    time '8h'
    errorStrategy 'terminate'
    maxRetries 0
    input:
    val command
    output:
    path 'cellvit-action-receipt.json'
    script:
    """
    set -euo pipefail
    ${command} > cellvit-action-receipt.json
    """
}

process STORE_ONE_CELLVIT_FULLSLIDE_SHARD {
    label 'compute_medium'
    container '' // host action launches the pinned SIF where required
    cpus 2
    memory '8 GB'
    time '4h'
    errorStrategy 'terminate'
    maxRetries 0
    input:
    val command
    output:
    path 'cellvit-action-receipt.json'
    script:
    """
    set -euo pipefail
    ${command} > cellvit-action-receipt.json
    """
}

process RUN_CELLVIT_FULLSLIDE {
    label 'cellvit_host_lifecycle'
    container '' // lifecycle checks must observe the host, outside the SIF
    cpus 2
    memory '12 GB'
    time '14d'
    errorStrategy 'terminate'
    maxRetries 0
    input:
    val command
    output:
    path 'cellvit-action-receipt.json'
    script:
    """
    set -euo pipefail
    ${command} > cellvit-action-receipt.json
    """
}

workflow {
    def pval = { key, fallback ->
        def value = params.containsKey(key) ? params[key] : null
        value == null || (value instanceof CharSequence && !value.toString().trim()) ? fallback : value
    }
    def action = pval('cellvit_fullslide_action', '').toString().trim()
    if (!(action in ['preflight', 'plan', 'adopt', 'shard', 'pack', 'compact', 'finalize', 'run'])) {
        error 'Choose --cellvit_fullslide_action run, preflight, plan, adopt, pack, compact or finalize.'
    }
    if (action == 'shard') {
        error 'No compatible direct GPU supervisor is approved. Run the separately approved frozen Chiamaka one-case adapter, then import its checked result with bin/cellvit_approved_case_import.py.'
    }
    def required = ['cellvit_fullslide_image', 'cellvit_fullslide_shift',
                    'cellvit_fullslide_clean_mask', 'cellvit_fullslide_bundle',
                    'cellvit_classifier_cache']
    def missing = required.findAll { !pval(it, '').toString().trim() }
    if (missing) error "Missing full-slide parameters: ${missing.join(', ')}"
    def absolute = required.findAll { !new File(params[it].toString()).isAbsolute() }
    if (absolute) error "Full-slide paths must be absolute and stable across actions: ${absolute.join(', ')}"
    if (action in ['adopt', 'shard', 'pack', 'compact'] && !pval('cellvit_fullslide_shard_id', '').toString().trim()) {
        error 'One explicit --cellvit_fullslide_shard_id is required; no automatic successor exists.'
    }
    if (action == 'adopt') {
        for (key in ['cellvit_fullslide_adopt_from', 'cellvit_fullslide_adoption_cleanup_receipt',
                     'cellvit_fullslide_expected_checkpoint_sha256',
                     'cellvit_fullslide_adoption_cleanup_receipt_sha256']) {
            if (!pval(key, '').toString().trim()) error "Adoption requires ${key}"
        }
        for (key in ['cellvit_fullslide_adopt_from', 'cellvit_fullslide_adoption_cleanup_receipt']) {
            if (!new File(params[key].toString()).isAbsolute()) error "Adoption ${key} must be absolute"
        }
    }
    def quote = { value -> "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    def executable = pval('cellvit_executable', 'cellvit-inference').toString()
    def defaultPython = executable.contains('/') ? "${new File(executable).parent}/python" : 'python3'
    def wrapperPython = action == 'shard' ||
                        (action == 'run' && !pval('cellvit_fullslide_runtime_sif', '').toString().trim()) ?
                        pval('cellvit_python', defaultPython).toString() :
                        pval('cellvit_fullslide_cpu_python', 'python3').toString()
    def cache = pval('cellvit_cache_dir', '').toString().trim()
    def args = [
        wrapperPython,
        "${projectDir}/bin/cellvit_fullslide_action.py",
        '--action', action,
        '--image', params.cellvit_fullslide_image,
        '--shift', params.cellvit_fullslide_shift,
        '--clean-tissue-mask', params.cellvit_fullslide_clean_mask,
        '--bundle-dir', params.cellvit_fullslide_bundle,
        '--classifier-cache', params.cellvit_classifier_cache,
        '--model', 'SAM',
        '--executable', executable,
        '--name', pval('cellvit_fullslide_name', 'CellViT++ full-slide cell review'),
        '--tissue-id', pval('cellvit_fullslide_tissue_id', 'tissue'),
        '--reserve-gib', pval('cellvit_fullslide_reserve_gib', 40),
        '--case-budget-gib', pval('cellvit_fullslide_case_budget_gib', 4),
        '--case-timeout-seconds', pval('cellvit_fullslide_case_timeout_seconds', 7200),
        '--memory-mb', pval('cellvit_fullslide_memory_mb', 12288),
        '--app-memory-mb', pval('cellvit_fullslide_app_memory_mb', 8192),
        '--case-tasks-max', pval('cellvit_fullslide_case_tasks_max', 512),
        '--batch-size', pval('cellvit_fullslide_batch_size', 2),
        '--expected-shards', pval('cellvit_fullslide_expected_shards', 0)
    ]
    if (action == 'run') {
        if (!pval('cellvit_fullslide_gpu_uuid', '').toString().trim()) {
            error 'Full-slide run requires the assigned --cellvit_fullslide_gpu_uuid.'
        }
        args += ['--gpu-uuid', params.cellvit_fullslide_gpu_uuid]
        args += ['--max-new-shards', pval('cellvit_fullslide_max_new_shards', 0)]
    }
    if (action in ['run', 'preflight', 'plan', 'pack', 'compact']) {
        if (pval('cellvit_fullslide_runtime_sif', '').toString().trim()) {
            def runtimeKeys = ['cellvit_fullslide_runtime_sif_sha256',
                               'cellvit_fullslide_runtime_python',
                               'cellvit_fullslide_runtime_executable']
            if (runtimeKeys.any { !pval(it, '').toString().trim() }) {
                error 'Pinned SIF run requires its SHA-256 and in-image CellViT Python/executable.'
            }
            args += ['--runtime-sif', params.cellvit_fullslide_runtime_sif,
                     '--runtime-sif-sha256', params.cellvit_fullslide_runtime_sif_sha256,
                     '--runtime-python', params.cellvit_fullslide_runtime_python,
                     '--runtime-executable', params.cellvit_fullslide_runtime_executable]
            def pydeps = pval('cellvit_fullslide_runtime_pydeps', '').toString().trim()
            def pydepsSha = pval('cellvit_fullslide_runtime_pydeps_sha256', '').toString().trim()
            if (pydeps || pydepsSha) {
                if (!pydeps || !pydepsSha) error 'Pinned runtime pydeps require their recursive SHA-256.'
                args += ['--runtime-pydeps', params.cellvit_fullslide_runtime_pydeps,
                         '--runtime-pydeps-sha256', params.cellvit_fullslide_runtime_pydeps_sha256]
            }
        }
    }
    if (action == 'run') {
        def boraKeys = ['cellvit_fullslide_bora_geojson', 'cellvit_fullslide_bora_receipt',
                        'cellvit_fullslide_expected_bora_sha256', 'cellvit_fullslide_combined_output']
        if (boraKeys.any { pval(it, '').toString().trim() }) {
            if (boraKeys.any { !pval(it, '').toString().trim() }) {
                error 'Bora merge requires GeoJSON, receipt, exact SHA-256 and combined output together.'
            }
            args += ['--bora-geojson', params.cellvit_fullslide_bora_geojson,
                     '--bora-receipt', params.cellvit_fullslide_bora_receipt,
                     '--expected-bora-sha256', params.cellvit_fullslide_expected_bora_sha256,
                     '--combined-output', params.cellvit_fullslide_combined_output]
        }
    }
    if (action in ['adopt', 'shard', 'pack', 'compact']) {
        args += ['--shard-id', params.cellvit_fullslide_shard_id]
    }
    if (pval('cellvit_amp', true)) args += ['--amp']
    if (action == 'adopt') args += [
        '--adopt-from', params.cellvit_fullslide_adopt_from,
        '--expected-checkpoint-sha256', params.cellvit_fullslide_expected_checkpoint_sha256,
        '--adoption-cleanup-receipt', params.cellvit_fullslide_adoption_cleanup_receipt,
        '--adoption-cleanup-receipt-sha256', params.cellvit_fullslide_adoption_cleanup_receipt_sha256]
    def environment = cache ? "CELLVIT_CACHE=${quote(cache)} " : ''
    def command = environment + args.collect(quote).join(' ')
    if (action in ['plan', 'preflight']) PLAN_CELLVIT_FULLSLIDE(Channel.value(command))
    if (action in ['adopt', 'pack', 'compact']) STORE_ONE_CELLVIT_FULLSLIDE_SHARD(Channel.value(command))
    if (action == 'finalize') FINALIZE_CELLVIT_FULLSLIDE(Channel.value(command))
    if (action == 'run') RUN_CELLVIT_FULLSLIDE(Channel.value(command))
}
