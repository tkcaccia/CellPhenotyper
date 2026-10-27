/*
 * Distinct local CPU continuation from a frozen RGB9/RGB19 replay.
 * No process in this DAG submits GPU work, retries, or invokes main.nf.
 * The Python action creates one durable result folder and one project file.
 */
nextflow.enable.dsl = 2

process BORA_CONTINUOUS_FULLSLIDE_CPU {
    label 'bora_continuous_cpu'
    executor 'local'
    cpus 2
    memory "${params.containsKey('bora_continuous_memory_gb') ? params.bora_continuous_memory_gb : 4} GB"
    time '25h'
    errorStrategy 'terminate'
    maxRetries 0
    maxForks 1
    input:
    val command
    output:
    path 'RUN_COMPLETE.json'
    path 'action-receipt.json'
    script:
    """
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export CUDA_VISIBLE_DEVICES=-1
    export HIP_VISIBLE_DEVICES=-1
    export ROCR_VISIBLE_DEVICES=-1
    export OMP_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    ${command} > action-receipt.json
    cp '${params.bora_continuous_output.toString().replace("'", "'\"'\"'")}/RUN_COMPLETE.json' RUN_COMPLETE.json
    """
    stub:
    """
    printf '%s\n' '{"format":"nextflow-stub-only","complete":false,"cpu_only":true}' > RUN_COMPLETE.json
    cp RUN_COMPLETE.json action-receipt.json
    """
}

workflow {
    if ((workflow.profile ?: '').toString().tokenize(',').any { it.trim() in ['singularity', 'docker'] }) {
        error 'This local Python continuation does not use container profiles; pin the running Python environment.'
    }
    def required = ['bora_continuous_replay', 'bora_continuous_registration',
                    'bora_continuous_support_tiff', 'bora_continuous_base_project',
                    'bora_continuous_source', 'bora_continuous_output',
                    'bora_continuous_source_sha256', 'bora_continuous_runtime_sha256']
    def missing = required.findAll { !params.containsKey(it) || !params[it]?.toString()?.trim() }
    if (missing) error "Missing continuous Bora parameters: ${missing.join(', ')}"
    def inputKeys = required.take(6)
    def nonAbsolute = inputKeys.findAll { !new File(params[it].toString()).isAbsolute() }
    if (nonAbsolute) error "Continuous Bora paths must be absolute: ${nonAbsolute.join(', ')}"
    if (params.compute_device?.toString()?.toLowerCase() != 'cpu') {
        error 'Continuous Bora requires --compute_device cpu.'
    }
    for (key in ['bora_continuous_source_sha256', 'bora_continuous_runtime_sha256']) {
        if (!(params[key].toString() ==~ /[0-9a-f]{64}/)) error "${key} must be a lowercase SHA-256."
    }
    for (key in ['bora_continuous_registration', 'bora_continuous_support_tiff',
                 'bora_continuous_base_project']) {
        if (!new File(params[key].toString()).isFile()) error "${key} is missing."
    }
    if (!new File(params.bora_continuous_replay.toString()).isDirectory() ||
        !new File(params.bora_continuous_replay.toString(), 'TOKEN_REPLAY_COMPLETE.json').isFile()) {
        error 'Frozen RGB9/RGB19 replay is missing.'
    }
    if (!new File(params.bora_continuous_source.toString(), 'src/bora').isDirectory() ||
        !new File(params.bora_continuous_source.toString(), 'pyproject.toml').isFile()) {
        error 'Pinned staged Bora source is missing.'
    }
    def output = new File(params.bora_continuous_output.toString())
    if (output.exists() || !output.parentFile.isDirectory()) {
        error 'Continuous output must be a new path with an existing parent.'
    }
    def sif = params.containsKey('bora_continuous_runtime_sif') ?
              params.bora_continuous_runtime_sif?.toString()?.trim() : ''
    if (sif && (!new File(sif).isAbsolute() || !new File(sif).isFile())) {
        error 'Pinned optional SIF must be an existing absolute path.'
    }
    def cores = params.containsKey('bora_continuous_core_pixels') ?
                params.bora_continuous_core_pixels.toString().toInteger() : 2048
    def halo = params.containsKey('bora_continuous_halo_pixels') ?
               params.bora_continuous_halo_pixels.toString().toInteger() : 128
    def memoryGb = params.containsKey('bora_continuous_memory_gb') ?
                   params.bora_continuous_memory_gb.toString().toDouble() : 4.0d
    def diskGb = params.containsKey('bora_continuous_disk_gb') ?
                 params.bora_continuous_disk_gb.toString().toDouble() : 16.0d
    if (cores < 1 || cores > 3072 || halo < 1 || halo > 512 ||
        memoryGb < 1 || memoryGb > 12 || diskGb < 4 || diskGb > 256) {
        error 'Continuous shard and resource limits are outside the audited CPU bounds.'
    }
    def quote = { value -> "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    def args = ['python3', "${projectDir}/bin/bora_continuous_fullslide_action.py", 'run',
                '--replay-folder', params.bora_continuous_replay,
                '--registration', params.bora_continuous_registration,
                '--registered-support-tiff', params.bora_continuous_support_tiff,
                '--base-project', params.bora_continuous_base_project,
                '--bora-source', params.bora_continuous_source,
                '--expected-bora-sha256', params.bora_continuous_source_sha256,
                '--expected-runtime-sha256', params.bora_continuous_runtime_sha256,
                '--output', params.bora_continuous_output,
                '--core-pixels', cores, '--halo-pixels', halo,
                '--memory-budget-bytes', (memoryGb * 1024d * 1024d * 1024d).longValue(),
                '--disk-budget-bytes', (diskGb * 1024d * 1024d * 1024d).longValue()]
    if (sif) args += ['--runtime-sif', sif]
    BORA_CONTINUOUS_FULLSLIDE_CPU(Channel.value(args.collect(quote).join(' ')))
}
