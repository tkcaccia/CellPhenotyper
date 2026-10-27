/*
 * One source-bound, CPU-only whole-field Bora run after a complete Virchow2
 * token inventory and direct-fit KODAMA receipt have been published.
 * There is no GPU process, retry, or downstream GPU successor in this DAG.
 * The caller must use -profile singularity with a separately qualified CPU
 * SIF and --compute_device cpu. Outputs are written to one new absolute path.
 */
nextflow.enable.dsl = 2

process BORA_GLOBAL_FULLSLIDE {
    label 'bora_global_cpu'
    cpus 2
    memory "${params.containsKey('bora_fullslide_memory_gb') ? params.bora_fullslide_memory_gb : 12} GB"
    time '25h'
    errorStrategy 'terminate'
    maxRetries 0
    maxForks 1
    container { params.bora_fullslide_cpu_sif }
    input:
    val command
    output:
    path 'CHAIN_COMPLETE.json'
    path 'bora-action-receipt.json'
    script:
    """
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    ${command} > bora-action-receipt.json
    cp '${params.bora_fullslide_output.toString().replace("'", "'\"'\"'")}/CHAIN_COMPLETE.json' CHAIN_COMPLETE.json
    """
    stub:
    """
    printf '%s\n' '{"schema":"nextflow-stub-only","complete":false}' > CHAIN_COMPLETE.json
    cp CHAIN_COMPLETE.json bora-action-receipt.json
    """
}

workflow {
    def algorithm = params.containsKey('bora_fullslide_algorithm') ? params.bora_fullslide_algorithm?.toString()?.trim() : ''
    if (algorithm != 'bora-global-hard-group-v1') {
        error 'bora_fullslide.nf is the legacy hard-group Bora route. Declare --bora_fullslide_algorithm bora-global-hard-group-v1 only for explicitly anonymous exploratory group review. Corrected semantic H&E-refined Bora requires the separate source-bound token/refined-chain project export; this workflow cannot publish it.'
    }
    def required = ['bora_fullslide_manifest', 'bora_fullslide_output', 'bora_fullslide_cpu_sif']
    def missing = required.findAll { !params.containsKey(it) || !params[it]?.toString()?.trim() }
    if (missing) error "Missing whole-field Bora parameters: ${missing.join(', ')}"
    def nonAbsolute = required.findAll { !new File(params[it].toString()).isAbsolute() }
    if (nonAbsolute) error "Whole-field Bora paths must be absolute: ${nonAbsolute.join(', ')}"
    if (!new File(params.bora_fullslide_manifest.toString()).isFile()) error 'Bora manifest is missing.'
    if (new File(params.bora_fullslide_output.toString()).exists()) error 'Bora output already exists; no overwrite or automatic retry.'
    if (!new File(params.bora_fullslide_output.toString()).parentFile.isDirectory()) error 'Create the Bora output parent directory before container binding.'
    if (!new File(params.bora_fullslide_cpu_sif.toString()).isFile()) error 'The qualified Bora CPU SIF is missing; do not use the GPU SIF.'
    if (params.compute_device?.toString()?.toLowerCase() != 'cpu') error 'Whole-field Bora requires --compute_device cpu.'
    def manifest = new groovy.json.JsonSlurper().parse(new File(params.bora_fullslide_manifest.toString()))
    if (manifest.algorithm_id != algorithm) error 'Bora manifest algorithm_id must match the explicit hard-group review intent; semantic/refined Bora cannot run through this workflow.'
    def requestedMemory = params.containsKey('bora_fullslide_memory_gb') ? params.bora_fullslide_memory_gb.toString().toDouble() : 12.0d
    if (requestedMemory < manifest.limits.memory_budget_bytes / (1024.0d * 1024.0d * 1024.0d) || requestedMemory > 12.0d) {
        error 'Bora process memory must cover the manifest memory budget and cannot exceed 12 GiB.'
    }
    def groups = params.containsKey('bora_fullslide_required_groups') && params.bora_fullslide_required_groups != null ?
        params.bora_fullslide_required_groups.toString().trim() : ''
    if (groups && !(groups ==~ /\d+/) || (groups && (groups.toInteger() < 2 || groups.toInteger() > 32))) {
        error 'bora_fullslide_required_groups must be an integer from 2 to 32.'
    }
    def quote = { value -> "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    def args = ['python3', "${projectDir}/bin/bora_fullslide_action.py",
                '--manifest', params.bora_fullslide_manifest,
                '--output', params.bora_fullslide_output,
                '--bora-executable', params.containsKey('bora_fullslide_executable') ? params.bora_fullslide_executable : 'bora']
    if (groups) args += ['--required-group-count', groups]
    BORA_GLOBAL_FULLSLIDE(Channel.value(args.collect(quote).join(' ')))
}
