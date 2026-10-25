/*
 * Explicit CPU continuation for three direct KODAMA groups and contextual Bora.
 * Full registered Virchow2 PCA tokens must already exist. This entrypoint
 * never invokes main.nf's older tissue-cluster route or a GPU job.
 */
nextflow.enable.dsl = 2

process BORA_CONTEXTUAL_FULLSLIDE_CPU {
    executor 'local'
    cpus 2
    memory '8 GB'
    time '25h'
    errorStrategy 'terminate'
    maxRetries 0
    maxForks 1

    input:
    val command

    output:
    path 'RUN_COMPLETE.json'

    script:
    def folder = params.bora_contextual_output.toString().replace("'", "'\"'\"'")
    """
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export CUDA_VISIBLE_DEVICES=-1
    export HIP_VISIBLE_DEVICES=-1
    export ROCR_VISIBLE_DEVICES=-1
    export OMP_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    ${command}
    cp '${folder}/RUN_COMPLETE.json' RUN_COMPLETE.json
    """

    stub:
    """
    printf '%s\n' '{"schema":"nextflow-stub-only","complete":false}' > RUN_COMPLETE.json
    """
}

workflow {
    def required = ['bora_contextual_manifest', 'bora_contextual_source',
                    'bora_contextual_source_sha256', 'bora_contextual_output']
    def missing = required.findAll { !params.containsKey(it) ||
        !params[it]?.toString()?.trim() }
    if (missing) error "Missing contextual Bora inputs: ${missing.join(', ')}"
    if (!(params.bora_contextual_source_sha256.toString() ==~ /[0-9a-f]{64}/)) {
        error 'Bora source pin must be a lowercase SHA-256.'
    }
    for (name in ['bora_contextual_manifest', 'bora_contextual_source',
                  'bora_contextual_output']) {
        if (!new File(params[name].toString()).isAbsolute()) {
            error "${name} must be absolute."
        }
    }
    if (!new File(params.bora_contextual_manifest.toString()).isFile()) {
        error 'Contextual Bora manifest is missing.'
    }
    if (!new File(params.bora_contextual_source.toString(),
                  'src/bora/kodama_bora.py').isFile()) {
        error 'Pinned Bora source lacks the three-group contextual API.'
    }
    if (new File(params.bora_contextual_output.toString()).exists()) {
        error 'Use a fresh contextual Bora result folder; no overwrite or retry.'
    }
    if ((workflow.profile ?: '').toString().tokenize(',').any {
            it.trim().toLowerCase() in ['singularity', 'docker'] }) {
        error 'This continuation requires its pinned local CPU Python runtime.'
    }
    def q = { value -> "'${value.toString().replace("'", "'\"'\"'")}'" }
    def cmd = "python3 ${q(projectDir + '/bin/bora_contextual_fullslide.py')} run" +
        " --manifest ${q(params.bora_contextual_manifest)}" +
        " --bora-source ${q(params.bora_contextual_source)}" +
        " --expected-bora-source-sha256 ${q(params.bora_contextual_source_sha256)}" +
        " --output ${q(params.bora_contextual_output)}" +
        " --deadline-seconds ${params.containsKey('bora_contextual_deadline_seconds') ? params.bora_contextual_deadline_seconds : 14400}" +
        " --disk-budget-bytes ${params.containsKey('bora_contextual_disk_budget_bytes') ? params.bora_contextual_disk_budget_bytes : 34359738368}"
    BORA_CONTEXTUAL_FULLSLIDE_CPU(cmd)
}
