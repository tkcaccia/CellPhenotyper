/*
 * CPU-only continuation of one completed Virchow2/KODAMA slide.
 * BUILD admits every frozen token case and direct-fit KODAMA receipt;
 * BORA runs only after BUILD_COMPLETE; project export is optional and runs
 * only after CHAIN_COMPLETE. No process submits, retries or queues GPU cases.
 */
nextflow.enable.dsl = 2

process BUILD_BORA_GLOBAL_MANIFEST {
    label 'bora_global_cpu'
    cpus 2
    memory "${params.containsKey('bora_fullslide_memory_gb') ? params.bora_fullslide_memory_gb : 12} GB"
    time '4h'
    errorStrategy 'terminate'
    maxRetries 0
    maxForks 1
    container { params.bora_fullslide_cpu_sif }
    input:
    tuple path(input_spec), val(manifest_dir)
    output:
    tuple val(manifest_dir), path('BUILD_COMPLETE.json')
    script:
    """
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    python3 '${projectDir}/bin/build_bora_global_manifest.py' \
      --inputs "\$PWD/${input_spec}" --output-dir '${manifest_dir.toString().replace("'", "'\"'\"'")}' > build-action-receipt.json
    cp '${manifest_dir.toString().replace("'", "'\"'\"'")}/BUILD_COMPLETE.json' BUILD_COMPLETE.json
    """
    stub:
    """
    printf '%s\n' '{"schema":"nextflow-stub-only","complete":false}' > BUILD_COMPLETE.json
    """
}

process RUN_ADMITTED_BORA_GLOBAL {
    label 'bora_global_cpu'
    cpus 2
    memory "${params.containsKey('bora_fullslide_memory_gb') ? params.bora_fullslide_memory_gb : 12} GB"
    time '25h'
    errorStrategy 'terminate'
    maxRetries 0
    maxForks 1
    container { params.bora_fullslide_cpu_sif }
    input:
    tuple val(manifest_dir), path(build_receipt)
    output:
    tuple val(manifest_dir), path('CHAIN_COMPLETE.json')
    script:
    def groups = params.containsKey('bora_fullslide_required_groups') && params.bora_fullslide_required_groups != null ?
        "--required-group-count ${params.bora_fullslide_required_groups}" : ''
    """
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    python3 '${projectDir}/bin/bora_fullslide_action.py' \
      --manifest '${manifest_dir.toString().replace("'", "'\"'\"'")}/manifest.json' \
      --build-receipt "\$PWD/${build_receipt}" \
      --output '${params.bora_fullslide_output.toString().replace("'", "'\"'\"'")}' \
      --bora-executable 'bora' ${groups} > bora-action-receipt.json
    cp '${params.bora_fullslide_output.toString().replace("'", "'\"'\"'")}/CHAIN_COMPLETE.json' CHAIN_COMPLETE.json
    """
    stub:
    """
    printf '%s\n' '{"schema":"nextflow-stub-only","complete":false}' > CHAIN_COMPLETE.json
    """
}

process EXPORT_ADMITTED_BORA_PROJECT {
    label 'bora_global_cpu'
    cpus 2
    memory '4 GB'
    time '4h'
    errorStrategy 'terminate'
    maxRetries 0
    maxForks 1
    container { params.bora_fullslide_cpu_sif }
    input:
    tuple val(manifest_dir), path(chain_receipt), path(base_project)
    output:
    path 'PROJECT_EXPORT_COMPLETE.json'
    script:
    """
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    python3 '${projectDir}/bin/export_bora_global_project.py' \
      --base-project "\$PWD/${base_project}" \
      --manifest '${manifest_dir.toString().replace("'", "'\"'\"'")}/manifest.json' \
      --bora-output '${params.bora_fullslide_output.toString().replace("'", "'\"'\"'")}' \
      --output '${params.bora_fullslide_project_output.toString().replace("'", "'\"'\"'")}' > PROJECT_EXPORT_COMPLETE.json
    """
    stub:
    """
    printf '%s\n' '{"schema":"nextflow-stub-only","complete":false}' > PROJECT_EXPORT_COMPLETE.json
    """
}

workflow {
    def required = ['bora_fullslide_input_spec', 'bora_fullslide_manifest_output',
                    'bora_fullslide_output', 'bora_fullslide_cpu_sif']
    def missing = required.findAll { !params.containsKey(it) || !params[it]?.toString()?.trim() }
    if (missing) error "Missing whole-field Bora orchestration parameters: ${missing.join(', ')}"
    def nonAbsolute = required.findAll { !new File(params[it].toString()).isAbsolute() }
    if (nonAbsolute) error "Whole-field Bora paths must be absolute: ${nonAbsolute.join(', ')}"
    if (!new File(params.bora_fullslide_input_spec.toString()).isFile()) error 'Versioned Bora input specification is missing.'
    if (!new File(params.bora_fullslide_cpu_sif.toString()).isFile()) error 'Qualified CPU SIF is missing; the GPU SIF is incompatible.'
    for (key in ['bora_fullslide_manifest_output', 'bora_fullslide_output']) {
        def target = new File(params[key].toString())
        if (target.exists() || !target.parentFile.isDirectory()) error "${key} must be a new path with an existing parent."
    }
    if (params.compute_device?.toString()?.toLowerCase() != 'cpu') error 'Global Bora orchestration requires --compute_device cpu.'
    def inputSpec = new groovy.json.JsonSlurper().parse(new File(params.bora_fullslide_input_spec.toString()))
    if (inputSpec.schema != 'cellphenotyper-bora-global-inputs-v1' || inputSpec.evidence_class != 'verified_gpu_inventory') {
        error 'Real-slide orchestration requires versioned verified GPU inventory input evidence.'
    }
    def groups = params.bora_fullslide_required_groups == null ? '' : params.bora_fullslide_required_groups.toString().trim()
    if (groups && (!(groups ==~ /\d+/) || groups.toInteger() < 2 || groups.toInteger() > 32 ||
                   inputSpec.required_group_count != groups.toInteger())) {
        error 'Requested groups must be a 2..32 direct-fit count matching the input specification.'
    }
    def requestedMemory = params.containsKey('bora_fullslide_memory_gb') ? params.bora_fullslide_memory_gb.toString().toDouble() : 12.0d
    if (requestedMemory < inputSpec.limits.memory_budget_bytes / (1024.0d * 1024.0d * 1024.0d) || requestedMemory > 12.0d) {
        error 'Bora process memory must cover its bounded manifest and cannot exceed 12 GiB.'
    }
    def base = params.containsKey('bora_fullslide_base_project') ? params.bora_fullslide_base_project?.toString()?.trim() : ''
    def exported = params.containsKey('bora_fullslide_project_output') ? params.bora_fullslide_project_output?.toString()?.trim() : ''
    if (!!base != !!exported) error 'Supply both an existing base project and a new project output, or neither.'
    if (base && (!new File(base).isAbsolute() || !new File(base).isFile() ||
                 !new File(exported).isAbsolute() || new File(exported).exists() ||
                 !new File(exported).parentFile.isDirectory())) {
        error 'Project export needs an existing absolute base project and a new absolute output path.'
    }
    BUILD_BORA_GLOBAL_MANIFEST(Channel.value(tuple(file(params.bora_fullslide_input_spec),
                                                   params.bora_fullslide_manifest_output.toString())))
    RUN_ADMITTED_BORA_GLOBAL(BUILD_BORA_GLOBAL_MANIFEST.out)
    if (base) {
        def exportInput = RUN_ADMITTED_BORA_GLOBAL.out.map { manifest_dir, chain_receipt ->
            tuple(manifest_dir, chain_receipt, file(base))
        }
        EXPORT_ADMITTED_BORA_PROJECT(exportInput)
    }
}
