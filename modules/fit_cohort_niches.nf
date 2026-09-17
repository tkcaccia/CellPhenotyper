process FIT_COHORT_NICHES {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'fit_cohort_niches', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([profiles]) }
    tag "cohort:${sample_ids.size()} specimens"
    label 'compute_medium'
    publishDir "${params.outdir_base}/25_cohort_niches", mode: (params.publish_dir_mode ?: 'copy'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'cohort_niches') }
    memory { TaskRuntime.memory(runtime_plan, 'cohort_niches') }
    time '24h'

    input:
    tuple val(sample_ids), path(profiles, stageAs: 'specimen??/*')
    val(runtime_plan)

    output:
    path('cohort_niches'), emit: cohort

    script:
    if (sample_ids.size() < 2 || sample_ids.unique(false).size() != sample_ids.size())
        error 'Cohort niche discovery requires at least two distinct specimens.'
    def settings = [max_k: params.cohort_niches_max_k, fixed_k: params.cohort_niches_fixed_k,
                    seed: params.cohort_niches_seed, repeats: params.cohort_niches_repeats,
                    fit_limit: params.cohort_niches_fit_limit, max_working_mb: params.cohort_niches_max_working_mb]
    if (!(String.valueOf(params.cell_neighborhood_row_batch_size) ==~ /[0-9]+/) ||
        (params.cell_neighborhood_row_batch_size as BigInteger) < 1)
        error 'cell_neighborhood_row_batch_size must be a positive integer'
    settings.each { name, value ->
        if (!(value.toString() ==~ /[0-9]+/)) error "cohort_niches_${name} must be an integer"
    }
    if ((settings.max_k as long) < 2 ||
        (settings.fixed_k as long) > (settings.max_k as long) || (settings.repeats as long) < 2 ||
        (settings.fit_limit as long) < 10 || (settings.max_working_mb as long) < 1)
        error 'Invalid cohort niche settings: max_k>=2, fixed_k=0..max_k, repeats>=2, fit_limit>=10, max_working_mb>=1 required.'
    if ((settings.max_working_mb as BigDecimal) * 1024 * 1024 > task.memory.toBytes())
        error 'cohort_niches_max_working_mb exceeds the task memory allocation; reduce the explicit working budget or increase the permitted runtime memory.'
    def profileFlags = profiles.collect { "'" + it.toString().replace("'", "'\\''") + "'" }.join(' ')
    def fixedFlag = (settings.fixed_k as long) > 0 ? "--fixed-k ${settings.fixed_k}" : ''
    def featureWeights = String.valueOf(params.cell_neighborhood_feature_weights ?: '{}').replace("'", "'\\''")
    """
    set -euo pipefail
    source "${projectDir}/bin/activate_source_python.sh"
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus} LOKY_MAX_CPU_COUNT=${task.cpus}
    "${params.cell_atlas_python}" "${projectDir}/bin/fit_cohort_niches.py" \
      --profiles ${profileFlags} --outdir cohort_niches \
      --max-k ${settings.max_k} ${fixedFlag} --seed ${settings.seed} \
      --repeats ${settings.repeats} --fit-limit ${settings.fit_limit} \
      --max-working-mb ${settings.max_working_mb} --feature-weights '${featureWeights}' \
      --row-batch-size ${params.cell_neighborhood_row_batch_size}
    test -f cohort_niches/cohort_niche_model.json
    test -f cohort_niches/cohort_niche_assignments.parquet
    test -f cohort_niches/cohort_niche_summary.json
    test -f cohort_niches/cohort_niches_completion.json
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p cohort_niches
    printf '{"stub":true}' > cohort_niches/cohort_niche_model.json
    printf '{"stub":true}' > cohort_niches/cohort_niche_summary.json
    printf '{"stub":true}' > cohort_niches/cohort_niches_completion.json
    touch cohort_niches/cohort_niche_assignments.parquet
    """
}
