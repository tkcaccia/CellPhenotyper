include { RUN_PATHSEGMENTOR_SEMANTICS } from '../modules/run_pathsegmentor_semantics'

workflow RUN_PATHSEGMENTOR_INFERENCE {
    take:
    prepared_ch
    run_current
    runtime_plan

    main:
    def runCurrent = run_current as boolean
    def bundleCh
    if (runCurrent) {
        def promptPanel = params.pathsegmentor_prompt_panel.toString().startsWith('/')
            ? params.pathsegmentor_prompt_panel
            : "${projectDir}/${params.pathsegmentor_prompt_panel}"
        RUN_PATHSEGMENTOR_SEMANTICS(
            prepared_ch,
            file(params.pathsegmentor_repo, checkIfExists: true),
            file(params.pathsegmentor_config, checkIfExists: true),
            file(params.pathsegmentor_checkpoint, checkIfExists: true),
            file(promptPanel, checkIfExists: true),
            runtime_plan,
        )
        bundleCh = RUN_PATHSEGMENTOR_SEMANTICS.out.semantic_bundle
    } else {
        bundleCh = prepared_ch.map { sample_id, _image, _tissue, _resolution ->
            def base = "${params.outdir_base}/09b_pathsegmentor/${sample_id}/pathsegmentor_${sample_id}"
            tuple(sample_id,
                file("${base}/pathsegmentor_probabilities.ome.tif", checkIfExists: true),
                file("${base}/pathsegmentor_manifest.json", checkIfExists: true))
        }
    }

    emit:
    bundle = bundleCh
}
