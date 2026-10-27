/*
 * CPU-only CellPhenotyper continuation for the recovered token-local Bora
 * classifier and registered native H&E boundary refinement. This starts only
 * after complete Virchow2 token materialization and an independently marked,
 * passing same-slide tissue holdout. It never calls the hard-group Bora route.
 * A new Chiamaka execution still requires its own authorized outer lifecycle.
 */
nextflow.enable.dsl = 2

process BORA_REFINED_SEMANTIC_FULLSLIDE_CPU {
    label 'bora_refined_semantic_cpu'
    executor 'local'
    cpus 2
    memory "${params.containsKey('bora_refined_memory_gb') ? params.bora_refined_memory_gb : 3} GB"
    time '25h'
    errorStrategy 'terminate'
    maxRetries 0
    maxForks 1

    input:
    val command

    output:
    path 'RUN_COMPLETE.json'
    path 'experiment.cellphenotyper.json'
    path 'action-receipt.json'

    script:
    def result = params.bora_refined_output.toString().replace("'", "'\"'\"'")
    """
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export CUDA_VISIBLE_DEVICES=-1
    export HIP_VISIBLE_DEVICES=-1
    export ROCR_VISIBLE_DEVICES=-1
    export OMP_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    ${command} > action-receipt.json
    cp '${result}/RUN_COMPLETE.json' RUN_COMPLETE.json
    cp '${result}/experiment.cellphenotyper.json' experiment.cellphenotyper.json
    """

    stub:
    """
    printf '%s\n' '{"format":"nextflow-stub-only","complete":false,"review_only":true}' > RUN_COMPLETE.json
    cp RUN_COMPLETE.json experiment.cellphenotyper.json
    cp RUN_COMPLETE.json action-receipt.json
    """
}

workflow {
    def algorithm = params.containsKey('bora_refined_algorithm') ?
        params.bora_refined_algorithm?.toString()?.trim() : ''
    if (algorithm != 'bora-wsi-native-he-refined-semantic-chain-v1') {
        error 'Select --bora_refined_algorithm bora-wsi-native-he-refined-semantic-chain-v1. The legacy hard-group route cannot substitute for token-local, H&E-refined semantic Bora.'
    }
    if ((workflow.profile ?: '').toString().tokenize(',').any { it.trim().toLowerCase() in ['singularity', 'docker'] }) {
        error 'The refined Bora continuation requires its pinned local CPU Python runtime; container profiles are not qualified for this entrypoint.'
    }
    if (params.compute_device?.toString()?.trim()?.toLowerCase() != 'cpu') {
        error 'The refined Bora continuation requires --compute_device cpu.'
    }
    def paths = ['bora_refined_roi_spec', 'bora_refined_evaluation_folder',
                 'bora_refined_materialization_folder', 'bora_refined_registration',
                 'bora_refined_support_tiff', 'bora_refined_base_project',
                 'bora_refined_source', 'bora_refined_output']
    def pins = ['bora_refined_source_sha256', 'bora_refined_runtime_sha256']
    def missing = (paths + pins).findAll { !params.containsKey(it) || !params[it]?.toString()?.trim() }
    if (missing) error "Missing refined Bora parameters: ${missing.join(', ')}"
    def nonAbsolute = paths.findAll { !new File(params[it].toString()).isAbsolute() }
    if (nonAbsolute) error "Refined Bora paths must be absolute: ${nonAbsolute.join(', ')}"
    for (key in pins) {
        if (!(params[key].toString() ==~ /[0-9a-f]{64}/)) error "${key} must be a lowercase SHA-256."
    }
    def source = new File(params.bora_refined_source.toString())
    if (!new File(source, 'src/bora/wsi_semantic_refined_chain.py').isFile() ||
        !new File(source, 'src/bora/wsi_semantic_probability.py').isFile()) {
        error 'The pinned Bora source lacks the token-local native H&E-refined semantic chain; hard-group fallback is forbidden.'
    }
    for (key in ['bora_refined_roi_spec', 'bora_refined_registration',
                 'bora_refined_support_tiff', 'bora_refined_base_project']) {
        if (!new File(params[key].toString()).isFile()) error "${key} is missing."
    }
    if (!new File(params.bora_refined_evaluation_folder.toString(), 'EVALUATION.json').isFile() ||
        !new File(params.bora_refined_evaluation_folder.toString(), 'local_semantic_model.npz').isFile()) {
        error 'A saved expert-trained model and same-slide holdout evaluation are required.'
    }
    def evaluationFile = new File(params.bora_refined_evaluation_folder.toString(), 'EVALUATION.json')
    if (evaluationFile.length() > 16 * 1024 * 1024) {
        error 'The expert holdout evaluation is oversized.'
    }
    def evaluation = null
    try {
        evaluation = new groovy.json.JsonSlurper().parse(evaluationFile)
    } catch (Throwable ignored) {
        error 'The expert holdout evaluation is not valid JSON.'
    }
    def requiredGates = ['holdout_material', 'macro_iou', 'per_class_iou',
                         'per_class_recall', 'no_single_class_collapse']
    if (!(evaluation instanceof Map) || evaluation.complete != true ||
        evaluation.status != 'same_slide_holdout_passed_independent_patient_validation_pending' ||
        evaluation.same_slide_holdout != true ||
        evaluation.annotation_source != 'declared_independent_expert' ||
        !(evaluation.gates instanceof Map) ||
        evaluation.gates.keySet().toSet() != requiredGates.toSet() ||
        requiredGates.any { evaluation.gates[it] != true }) {
        error 'A passing independent-expert same-slide holdout is required; unsupported PM016 or transferred labels cannot publish tissue classes.'
    }
    def ids = evaluation.class_ids
    def confusion = evaluation.confusion
    def classes = evaluation.per_class
    def names = evaluation.class_names
    def polygons = evaluation.per_polygon
    def tokenShape = evaluation.token_shape_yx
    if (!(ids instanceof List) || ids.size() < 2 ||
        ids.any { !(it instanceof Integer) || it < 1 || it > 253 } ||
        ids != ids.toList().sort() || ids.toSet().size() != ids.size() ||
        !(names instanceof Map) || names.keySet().toSet() != ids.collect { it.toString() }.toSet() ||
        !(classes instanceof Map) || classes.keySet().toSet() != names.keySet().toSet() ||
        !(confusion instanceof List) || confusion.size() != ids.size() ||
        confusion.any { !(it instanceof List) || it.size() != ids.size() ||
                        it.any { count -> !(count instanceof Integer) || count < 0 } } ||
        !(polygons instanceof List) || !polygons ||
        !(tokenShape instanceof List) || tokenShape.size() != 2 ||
        tokenShape.any { !(it instanceof Integer) || it < 1 }) {
        error 'The expert holdout lacks complete class, confusion, polygon or token-grid evidence.'
    }
    def polygonRois = ids.collectEntries { [(it): [:]] }
    def polygonTotal = 0
    def allPolygonTotal = 0
    for (row in polygons) {
        def count = row instanceof Map ? row.retained_supported_interior_tokens : null
        if (!(row instanceof Map) || !(count instanceof Integer) || count < 0 ||
            !(row.class_id in ids) || !(row.roi_id instanceof String) || !row.roi_id.trim() ||
            !(row.split in ['train', 'holdout'])) {
            error 'The expert per-polygon ROI counts are invalid.'
        }
        allPolygonTotal += count
        if (row.split == 'holdout') {
            polygonTotal += count
            def rois = polygonRois[row.class_id]
            rois[row.roi_id] = (rois[row.roi_id] ?: 0) + count
        }
    }
    def total = confusion.collect { it.sum() }.sum()
    if (!(evaluation.holdout_tokens instanceof Integer) || total != evaluation.holdout_tokens ||
        total != polygonTotal || allPolygonTotal > tokenShape[0].longValue() * tokenShape[1].longValue()) {
        error 'The expert holdout token totals differ from registered polygon evidence.'
    }
    def ious = []
    def recomputedGates = [holdout_material: true, macro_iou: false,
        per_class_iou: true, per_class_recall: true, no_single_class_collapse: false]
    def predictedClasses = 0
    for (int i = 0; i < ids.size(); i++) {
        def cid = ids[i]
        def actual = confusion[i].sum()
        def predicted = (0..<ids.size()).sum { k -> confusion[k][i] }
        if (predicted > 0) predictedClasses++
        def tp = confusion[i][i]
        def union = actual + predicted - tp
        def classRow = classes[cid.toString()]
        def rois = polygonRois[cid]
        def close = { claimed, expected -> claimed instanceof Number &&
            Double.isFinite(claimed.doubleValue()) &&
            Math.abs(claimed.doubleValue() - expected) <= 1e-12 }
        if (actual < 1 || predicted < 1 || union < 1 ||
            !(classRow instanceof Map) || classRow.name != names[cid.toString()] ||
            !(classRow.support instanceof Integer) || classRow.support != actual ||
            classRow.holdout_roi_counts != rois || rois.values().sum() != actual ||
            !close(classRow.precision, predicted > 0 ? tp / (double) predicted : -1d) ||
            !close(classRow.recall, actual > 0 ? tp / (double) actual : -1d) ||
            !close(classRow.iou, union > 0 ? tp / (double) union : -1d)) {
            error 'The expert holdout class metrics or ROI counts differ from confusion.'
        }
        def iou = tp / (double) union
        def recall = tp / (double) actual
        ious << iou
        recomputedGates.holdout_material &= actual >= 100 && rois.values().count { it > 0 } >= 2
        recomputedGates.per_class_iou &= iou >= 0.60d
        recomputedGates.per_class_recall &= recall >= 0.70d
    }
    def macro = ious.sum() / ids.size()
    recomputedGates.macro_iou = macro >= 0.70d
    recomputedGates.no_single_class_collapse = predictedClasses >= 2
    if (!(evaluation.macro_iou instanceof Number) ||
        !Double.isFinite(evaluation.macro_iou.doubleValue()) ||
        Math.abs(evaluation.macro_iou.doubleValue() - macro) > 1e-12 ||
        evaluation.gates != recomputedGates ||
        evaluation.gate_thresholds != [holdout_tokens_per_class: 100,
            holdout_rois_per_class: 2, macro_iou: 0.70d,
            class_iou: 0.60d, class_recall: 0.70d]) {
        error 'The expert holdout gates or macro IoU disagree with confusion counts.'
    }
    if (!new File(params.bora_refined_materialization_folder.toString(), 'MATERIALIZATION_COMPLETE.json').isFile()) {
        error 'A complete, source-bound Virchow2 token materialization is required.'
    }
    def output = new File(params.bora_refined_output.toString())
    if (output.exists() || !output.parentFile.isDirectory()) {
        error 'Refined Bora output must be a new path beneath an existing parent; no overwrite or retry.'
    }
    def refinement = params.containsKey('bora_refined_refinement_spec') ?
        params.bora_refined_refinement_spec?.toString()?.trim() : ''
    if (refinement && (!new File(refinement).isAbsolute() || !new File(refinement).isFile())) {
        error 'An explicit H&E refinement spec must be an existing absolute JSON file.'
    }
    def cores = params.containsKey('bora_refined_core_pixels') ? params.bora_refined_core_pixels.toString().toInteger() : 2048
    def sourceHalo = params.containsKey('bora_refined_source_halo_pixels') ? params.bora_refined_source_halo_pixels.toString().toInteger() : 96
    def outputHalo = params.containsKey('bora_refined_output_halo_pixels') ? params.bora_refined_output_halo_pixels.toString().toInteger() : 3
    def memoryGb = params.containsKey('bora_refined_memory_gb') ? params.bora_refined_memory_gb.toString().toDouble() : 3.0d
    def diskGb = params.containsKey('bora_refined_disk_gb') ? params.bora_refined_disk_gb.toString().toDouble() : 64.0d
    def deadline = params.containsKey('bora_refined_deadline_seconds') ? params.bora_refined_deadline_seconds.toString().toInteger() : 86400
    if (cores < 1 || cores > 3072 || sourceHalo < 1 || sourceHalo > 512 ||
        outputHalo < 1 || outputHalo > 256 || memoryGb < 0.5d || memoryGb > 12.0d ||
        diskGb < 2.0d || diskGb > 1024.0d || deadline < 60 || deadline > 86400) {
        error 'Refined Bora shard, resource or deadline settings exceed the reviewed CPU bounds.'
    }
    def quote = { value -> "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    def args = ['python3', "${projectDir}/bin/bora_refined_semantic_fullslide_action.py", 'run',
                '--roi-spec', params.bora_refined_roi_spec,
                '--evaluation-folder', params.bora_refined_evaluation_folder,
                '--materialization-folder', params.bora_refined_materialization_folder,
                '--registration', params.bora_refined_registration,
                '--registered-support-tiff', params.bora_refined_support_tiff,
                '--base-project', params.bora_refined_base_project,
                '--bora-source', params.bora_refined_source,
                '--expected-bora-sha256', params.bora_refined_source_sha256,
                '--expected-runtime-sha256', params.bora_refined_runtime_sha256,
                '--output', params.bora_refined_output,
                '--compute-device', 'cpu',
                '--core-pixels', cores,
                '--source-halo-pixels', sourceHalo,
                '--output-halo-pixels', outputHalo,
                '--memory-budget-bytes', (memoryGb * 1024d * 1024d * 1024d).longValue(),
                '--disk-budget-bytes', (diskGb * 1024d * 1024d * 1024d).longValue(),
                '--deadline-seconds', deadline]
    if (refinement) args += ['--refinement-spec', refinement]
    BORA_REFINED_SEMANTIC_FULLSLIDE_CPU(Channel.value(args.collect(quote).join(' ')))
}
