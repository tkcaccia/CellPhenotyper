include { RUN_CELLVITPP } from './run_cellvitpp'
include { REVIEW_CELL_SEGMENTATION } from './review_cell_segmentation'

workflow RUN_CELL_IDENTIFICATION {
    take:
    crop_roi_ch
    shift_json_ch
    tissue_mask_ch
    stardist_labels_ch
    stardist_frame_ch
    review_enabled
    resolved_settings

    main:
    def cellvit_input_ch = crop_roi_ch.join(shift_json_ch).join(tissue_mask_ch)
      .map { sample_id, crop_tif, shift_json, clean_tissue_mask -> tuple(sample_id, crop_tif, shift_json, clean_tissue_mask) }
    RUN_CELLVITPP(cellvit_input_ch, resolved_settings)
    def qc_ch = Channel.empty()
    if (review_enabled) {
      def review_input_ch = crop_roi_ch.join(RUN_CELLVITPP.out.cells_json)
        .join(RUN_CELLVITPP.out.metadata_json).join(stardist_labels_ch).join(stardist_frame_ch)
        .map { sample_id, crop_tif, cells_json, cellvit_metadata, stardist_labels, stardist_frame ->
          tuple(sample_id, crop_tif, cells_json, cellvit_metadata, stardist_labels, stardist_frame)
        }
      REVIEW_CELL_SEGMENTATION(review_input_ch)
      qc_ch = REVIEW_CELL_SEGMENTATION.out.qc_json
    }

    emit:
    cells_json = RUN_CELLVITPP.out.cells_json
    metadata_json = RUN_CELLVITPP.out.metadata_json
    feature_manifest = RUN_CELLVITPP.out.feature_manifest
    cell_features = RUN_CELLVITPP.out.cell_features
    feature_cells = RUN_CELLVITPP.out.feature_cells
    classification_json = RUN_CELLVITPP.out.classification_json
    classification_csv = RUN_CELLVITPP.out.classification_csv
    model_preflight = RUN_CELLVITPP.out.model_preflight
    qc_json = qc_ch
}

workflow RUN_CELL_IDENTIFICATION_FULLSLIDE {
    take:
    full_image_ch
    full_mask_ch
    resolved_settings
    sample_count

    main:
    def fullslidePath = { String key, String sample_id ->
      def value = params[key].toString().trim()
      if (sample_count > 1 && !value.contains('{sample_id}')) {
        error "--${key} must contain {sample_id} for multiple images."
      }
      value.replace('{sample_id}', sample_id)
    }
    def fullslide_input_ch = full_image_ch.join(full_mask_ch)
      .map { sample_id, converted_image, grandqc_mask ->
        tuple(sample_id, converted_image, grandqc_mask,
          fullslidePath('cellvit_fullslide_image', sample_id),
          fullslidePath('cellvit_fullslide_shift', sample_id),
          fullslidePath('cellvit_fullslide_clean_mask', sample_id),
          fullslidePath('cellvit_fullslide_bundle', sample_id),
          (params.cellvit_fullslide_bora_geojson ?: '').toString().trim(),
          (params.cellvit_fullslide_bora_receipt ?: '').toString().trim(),
          (params.cellvit_fullslide_expected_bora_sha256 ?: '').toString().trim(),
          (params.cellvit_fullslide_combined_output ?: '').toString().trim())
      }
    RUN_CELLVIT_FULLSLIDE(fullslide_input_ch, resolved_settings)

    emit:
    completion_receipt = RUN_CELLVIT_FULLSLIDE.out.completion_receipt
    project_pointer = RUN_CELLVIT_FULLSLIDE.out.project_pointer
}

// Host-owned lifecycle control is required for the bounded sequential WSI
// campaign. The CellViT model itself runs inside its pinned Singularity SIF.
// No project pointer is emitted until all planned cores are verified.
process RUN_CELLVIT_FULLSLIDE {
    tag "${sample_id}"
    label 'host_only'
    cpus 2
    memory '12 GB'
    time { (params.cellvit_fullslide_time ?: '14d').toString() }
    errorStrategy 'terminate'
    maxRetries 0
    container ''

    input:
    tuple val(sample_id), path(converted_image), path(grandqc_mask),
          val(full_image), val(full_shift), val(full_mask), val(bundle_dir),
          val(bora_geojson), val(bora_receipt), val(bora_sha), val(combined_output)
    val resolved_settings

    output:
    tuple val(sample_id), path('cellvit_fullslide_complete.json'), emit: completion_receipt
    tuple val(sample_id), path('cellvit_project_pointer.json'), emit: project_pointer

    script:
    def q = { value -> "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    if (!(resolved_settings instanceof Map) || resolved_settings.enabled != true ||
        resolved_settings.input_scope != 'full_slide' || resolved_settings.taxonomies != ['all'] ||
        resolved_settings.compute_device != 'gpu') {
      error 'RUN_CELLVIT_FULLSLIDE requires validated full-slide, eight-head GPU settings.'
    }
    def cpuPython = (params.cellvit_fullslide_cpu_python ?: 'python3').toString()
    def source = "${projectDir}/bin/cellvit_fullslide_action.py"
    def pydeps = (params.cellvit_fullslide_runtime_pydeps ?: '').toString().trim()
    def pydepsHash = (params.cellvit_fullslide_runtime_pydeps_sha256 ?: '').toString().trim()
    if (!!pydeps != !!pydepsHash) error 'Full-slide runtime pydeps path and SHA-256 must be provided together.'
    def pydepsArgs = pydeps ? "--runtime-pydeps ${q(pydeps)} --runtime-pydeps-sha256 ${q(pydepsHash)}" : ''
    def expectedShards = (params.cellvit_fullslide_expected_shards ?: 0) as int
    def reserve = (params.cellvit_fullslide_reserve_gib ?: 40).toString()
    def budget = (params.cellvit_fullslide_case_budget_gib ?: 4).toString()
    def timeout = (params.cellvit_fullslide_case_timeout_seconds ?: 7200).toString()
    def boraArgs = bora_geojson ? "--bora-geojson ${q(bora_geojson)} --bora-receipt ${q(bora_receipt)} --expected-bora-sha256 ${q(bora_sha)} --combined-output ${q(combined_output)}" : ''
    def selectedProject = combined_output ?: "${bundle_dir}/experiment.cellphenotyper.json"
    """
    set -euo pipefail
    command -v systemd-run >/dev/null
    command -v singularity >/dev/null
    test -f ${q(full_image)}
    test -f ${q(full_shift)}
    test -f ${q(full_mask)}
    test -d ${q(params.cellvit_classifier_cache)}
    test "\$(sha256sum -- ${q(converted_image)} | cut -d' ' -f1)" = "\$(sha256sum -- ${q(full_image)} | cut -d' ' -f1)" || { echo 'Full-slide image differs from the converted pipeline image' >&2; exit 2; }
    test "\$(sha256sum -- ${q(grandqc_mask)} | cut -d' ' -f1)" = "\$(sha256sum -- ${q(full_mask)} | cut -d' ' -f1)" || { echo 'Full-slide mask differs from GrandQC output' >&2; exit 2; }
    ${q(cpuPython)} ${q(source)} --action run \
      --image ${q(full_image)} --shift ${q(full_shift)} --clean-tissue-mask ${q(full_mask)} \
      --bundle-dir ${q(bundle_dir)} --classifier-cache ${q(params.cellvit_classifier_cache)} \
      --model ${q(params.cellvit_fullslide_model ?: 'SAM')} --gpu-uuid ${q(params.cellvit_fullslide_gpu_uuid)} \
      --runtime-sif ${q(params.cellvit_fullslide_runtime_sif)} \
      --runtime-sif-sha256 ${q(params.cellvit_fullslide_runtime_sif_sha256)} \
      --runtime-python ${q(params.cellvit_fullslide_runtime_python)} \
      --runtime-executable ${q(params.cellvit_fullslide_runtime_executable)} \
      --cpus 2 --memory-mb 12288 --app-memory-mb 8192 \
      --reserve-gib ${reserve} --case-budget-gib ${budget} \
      --case-timeout-seconds ${timeout} --expected-shards ${expectedShards} --amp \
      ${boraArgs} \
      ${pydepsArgs} > cellvit_fullslide_action_result.json
    export CELLVIT_BUNDLE=${q(bundle_dir)} CELLVIT_SAMPLE=${q(sample_id)} CELLVIT_SELECTED_PROJECT=${q(selectedProject)}
    ${q(cpuPython)} - <<'PY'
import hashlib, json, os
from pathlib import Path
bundle = Path(os.environ['CELLVIT_BUNDLE'])
receipt = bundle / 'CELLVIT_PROJECT_COMPLETE.json'
project = Path(os.environ['CELLVIT_SELECTED_PROJECT'])
index = bundle / 'cellvit_cells.sqlite'
if not all(path.is_file() for path in (receipt, project, index)):
    raise SystemExit('Full-slide CellViT returned without complete project, receipt and index')
complete = json.loads(receipt.read_text())
if complete.get('schema') != 'cellphenotyper-cellvit-fullslide-complete-v2':
    raise SystemExit('Full-slide CellViT completion receipt has an unexpected schema')
Path('cellvit_fullslide_complete.json').write_bytes(receipt.read_bytes())
Path('cellvit_project_pointer.json').write_text(json.dumps({
    'sample_id': os.environ['CELLVIT_SAMPLE'], 'project_path': str(project),
    'index_path': str(index), 'completion_receipt_path': str(receipt),
    'completion_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
    'complete': True}, sort_keys=True) + '\n')
PY
    """
}
