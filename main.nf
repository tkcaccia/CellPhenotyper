nextflow.enable.dsl = 2

include { PREPARE_INPUT_OMETIFF } from './modules/prepare_input_ometiff'
include { RUN_STARDIST_ROI_SEGMENTATION } from './modules/run_stardist_roi_segmentation'
include { BUILD_TISSUE_MASK } from './modules/build_tissue_mask'
include { CONVERT_TISSUE_MASK_TO_GEOJSON } from './modules/convert_tissue_mask_to_geojson'
include { MAP_CELLS_TO_ROI_POLYGONS } from './modules/map_cells_to_roi_polygons'
include { EXPAND_LABELS_TO_CYTOPLASM as EXPAND_LABELS_TO_CYTOPLASM_PRIMARY } from './modules/expand_labels_to_cytoplasm'
include { EXPAND_LABELS_TO_CYTOPLASM as EXPAND_LABELS_TO_CYTOPLASM_FULL } from './modules/expand_labels_to_cytoplasm'
include { EXTRACT_UNI2_EMBEDDINGS } from './modules/extract_uni2_embeddings'
include { RUN_KODAMA_ANALYSIS } from './modules/run_kodama_analysis'

workflow {
    def roi_geojson = file(params.roi_geojson, checkIfExists: true)
    def run_full_pipeline = params.run_full_pipeline as boolean
    def tissue_mask_from_input = params.tissue_mask_from_input as boolean

    def stage_aliases = [
        convert         : 'convert',
        image_conversion: 'convert',
        prepare_input   : 'convert',
        stardist        : 'stardist',
        startdist       : 'stardist',
        tissue_mask     : 'tissue_mask',
        mask_tissue     : 'tissue_mask',
        tissue_geojson  : 'tissue_geojson',
        geojson         : 'tissue_geojson',
        cell_assignment : 'cell_assignment',
        assign          : 'cell_assignment',
        cytoplasm       : 'cytoplasm',
        uni2            : 'uni2',
        uni_2           : 'uni2',
        'uni-2'         : 'uni2',
        embeddings      : 'uni2',
        kodama          : 'kodama'
    ]
    def stage_order = ['convert', 'stardist', 'tissue_mask', 'tissue_geojson', 'cell_assignment', 'cytoplasm', 'uni2', 'kodama']
    def stage_index = stage_order.withIndex().collectEntries { stage_name, idx -> [(stage_name): idx] }

    def normalize_stage = { raw_value, fallback_value ->
        def key = (raw_value ?: fallback_value).toString().trim().toLowerCase()
        if (key == 'auto') {
            key = fallback_value
        }
        key = stage_aliases.getOrDefault(key, key)
        if (!stage_index.containsKey(key)) {
            error "Invalid stage '${raw_value}'. Allowed stages: ${stage_order.join(', ')}"
        }
        key
    }

    def default_end_point = run_full_pipeline ? 'kodama' : 'tissue_geojson'
    def start_point = normalize_stage(params.start_point, 'convert')
    def end_point = normalize_stage(params.end_point, default_end_point)

    if (stage_index[start_point] > stage_index[end_point]) {
        error "Invalid stage window: start_point '${start_point}' occurs after end_point '${end_point}'"
    }

    params._resolved_start_point = start_point
    params._resolved_end_point = end_point

    def should_run_stage = { stage_name ->
        def idx = stage_index[stage_name]
        idx >= stage_index[start_point] && idx <= stage_index[end_point]
    }

    def run_convert = should_run_stage('convert')
    def run_stardist = should_run_stage('stardist')
    def run_tissue_mask = should_run_stage('tissue_mask')
    def run_tissue_geojson = should_run_stage('tissue_geojson')
    def run_cell_assignment = should_run_stage('cell_assignment')
    def run_cytoplasm = should_run_stage('cytoplasm')
    def run_uni2 = should_run_stage('uni2')
    def run_kodama = should_run_stage('kodama')

    println "Pipeline stage window: ${start_point} -> ${end_point}"

    Channel
        .fromPath(params.image_input, checkIfExists: true)
        .map { image_file ->
            def sample_id = image_file.name
                .replaceFirst(/\.ome\.tif$/, '')
                .replaceFirst(/\.btf$/, '')
            tuple(sample_id, image_file)
        }
        .set { image_input_ch }

    def ome_tif_ch
    if (run_convert) {
        PREPARE_INPUT_OMETIFF(image_input_ch)
        ome_tif_ch = PREPARE_INPUT_OMETIFF.out.ome_tif
    } else {
        ome_tif_ch = image_input_ch.map { sample_id, image_file ->
            def is_ome = image_file.name.toLowerCase().endsWith('.ome.tif')
            def ome_tif = is_ome
                ? image_file
                : file("${params.outdir_base}/01_input/${sample_id}.ome.tif", checkIfExists: true)
            tuple(sample_id, ome_tif)
        }
    }

    def need_stardist_outputs = run_stardist || (!tissue_mask_from_input && run_tissue_mask) || run_cell_assignment || run_cytoplasm || run_uni2

    def crop_roi_ch = Channel.empty()
    def labels_tif_ch = Channel.empty()
    def labels_full_tif_ch = Channel.empty()
    def objects_csv_ch = Channel.empty()
    def shift_json_ch = Channel.empty()

    if (need_stardist_outputs) {
        if (run_stardist) {
            def stardist_input_ch = ome_tif_ch.map { sample_id, ome_tif ->
                tuple(sample_id, ome_tif, roi_geojson)
            }
            RUN_STARDIST_ROI_SEGMENTATION(stardist_input_ch)
            crop_roi_ch = RUN_STARDIST_ROI_SEGMENTATION.out.crop_roi
            labels_tif_ch = RUN_STARDIST_ROI_SEGMENTATION.out.labels_tif
            labels_full_tif_ch = RUN_STARDIST_ROI_SEGMENTATION.out.labels_full_tif
            objects_csv_ch = RUN_STARDIST_ROI_SEGMENTATION.out.objects_csv
            shift_json_ch = RUN_STARDIST_ROI_SEGMENTATION.out.shift_json
        } else {
            def stardist_base_dir = "${params.outdir_base}/02_stardist/stardist_out"
            def crop_roi_existing = file("${stardist_base_dir}/crop_roi.tif", checkIfExists: true)
            def labels_existing = file("${stardist_base_dir}/labels.tif", checkIfExists: true)
            def objects_existing = file("${stardist_base_dir}/objects.csv", checkIfExists: true)
            def shift_existing = file("${stardist_base_dir}/shift.json", checkIfExists: true)
            def labels_full_existing = file("${stardist_base_dir}/labels_full.tif")

            crop_roi_ch = image_input_ch.map { sample_id, _ -> tuple(sample_id, crop_roi_existing) }
            labels_tif_ch = image_input_ch.map { sample_id, _ -> tuple(sample_id, labels_existing) }
            objects_csv_ch = image_input_ch.map { sample_id, _ -> tuple(sample_id, objects_existing) }
            shift_json_ch = image_input_ch.map { sample_id, _ -> tuple(sample_id, shift_existing) }
            labels_full_tif_ch = labels_full_existing.exists()
                ? image_input_ch.map { sample_id, _ -> tuple(sample_id, labels_full_existing) }
                : Channel.empty()
        }
    }

    def tissue_mask_ch = Channel.empty()
    if (run_tissue_mask) {
        def tissue_mask_input_ch = tissue_mask_from_input ? ome_tif_ch : crop_roi_ch
        BUILD_TISSUE_MASK(tissue_mask_input_ch)
        tissue_mask_ch = BUILD_TISSUE_MASK.out.tissue_mask
    } else if (run_tissue_geojson) {
        tissue_mask_ch = image_input_ch.map { sample_id, _ ->
            tuple(sample_id, file("${params.outdir_base}/03_tissue_mask/${sample_id}_tissue_mask.tif", checkIfExists: true))
        }
    }

    if (run_tissue_geojson) {
        CONVERT_TISSUE_MASK_TO_GEOJSON(tissue_mask_ch)
    }

    if (run_cell_assignment || run_cytoplasm || run_uni2 || run_kodama) {
        def objects_assigned_ch = Channel.empty()
        if (run_cell_assignment) {
            def assign_input_ch = objects_csv_ch
                .join(shift_json_ch)
                .map { sample_id, objects_csv, shift_json ->
                    tuple(sample_id, objects_csv, roi_geojson, shift_json)
                }
            MAP_CELLS_TO_ROI_POLYGONS(assign_input_ch)
            objects_assigned_ch = MAP_CELLS_TO_ROI_POLYGONS.out.objects_assigned
        } else {
            objects_assigned_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/05_cell_assignments/${sample_id}_objects_assigned.csv", checkIfExists: true))
            }
        }

        def cyto_mask_ch = Channel.empty()
        if (run_cytoplasm) {
            def expand_primary_ch = labels_tif_ch.map { sample_id, labels_tif ->
                tuple(sample_id, labels_tif, 'labels_cyto')
            }
            EXPAND_LABELS_TO_CYTOPLASM_PRIMARY(expand_primary_ch)

            if (params.expand_full_labels as boolean) {
                def expand_full_ch = labels_full_tif_ch.map { sample_id, labels_full_tif ->
                    tuple(sample_id, labels_full_tif, 'labels_full_cyto')
                }
                EXPAND_LABELS_TO_CYTOPLASM_FULL(expand_full_ch)
            }

            cyto_mask_ch = EXPAND_LABELS_TO_CYTOPLASM_PRIMARY.out.expanded_labels
                .filter { sample_id, expanded_mask, label_kind -> label_kind == 'labels_cyto' }
                .map { sample_id, expanded_mask, label_kind -> tuple(sample_id, expanded_mask) }
        } else if (run_uni2) {
            cyto_mask_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/06_cytoplasm/${sample_id}_labels_cyto.tif", checkIfExists: true))
            }
        }

        def tile_embeddings_ch = Channel.empty()
        if (run_uni2) {
            def uni2_cyto_input_ch = crop_roi_ch
                .join(cyto_mask_ch)
                .map { sample_id, crop_roi_tif, cyto_mask_tif ->
                    tuple(sample_id, crop_roi_tif, cyto_mask_tif, 'cyto', true)
                }

            def uni2_tile_input_ch = crop_roi_ch
                .join(labels_tif_ch)
                .map { sample_id, crop_roi_tif, labels_tif ->
                    tuple(sample_id, crop_roi_tif, labels_tif, 'tile', false)
                }

            EXTRACT_UNI2_EMBEDDINGS(uni2_cyto_input_ch.mix(uni2_tile_input_ch))
            tile_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
                .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'tile' }
                .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
        } else if (run_kodama) {
            tile_embeddings_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/07_embeddings/embeddings_${sample_id}_tile", checkIfExists: true))
            }
        }

        if (run_kodama) {
            def kodama_input_ch = tile_embeddings_ch
                .join(objects_assigned_ch)
                .map { sample_id, embeddings_dir, objects_assigned ->
                    tuple(sample_id, embeddings_dir, objects_assigned)
                }
            RUN_KODAMA_ANALYSIS(kodama_input_ch)
        }
    }
}

workflow.onComplete {
    if (workflow.success) {
        println "PIPELINE COMPLETED SUCCESSFULLY"
        println "Stage window: ${params._resolved_start_point ?: 'convert'} -> ${params._resolved_end_point ?: 'tissue_geojson'}"
        if ((params._resolved_end_point ?: '') in ['tissue_geojson', 'cell_assignment', 'cytoplasm', 'uni2', 'kodama']) {
            println "Tissue GeoJSON output dir: ${params.outdir_base}/04_tissue_geojson"
        }
    }
}
