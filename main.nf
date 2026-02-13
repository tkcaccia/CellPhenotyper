nextflow.enable.dsl = 2

include { PREPARE_INPUT_OMETIFF } from './modules/prepare_input_ometiff'
include { RUN_STARDIST_ROI_SEGMENTATION } from './modules/run_stardist_roi_segmentation'
include { BUILD_TISSUE_MASK } from './modules/build_tissue_mask'
include { CONVERT_TISSUE_MASK_TO_GEOJSON } from './modules/convert_tissue_mask_to_geojson'
include { MAP_CELLS_TO_ROI_POLYGONS } from './modules/map_cells_to_roi_polygons'
include { EXPAND_LABELS_TO_CYTOPLASM } from './modules/expand_labels_to_cytoplasm'
include { EXTRACT_UNI2_EMBEDDINGS } from './modules/extract_uni2_embeddings'
include { RUN_KODAMA_ANALYSIS } from './modules/run_kodama_analysis'

workflow {
    def roi_geojson = file(params.roi_geojson, checkIfExists: true)
    def run_full_pipeline = params.run_full_pipeline as boolean
    def tissue_mask_from_input = params.tissue_mask_from_input as boolean
    def need_stardist = run_full_pipeline || !tissue_mask_from_input

    Channel
        .fromPath(params.image_input, checkIfExists: true)
        .map { image_file ->
            def sample_id = image_file.name
                .replaceFirst(/\.ome\.tif$/, '')
                .replaceFirst(/\.btf$/, '')
            tuple(sample_id, image_file)
        }
        .set { image_input_ch }

    PREPARE_INPUT_OMETIFF(image_input_ch)

    def ome_tif_ch = PREPARE_INPUT_OMETIFF.out.ome_tif

    if (need_stardist) {
        def stardist_input_ch = ome_tif_ch.map { sample_id, ome_tif ->
            tuple(sample_id, ome_tif, roi_geojson)
        }
        RUN_STARDIST_ROI_SEGMENTATION(stardist_input_ch)
    }

    def tissue_mask_input_ch = tissue_mask_from_input
        ? ome_tif_ch
        : RUN_STARDIST_ROI_SEGMENTATION.out.crop_roi

    BUILD_TISSUE_MASK(tissue_mask_input_ch)
    CONVERT_TISSUE_MASK_TO_GEOJSON(BUILD_TISSUE_MASK.out.tissue_mask)

    if (run_full_pipeline) {
        def assign_input_ch = RUN_STARDIST_ROI_SEGMENTATION.out.objects_csv
            .join(RUN_STARDIST_ROI_SEGMENTATION.out.shift_json)
            .map { sample_id, objects_csv, shift_json ->
                tuple(sample_id, objects_csv, roi_geojson, shift_json)
            }
        MAP_CELLS_TO_ROI_POLYGONS(assign_input_ch)

        def expand_primary_ch = RUN_STARDIST_ROI_SEGMENTATION.out.labels_tif.map { sample_id, labels_tif ->
            tuple(sample_id, labels_tif, 'labels_cyto')
        }
        EXPAND_LABELS_TO_CYTOPLASM(expand_primary_ch)

        def expand_full_ch = RUN_STARDIST_ROI_SEGMENTATION.out.labels_full_tif.map { sample_id, labels_full_tif ->
            tuple(sample_id, labels_full_tif, 'labels_full_cyto')
        }
        EXPAND_LABELS_TO_CYTOPLASM(expand_full_ch)

        def cyto_mask_ch = EXPAND_LABELS_TO_CYTOPLASM.out.expanded_labels
            .filter { sample_id, expanded_mask, label_kind -> label_kind == 'labels_cyto' }
            .map { sample_id, expanded_mask, label_kind -> tuple(sample_id, expanded_mask) }

        def uni2_cyto_input_ch = RUN_STARDIST_ROI_SEGMENTATION.out.crop_roi
            .join(cyto_mask_ch)
            .map { sample_id, crop_roi_tif, cyto_mask_tif ->
                tuple(sample_id, crop_roi_tif, cyto_mask_tif, 'cyto', true)
            }

        def uni2_tile_input_ch = RUN_STARDIST_ROI_SEGMENTATION.out.crop_roi
            .join(RUN_STARDIST_ROI_SEGMENTATION.out.labels_tif)
            .map { sample_id, crop_roi_tif, labels_tif ->
                tuple(sample_id, crop_roi_tif, labels_tif, 'tile', false)
            }

        EXTRACT_UNI2_EMBEDDINGS(uni2_cyto_input_ch.mix(uni2_tile_input_ch))

        def kodama_input_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
            .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'tile' }
            .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
            .join(MAP_CELLS_TO_ROI_POLYGONS.out.objects_assigned)
            .map { sample_id, embeddings_dir, objects_assigned ->
                tuple(sample_id, embeddings_dir, objects_assigned)
            }

        RUN_KODAMA_ANALYSIS(kodama_input_ch)
    }
}

workflow.onComplete {
    if (workflow.success) {
        println "PIPELINE COMPLETED SUCCESSFULLY"
        println "Tissue GeoJSON output dir: ${params.outdir_base}/04_tissue_geojson"
    }
}
