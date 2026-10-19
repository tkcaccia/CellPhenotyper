include { EXTRACT_UNI2_EMBEDDINGS_SHARED as EXTRACT_UNI2_CELL_AUXILIARY } from '../modules/extract_uni2_embeddings_shared'
include { RUN_KODAMA_ANALYSIS as RUN_KODAMA_CELL_AUXILIARY } from '../modules/run_kodama_analysis'

workflow RUN_AUXILIARY_CELL_UNI2 {
    take:
    image_input_ch
    crop_roi_ch
    labels_tif_ch
    objects_assigned_ch
    primary_tile_embeddings_ch
    run_uni2
    run_kodama

    main:
    def suffix = '__cells'
    def empty_dir = file("${projectDir}/resources/empty_embeddings_placeholder", checkIfExists: true)
    def empty_observations = file("${projectDir}/resources/empty_uni2_observations.csv", checkIfExists: true)
    def empty_resolution = file("${projectDir}/resources/empty_uni2_resolution.json", checkIfExists: true)
    def auxiliary_objects_ch = objects_assigned_ch.map { sample_id, objects_csv ->
        tuple("${sample_id}${suffix}", objects_csv)
    }
    def tile_embeddings_ch = Channel.empty()
    def inner_embeddings_ch = Channel.empty()

    if (run_uni2) {
        // The primary output acts as a GPU semaphore: the comparison starts only
        // after grid UNI-2 has released its model and device allocations.
        def primary_complete_ch = primary_tile_embeddings_ch.map { sample_id, _dir ->
            tuple(sample_id, true)
        }
        def auxiliary_uni2_input_ch = crop_roi_ch
            .join(labels_tif_ch)
            .join(primary_complete_ch)
            .map { sample_id, image_tif, labels_tif, _primary_complete ->
                tuple("${sample_id}${suffix}", image_tif, labels_tif, 'cell', empty_observations, empty_resolution)
            }
        EXTRACT_UNI2_CELL_AUXILIARY(auxiliary_uni2_input_ch)
        tile_embeddings_ch = EXTRACT_UNI2_CELL_AUXILIARY.out.tile_embeddings_dir
            .map { sample_id, _mode, directory -> tuple(sample_id, directory) }
        inner_embeddings_ch = EXTRACT_UNI2_CELL_AUXILIARY.out.inner_square_embeddings_dir
            .map { sample_id, _mode, directory -> tuple(sample_id, directory) }
    } else {
        tile_embeddings_ch = image_input_ch.map { sample_id, _image ->
            def auxiliary_id = "${sample_id}${suffix}"
            tuple(auxiliary_id, file("${params.outdir_base}/09_embeddings/${auxiliary_id}/embeddings_${auxiliary_id}_tile", checkIfExists: true))
        }
        inner_embeddings_ch = image_input_ch.map { sample_id, _image ->
            def auxiliary_id = "${sample_id}${suffix}"
            tuple(auxiliary_id, file("${params.outdir_base}/09_embeddings/${auxiliary_id}/embeddings_${auxiliary_id}_inner_square", checkIfExists: true))
        }
    }

    def kodama_dir_ch = Channel.empty()
    if (run_kodama) {
        def cyto_placeholder_ch = image_input_ch.map { sample_id, _image ->
            tuple("${sample_id}${suffix}", empty_dir)
        }
        def nuclei_placeholder_ch = image_input_ch.map { sample_id, _image ->
            tuple("${sample_id}${suffix}", empty_dir)
        }
        def kodama_input_ch = tile_embeddings_ch
            .join(cyto_placeholder_ch)
            .join(inner_embeddings_ch)
            .join(nuclei_placeholder_ch)
            .join(auxiliary_objects_ch)
            .map { sample_id, tile_dir, cyto_dir, inner_dir, nuclei_dir, objects_csv ->
                tuple(sample_id, tile_dir, cyto_dir, inner_dir, nuclei_dir, objects_csv)
            }
        RUN_KODAMA_CELL_AUXILIARY(kodama_input_ch)
        kodama_dir_ch = RUN_KODAMA_CELL_AUXILIARY.out.kodama_dir
    }

    emit:
    tile_embeddings = tile_embeddings_ch
    inner_embeddings = inner_embeddings_ch
    kodama_dir = kodama_dir_ch
}
