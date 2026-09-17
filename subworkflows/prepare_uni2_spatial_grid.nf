include { BUILD_UNI2_SPATIAL_GRID } from '../modules/build_uni2_spatial_grid'

workflow PREPARE_UNI2_SPATIAL_GRID {
    take:
    image_input_ch
    crop_roi_ch
    tissue_mask_ch
    resolution_report_ch
    run_grid_tiles
    grid_artifacts_needed

    main:
    def gridObjectsCh = Channel.empty()
    def gridMetadataCh = Channel.empty()
    def gridPreviewCh = Channel.empty()
    if (grid_artifacts_needed as boolean) {
        if (run_grid_tiles as boolean) {
            def gridInputCh = crop_roi_ch
                .join([failOnDuplicate: true, failOnMismatch: true], tissue_mask_ch)
                .join([failOnDuplicate: true, failOnMismatch: true], resolution_report_ch)
                .map { sample_id, crop_tif, tissue_mask_tif, resolution_json ->
                    tuple(sample_id, crop_tif, tissue_mask_tif, resolution_json)
                }
            BUILD_UNI2_SPATIAL_GRID(gridInputCh)
            gridObjectsCh = BUILD_UNI2_SPATIAL_GRID.out.grid_objects
            gridMetadataCh = BUILD_UNI2_SPATIAL_GRID.out.grid_metadata
            gridPreviewCh = BUILD_UNI2_SPATIAL_GRID.out.grid_preview
        } else {
            gridObjectsCh = image_input_ch.map { sample_id, _image_input ->
                tuple(sample_id, file("${params.outdir_base}/09_grid_tiles/${sample_id}/${sample_id}_uni2_grid_objects.csv", checkIfExists: true))
            }
            gridMetadataCh = image_input_ch.map { sample_id, _image_input ->
                tuple(sample_id, file("${params.outdir_base}/09_grid_tiles/${sample_id}/${sample_id}_uni2_grid_metadata.json", checkIfExists: true))
            }
            gridPreviewCh = image_input_ch.map { sample_id, _image_input ->
                tuple(sample_id, file("${params.outdir_base}/09_grid_tiles/${sample_id}/${sample_id}_uni2_grid_preview.png", checkIfExists: true))
            }
        }
    }

    emit:
    grid_objects = gridObjectsCh
    grid_metadata = gridMetadataCh
    grid_preview = gridPreviewCh
}
