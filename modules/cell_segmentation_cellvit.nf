include { RUN_CELL_IDENTIFICATION; RUN_CELL_IDENTIFICATION_FULLSLIDE } from './cell_identification'
include { EXPORT_CELLVIT_PRIMARY_SEGMENTATION } from './export_cellvit_primary_segmentation'

// Whole-slide completion and the common crop-frame raster/object contract are
// joined here. No StarDist or HoVer-Net process is imported by this route.
workflow RUN_PRIMARY_CELLVIT {
    take:
    crop_ch
    shift_ch
    support_ch
    full_image_ch
    full_support_ch
    settings
    sample_count

    main:
    def source_ch = Channel.empty()
    def metadata_ch = Channel.empty()
    def fullslide = settings.input_scope == 'full_slide'
    if (fullslide) {
        RUN_CELL_IDENTIFICATION_FULLSLIDE(full_image_ch, full_support_ch, settings, sample_count)
        source_ch = RUN_CELL_IDENTIFICATION_FULLSLIDE.out.project_pointer
        metadata_ch = RUN_CELL_IDENTIFICATION_FULLSLIDE.out.completion_receipt
    } else {
        RUN_CELL_IDENTIFICATION(crop_ch, shift_ch, support_ch, Channel.empty(), Channel.empty(), false, settings)
        source_ch = RUN_CELL_IDENTIFICATION.out.cells_json
        metadata_ch = RUN_CELL_IDENTIFICATION.out.metadata_json
    }
    def primary_input_ch = crop_ch.join(shift_ch).join(support_ch).join(source_ch).join(metadata_ch)
      .map { sample_id, image, shift, support, source, metadata ->
        tuple(sample_id, image, shift, support, fullslide ? 'full_slide' : 'roi', source, metadata)
      }
    EXPORT_CELLVIT_PRIMARY_SEGMENTATION(primary_input_ch)

    emit:
    labels_tif = EXPORT_CELLVIT_PRIMARY_SEGMENTATION.out.labels_tif
    objects_csv = EXPORT_CELLVIT_PRIMARY_SEGMENTATION.out.objects_csv
    metadata_json = EXPORT_CELLVIT_PRIMARY_SEGMENTATION.out.metadata_json
    annotations_geojson = EXPORT_CELLVIT_PRIMARY_SEGMENTATION.out.annotations_geojson
}
