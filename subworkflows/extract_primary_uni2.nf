include { EXTRACT_UNI2_EMBEDDINGS } from '../modules/extract_uni2_embeddings'
include { EXTRACT_UNI2_EMBEDDINGS_SHARED } from '../modules/extract_uni2_embeddings_shared'

workflow EXTRACT_PRIMARY_UNI2 {
    take:
    crop_roi_ch
    ome_tif_ch
    labels_tif_ch
    labels_full_ch
    tissue_mask_ch
    cyto_mask_ch
    cyto_mask_full_ch
    grid_objects_ch
    grid_metadata_ch
    resolution_ch
    flags
    runtime_plan

    main:
    def use_roi_crop_for_uni2 = flags.use_crop as boolean
    def uni2_grid_mode = flags.grid as boolean
    def fuse_tile_inner_square_uni2 = flags.fuse as boolean
    def include_uni2_inner_square = flags.inner as boolean
    def include_uni2_cyto = flags.cyto as boolean
    def include_uni2_nuclei = flags.nuclei as boolean
    def tile_embeddings_ch = Channel.empty()
    def inner_square_embeddings_ch = Channel.empty()
    def nuclei_embeddings_ch = Channel.empty()
    def cyto_embeddings_ch = Channel.empty()
    def placeholder_observations_file = file("${projectDir}/resources/empty_uni2_observations.csv", checkIfExists: true)
    def strictJoin = [failOnDuplicate: true, failOnMismatch: true]
  def uni2_image_ch = use_roi_crop_for_uni2 ? crop_roi_ch : ome_tif_ch
  def uni2_label_mask_ch = uni2_grid_mode
    ? tissue_mask_ch
    : (use_roi_crop_for_uni2 ? labels_tif_ch : labels_full_ch)
  def uni2_cyto_mask_source_ch = use_roi_crop_for_uni2 ? cyto_mask_ch : cyto_mask_full_ch
  def pairedFoundationEncoders = ['uni2-h', 'virchow', 'virchow2', 'phikon-v2'] as Set
  def selectedFoundationEncoder = params.uni2_encoder?.toString()?.toLowerCase()
  def canFuseTileInnerSquare = fuse_tile_inner_square_uni2 &&
    include_uni2_inner_square &&
    pairedFoundationEncoders.contains(selectedFoundationEncoder)

  def uni2_input_ch = Channel.empty()
  def haveSeparateUni2Inputs = false
  if (canFuseTileInnerSquare) {
    def uni2_shared_input_ch = uni2_grid_mode
      ? uni2_image_ch
        .join(strictJoin, uni2_label_mask_ch)
        .join(strictJoin, grid_objects_ch)
        .join(strictJoin, grid_metadata_ch)
        .join(strictJoin, resolution_ch)
        .map { sample_id, image_tif, tissue_mask_tif, grid_objects_csv, grid_metadata_json, resolution_json ->
          tuple(sample_id, image_tif, tissue_mask_tif, 'grid', grid_objects_csv, resolution_json)
        }
      : uni2_image_ch
        .join(strictJoin, uni2_label_mask_ch)
        .join(strictJoin, resolution_ch)
        .map { sample_id, image_tif, labels_tif, resolution_json ->
          tuple(sample_id, image_tif, labels_tif, 'cell', placeholder_observations_file, resolution_json)
        }
    EXTRACT_UNI2_EMBEDDINGS_SHARED(uni2_shared_input_ch, runtime_plan)
    tile_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS_SHARED.out.tile_embeddings_dir
      .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
    tile_embeddings_ch = tile_embeddings_ch.ifEmpty { error 'Requested primary UNI-2 extraction emitted no observations' }
    inner_square_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS_SHARED.out.inner_square_embeddings_dir
      .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
  } else {
    def uni2_tile_input_ch = uni2_image_ch
      .join(strictJoin, uni2_label_mask_ch)
      .join(strictJoin, resolution_ch)
      .map { sample_id, image_tif, labels_tif, resolution_json ->
        tuple(sample_id, image_tif, labels_tif, resolution_json, 'tile', false, 'none', 255)
      }
    uni2_input_ch = uni2_tile_input_ch
    haveSeparateUni2Inputs = true
  }
  if (include_uni2_cyto) {
    def uni2_cyto_input_ch = uni2_image_ch
      .join(strictJoin, uni2_cyto_mask_source_ch)
      .join(strictJoin, resolution_ch)
      .map { sample_id, image_tif, cyto_mask_tif, resolution_json ->
        tuple(sample_id, image_tif, cyto_mask_tif, resolution_json, 'cyto', true, 'label', 255)
      }
    uni2_input_ch = haveSeparateUni2Inputs ? uni2_input_ch.mix(uni2_cyto_input_ch) : uni2_cyto_input_ch
    haveSeparateUni2Inputs = true
  }
  if (include_uni2_inner_square && !canFuseTileInnerSquare) {
    def uni2_inner_square_input_ch = uni2_image_ch
      .join(strictJoin, uni2_cyto_mask_source_ch)
      .join(strictJoin, resolution_ch)
      .map { sample_id, image_tif, cyto_mask_tif, resolution_json ->
        tuple(sample_id, image_tif, cyto_mask_tif, resolution_json, 'inner_square', true, 'inner_square', 255)
      }
    uni2_input_ch = haveSeparateUni2Inputs ? uni2_input_ch.mix(uni2_inner_square_input_ch) : uni2_inner_square_input_ch
    haveSeparateUni2Inputs = true
  }
  if (include_uni2_nuclei) {
    def uni2_nuclei_input_ch = uni2_image_ch
      .join(strictJoin, uni2_label_mask_ch)
      .join(strictJoin, resolution_ch)
      .map { sample_id, image_tif, labels_tif, resolution_json ->
        tuple(sample_id, image_tif, labels_tif, resolution_json, 'nuclei', true, 'label', 255)
      }
    uni2_input_ch = haveSeparateUni2Inputs ? uni2_input_ch.mix(uni2_nuclei_input_ch) : uni2_nuclei_input_ch
    haveSeparateUni2Inputs = true
  }

  if (haveSeparateUni2Inputs) {
    EXTRACT_UNI2_EMBEDDINGS(uni2_input_ch, runtime_plan)
    if (!canFuseTileInnerSquare) {
      tile_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
        .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'tile' }
        .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
      tile_embeddings_ch = tile_embeddings_ch.ifEmpty { error 'Requested primary UNI-2 extraction emitted no observations' }
      inner_square_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
        .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'inner_square' }
        .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
    }
    nuclei_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
      .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'nuclei' }
      .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
    cyto_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
      .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'cyto' }
      .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
  }

    emit:
    tile = tile_embeddings_ch
    local = inner_square_embeddings_ch
    nuclei = nuclei_embeddings_ch
    cyto = cyto_embeddings_ch
}
