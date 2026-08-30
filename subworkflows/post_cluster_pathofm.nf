include { SELECT_NEOPLASTIC_SECTION } from '../modules/select_neoplastic_section'
include { EXTRACT_TITAN_SECTION_EMBEDDING } from '../modules/extract_titan_section_embedding'
include { RUN_PATHOFMPRED } from '../modules/run_pathofmpred'

workflow POST_CLUSTER_PATHOFM {
    take:
    image_input_ch
    final_cluster_geojson_input_ch
    objects_csv_ch
    crop_roi_ch
    shift_json_ch

    main:
    def stageOrder = ['cluster_geojson', 'neoplastic_section', 'titan', 'pathofmpred']
    def stageAliases = [
        cluster_geojson: 'cluster_geojson',
        geojson: 'cluster_geojson',
        neoplastic_section: 'neoplastic_section',
        tumor_section: 'neoplastic_section',
        tumour_section: 'neoplastic_section',
        titan: 'titan',
        titan_embedding: 'titan',
        pathofmpred: 'pathofmpred',
        pathofm: 'pathofmpred',
    ]
    def normalizeStage = { rawValue, fallback ->
        def key = (rawValue ?: fallback).toString().trim().toLowerCase().replace('-', '_')
        stageAliases.getOrDefault(key, key)
    }
    def startIndex = stageOrder.indexOf(normalizeStage.call(params.start_point, 'cluster_geojson'))
    def endIndex = stageOrder.indexOf(normalizeStage.call(params.end_point, 'pathofmpred'))
    if (startIndex < 0) startIndex = 0
    if (endIndex < 0) endIndex = stageOrder.size() - 1
    def inWindow = { name ->
        def index = stageOrder.indexOf(name)
        index >= startIndex && index <= endIndex
    }
    def runNeoplasticSection = inWindow.call('neoplastic_section')
    def runTitan = inWindow.call('titan')
    def runPathoFMPred = inWindow.call('pathofmpred')
    def runClusterGeoJSON = inWindow.call('cluster_geojson')

    def clusterVariants = [(params.cluster_primary_variant ?: 'standard').toString().trim() ?: 'standard']
    def secondaryVariant = (params.cluster_secondary_variant ?: '').toString().trim()
    if (secondaryVariant && !(secondaryVariant.toLowerCase() in ['none', 'false', 'off', '0'])) {
        if (!clusterVariants.contains(secondaryVariant)) clusterVariants << secondaryVariant
    }

    def finalClusterGeoJSONCh = final_cluster_geojson_input_ch
    def selectedImageCh = Channel.empty()
    def selectedMaskCh = Channel.empty()
    def selectedShiftCh = Channel.empty()
    def selectedSummaryCh = Channel.empty()

    if (runNeoplasticSection) {
        if (!runClusterGeoJSON) {
            finalClusterGeoJSONCh = image_input_ch.flatMap { sample_id, _image_input ->
                clusterVariants.collect { cluster_variant ->
                    def sample_key = "${sample_id}::${cluster_variant}"
                    tuple(
                        sample_key, sample_id, cluster_variant,
                        file("${params.outdir_base}/15_cluster_geojson/${sample_id}/${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson", checkIfExists: true),
                    )
                }
            }
        }
        def sectionInputCh = finalClusterGeoJSONCh
            .map { sample_key, sample_id, cluster_variant, cluster_geojson ->
                tuple(sample_id, sample_key, cluster_variant, cluster_geojson)
            }
            .combine(objects_csv_ch, by: 0)
            .combine(crop_roi_ch, by: 0)
            .combine(shift_json_ch, by: 0)
            .map { sample_id, sample_key, cluster_variant, cluster_geojson, objects_csv, image_tif, shift_json ->
                tuple(sample_key, sample_id, cluster_variant, cluster_geojson, objects_csv, image_tif, shift_json)
            }
        SELECT_NEOPLASTIC_SECTION(sectionInputCh)
        selectedImageCh = SELECT_NEOPLASTIC_SECTION.out.selected_image
        selectedMaskCh = SELECT_NEOPLASTIC_SECTION.out.selected_mask
        selectedShiftCh = SELECT_NEOPLASTIC_SECTION.out.selected_shift
        selectedSummaryCh = SELECT_NEOPLASTIC_SECTION.out.selected_summary
    }

    def titanEmbeddingCh = Channel.empty()
    if (runTitan) {
        if (!runNeoplasticSection) {
            selectedImageCh = image_input_ch.flatMap { sample_id, _image_input ->
                clusterVariants.collect { cluster_variant ->
                    def sample_key = "${sample_id}::${cluster_variant}"
                    tuple(sample_key, sample_id, cluster_variant, file("${params.outdir_base}/16_neoplastic_section/${sample_id}/neoplastic_${sample_id}_${cluster_variant}/selected_section.ome.tif", checkIfExists: true))
                }
            }
            selectedMaskCh = image_input_ch.flatMap { sample_id, _image_input ->
                clusterVariants.collect { cluster_variant ->
                    def sample_key = "${sample_id}::${cluster_variant}"
                    tuple(sample_key, sample_id, cluster_variant, file("${params.outdir_base}/16_neoplastic_section/${sample_id}/neoplastic_${sample_id}_${cluster_variant}/selected_section_mask.tif", checkIfExists: true))
                }
            }
            selectedShiftCh = image_input_ch.flatMap { sample_id, _image_input ->
                clusterVariants.collect { cluster_variant ->
                    def sample_key = "${sample_id}::${cluster_variant}"
                    tuple(sample_key, sample_id, cluster_variant, file("${params.outdir_base}/16_neoplastic_section/${sample_id}/neoplastic_${sample_id}_${cluster_variant}/selected_section_shift.json", checkIfExists: true))
                }
            }
            selectedSummaryCh = image_input_ch.flatMap { sample_id, _image_input ->
                clusterVariants.collect { cluster_variant ->
                    def sample_key = "${sample_id}::${cluster_variant}"
                    tuple(sample_key, sample_id, cluster_variant, file("${params.outdir_base}/16_neoplastic_section/${sample_id}/neoplastic_${sample_id}_${cluster_variant}/selected_section_summary.json", checkIfExists: true))
                }
            }
        }
        def titanInputCh = selectedImageCh
            .join(selectedMaskCh)
            .join(selectedShiftCh)
            .join(selectedSummaryCh)
            .map { sample_key, sample_id, cluster_variant, selected_image,
                   ignored_sample_id_1, ignored_variant_1, selected_mask,
                   ignored_sample_id_2, ignored_variant_2, selected_shift,
                   ignored_sample_id_3, ignored_variant_3, selected_summary ->
                tuple(sample_key, sample_id, cluster_variant, selected_image, selected_mask, selected_shift, selected_summary)
            }
        EXTRACT_TITAN_SECTION_EMBEDDING(titanInputCh)
        titanEmbeddingCh = EXTRACT_TITAN_SECTION_EMBEDDING.out.embedding_csv
    } else if (runPathoFMPred) {
        titanEmbeddingCh = image_input_ch.flatMap { sample_id, _image_input ->
            clusterVariants.collect { cluster_variant ->
                def sample_key = "${sample_id}::${cluster_variant}"
                tuple(sample_key, sample_id, cluster_variant, file("${params.outdir_base}/17_titan/${sample_id}/titan_${sample_id}_${cluster_variant}/titan_embedding.csv", checkIfExists: true))
            }
        }
    }

    if (runPathoFMPred) RUN_PATHOFMPRED(titanEmbeddingCh)

    emit:
    selected_sections = runNeoplasticSection ? SELECT_NEOPLASTIC_SECTION.out.selected_dir : Channel.empty()
    titan_embeddings = runTitan ? EXTRACT_TITAN_SECTION_EMBEDDING.out.embedding_csv : Channel.empty()
    predictions = runPathoFMPred ? RUN_PATHOFMPRED.out.predictions : Channel.empty()
}
