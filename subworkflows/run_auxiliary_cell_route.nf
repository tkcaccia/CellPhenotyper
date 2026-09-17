include { COMPARE_UNI2_ROUTES } from '../modules/compare_uni2_routes'
include { RUN_AUXILIARY_CELL_UNI2 } from './run_auxiliary_cell_uni2'
include { RUN_AUXILIARY_CELL_SPATIAL } from './run_auxiliary_cell_spatial'

workflow RUN_AUXILIARY_CELL_ROUTE {
    take:
    image_input_ch
    crop_roi_ch
    labels_tif_ch
    objects_assigned_ch
    primary_tile_embeddings_ch
    primary_kodama_dir_ch
    grid_objects_ch
    tissue_mask_ch
    resolution_ch
    verified_resolution_ch
    cyto_mask_ch
    current_marker_quant_dir_ch
    flags
    runtime_plan

    main:
    def runUni2 = flags.run_uni2 as boolean
    def runKodama = flags.run_kodama as boolean
    def runClustering = flags.run_clustering as boolean
    def runClusterMask = flags.run_cluster_mask as boolean
    def runGrowTissue = flags.run_grow_tissue as boolean
    def runMedsamRefine = flags.run_medsam_refine as boolean
    def runClusterGeoJSON = flags.run_cluster_geojson as boolean
    def runCytoplasm = flags.run_cytoplasm as boolean
    def useCurrentMarkers = flags.use_current_markers as boolean
    def markersConfigured = flags.markers_configured as boolean
    def placeholder = file("${projectDir}/resources/empty_embeddings_placeholder", checkIfExists: true)

    def markerQuantDirCh = useCurrentMarkers
        ? current_marker_quant_dir_ch
        : (markersConfigured
            ? image_input_ch.map { sample_id, _image ->
                tuple(sample_id, file("${params.outdir_base}/05_gigatime/${sample_id}/quantification_${sample_id}", checkIfExists: true))
            }
            : image_input_ch.map { sample_id, _image -> tuple(sample_id, placeholder) })

    def auxiliaryKodamaRawCh = Channel.empty()
    def auxiliaryKodamaMatchedCh = Channel.empty()
    def cellTileCh = Channel.empty()
    def cellLocalCh = Channel.empty()
    if (runUni2 || runKodama) {
        RUN_AUXILIARY_CELL_UNI2(
            image_input_ch,
            crop_roi_ch,
            labels_tif_ch,
            objects_assigned_ch,
            primary_tile_embeddings_ch,
            verified_resolution_ch,
            runUni2,
            runKodama,
            runtime_plan,
        )
        cellTileCh = RUN_AUXILIARY_CELL_UNI2.out.tile_embeddings
        cellLocalCh = RUN_AUXILIARY_CELL_UNI2.out.inner_embeddings
        if (runKodama) {
            auxiliaryKodamaRawCh = RUN_AUXILIARY_CELL_UNI2.out.kodama_dir
                .ifEmpty { error 'The auxiliary cell-centred KODAMA route emitted no output; check sample-ID joins and cached embeddings.' }
            auxiliaryKodamaMatchedCh = auxiliaryKodamaRawCh.map { auxiliary_id, directory ->
                tuple(auxiliary_id.replaceFirst(/__cells$/, ''), directory)
            }
        }
    } else if (runClustering) {
        auxiliaryKodamaRawCh = image_input_ch.map { sample_id, _image ->
            def auxiliaryId = "${sample_id}__cells"
            tuple(auxiliaryId, file("${params.outdir_base}/10_kodama/${auxiliaryId}/kodama_output", checkIfExists: true))
        }
    }

    def comparisonCh = Channel.empty()
    if (runKodama) {
        def comparisonInputCh = primary_kodama_dir_ch
            .join(auxiliaryKodamaMatchedCh)
            .join(grid_objects_ch)
            .join(objects_assigned_ch)
            .join(markerQuantDirCh)
            .map { sample_id, grid_kodama_dir, cell_kodama_dir, grid_objects_csv, cell_objects_csv, marker_quant_dir ->
                tuple(sample_id, grid_kodama_dir, cell_kodama_dir, grid_objects_csv, cell_objects_csv, marker_quant_dir)
            }
        COMPARE_UNI2_ROUTES(comparisonInputCh)
        comparisonCh = COMPARE_UNI2_ROUTES.out.comparison_dir
            .ifEmpty { error 'The cell-versus-grid UNI-2 comparison emitted no output; check matched sample IDs and route artifacts.' }
    }

    def cellLabelsCh = runClusterMask
        ? (runCytoplasm
            ? cyto_mask_ch
            : image_input_ch.map { sample_id, _image ->
                tuple(sample_id, file("${params.outdir_base}/08_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif", checkIfExists: true))
            })
        : Channel.empty()
    def markerContextCh = runClustering ? markerQuantDirCh : Channel.empty()

    def clusterCsvCh = Channel.empty()
    def clusterAssessmentCh = Channel.empty()
    def clusterMaskCh = Channel.empty()
    def grownMaskCh = Channel.empty()
    def refinedMaskCh = Channel.empty()
    def clusterGeoJSONCh = Channel.empty()
    if (runClustering || runClusterMask || runGrowTissue || runMedsamRefine || runClusterGeoJSON) {
        RUN_AUXILIARY_CELL_SPATIAL(
            image_input_ch,
            crop_roi_ch,
            tissue_mask_ch,
            resolution_ch,
            objects_assigned_ch,
            cellLabelsCh,
            markerContextCh,
            auxiliaryKodamaRawCh,
            runClustering,
            runClusterMask,
            runGrowTissue,
            runMedsamRefine,
            runClusterGeoJSON,
            runtime_plan,
        )
        clusterCsvCh = RUN_AUXILIARY_CELL_SPATIAL.out.cluster_csv
        clusterAssessmentCh = RUN_AUXILIARY_CELL_SPATIAL.out.cluster_assessment
        clusterMaskCh = RUN_AUXILIARY_CELL_SPATIAL.out.cluster_mask
        grownMaskCh = RUN_AUXILIARY_CELL_SPATIAL.out.grown_mask
        refinedMaskCh = RUN_AUXILIARY_CELL_SPATIAL.out.refined_mask
        clusterGeoJSONCh = RUN_AUXILIARY_CELL_SPATIAL.out.cluster_geojson
    }

    emit:
    tile_embeddings = cellTileCh
    inner_embeddings = cellLocalCh
    kodama_dir = auxiliaryKodamaRawCh
    comparison = comparisonCh
    cluster_csv = clusterCsvCh
    cluster_assessment = clusterAssessmentCh
    cluster_mask = clusterMaskCh
    grown_mask = grownMaskCh
    refined_mask = refinedMaskCh
    cluster_geojson = clusterGeoJSONCh
}
