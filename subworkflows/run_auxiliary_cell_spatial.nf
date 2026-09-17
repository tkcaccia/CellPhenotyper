include { RUN_RCODE_CLUSTERING as RUN_RCODE_CELL_AUXILIARY_CLUSTERING } from '../modules/run_rcode_clustering'
include { ASSESS_CLUSTER_INTERPRETATION as ASSESS_CELL_AUXILIARY_CLUSTERING } from '../modules/assess_cluster_interpretation'
include { LABELS_TO_CLUSTER_MASK as LABELS_TO_CELL_AUXILIARY_CLUSTER_MASK } from '../modules/labels_to_cluster_mask'
include { GROW_TO_TISSUE as GROW_CELL_AUXILIARY_TO_TISSUE } from '../modules/grow_to_tissue'
include { REFINE_GROWN_TISSUE_MEDSAM as REFINE_CELL_AUXILIARY_MEDSAM } from '../modules/refine_grown_tissue_medsam'
include { MASK_TO_GEOJSON as CELL_AUXILIARY_MASK_TO_GEOJSON } from '../modules/mask_to_geojson'

workflow RUN_AUXILIARY_CELL_SPATIAL {
    take:
    image_input_ch
    crop_roi_ch
    tissue_mask_ch
    resolution_ch
    objects_assigned_ch
    cell_label_mask_ch
    marker_quant_dir_ch
    auxiliary_kodama_dir_ch
    run_clustering
    run_cluster_mask
    run_grow_tissue
    run_medsam_refine
    run_cluster_geojson
    runtime_plan

    main:
    def suffix = '__cells'
    def runClustering = run_clustering as boolean
    def runClusterMask = run_cluster_mask as boolean
    def runGrowTissue = run_grow_tissue as boolean
    def runMedsamRefine = run_medsam_refine as boolean
    def runClusterGeoJSON = run_cluster_geojson as boolean
    def grownRefineMethod = (params.grown_tissue_refine_method ?: 'medsam_border_refine').toString().trim().toLowerCase()
    def clusterResolution = (params.cluster_resolution ?: 'auto').toString().trim() ?: 'auto'
    def clusterVariants = [
        [
            variant: (params.cluster_primary_variant ?: 'standard').toString().trim() ?: 'standard',
            profile: 'standard',
        ]
    ]
    def secondaryVariant = (params.cluster_secondary_variant ?: '').toString().trim()
    if (secondaryVariant && !(secondaryVariant.toLowerCase() in ['none', 'false', 'off', '0'])) {
        if (clusterVariants*.variant.contains(secondaryVariant)) {
            error 'cluster_primary_variant and cluster_secondary_variant must be different.'
        }
        clusterVariants << [
            variant: secondaryVariant,
            profile: (params.cluster_secondary_profile ?: 'fine').toString().trim().toLowerCase() ?: 'fine',
        ]
    }

    def auxiliarySamplesCh = image_input_ch.map { sample_id, image_input ->
        tuple("${sample_id}${suffix}", image_input)
    }
    def auxiliaryImagesCh = crop_roi_ch.map { sample_id, image_tif ->
        tuple("${sample_id}${suffix}", image_tif)
    }
    def auxiliaryTissueMasksCh = tissue_mask_ch.map { sample_id, tissue_mask_tif ->
        tuple("${sample_id}${suffix}", tissue_mask_tif)
    }
    def auxiliaryResolutionCh = resolution_ch.map { sample_id, resolution_json ->
        tuple("${sample_id}${suffix}", resolution_json)
    }
    def auxiliaryObjectsCh = objects_assigned_ch.map { sample_id, objects_csv ->
        tuple("${sample_id}${suffix}", objects_csv)
    }
    def auxiliaryLabelsCh = cell_label_mask_ch.map { sample_id, labels_tif ->
        tuple("${sample_id}${suffix}", labels_tif)
    }
    def auxiliaryMarkersCh = marker_quant_dir_ch.map { sample_id, marker_quant_dir ->
        tuple("${sample_id}${suffix}", marker_quant_dir)
    }

    def clusterCsvCh = Channel.empty()
    def clusterMembershipPngCh = Channel.empty()
    def clusterAssessmentCh = Channel.empty()
    if (runClustering) {
        def clusteringInputCh = auxiliary_kodama_dir_ch
            .join(auxiliaryObjectsCh)
            .flatMap { sample_id, kodama_dir, objects_csv ->
                clusterVariants.collect { spec ->
                    def sampleKey = "${sample_id}::${spec.variant}"
                    tuple(sampleKey, sample_id, spec.variant, spec.profile, clusterResolution, kodama_dir, objects_csv)
                }
            }
        RUN_RCODE_CELL_AUXILIARY_CLUSTERING(clusteringInputCh)
        clusterCsvCh = RUN_RCODE_CELL_AUXILIARY_CLUSTERING.out.cluster_csv
            .ifEmpty { error 'Auxiliary cell-centred clustering emitted no output; check KODAMA and cell-observation joins.' }
        clusterMembershipPngCh = RUN_RCODE_CELL_AUXILIARY_CLUSTERING.out.membership_png

        def assessmentContextCh = auxiliaryObjectsCh
            .join(auxiliaryMarkersCh)
            .flatMap { sample_id, objects_csv, marker_quant_dir ->
                clusterVariants.collect { spec ->
                    tuple("${sample_id}::${spec.variant}", objects_csv, marker_quant_dir)
                }
            }
        def assessmentInputCh = clusterCsvCh
            .join(assessmentContextCh)
            .map { sample_key, sample_id, cluster_variant, cluster_csv, objects_csv, marker_quant_dir ->
                tuple(sample_key, sample_id, cluster_variant, cluster_csv, objects_csv, marker_quant_dir)
            }
        ASSESS_CELL_AUXILIARY_CLUSTERING(assessmentInputCh)
        clusterAssessmentCh = ASSESS_CELL_AUXILIARY_CLUSTERING.out.assessment_dir
    } else if (runClusterMask) {
        clusterCsvCh = auxiliarySamplesCh.flatMap { sample_id, _image_input ->
            clusterVariants.collect { spec ->
                def sampleKey = "${sample_id}::${spec.variant}"
                tuple(sampleKey, sample_id, spec.variant, file("${params.outdir_base}/11_clustering/${sample_id}/${sample_id}_${spec.variant}_cluster.csv", checkIfExists: true))
            }
        }
    }
    if (!runClustering && runMedsamRefine) {
        clusterMembershipPngCh = auxiliarySamplesCh.flatMap { sample_id, _image_input ->
            clusterVariants.collect { spec ->
                def sampleKey = "${sample_id}::${spec.variant}"
                tuple(sampleKey, sample_id, spec.variant, file("${params.outdir_base}/11_clustering/${sample_id}/${sample_id}_${spec.variant}_cluster_kodama_membership.png", checkIfExists: true))
            }
        }
    }

    def clusterMaskCh = Channel.empty()
    def clusterUncertaintyCh = Channel.empty()
    if (runClusterMask) {
        def labelsVariantCh = auxiliaryLabelsCh.flatMap { sample_id, labels_tif ->
            clusterVariants.collect { spec -> tuple("${sample_id}::${spec.variant}", labels_tif) }
        }
        def previewVariantCh = auxiliaryImagesCh.flatMap { sample_id, image_tif ->
            clusterVariants.collect { spec -> tuple("${sample_id}::${spec.variant}", image_tif) }
        }
        def clusterMaskInputCh = clusterCsvCh
            .join(labelsVariantCh)
            .join(previewVariantCh)
            .map { sample_key, sample_id, cluster_variant, cluster_csv, labels_tif, image_tif ->
                tuple(sample_key, sample_id, cluster_variant, labels_tif, cluster_csv, image_tif)
            }
        LABELS_TO_CELL_AUXILIARY_CLUSTER_MASK(clusterMaskInputCh)
        clusterMaskCh = LABELS_TO_CELL_AUXILIARY_CLUSTER_MASK.out.cluster_mask
            .ifEmpty { error 'Auxiliary cell-centred cluster-mask generation emitted no output; check cluster and cytoplasm-label joins.' }
        clusterUncertaintyCh = LABELS_TO_CELL_AUXILIARY_CLUSTER_MASK.out.uncertainty_mask
    } else if (runGrowTissue || runMedsamRefine) {
        clusterMaskCh = auxiliarySamplesCh.flatMap { sample_id, _image_input ->
            clusterVariants.collect { spec ->
                def sampleKey = "${sample_id}::${spec.variant}"
                tuple(sampleKey, sample_id, spec.variant, file("${params.outdir_base}/12_cluster_mask/${sample_id}/${sample_id}_${spec.variant}_cluster_mask.tif", checkIfExists: true))
            }
        }
        clusterUncertaintyCh = auxiliarySamplesCh.flatMap { sample_id, _image_input ->
            clusterVariants.collect { spec ->
                tuple("${sample_id}::${spec.variant}", sample_id, spec.variant, file("${params.outdir_base}/12_cluster_mask/${sample_id}/${sample_id}_${spec.variant}_cluster_uncertainty_mask.tif", checkIfExists: true))
            }
        }
    }

    def imageVariantCh = Channel.empty()
    def tissueVariantCh = Channel.empty()
    def resolutionVariantCh = Channel.empty()
    if (runGrowTissue || runMedsamRefine) {
        imageVariantCh = auxiliaryImagesCh.flatMap { sample_id, image_tif ->
            clusterVariants.collect { spec -> tuple("${sample_id}::${spec.variant}", image_tif) }
        }
        tissueVariantCh = auxiliaryTissueMasksCh.flatMap { sample_id, tissue_mask_tif ->
            clusterVariants.collect { spec -> tuple("${sample_id}::${spec.variant}", tissue_mask_tif) }
        }
        resolutionVariantCh = auxiliaryResolutionCh.flatMap { sample_id, resolution_json ->
            clusterVariants.collect { spec -> tuple("${sample_id}::${spec.variant}", resolution_json) }
        }
    }

    def grownMaskCh = Channel.empty()
    if (runGrowTissue) {
        def growInputCh = clusterMaskCh
            .join(imageVariantCh)
            .join(tissueVariantCh)
            .join(resolutionVariantCh)
            .map { sample_key, sample_id, cluster_variant, cluster_mask_tif, image_tif, tissue_mask_tif, resolution_json ->
                tuple(sample_key, sample_id, cluster_variant, image_tif, cluster_mask_tif, tissue_mask_tif, resolution_json)
            }
        GROW_CELL_AUXILIARY_TO_TISSUE(growInputCh)
        grownMaskCh = GROW_CELL_AUXILIARY_TO_TISSUE.out.grown_mask
            .ifEmpty { error 'Auxiliary cell-centred tissue growth emitted no output; check mask, image, tissue-support and resolution joins.' }
    } else if (runMedsamRefine || (runClusterGeoJSON && grownRefineMethod in ['none', ''])) {
        grownMaskCh = auxiliarySamplesCh.flatMap { sample_id, _image_input ->
            clusterVariants.collect { spec ->
                def sampleKey = "${sample_id}::${spec.variant}"
                tuple(sampleKey, sample_id, spec.variant, file("${params.outdir_base}/13_grown_tissue/${sample_id}/${sample_id}_${spec.variant}_grown_mask.ome.tif", checkIfExists: true))
            }
        }
    }

    def finalMaskCh = grownMaskCh
    def refinedMaskCh = Channel.empty()
    def finalUncertaintyCh = clusterUncertaintyCh
    def finalProvenanceCh = Channel.empty()
    def provenanceMetadataCh = Channel.empty()
    if (runMedsamRefine && grownRefineMethod == 'medsam_border_refine') {
        def clusterMaskFileCh = clusterMaskCh.map { sample_key, sample_id, cluster_variant, cluster_mask_tif ->
            tuple(sample_key, cluster_mask_tif)
        }
        def membershipFileCh = clusterMembershipPngCh.map { sample_key, sample_id, cluster_variant, membership_png ->
            tuple(sample_key, membership_png)
        }
        def uncertaintyFileCh = clusterUncertaintyCh.map { sample_key, sample_id, cluster_variant, uncertainty_tif ->
            tuple(sample_key, uncertainty_tif)
        }
        def refineInputCh = grownMaskCh
            .join(imageVariantCh)
            .join(clusterMaskFileCh)
            .join(tissueVariantCh)
            .join(membershipFileCh)
            .join(resolutionVariantCh)
            .join(uncertaintyFileCh, failOnMismatch: true, failOnDuplicate: true)
            .map { sample_key, sample_id, cluster_variant, grown_mask_tif, image_tif, cluster_mask_tif, tissue_mask_tif, membership_png, resolution_json, uncertainty_tif ->
                tuple(sample_key, sample_id, cluster_variant, image_tif, cluster_mask_tif, grown_mask_tif, tissue_mask_tif, membership_png, resolution_json, uncertainty_tif)
            }
        REFINE_CELL_AUXILIARY_MEDSAM(refineInputCh, runtime_plan)
        refinedMaskCh = REFINE_CELL_AUXILIARY_MEDSAM.out.refined_mask
            .ifEmpty { error 'Auxiliary cell-centred MedSAM refinement emitted no output; check route-specific masks and clustering preview.' }
        finalMaskCh = refinedMaskCh
        finalUncertaintyCh = REFINE_CELL_AUXILIARY_MEDSAM.out.refined_uncertainty
        finalProvenanceCh = REFINE_CELL_AUXILIARY_MEDSAM.out.refined_provenance
        provenanceMetadataCh = REFINE_CELL_AUXILIARY_MEDSAM.out.provenance_metadata
    } else if (runMedsamRefine && !(grownRefineMethod in ['none', ''])) {
        error "Unsupported grown_tissue_refine_method: ${grownRefineMethod}"
    } else if (runClusterGeoJSON && grownRefineMethod == 'medsam_border_refine') {
        finalMaskCh = auxiliarySamplesCh.flatMap { sample_id, _image_input ->
            clusterVariants.collect { spec ->
                def sampleKey = "${sample_id}::${spec.variant}"
                tuple(sampleKey, sample_id, spec.variant, file("${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/${sample_id}_${spec.variant}_grown_mask_refined.ome.tif", checkIfExists: true))
            }
        }
        finalUncertaintyCh = finalMaskCh.map { sample_key, sample_id, variant, mask ->
            tuple(sample_key, sample_id, variant, file("${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/${sample_id}_${variant}_grown_mask_refined_uncertainty.tif", checkIfExists: true))
        }
        finalProvenanceCh = finalMaskCh.map { sample_key, sample_id, variant, mask ->
            tuple(sample_key, sample_id, variant, file("${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/${sample_id}_${variant}_grown_mask_refined_provenance.tif", checkIfExists: true))
        }
        provenanceMetadataCh = finalMaskCh.map { sample_key, sample_id, variant, mask ->
            tuple(sample_key, sample_id, variant, file("${params.outdir_base}/14_medsam_refine_tissue/${sample_id}/${sample_id}_${variant}_grown_mask_refined.ome.tif.provenance.json", checkIfExists: true))
        }
    } else if (runClusterGeoJSON && !(grownRefineMethod in ['none', ''])) {
        error "Unsupported grown_tissue_refine_method: ${grownRefineMethod}"
    }

    def clusterGeoJSONCh = Channel.empty()
    def vectorProvenanceSummaryCh = Channel.empty()
    if (runClusterGeoJSON) {
        def vectorInputCh
        if (grownRefineMethod == 'medsam_border_refine') {
            vectorInputCh = finalMaskCh
                .join(finalUncertaintyCh, by: [0, 1, 2], failOnMismatch: true, failOnDuplicate: true)
                .join(finalProvenanceCh, by: [0, 1, 2], failOnMismatch: true, failOnDuplicate: true)
                .join(provenanceMetadataCh, by: [0, 1, 2], failOnMismatch: true, failOnDuplicate: true)
                .map { key, id, variant, mask, uncertainty, provenance, metadata ->
                    tuple(key, id, variant, mask, uncertainty, provenance, metadata, [uncertainty: true, refinement: true])
                }
        } else {
            vectorInputCh = finalMaskCh
                .join(finalUncertaintyCh, by: [0, 1, 2], remainder: true, failOnDuplicate: true)
                .map { key, id, variant, mask, uncertainty ->
                    if (mask == null) error "Unmatched auxiliary uncertainty sidecar for vectorization: ${key}"
                    tuple(key, id, variant, mask, uncertainty ?: mask, mask, mask, [uncertainty: uncertainty != null, refinement: false])
                }
        }
        CELL_AUXILIARY_MASK_TO_GEOJSON(vectorInputCh)
        clusterGeoJSONCh = CELL_AUXILIARY_MASK_TO_GEOJSON.out.cluster_geojson
            .ifEmpty { error 'Auxiliary cell-centred GeoJSON export emitted no output; check the selected final mask.' }
        vectorProvenanceSummaryCh = CELL_AUXILIARY_MASK_TO_GEOJSON.out.provenance_summary
    }

    emit:
    cluster_csv = clusterCsvCh
    cluster_assessment = clusterAssessmentCh
    cluster_mask = clusterMaskCh
    grown_mask = grownMaskCh
    refined_mask = refinedMaskCh
    final_uncertainty = finalUncertaintyCh
    final_provenance = finalProvenanceCh
    provenance_metadata = provenanceMetadataCh
    cluster_geojson = clusterGeoJSONCh
    vector_provenance_summary = vectorProvenanceSummaryCh
}
