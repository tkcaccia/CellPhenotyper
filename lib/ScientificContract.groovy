import groovy.json.JsonOutput

class ScientificContract {
    private static final Map ALIASES = [
        exploratory_analysis: 'exploratory',
        segmentation: 'cell_segmentation',
        cell_identification: 'cell_segmentation',
        phenotype: 'cell_phenotyping',
        phenotyping: 'cell_phenotyping',
        tissue_domains: 'tissue_domain_discovery',
        tissue_segmentation: 'tissue_domain_discovery',
        virtual_mif: 'virtual_staining',
        prediction: 'outcome_prediction'
    ]

    private static final Map DEFINITIONS = [
        exploratory: ['Exploratory multimodal tissue characterization', null],
        cell_segmentation: ['Research-use nuclear instance identification', 'nuclear_instance'],
        cell_phenotyping: ['Research-use cell-centred contextual phenotype discovery', 'contextual_cell'],
        tissue_domain_discovery: ['Research-use spatial tissue-domain discovery and segmentation', 'tissue_grid'],
        virtual_staining: ['Research-use virtual mIF generation and marker quantification', 'image_pixel_and_cell_region'],
        outcome_prediction: ['Research-use section representation and outcome prediction', 'selected_tissue_section']
    ]

    static Map resolve(
        def requested,
        String uni2Mode,
        String cellDetectionMode,
        boolean gigatimeEnabled,
        boolean titanEnabled,
        boolean pathofmpredEnabled,
        String startPoint,
        String endPoint
    ) {
        String intent = (requested ?: 'exploratory').toString().trim().toLowerCase().replace('-', '_')
        intent = ALIASES.getOrDefault(intent, intent)
        if (!DEFINITIONS.containsKey(intent)) {
            throw new IllegalArgumentException("Invalid --analysis_intent '${requested}'. Use ${DEFINITIONS.keySet().join(', ')}.")
        }
        boolean gridPrimary = uni2Mode in ['grid', 'both']
        if (intent == 'cell_phenotyping' && uni2Mode != 'cells') {
            throw new IllegalArgumentException('analysis_intent=cell_phenotyping requires --uni2_sampling_mode cells so cell-centred observations remain the primary clustering route.')
        }
        if (intent == 'tissue_domain_discovery' && !gridPrimary) {
            throw new IllegalArgumentException('analysis_intent=tissue_domain_discovery requires --uni2_sampling_mode grid or both.')
        }
        if (intent == 'virtual_staining' && !gigatimeEnabled) {
            throw new IllegalArgumentException('analysis_intent=virtual_staining requires --gigatime_enable true.')
        }
        if (intent == 'outcome_prediction' && !pathofmpredEnabled) {
            throw new IllegalArgumentException('analysis_intent=outcome_prediction requires --pathofmpred_enable true and an explicitly configured cancer type.')
        }

        String observationUnit = DEFINITIONS[intent][1] ?: (gridPrimary ? 'tissue_grid' : 'contextual_cell')
        [
            schema_version: 2,
            pipeline_primary_intended_use: 'Research-use exploratory multimodal characterization of H&E whole-slide images',
            analysis_intent: intent,
            intended_use: DEFINITIONS[intent][0],
            observation_unit: observationUnit,
            research_use_only: true,
            clinical_use_validated: false,
            stage_window: [start: startPoint, end: endPoint],
            scientific_route: [
                cell_detection_mode: cellDetectionMode,
                uni2_sampling_mode: uni2Mode,
                primary_embedding_route: gridPrimary ? 'tissue_grid' : 'contextual_cell',
                auxiliary_cell_comparison: uni2Mode == 'both',
                cell_mask_growth_applicable: !gridPrimary,
                gigatime_enabled: gigatimeEnabled,
                titan_enabled: titanEnabled,
                pathofmpred_enabled: pathofmpredEnabled
            ],
            uncertainty_policy: [
                physical_calibration: [
                    evidence: 'source metadata plus recorded fallback provenance',
                    abstention: 'strict-MPP stages fail when scale cannot be resolved'
                ],
                grandqc: [
                    evidence: 'model class scores, masks, artifact overlays and resolution metadata',
                    abstention: 'artifact and background are excluded from downstream support; scores are not accuracy estimates'
                ],
                cell_instances: [
                    evidence: 'broad-scope detector agreement, separately reported scoped support, detector-specific probabilities and geometry agreement tier',
                    abstention: 'missing broad consensus, excessive broad-detector distance, insufficient support or low agreement remain in the alignment audit but are excluded from canonical instances'
                ],
                cell_phenotype: [
                    evidence: 'detector-specific phenotype labels remain separate',
                    abstention: 'no cross-taxonomy consensus phenotype is fabricated'
                ],
                virtual_markers: [
                    evidence: 'raw virtual-marker scores plus block-boundary seam QC',
                    abstention: 'seam gate can fail publication output; marker values remain uncalibrated biological proxies'
                ],
                representation: [
                    evidence: 'grid/cell route coverage, Procrustes agreement, neighborhood overlap, spatial coherence and marker-proxy prediction',
                    abstention: 'route comparison never selects a preferred biological interpretation automatically'
                ],
                clustering: [
                    evidence: 'landmark vote fraction/margin and repeated-seed adjusted Rand index',
                    abstention: 'ambiguous or seed-unstable observations have no interpretable_cluster when enabled'
                ],
                medsam: [
                    evidence: 'added, removed and relabelled pixels plus native-resolution QC crops and GrandQC leakage assertion',
                    abstention: 'forbidden support fails; edit magnitude is QC evidence rather than calibrated confidence'
                ],
                outcome_prediction: [
                    evidence: 'model outputs and provenance',
                    abstention: 'research-only scores are not calibrated clinical probabilities'
                ]
            ],
            interpretation_limits: [
                'Pipeline outputs are research measurements and are not clinical diagnoses.',
                'Detector agreement is not accuracy against an expert reference standard.',
                'Virtual marker scores are not measured protein abundance or calibrated probabilities.',
                'Unsupervised clusters require stability analysis and independent biological interpretation.'
            ]
        ]
    }

    static void write(Map contract, File outputFile) {
        outputFile.parentFile.mkdirs()
        outputFile.text = JsonOutput.prettyPrint(JsonOutput.toJson(contract)) + System.lineSeparator()
    }
}
