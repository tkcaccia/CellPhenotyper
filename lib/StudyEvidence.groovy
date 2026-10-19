import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import java.security.MessageDigest

class StudyEvidence {
    private static final Set PHASES = [
        'exploratory', 'development', 'internal_testing', 'external_testing'
    ] as Set
    private static final Set GATE_MODES = ['off', 'warn', 'fail'] as Set

    private static String textValue(def value) {
        value == null ? '' : value.toString().trim()
    }

    private static int intValue(def value) {
        try {
            return Math.max(0, value as int)
        } catch (Throwable ignored) {
            return 0
        }
    }

    private static String sha256(File file) {
        MessageDigest digest = MessageDigest.getInstance('SHA-256')
        file.withInputStream { input ->
            byte[] buffer = new byte[1024 * 1024]
            int count
            while ((count = input.read(buffer)) > 0) digest.update(buffer, 0, count)
        }
        digest.digest().encodeHex().toString()
    }

    static Map resolve(def requestedPath, def requestedGateMode, String analysisIntent) {
        String gateMode = textValue(requestedGateMode ?: 'warn').toLowerCase()
        if (!GATE_MODES.contains(gateMode)) {
            throw new IllegalArgumentException("Invalid --evidence_gate_mode '${requestedGateMode}'. Use off, warn, or fail.")
        }

        String path = textValue(requestedPath)
        if (!path) {
            return [
                schema_version: 1,
                status: 'not_declared',
                gate_mode: gateMode,
                gate_passed: false,
                study_phase: 'not_declared',
                analysis_intent: analysisIntent,
                source_manifest: null,
                source_manifest_sha256: null,
                issues: [
                    'No study manifest was supplied.',
                    'A primary endpoint and statistical unit have not been declared.',
                    'Cohort independence, reference standard and prespecification have not been declared.'
                ],
                claim_ceiling: 'engineering_feasibility_and_exploratory_description_only',
                interpretation: 'Pipeline completion does not establish biological accuracy, clinical validity, or generalizability.'
            ]
        }

        File source = new File(path).canonicalFile
        if (!source.exists() || !source.isFile()) {
            throw new IllegalArgumentException("Study manifest does not exist or is not a file: ${source}")
        }
        def parsed = new JsonSlurper().parse(source)
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("Study manifest must contain one JSON object: ${source}")
        }
        Map manifest = (Map) parsed
        List<String> issues = []
        String phase = textValue(manifest.study_phase).toLowerCase()
        if (!PHASES.contains(phase)) issues << "study_phase must be one of ${PHASES.join(', ')}"
        if (!textValue(manifest.study_id)) issues << 'study_id is required'
        if (!textValue(manifest.intended_use)) issues << 'intended_use is required'
        String manifestIntent = textValue(manifest.analysis_intent).toLowerCase()
        if (!manifestIntent) {
            issues << 'analysis_intent is required'
        } else if (manifestIntent != analysisIntent) {
            issues << "analysis_intent '${manifestIntent}' does not match pipeline intent '${analysisIntent}'"
        }

        Map endpoint = manifest.primary_endpoint instanceof Map ? (Map) manifest.primary_endpoint : [:]
        if (!textValue(endpoint.name)) issues << 'primary_endpoint.name is required'
        if (!textValue(endpoint.statistical_unit)) issues << 'primary_endpoint.statistical_unit is required'
        if (!textValue(endpoint.metric)) issues << 'primary_endpoint.metric is required'

        Map cohorts = manifest.cohorts instanceof Map ? (Map) manifest.cohorts : [:]
        String splitUnit = textValue(cohorts.split_unit).toLowerCase()
        if (!splitUnit) issues << 'cohorts.split_unit is required'
        if (phase in ['internal_testing', 'external_testing']) {
            if (!(splitUnit in ['patient', 'participant', 'subject'])) {
                issues << 'testing cohorts must be split at patient/participant/subject level'
            }
            if (intValue(cohorts.test_subjects) < 1) issues << 'cohorts.test_subjects must be greater than zero'
        }
        if (phase == 'external_testing') {
            if (intValue(cohorts.external_test_subjects) < 1) issues << 'cohorts.external_test_subjects must be greater than zero'
            if (intValue(cohorts.external_sites) < 1) issues << 'cohorts.external_sites must be greater than zero'
        }

        Map reference = manifest.reference_standard instanceof Map ? (Map) manifest.reference_standard : [:]
        Map prespecification = manifest.prespecification instanceof Map ? (Map) manifest.prespecification : [:]
        if (phase in ['internal_testing', 'external_testing']) {
            if (textValue(reference.status).toLowerCase() != 'available') issues << 'reference_standard.status must be available for testing'
            if (!textValue(reference.definition)) issues << 'reference_standard.definition is required for testing'
            if (!textValue(reference.rationale)) issues << 'reference_standard.rationale is required for testing'
            if (intValue(reference.annotator_count) < 1) issues << 'reference_standard.annotator_count must be greater than zero'
            if (!textValue(reference.annotator_expertise)) issues << 'reference_standard.annotator_expertise is required for testing'
            if (prespecification.locked_before_testing != true) issues << 'prespecification.locked_before_testing must be true for testing'
            if (!textValue(prespecification.protocol_uri)) issues << 'prespecification.protocol_uri is required for testing'
        }

        boolean ready = issues.isEmpty()
        String status = ready ? "declared_${phase ?: 'unknown'}" : 'incomplete'
        String claimCeiling
        if (!ready || phase in ['exploratory', 'development']) {
            claimCeiling = 'engineering_feasibility_and_exploratory_description_only'
        } else if (phase == 'internal_testing') {
            claimCeiling = 'internal_testing_results_only_pending_metric_verification_and_independent_review'
        } else {
            claimCeiling = 'external_testing_results_pending_metric_verification_and_independent_review'
        }
        [
            schema_version: 1,
            status: status,
            gate_mode: gateMode,
            gate_passed: ready,
            study_phase: phase ?: 'invalid',
            analysis_intent: analysisIntent,
            source_manifest: source.absolutePath,
            source_manifest_sha256: sha256(source),
            declared_design: manifest,
            issues: issues,
            claim_ceiling: claimCeiling,
            reporting_frameworks: [
                DOME: 'https://doi.org/10.1038/s41592-021-01205-4',
                CLAIM_2024: 'https://doi.org/10.1148/ryai.240300'
            ],
            interpretation: 'This is a machine-check of declared study design fields, not verification of the evidence or reported metrics. Use the term reference standard for the benchmark.'
        ]
    }

    static void write(Map report, File outputFile) {
        outputFile.parentFile.mkdirs()
        outputFile.text = JsonOutput.prettyPrint(JsonOutput.toJson(report)) + System.lineSeparator()
    }

    static String writeReports(Map contract, Map report, def requestedOutputRoot) {
        String outputRoot = textValue(requestedOutputRoot)
        if (!outputRoot || outputRoot.contains('://')) return ''
        File executionDir = new File(outputRoot, '00_execution')
        File contractFile = new File(executionDir, 'analysis_contract.json')
        ScientificContract.write(contract, contractFile)
        write(report, new File(executionDir, 'validation_readiness.json'))
        contractFile.absolutePath
    }

    static String gateMessage(Map report) {
        "Study evidence gate: status=${report.status}; claim ceiling=${report.claim_ceiling}; issues=${report.issues.join(' | ')}"
    }
}
