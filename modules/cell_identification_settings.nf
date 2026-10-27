// Parameter-only validation; no subprocess, model loading or scheduler action.
def resolveCellIdentificationSettings(Map params, String resolved_compute_device) {
    def cell_identification_enabled = (((params.cell_identification_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
    def input_scope = (params.cellvit_input_scope ?: 'full_slide').toString().trim().toLowerCase()
    if (!(input_scope in ['full_slide', 'roi'])) {
      throw new IllegalArgumentException("Invalid --cellvit_input_scope. Use full_slide or roi.")
    }
    def cell_segmentation_review_enabled = (((params.cell_segmentation_review_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
    def cellvit_taxonomies = (params.cellvit_taxonomies ?: 'all').toString().trim().toLowerCase().split(/[\s,]+/).findAll { it }.unique()
    def known_cellvit_taxonomies = ['all', 'pannuke', 'consep', 'lizard', 'nucls_main', 'nucls_super', 'ocelot', 'midog', 'panoptils']
    if (!cellvit_taxonomies || cellvit_taxonomies.any { !(it in known_cellvit_taxonomies) } || (cellvit_taxonomies.contains('all') && cellvit_taxonomies.size() != 1)) {
      throw new IllegalArgumentException("Invalid --cellvit_taxonomies. Use all, or a comma/space-separated list of pannuke,consep,lizard,nucls_main,nucls_super,ocelot,midog,panoptils.")
    }
    if (cell_identification_enabled && resolved_compute_device != 'gpu') {
      throw new IllegalArgumentException("CellViT segmentation for --cell_identification_enable requires an explicitly resolved GPU runtime; cached-head classification can instead use bin/classify_cellvit_features.py on CPU.")
    }
    if (cell_segmentation_review_enabled && !cell_identification_enabled) {
      throw new IllegalArgumentException("--cell_segmentation_review_enable requires --cell_identification_enable true; review does not replace the canonical detector population.")
    }
    if (cell_identification_enabled && (params.cellvit_classification_device ?: 'cpu').toString() != 'cpu') {
      throw new IllegalArgumentException("The pipeline's cached cell-type heads run on CPU; use --cellvit_classification_device cpu.")
    }
    if (cell_identification_enabled && input_scope == 'full_slide' && cellvit_taxonomies != ['all']) {
      throw new IllegalArgumentException("Full-slide CellViT requires all eight pretrained taxonomies; use --cellvit_taxonomies all.")
    }
    def feature_cache_path = (params.virchow2_feature_cache ?: '').toString().trim()
    if (feature_cache_path) {
      def cacheDir = new File(feature_cache_path)
      if (!cacheDir.isAbsolute() || !cacheDir.isDirectory() || !cacheDir.canWrite()) {
        throw new IllegalArgumentException("--virchow2_feature_cache must be an existing writable absolute directory shared by extraction and Bora; create and mount it explicitly.")
      }
      params.virchow2_feature_cache = cacheDir.canonicalPath
    }
    def classifier_cache_path = (params.cellvit_classifier_cache ?: '').toString().trim()
    if (cell_identification_enabled && classifier_cache_path && (!new File(classifier_cache_path).isAbsolute() || !new File(classifier_cache_path).isDirectory())) {
      throw new IllegalArgumentException("--cellvit_classifier_cache must be an existing absolute classifier-cache directory; models are never downloaded by this route.")
    }
    return [enabled: cell_identification_enabled, review: cell_segmentation_review_enabled,
            input_scope: input_scope, taxonomies: cellvit_taxonomies, feature_cache_path: feature_cache_path,
            classification_device: 'cpu', compute_device: resolved_compute_device,
            default_mpp: cell_identification_enabled ? 0.0 :
                (params.cellvit_default_mpp != null ? params.cellvit_default_mpp : 0.25)]
}

def annotateCellIdentificationContract(Map contract, Map settings, String endpoint) {
    contract.scientific_route.cell_identification_enabled = settings.enabled
    contract.scientific_route.cellvit_input_scope = settings.enabled ? settings.input_scope : 'disabled'
    contract.scientific_route.cellvit_taxonomies = settings.enabled ? settings.taxonomies : ['pannuke']
    contract.scientific_route.cell_segmentation_review_enabled = settings.review
    contract.scientific_route.cell_segmentation_review_changes_canonical_population = false
    contract.scientific_route.virchow2_shared_cache_enabled = !!settings.feature_cache_path
    if (endpoint == 'cell_identification') {
        contract.observation_unit = 'detected_nuclear_instance'
        contract.scientific_route.primary_embedding_route = 'cellvit_cached_cell'
    }
    return contract
}

def validateFullslideCellIdentificationInputs(Map params, int sample_count) {
    if ((params.cellvit_fullslide_model ?: 'SAM').toString() != 'SAM' ||
        !(((params.cellvit_amp == null) ? true : params.cellvit_amp).toString().trim().toLowerCase() in ['true', '1', 'yes', 'y', 'on'])) {
      throw new IllegalArgumentException('Full-slide CellViT requires the SAM backbone and AMP for its pinned eight-head contract.')
    }
    def required = ['cellvit_fullslide_image', 'cellvit_fullslide_shift',
      'cellvit_fullslide_clean_mask', 'cellvit_fullslide_bundle',
      'cellvit_classifier_cache', 'cellvit_fullslide_gpu_uuid',
      'cellvit_fullslide_runtime_sif', 'cellvit_fullslide_runtime_sif_sha256',
      'cellvit_fullslide_runtime_python', 'cellvit_fullslide_runtime_executable']
    def missing = required.findAll { !((params[it] ?: '').toString().trim()) }
    if (missing) throw new IllegalArgumentException(
      "Full-slide CellViT requires persistent, pinned inputs: ${missing.join(', ')}. The crop shift cannot be reused.")
    for (key in ['cellvit_fullslide_image', 'cellvit_fullslide_shift',
                 'cellvit_fullslide_clean_mask', 'cellvit_fullslide_bundle']) {
      def value = params[key].toString().trim()
      if (sample_count > 1 && !value.contains('{sample_id}')) {
        throw new IllegalArgumentException("--${key} must contain {sample_id} when processing multiple images.")
      }
      if (!new File(value.replace('{sample_id}', 'sample')).isAbsolute()) {
        throw new IllegalArgumentException("--${key} must be an absolute persistent path.")
      }
    }
    def boraKeys = ['cellvit_fullslide_bora_geojson', 'cellvit_fullslide_bora_receipt',
                    'cellvit_fullslide_expected_bora_sha256', 'cellvit_fullslide_combined_output']
    def suppliedBora = boraKeys.findAll { (params[it] ?: '').toString().trim() }
    if (suppliedBora && suppliedBora.size() != boraKeys.size()) {
      throw new IllegalArgumentException('Full-slide Bora merge requires GeoJSON, accepted receipt, exact SHA-256 and combined output together.')
    }
    if (suppliedBora) {
      if (sample_count > 1) {
        throw new IllegalArgumentException('Main-pipeline Bora merge currently requires one image per run so its GeoJSON and exact SHA-256 cannot be applied to another tissue.')
      }
      for (key in ['cellvit_fullslide_bora_geojson', 'cellvit_fullslide_bora_receipt',
                   'cellvit_fullslide_combined_output']) {
        if (!new File(params[key].toString().trim()).isAbsolute()) {
          throw new IllegalArgumentException("--${key} must be an absolute persistent path.")
        }
      }
      if (!(params.cellvit_fullslide_expected_bora_sha256.toString().trim() ==~ /[0-9a-f]{64}/)) {
        throw new IllegalArgumentException('--cellvit_fullslide_expected_bora_sha256 must be an exact lower-case SHA-256.')
      }
    }
    return true
}
