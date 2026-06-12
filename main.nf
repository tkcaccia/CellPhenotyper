import groovy.io.FileType
import groovy.json.JsonOutput

nextflow.enable.dsl = 2

include { PREPARE_INPUT_OMETIFF } from './modules/prepare_input_ometiff'
include { RUN_GRANDQC_ARTIFACT_ANALYSIS } from './modules/run_grandqc_artifact_analysis'
include { PREPARE_ROI_GEOJSON } from './modules/prepare_roi_geojson'
include { PREPARE_STARDIST_AUTO_ROI } from './modules/prepare_stardist_auto_roi'
include { RUN_STARDIST_ROI_SEGMENTATION } from './modules/run_stardist_roi_segmentation'
include { DETECT_TMA_SPOTS } from './modules/detect_tma_spots'
include { RUN_GIGATIME_ON_CROP } from './modules/run_gigatime_on_crop'
include { ROI_GEOJSON_TO_MASK } from './modules/roi_geojson_to_mask'
include { BUILD_TISSUE_MASK } from './modules/build_tissue_mask'
include { MAP_CELLS_TO_ROI_POLYGONS } from './modules/map_cells_to_roi_polygons'
include { EXPAND_LABELS_TO_CYTOPLASM as EXPAND_LABELS_TO_CYTOPLASM_PRIMARY } from './modules/expand_labels_to_cytoplasm'
include { EXPAND_LABELS_TO_CYTOPLASM as EXPAND_LABELS_TO_CYTOPLASM_FULL } from './modules/expand_labels_to_cytoplasm'
include { QUANTIFY_GIGATIME_INTENSITY } from './modules/quantify_gigatime_intensity'
include { EXTRACT_UNI2_EMBEDDINGS } from './modules/extract_uni2_embeddings'
include { EXTRACT_UNI2_EMBEDDINGS_SHARED } from './modules/extract_uni2_embeddings_shared'
include { RUN_KODAMA_ANALYSIS } from './modules/run_kodama_analysis'
include { RUN_RCODE_CLUSTERING } from './modules/run_rcode_clustering'
include { LABELS_TO_CLUSTER_MASK } from './modules/labels_to_cluster_mask'
include { GROW_TO_TISSUE } from './modules/grow_to_tissue'
include { REFINE_GROWN_TISSUE_MEDSAM } from './modules/refine_grown_tissue_medsam'
include { MASK_TO_GEOJSON } from './modules/mask_to_geojson'

workflow {
  def run_full_pipeline = params.run_full_pipeline as boolean
  def tissue_mask_from_input = params.tissue_mask_from_input as boolean
  def active_profiles = (workflow.profile ?: '')
    .split(',')
    .collect { it.trim().toLowerCase() }
    .findAll { it }

  def runtime_profiles = active_profiles.intersect(['singularity', 'docker'])
  if (runtime_profiles.size() > 1) {
    error "Select only one runtime profile: use either '-profile singularity' or '-profile docker' (not both)."
  }

  def normalize_arch = { raw_arch ->
    def value = (raw_arch ?: '').toString().trim().toLowerCase()
    if (value in ['x86_64', 'amd64', 'x64', 'x86-64']) return 'amd64'
    if (value in ['aarch64', 'arm64', 'arm64v8', 'arm64/v8', 'armv8', 'armv8l']) return 'arm64'
    if (value == 'auto') return 'auto'
    return value ?: 'unknown'
  }

  def command_output = { String cmd ->
    try {
      def proc = ['bash', '-lc', cmd].execute()
      def finished = proc.waitFor(2, java.util.concurrent.TimeUnit.SECONDS)
      if (!finished) {
        proc.destroyForcibly()
        return ''
      }
      if (proc.exitValue() != 0) return ''
      return proc.in.text.trim()
    } catch (Throwable ignored) {
      return ''
    }
  }

  def command_succeeds = { String cmd ->
    try {
      def proc = ['bash', '-lc', cmd].execute()
      def finished = proc.waitFor(2, java.util.concurrent.TimeUnit.SECONDS)
      if (!finished) {
        proc.destroyForcibly()
        return false
      }
      return proc.exitValue() == 0
    } catch (Throwable ignored) {
      return false
    }
  }

  def parse_positive_int = { raw_value, fallback ->
    try {
      def parsed = (raw_value ?: fallback).toString().trim().toInteger()
      return parsed > 0 ? parsed : fallback
    } catch (Throwable ignored) {
      return fallback
    }
  }

  def paramOr = { String key, def fallback ->
    params.containsKey(key) ? params[key] : fallback
  }

  // Use the user-configured CPU cap directly (validated as positive integer).
  def configured_max_cpus = parse_positive_int(paramOr('max_cpus', 4), 4)
  params.max_cpus = configured_max_cpus

  def detect_host_memory_gb = {
    def candidates = [
      command_output('awk \'/MemTotal/ {printf "%d", int($2/1024/1024)}\' /proc/meminfo'),
      command_output('free -g | awk \'/^Mem:/ {print $2}\''),
      command_output('sysctl -n hw.memsize 2>/dev/null | awk \'{printf "%d", int($1/1024/1024/1024)}\'')
    ].findAll { it && it ==~ /\d+/ }
    if (!candidates) return 8
    try {
      return Math.max(1, candidates[0].toInteger())
    } catch (Throwable ignored) {
      return 8
    }
  }
  def configured_max_memory_gb = parse_positive_int(paramOr('max_memory_gb', 8), 8)
  def host_memory_gb = detect_host_memory_gb()
  def effective_max_memory_gb = Math.max(2, Math.min(configured_max_memory_gb, host_memory_gb))
  if (effective_max_memory_gb != configured_max_memory_gb) {
    println "WARN: Reducing max_memory_gb from ${configured_max_memory_gb} to ${effective_max_memory_gb} (host total RAM: ${host_memory_gb} GB)."
  }
  params.max_memory_gb = effective_max_memory_gb

  def runtime_image_mode = (paramOr('runtime_image_mode', 'auto') ?: 'auto').toString().trim().toLowerCase()
  if (!(runtime_image_mode in ['auto', 'manual'])) {
    runtime_image_mode = 'auto'
  }

  def requested_arch_raw = (paramOr('host_arch', 'auto') ?: 'auto').toString().trim().toLowerCase()
  def requested_arch = normalize_arch(requested_arch_raw)
  if (!(requested_arch in ['auto', 'amd64', 'arm64'])) {
    requested_arch = 'auto'
  }

  def detected_arch_candidates = [
    normalize_arch(System.getProperty('os.arch')),
    normalize_arch(System.getenv('NXF_HOST_ARCH')),
    normalize_arch(System.getenv('TARGETARCH')),
    normalize_arch(command_output('uname -m')),
    normalize_arch(command_output('dpkg --print-architecture'))
  ].findAll { it && it != 'unknown' && it != 'auto' }

  def detected_arch = (requested_arch in ['amd64', 'arm64'])
    ? requested_arch
    : (detected_arch_candidates ? detected_arch_candidates[0] : 'unknown')
  if (detected_arch == 'unknown') {
    error "Could not detect host architecture. Use --host_arch amd64 or --host_arch arm64."
  }

  def requested_compute_device = (paramOr('compute_device', 'auto') ?: 'auto').toString().trim().toLowerCase()
  if (!(requested_compute_device in ['cpu', 'gpu', 'auto'])) {
    requested_compute_device = 'auto'
  }
  def enable_gpu_on_arm64 = ((paramOr('enable_gpu_on_arm64', false) ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on']
  def enable_stardist_gpu_on_arm64 = ((paramOr('enable_stardist_gpu_on_arm64', false) ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on']

  def nvidia_visible = (System.getenv('NVIDIA_VISIBLE_DEVICES') ?: '').toString().trim().toLowerCase()
  def cuda_visible = (System.getenv('CUDA_VISIBLE_DEVICES') ?: '').toString().trim().toLowerCase()
  def is_positive = { String value -> value && !(value in ['none', 'void', 'no', 'false', '-1']) }
  def detected_nvidia = is_positive(nvidia_visible) || is_positive(cuda_visible)
  if (!detected_nvidia) {
    detected_nvidia = command_succeeds('command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1')
  }

  def gpu_allowed_on_arch = (detected_arch == 'amd64') || (detected_arch == 'arm64' && enable_gpu_on_arm64)
  def resolved_compute_device = requested_compute_device == 'auto'
    ? ((gpu_allowed_on_arch && detected_nvidia) ? 'gpu' : 'cpu')
    : requested_compute_device
  if (resolved_compute_device == 'gpu' && !gpu_allowed_on_arch) {
    log.warn "compute_device='gpu' requested on ${detected_arch} but enable_gpu_on_arm64=${enable_gpu_on_arm64}. Falling back to CPU."
    resolved_compute_device = 'cpu'
  } else if (resolved_compute_device == 'gpu' && detected_arch == 'arm64') {
    log.warn "compute_device='gpu' enabled on arm64. GPU processes will run only if an arm64-compatible GPU container is available."
  }

  def container_repo = (paramOr('container_repo', 'ghcr.io/tkcaccia/cellphenotyper') ?: 'ghcr.io/tkcaccia/cellphenotyper').toString().trim()
  if (!container_repo) {
    container_repo = 'ghcr.io/tkcaccia/cellphenotyper'
  }
  def default_cpu_tag_generic = detected_arch == 'amd64' ? '0.2.0-amd64' : '0.2.0'
  def container_cpu_tag = (paramOr('container_cpu_tag', default_cpu_tag_generic) ?: default_cpu_tag_generic).toString().trim()
  if (!container_cpu_tag) {
    container_cpu_tag = default_cpu_tag_generic
  }
  def container_cpu_tag_amd64 = (paramOr('container_cpu_tag_amd64', '0.2.0-amd64') ?: '0.2.0-amd64').toString().trim()
  if (!container_cpu_tag_amd64) {
    container_cpu_tag_amd64 = '0.2.0-amd64'
  }
  def container_cpu_tag_arm64 = (paramOr('container_cpu_tag_arm64', '0.2.0') ?: '0.2.0').toString().trim()
  if (!container_cpu_tag_arm64) {
    container_cpu_tag_arm64 = '0.2.0'
  }
  def container_gpu_tag = (paramOr('container_gpu_tag', '0.2.0-gpu') ?: '0.2.0-gpu').toString().trim()
  if (!container_gpu_tag) {
    container_gpu_tag = '0.2.0-gpu'
  }

  def selected_cpu_tag = detected_arch == 'amd64' ? container_cpu_tag_amd64 : container_cpu_tag_arm64
  def auto_cpu_docker_image = "${container_repo}:${selected_cpu_tag}"
  def auto_gpu_docker_image = "${container_repo}:${container_gpu_tag}"
  def auto_docker_image = resolved_compute_device == 'gpu'
    ? (detected_arch == 'amd64' ? auto_gpu_docker_image : auto_cpu_docker_image)
    : auto_cpu_docker_image
  def auto_docker_singularity_image = "docker://${auto_docker_image}"

  def singularity_image_source = (paramOr('singularity_image_source', 'auto') ?: 'auto').toString().trim().toLowerCase()
  if (!(singularity_image_source in ['auto', 'oras', 'release', 'docker'])) {
    singularity_image_source = 'auto'
  }
  def singularity_oras_repo = (paramOr('singularity_oras_repo', container_repo) ?: container_repo).toString().trim()
  def singularity_release_repo = (paramOr('singularity_release_repo', 'tkcaccia/CellPhenotyper') ?: 'tkcaccia/CellPhenotyper').toString().trim()
  def singularity_release_tag = (paramOr('singularity_release_tag', 'v2.2') ?: 'v2.2').toString().trim()
  def singularity_cpu_asset_amd64 = (paramOr('singularity_cpu_asset_amd64', 'cellphenotyper-2.2-amd64.sif') ?: 'cellphenotyper-2.2-amd64.sif').toString().trim()
  def singularity_cpu_asset_arm64 = (paramOr('singularity_cpu_asset_arm64', 'cellphenotyper-2.2-arm64.sif') ?: 'cellphenotyper-2.2-arm64.sif').toString().trim()
  def singularity_gpu_asset_amd64 = (paramOr('singularity_gpu_asset_amd64', 'cellphenotyper-2.2-gpu-amd64.sif') ?: 'cellphenotyper-2.2-gpu-amd64.sif').toString().trim()
  def singularity_gpu_asset_arm64 = (paramOr('singularity_gpu_asset_arm64', 'cellphenotyper-2.2-gpu-arm64.sif') ?: 'cellphenotyper-2.2-gpu-arm64.sif').toString().trim()
  def singularity_cpu_oras_tag_amd64 = (paramOr('singularity_cpu_oras_tag_amd64', '2.2-sif-amd64') ?: '2.2-sif-amd64').toString().trim()
  def singularity_cpu_oras_tag_arm64 = (paramOr('singularity_cpu_oras_tag_arm64', '2.2-sif-arm64') ?: '2.2-sif-arm64').toString().trim()
  def singularity_gpu_oras_tag_amd64 = (paramOr('singularity_gpu_oras_tag_amd64', '2.2-sif-gpu-amd64') ?: '2.2-sif-gpu-amd64').toString().trim()
  def singularity_gpu_oras_tag_arm64 = (paramOr('singularity_gpu_oras_tag_arm64', '2.2-sif-gpu-arm64') ?: '2.2-sif-gpu-arm64').toString().trim()
  def singularity_local_dir = (paramOr('singularity_local_dir', '') ?: '').toString().trim()
  def cpu_container_image = (paramOr('cpu_container_image', '') ?: '').toString().trim()
  def gpu_container_image = (paramOr('gpu_container_image', '') ?: '').toString().trim()

  def selected_release_asset = ''
  def selected_oras_tag = ''
  if (resolved_compute_device == 'gpu') {
    if (detected_arch == 'amd64') {
      selected_release_asset = singularity_gpu_asset_amd64
      selected_oras_tag = singularity_gpu_oras_tag_amd64
    } else if (detected_arch == 'arm64') {
      selected_release_asset = singularity_gpu_asset_arm64 ?: ''
      selected_oras_tag = singularity_gpu_oras_tag_arm64 ?: ''
    } else {
      selected_release_asset = singularity_gpu_asset_amd64
      selected_oras_tag = singularity_gpu_oras_tag_amd64
    }
  } else {
    selected_release_asset = (detected_arch == 'amd64' ? singularity_cpu_asset_amd64 : singularity_cpu_asset_arm64)
    selected_oras_tag = (detected_arch == 'amd64' ? singularity_cpu_oras_tag_amd64 : singularity_cpu_oras_tag_arm64)
  }
  def local_sif_candidates = []
  if (selected_release_asset) {
    if (singularity_local_dir) {
      local_sif_candidates << new File(singularity_local_dir, selected_release_asset)
    }
    local_sif_candidates << new File(baseDir.toString(), selected_release_asset)
    local_sif_candidates << new File(baseDir.toString(), "singularity/${selected_release_asset}")
  }
  def local_sif_file = local_sif_candidates.find { it.exists() && it.isFile() }
  def auto_local_singularity_image = local_sif_file ? local_sif_file.absolutePath : ''
  def auto_oras_singularity_image = (singularity_oras_repo && selected_oras_tag)
    ? "oras://${singularity_oras_repo}:${selected_oras_tag}"
    : ''
  def auto_release_singularity_image = (singularity_release_repo && singularity_release_tag && selected_release_asset)
    ? "https://github.com/${singularity_release_repo}/releases/download/${singularity_release_tag}/${selected_release_asset}"
    : ''

  def release_asset_reachable = false
  if (runtime_profiles.contains('singularity') && singularity_image_source in ['auto', 'release'] && auto_release_singularity_image) {
    def escaped_release_url = auto_release_singularity_image.replace("'", "'\"'\"'")
    release_asset_reachable = command_succeeds("curl -fsIL --max-time 12 '${escaped_release_url}' >/dev/null 2>&1")
  }

  def auto_singularity_image = auto_docker_singularity_image
  def auto_singularity_origin = 'docker'
  if (runtime_profiles.contains('singularity') && singularity_image_source in ['auto', 'oras', 'release'] && auto_local_singularity_image) {
    auto_singularity_image = auto_local_singularity_image
    auto_singularity_origin = 'local'
  } else if (runtime_profiles.contains('singularity') && singularity_image_source in ['auto', 'oras'] && auto_oras_singularity_image) {
    auto_singularity_image = auto_oras_singularity_image
    auto_singularity_origin = 'oras'
  } else if (runtime_profiles.contains('singularity') && singularity_image_source in ['auto', 'release'] && auto_release_singularity_image && release_asset_reachable) {
    auto_singularity_image = auto_release_singularity_image
    auto_singularity_origin = 'release'
  } else if (runtime_profiles.contains('singularity') && singularity_image_source in ['auto', 'oras', 'release']) {
    auto_singularity_image = auto_docker_singularity_image
    auto_singularity_origin = 'docker'
    if (singularity_image_source == 'release') {
      println "WARN: Release-hosted Singularity image is not reachable (${auto_release_singularity_image}); falling back to ${auto_docker_singularity_image}"
    } else if (singularity_image_source == 'oras') {
      println "WARN: ORAS-hosted Singularity image is not configured; falling back to ${auto_docker_singularity_image}"
    }
  }
  if (resolved_compute_device == 'gpu' && detected_arch == 'arm64' && auto_singularity_origin == 'docker') {
    log.warn "No arm64 GPU Singularity asset was found/reachable. Auto-selected fallback container is CPU-oriented (${auto_singularity_image})."
  }

  def raw_singularity_image = (paramOr('singularity_image', '') ?: '').toString().trim()
  def resolved_singularity_image = raw_singularity_image
  if (runtime_image_mode == 'auto' || !raw_singularity_image) {
    if (resolved_compute_device == 'gpu' && gpu_container_image) {
      resolved_singularity_image = gpu_container_image
    } else if (cpu_container_image) {
      resolved_singularity_image = cpu_container_image
    } else {
      resolved_singularity_image = auto_singularity_image
    }
  } else if (runtime_image_mode == 'manual' && resolved_compute_device == 'gpu' && gpu_container_image) {
    resolved_singularity_image = gpu_container_image
  } else if (runtime_image_mode == 'manual' && cpu_container_image) {
    resolved_singularity_image = cpu_container_image
  } else if (raw_singularity_image.startsWith('docker://')) {
    def ref = raw_singularity_image.replaceFirst('^docker://', '')
    if (!ref.contains('/') || ref.endsWith('.sif')) {
      resolved_singularity_image = auto_singularity_image
    }
  } else if (raw_singularity_image.toLowerCase().endsWith('.sif') && !raw_singularity_image.contains('/')) {
    resolved_singularity_image = auto_singularity_image
  }

  def raw_docker_image = (paramOr('docker_image', '') ?: '').toString().trim()
  def resolved_docker_image = raw_docker_image
  if (runtime_image_mode == 'auto' || !raw_docker_image || raw_docker_image == 'cellphenotyper:full-cpu') {
    if (resolved_compute_device == 'gpu' && gpu_container_image) {
      resolved_docker_image = gpu_container_image.replaceFirst('^docker://', '')
    } else if (cpu_container_image) {
      resolved_docker_image = cpu_container_image.replaceFirst('^docker://', '')
    } else {
      resolved_docker_image = auto_docker_image
    }
  }

  if (runtime_profiles.contains('singularity')) {
    def singularity_image = resolved_singularity_image
    if (!singularity_image) {
      error "Parameter 'singularity_image' is empty. Use e.g. docker://ghcr.io/tkcaccia/cellphenotyper:0.2.0"
    }

    if (singularity_image.startsWith('docker://')) {
      def docker_ref = singularity_image.replaceFirst('^docker://', '')
      def likely_invalid_ref = (!docker_ref.contains('/')) || docker_ref.endsWith('.sif')
      if (likely_invalid_ref) {
        error "Invalid singularity_image '${singularity_image}'. Use a valid OCI reference, e.g. docker://ghcr.io/tkcaccia/cellphenotyper:0.2.0"
      }
    }

    def resolved_arch = detected_arch
    def image_lc = singularity_image.toLowerCase()
    if (resolved_arch == 'amd64' && image_lc.contains('arm64')) {
      error "Detected host architecture amd64 but singularity_image appears arm64: ${singularity_image}. Set --host_arch amd64 and/or override --singularity_image."
    }
    if (resolved_arch == 'arm64' && (image_lc.contains('amd64') || image_lc.contains('x86_64'))) {
      error "Detected host architecture arm64 but singularity_image appears amd64/x86_64: ${singularity_image}. Set --host_arch arm64 and/or override --singularity_image."
    }
  }

  def stage_aliases = [
    convert         : 'convert',
    image_conversion: 'convert',
    prepare_input   : 'convert',
    grandqc         : 'grandqc',
    qc              : 'grandqc',
    artifact        : 'grandqc',
    artifacts       : 'grandqc',
    stardist        : 'stardist',
    startdist       : 'stardist',
    tma             : 'tma',
    tma_spots       : 'tma',
    tissue_microarray: 'tma',
    gigatime        : 'gigatime',
    virtual_mif     : 'gigatime',
    mif             : 'gigatime',
    tissue_mask     : 'tissue_mask',
    mask_tissue     : 'tissue_mask',
    tissue_geojson  : 'tissue_mask',
    geojson         : 'cluster_geojson',
    cell_assignment : 'cell_assignment',
    assign          : 'cell_assignment',
    cytoplasm       : 'cytoplasm',
    quantification  : 'marker_quantification',
    marker_quantification: 'marker_quantification',
    marker_intensity : 'marker_quantification',
    gigatime_quantification: 'marker_quantification',
    uni2            : 'uni2',
    uni_2           : 'uni2',
    'uni-2'         : 'uni2',
    embeddings      : 'uni2',
    kodama          : 'kodama',
    clustering      : 'clustering',
    rcode_clustering: 'clustering',
    cluster_mask    : 'cluster_mask',
    labels_to_cluster_mask: 'cluster_mask',
    grow_tissue     : 'grow_tissue',
    grow_to_tissue  : 'grow_tissue',
    cluster_geojson : 'cluster_geojson',
    mask_to_geojson : 'cluster_geojson',
    final_geojson   : 'cluster_geojson'
  ]
  def stage_order = ['convert', 'grandqc', 'stardist', 'tma', 'tissue_mask', 'cell_assignment', 'cytoplasm', 'gigatime', 'marker_quantification', 'uni2', 'kodama', 'clustering', 'cluster_mask', 'grow_tissue', 'cluster_geojson']
  def stage_index = stage_order.withIndex().collectEntries { stage_name, idx -> [(stage_name): idx] }

  def normalize_stage = { raw_value, fallback_value ->
    def key = (raw_value ?: fallback_value).toString().trim().toLowerCase()
    if (key == 'auto') {
      key = fallback_value
    }
    key = stage_aliases.getOrDefault(key, key)
    if (!stage_index.containsKey(key)) {
      error "Invalid stage '${raw_value}'. Allowed stages: ${stage_order.join(', ')}"
    }
    key
  }

  def default_end_point = run_full_pipeline ? 'cluster_geojson' : 'tissue_mask'
  def start_point = normalize_stage(params.start_point, 'convert')
  def end_point = normalize_stage(params.end_point, default_end_point)

  if (stage_index[start_point] > stage_index[end_point]) {
    error "Invalid stage window: start_point '${start_point}' occurs after end_point '${end_point}'"
  }

  params._resolved_start_point = start_point
  params._resolved_end_point = end_point

  def resolveLabelsFullArtifact = { sample_id ->
    def base = "${params.outdir_base}/03_stardist/${sample_id}/stardist_out"
    def candidates = [
      file("${base}/labels_full.zarr", checkIfExists: false),
      file("${base}/labels_full.tif", checkIfExists: false),
    ]
    def existing = candidates.find { it.exists() }
    existing ?: file("${base}/labels_full.${(params.full_format ?: 'tif').toString()}", checkIfExists: true)
  }

  def should_run_stage = { stage_name ->
    def idx = stage_index[stage_name]
    idx >= stage_index[start_point] && idx <= stage_index[end_point]
  }

  def gigatime_enabled = (((params.gigatime_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
  def marker_quantification_enabled = (((params.marker_quantification_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
  def grandqc_enabled = (((params.grandqc_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
  def tma_enabled = (((params.tma_enable ?: true).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
  def uni2_reuse_existing = (((params.uni2_reuse_existing ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
  def run_convert = should_run_stage('convert')
  def run_grandqc = should_run_stage('grandqc') && grandqc_enabled
  def run_stardist = should_run_stage('stardist')
  def run_tma = should_run_stage('tma') && tma_enabled
  def run_gigatime = should_run_stage('gigatime') && gigatime_enabled
  def run_tissue_mask = should_run_stage('tissue_mask')
  def run_cell_assignment = should_run_stage('cell_assignment')
  def run_cytoplasm = should_run_stage('cytoplasm')
  def run_marker_quantification = should_run_stage('marker_quantification') && gigatime_enabled && marker_quantification_enabled
  def run_uni2 = should_run_stage('uni2') && !uni2_reuse_existing
  def run_kodama = should_run_stage('kodama')
  def run_clustering = should_run_stage('clustering')
  def run_cluster_mask = should_run_stage('cluster_mask')
  def run_grow_tissue = should_run_stage('grow_tissue')
  def run_cluster_geojson = should_run_stage('cluster_geojson')
  def resolveKodamaModes = { rawValue ->
    def raw = (rawValue == null ? '' : rawValue.toString()).trim().toLowerCase()
    if (!raw || raw in ['all', 'default']) {
      return ['tile', 'inner_square']
    }
    if (raw in ['full', 'all4', 'all_four', 'full_stack']) {
      return ['tile', 'nuclei', 'cyto', 'inner_square']
    }
    def mapped = raw
      .replace('+', ',')
      .split(',')
      .collect { it.trim() }
      .findAll { it }
      .collect { token ->
        switch (token) {
          case ['tile', 'full', 'full_tile', 'full-tile']:
            return 'tile'
          case ['nuclei', 'nucleus', 'nuclear', 'label', 'labels']:
            return 'nuclei'
          case ['cyto', 'cytoplasm']:
            return 'cyto'
          case ['inner', 'inner_square', 'inner-square', 'square']:
            return 'inner_square'
          default:
            error "Unknown KODAMA embedding mode token: ${token}"
        }
      }
      .unique()
    if (!mapped) {
      error "KODAMA embedding mode resolved to an empty set."
    }
    return mapped
  }
  def kodama_requested_modes = resolveKodamaModes(params.kodama_embedding_mode)
  def include_uni2_nuclei = params.uni2_include_nuclei == null ? kodama_requested_modes.contains('nuclei') : (params.uni2_include_nuclei as boolean)
  def include_uni2_cyto = params.uni2_include_cyto == null ? kodama_requested_modes.contains('cyto') : (params.uni2_include_cyto as boolean)
  def include_uni2_inner_square = params.uni2_include_inner_square == null ? kodama_requested_modes.contains('inner_square') : (params.uni2_include_inner_square as boolean)
  def use_roi_crop_for_uni2 = params.uni2_use_roi_crop == null ? true : (params.uni2_use_roi_crop as boolean)
  def fuse_tile_inner_square_uni2 = params.uni2_fuse_tile_inner_square == null ? true : (params.uni2_fuse_tile_inner_square as boolean)
  if (should_run_stage('uni2') && uni2_reuse_existing) {
    println "UNI-2 stage is inside the requested window, but --uni2_reuse_existing=true; published 09_embeddings will be used instead of recomputing UNI-2."
  }
  if (run_kodama && run_uni2) {
    def missingKodamaModes = []
    if (kodama_requested_modes.contains('nuclei') && !include_uni2_nuclei) missingKodamaModes << 'nuclei'
    if (kodama_requested_modes.contains('cyto') && !include_uni2_cyto) missingKodamaModes << 'cyto'
    if (kodama_requested_modes.contains('inner_square') && !include_uni2_inner_square) missingKodamaModes << 'inner_square'
    if (missingKodamaModes) {
      error "KODAMA stage requires UNI-2 embedding families enabled for: ${missingKodamaModes.join(', ')}"
    }
  }

  println "Runtime auto-selection: runtime_image_mode=${runtime_image_mode}, requested_arch=${requested_arch_raw ?: 'auto'}, detected_arch=${detected_arch}, arch_candidates=${detected_arch_candidates.join(',')}, requested_compute_device=${requested_compute_device}, resolved_compute_device=${resolved_compute_device}, enable_gpu_on_arm64=${enable_gpu_on_arm64}, enable_stardist_gpu_on_arm64=${enable_stardist_gpu_on_arm64}, singularity_image_source=${singularity_image_source}, singularity_origin=${auto_singularity_origin}, singularity_oras_tag=${selected_oras_tag ?: 'none'}, singularity_asset=${selected_release_asset ?: 'none'}, cpu_container_image=${cpu_container_image ?: 'auto'}, gpu_container_image=${gpu_container_image ?: 'auto'}, singularity_image=${resolved_singularity_image}, docker_image=${resolved_docker_image}"
  println "Pipeline stage window: ${start_point} -> ${end_point}"

  def supported_image_suffixes = [
    [suffix: '.ome.tif',  priority: 80],
    [suffix: '.ome.tiff', priority: 75],
    [suffix: '.btf',      priority: 70],
    [suffix: '.czi',      priority: 69],
    [suffix: '.svs',      priority: 68],
    [suffix: '.ndpi',     priority: 67],
    [suffix: '.scn',      priority: 66],
    [suffix: '.mrxs',     priority: 65],
    [suffix: '.vms',      priority: 64],
    [suffix: '.vmu',      priority: 63],
    [suffix: '.tif',      priority: 60],
    [suffix: '.tiff',     priority: 55],
    [suffix: '.png',      priority: 50],
    [suffix: '.jpg',      priority: 45],
    [suffix: '.jpeg',     priority: 40]
  ]

  def detectImageSuffix = { String fileName ->
    def lower = (fileName ?: '').toLowerCase()
    def hit = supported_image_suffixes.find { lower.endsWith(it.suffix as String) }
    hit?.suffix ?: ''
  }

  def imageSuffixPriority = { String suffix ->
    def hit = supported_image_suffixes.find { it.suffix == suffix }
    (hit?.priority ?: 0) as int
  }

  def deriveSampleId = { imageFile ->
    def fileObj = imageFile instanceof File ? imageFile : new File(imageFile.toString())
    def name = fileObj.name
    def suffix = detectImageSuffix(name)
    if (suffix) {
      return name.substring(0, name.length() - suffix.length())
    }
    def dot = name.lastIndexOf('.')
    dot > 0 ? name.substring(0, dot) : name
  }

  def extractCziRegionLabel = { String rawName ->
    def matcher = ((rawName ?: '') =~ /(?i)(ScanRegion\d+)/)
    matcher.find() ? matcher.group(1) : ''
  }

  def extractCziRegionOrder = { String rawName ->
    def matcher = ((rawName ?: '') =~ /(?i)ScanRegion(\d+)/)
    if (!matcher.find()) {
      return Integer.MAX_VALUE
    }
    try {
      return matcher.group(1).toInteger()
    } catch (Throwable ignored) {
      return Integer.MAX_VALUE
    }
  }

  def buildSampleId = { imageFile, String regionLabel = '' ->
    def baseId = deriveSampleId(imageFile)
    regionLabel ? "${baseId}__${regionLabel}" : baseId
  }

  def folder_input = (params.folder_input ?: '').toString().trim()
  def image_input_param = (params.image_input ?: '').toString().trim()
  def roi_geojson_param = (params.roi_geojson ?: '').toString().trim()
  def require_matching_roi = (((params.require_matching_roi ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
  def sample_rows = []

  if (folder_input) {
    if (image_input_param || roi_geojson_param) {
      println "WARN: --folder_input is set; --image_input/--roi_geojson are ignored."
    }
    def input_dir = new File(folder_input).canonicalFile
    if (!input_dir.exists()) {
      error "--folder_input does not exist: ${input_dir}"
    }
    if (!input_dir.isDirectory()) {
      error "--folder_input must be a directory. Got: ${input_dir}"
    }

    def candidates = []
    input_dir.eachFile(FileType.FILES) { File f ->
      def suffix = detectImageSuffix(f.name)
      if (suffix) {
        candidates << [file: f, suffix: suffix]
      }
    }
    if (!candidates) {
      error "No supported image files found in --folder_input ${input_dir}. Supported extensions: ${supported_image_suffixes.collect { it.suffix }.join(', ')}"
    }

    def sample_map = [:]
    candidates.each { def candidate ->
      def f = candidate.file as File
      def suffix = candidate.suffix as String
      def sample_id = deriveSampleId(f)
      def priority = imageSuffixPriority(suffix)
      def prev = sample_map[sample_id]
      if (prev == null || priority > prev.priority) {
        sample_map[sample_id] = [file: f, priority: priority]
      }
    }

    sample_map.keySet().sort().each { sample_id ->
      def image_file = sample_map[sample_id].file as File
      def suffix = detectImageSuffix(image_file.name)
      if (suffix == '.czi') {
        def czi_geojsons = []
        input_dir.eachFile(FileType.FILES) { File roiFile ->
          if (!roiFile.name.toLowerCase().endsWith('.geojson')) {
            return
          }
          def matchesImagePrefix = roiFile.name.startsWith("${image_file.name} - ")
          if (!matchesImagePrefix) {
            return
          }
          def regionLabel = extractCziRegionLabel(roiFile.name)
          if (!regionLabel) {
            return
          }
          czi_geojsons << [file: roiFile, region: regionLabel, order: extractCziRegionOrder(roiFile.name)]
        }

        if (czi_geojsons) {
          czi_geojsons
            .sort { a, b -> (a.order as int) <=> (b.order as int) ?: (a.file.name as String) <=> (b.file.name as String) }
            .each { entry ->
              def roi_file = entry.file as File
              def region_label = entry.region as String
              def region_sample_id = buildSampleId(image_file, region_label)
              sample_rows << tuple(
                region_sample_id,
                file(image_file.absolutePath, checkIfExists: true),
                region_label,
                roi_file.name,
                roi_file.bytes.encodeBase64().toString()
              )
            }
          return
        }

        if (require_matching_roi) {
          println "WARN: Skipping CZI input ${image_file.name} because no region-specific ScanRegion GeoJSON files were found and --require_matching_roi is enabled."
          return
        }
        println "WARN: No region-specific ScanRegion GeoJSON files found for CZI input ${image_file.name}. The pipeline will treat it as a single sample."
      }

      def roi_candidate = new File(input_dir, "${sample_id}.geojson")
      if (!roi_candidate.exists() && require_matching_roi) {
        println "WARN: Skipping image ${image_file.name} because matching ROI GeoJSON ${sample_id}.geojson was not found and --require_matching_roi is enabled."
        return
      }
      def roi_hint_name = roi_candidate.exists() ? roi_candidate.name : ''
      def roi_hint_b64 = roi_candidate.exists() ? roi_candidate.bytes.encodeBase64().toString() : ''
      sample_rows << tuple(sample_id, file(image_file.absolutePath, checkIfExists: true), '', roi_hint_name, roi_hint_b64)
    }
  } else {
    if (!image_input_param) {
      error "Set either --folder_input (directory with images) or --image_input (single image file)."
    }
    def image_file = file(image_input_param, checkIfExists: true)
    def single_suffix = detectImageSuffix(image_file.name)
    if (!single_suffix) {
      error "Unsupported image extension for --image_input '${image_file.name}'. Supported extensions: ${supported_image_suffixes.collect { it.suffix }.join(', ')}"
    }
    def base_sample_id = deriveSampleId(image_file)
    def roi_hint_name = ''
    def roi_hint_b64 = ''
    def input_region = ''
    if (roi_geojson_param) {
      def roi_file = file(roi_geojson_param, checkIfExists: true)
      roi_hint_name = roi_file.name
      roi_hint_b64 = roi_file.bytes.encodeBase64().toString()
      if (single_suffix == '.czi') {
        input_region = extractCziRegionLabel(roi_file.name)
        if (!input_region) {
          println "WARN: --roi_geojson ${roi_file.name} does not contain a ScanRegion selector for CZI input ${image_file.name}."
        }
      }
    } else {
      def image_file_obj = image_file instanceof File ? image_file : new File(image_file.toString())
      def roi_candidate = new File(image_file_obj.parentFile, "${base_sample_id}.geojson")
      if (roi_candidate.exists()) {
        roi_hint_name = roi_candidate.name
        roi_hint_b64 = roi_candidate.bytes.encodeBase64().toString()
        if (single_suffix == '.czi') {
          input_region = extractCziRegionLabel(roi_candidate.name)
        }
      } else if (require_matching_roi) {
        error "Matching ROI GeoJSON not found for ${image_file.name} and --require_matching_roi is enabled. Expected: ${roi_candidate}"
      }
    }
    def sample_id = buildSampleId(image_file, input_region)
    sample_rows << tuple(sample_id, image_file, input_region, roi_hint_name, roi_hint_b64)
  }

  if (!sample_rows) {
    error "No input samples were resolved."
  }
  println "Resolved input samples (${sample_rows.size()}): ${sample_rows.collect { it[0] }.join(', ')}"

  Channel
    .fromList(sample_rows)
    .set { input_spec_ch }

  def convert_input_ch = input_spec_ch
    .map { sample_id, image_file, input_region, roi_hint_name, roi_hint_b64 ->
      tuple(sample_id, image_file, input_region ?: '')
    }

  def image_input_ch = input_spec_ch
    .map { sample_id, image_file, input_region, roi_hint_name, roi_hint_b64 ->
      tuple(sample_id, image_file)
    }

  def roi_hint_ch = input_spec_ch
    .map { sample_id, image_file, input_region, roi_hint_name, roi_hint_b64 ->
      tuple(sample_id, roi_hint_name ?: '', roi_hint_b64 ?: '')
    }
  def roi_provided_ch = roi_hint_ch
    .map { sample_id, roi_hint_name, roi_hint_b64 ->
      tuple(sample_id, ((roi_hint_b64 ?: '').toString().trim() ? true : false))
    }

  def ome_tif_ch
  if (run_convert) {
    PREPARE_INPUT_OMETIFF(convert_input_ch)
    ome_tif_ch = PREPARE_INPUT_OMETIFF.out.ome_tif
  } else {
    ome_tif_ch = convert_input_ch.map { sample_id, image_file, input_region ->
      def is_ome = image_file.name.toLowerCase().endsWith('.ome.tif')
      def ome_tif = is_ome
        ? image_file
        : file("${params.outdir_base}/01_input/${sample_id}/${sample_id}.ome.tif", checkIfExists: true)
      tuple(sample_id, ome_tif)
    }
  }

  def grandqc_dir_ch = Channel.empty()
  def grandqc_clean_tissue_mask_ch = Channel.empty()
  def grandqc_artifact_mask_ch = Channel.empty()
  def grandqc_geojson_ch = Channel.empty()
  if (run_grandqc) {
    RUN_GRANDQC_ARTIFACT_ANALYSIS(ome_tif_ch)
    grandqc_dir_ch = RUN_GRANDQC_ARTIFACT_ANALYSIS.out.grandqc_dir
    grandqc_clean_tissue_mask_ch = RUN_GRANDQC_ARTIFACT_ANALYSIS.out.clean_tissue_mask
    grandqc_artifact_mask_ch = RUN_GRANDQC_ARTIFACT_ANALYSIS.out.artifact_mask
    grandqc_geojson_ch = RUN_GRANDQC_ARTIFACT_ANALYSIS.out.artifact_geojson
  }

  def roi_geojson_ch = Channel.empty()
  if (run_stardist) {
    def roi_prepare_input_ch = ome_tif_ch
      .join(roi_hint_ch)
      .map { sample_id, ome_tif, roi_hint_name, roi_hint_b64 ->
        tuple(sample_id, ome_tif, roi_hint_name ?: '', roi_hint_b64 ?: '')
      }
    PREPARE_ROI_GEOJSON(roi_prepare_input_ch)
    roi_geojson_ch = PREPARE_ROI_GEOJSON.out.roi_geojson
  } else if (run_cell_assignment) {
    roi_geojson_ch = image_input_ch.map { sample_id, _ ->
      tuple(sample_id, file("${params.outdir_base}/06_roi/${sample_id}/${sample_id}.roi.geojson", checkIfExists: true))
    }
  }

  def stardist_roi_geojson_ch = roi_geojson_ch
  if (run_stardist && (params.stardist_auto_roi_from_tissue as boolean)) {
    def provided_roi_for_stardist_ch = roi_geojson_ch
      .join(roi_provided_ch)
      .filter { sample_id, roi_geojson, roi_provided -> roi_provided }
      .map { sample_id, roi_geojson, roi_provided ->
        tuple(sample_id, roi_geojson)
      }
    def auto_roi_input_ch = image_input_ch
      .join(roi_provided_ch)
      .filter { sample_id, image_input, roi_provided -> !roi_provided }
      .map { sample_id, image_input, roi_provided ->
        tuple(sample_id, image_input)
      }
    PREPARE_STARDIST_AUTO_ROI(auto_roi_input_ch)
    stardist_roi_geojson_ch = provided_roi_for_stardist_ch.mix(PREPARE_STARDIST_AUTO_ROI.out.roi_geojson)
  }

  def need_stardist_outputs = run_stardist || run_tma || (!tissue_mask_from_input && run_tissue_mask) || run_cell_assignment || run_cytoplasm || run_uni2

  def crop_roi_ch = Channel.empty()
  def labels_tif_ch = Channel.empty()
  def labels_full_ch = Channel.empty()
  def objects_csv_ch = Channel.empty()
  def roi_crop_geojson_ch = Channel.empty()
  def shift_json_ch = Channel.empty()

  if (need_stardist_outputs) {
    if (run_stardist) {
      def require_labels_full = (!use_roi_crop_for_uni2 && run_uni2) || ((run_cytoplasm && (params.expand_full_labels as boolean)) && !use_roi_crop_for_uni2)
      params._resolved_stardist_write_full_labels = require_labels_full
      params._resolved_stardist_full_format = require_labels_full ? (params.full_format ?: 'tif') : (params.full_format ?: 'tif')
      params._resolved_allow_huge_tif = require_labels_full ? (params.allow_huge_tif as boolean) : false
      def stardist_input_ch = ome_tif_ch
        .join(stardist_roi_geojson_ch)
        .map { sample_id, ome_tif, stardist_roi_geojson ->
          tuple(sample_id, ome_tif, stardist_roi_geojson)
        }
      RUN_STARDIST_ROI_SEGMENTATION(stardist_input_ch)
      crop_roi_ch = RUN_STARDIST_ROI_SEGMENTATION.out.crop_roi
      labels_tif_ch = RUN_STARDIST_ROI_SEGMENTATION.out.labels_tif
      labels_full_ch = RUN_STARDIST_ROI_SEGMENTATION.out.labels_full
      objects_csv_ch = RUN_STARDIST_ROI_SEGMENTATION.out.objects_csv
      roi_crop_geojson_ch = RUN_STARDIST_ROI_SEGMENTATION.out.roi_crop_geojson
      shift_json_ch = RUN_STARDIST_ROI_SEGMENTATION.out.shift_json
    } else {
      def require_labels_full = (!use_roi_crop_for_uni2 && run_uni2) || ((run_cytoplasm && (params.expand_full_labels as boolean)) && !use_roi_crop_for_uni2)
      crop_roi_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/crop_roi.tif", checkIfExists: true))
      }
      labels_tif_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/labels.tif", checkIfExists: true))
      }
      objects_csv_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/objects.csv", checkIfExists: true))
      }
      roi_crop_geojson_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/roi_all_crop.geojson", checkIfExists: true))
      }
      shift_json_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/shift.json", checkIfExists: true))
      }
      labels_full_ch = require_labels_full
        ? image_input_ch.map { sample_id, _ ->
          tuple(sample_id, resolveLabelsFullArtifact(sample_id))
        }
        : Channel.empty()
    }
  }

  def objects_for_assignment_ch = objects_csv_ch
  def tma_outputs_available = stage_index[end_point] >= stage_index['tma'] && tma_enabled
  if (run_tma) {
    def tma_input_ch = crop_roi_ch
      .join(objects_csv_ch)
      .join(shift_json_ch)
      .map { sample_id, crop_tif, objects_csv, shift_json ->
        tuple(sample_id, crop_tif, objects_csv, shift_json)
      }
    DETECT_TMA_SPOTS(tma_input_ch)
    objects_for_assignment_ch = DETECT_TMA_SPOTS.out.objects_tma_assigned
  } else if (tma_outputs_available && (run_cell_assignment || run_cytoplasm || run_marker_quantification || run_uni2 || run_kodama || run_clustering || run_cluster_mask || run_grow_tissue || run_cluster_geojson)) {
    objects_for_assignment_ch = image_input_ch.map { sample_id, _ ->
      tuple(sample_id, file("${params.outdir_base}/04_TMA/${sample_id}/tma_${sample_id}/${sample_id}_objects_tma_assigned.csv", checkIfExists: true))
    }
  }

  def stardist_outputs_available = stage_index[end_point] >= stage_index['stardist']
  def gigatime_outputs_available = stage_index[end_point] >= stage_index['gigatime'] && gigatime_enabled
  def crop_roi_for_masks_ch = stardist_outputs_available
    ? (run_stardist
      ? crop_roi_ch
      : image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/crop_roi.tif", checkIfExists: true))
      })
    : Channel.empty()
  def roi_crop_geojson_for_masks_ch = stardist_outputs_available
    ? (run_stardist
      ? roi_crop_geojson_ch
      : image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/roi_all_crop.geojson", checkIfExists: true))
      })
    : Channel.empty()
  def shift_json_for_masks_ch = stardist_outputs_available
    ? (run_stardist
      ? shift_json_ch
      : image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/shift.json", checkIfExists: true))
      })
    : Channel.empty()
  def gigatime_image_ch = Channel.empty()
  def gigatime_quant_dir_ch = Channel.empty()

  if (stardist_outputs_available) {
    if (!run_gigatime && run_marker_quantification) {
      gigatime_image_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/05_gigatime/${sample_id}/gigatime_${sample_id}", checkIfExists: true))
      }
      gigatime_quant_dir_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/05_gigatime/${sample_id}/quantification_${sample_id}", checkIfExists: true))
      }
    }
    def roi_mask_input_ch = roi_crop_geojson_for_masks_ch
      .join(crop_roi_for_masks_ch)
      .join(roi_provided_ch)
      .filter { sample_id, roi_crop_geojson, crop_roi_tif, roi_provided -> roi_provided }
      .map { sample_id, roi_crop_geojson, crop_roi_tif, roi_provided ->
        tuple(sample_id, roi_crop_geojson, crop_roi_tif)
      }
    ROI_GEOJSON_TO_MASK(roi_mask_input_ch)
  }

  def tissue_mask_ch = Channel.empty()
  if (run_tissue_mask) {
    def tissue_mask_input_ch = tissue_mask_from_input ? ome_tif_ch : crop_roi_ch
    BUILD_TISSUE_MASK(tissue_mask_input_ch)
    tissue_mask_ch = BUILD_TISSUE_MASK.out.tissue_mask
  } else if (run_grow_tissue || run_cluster_geojson) {
    tissue_mask_ch = image_input_ch.map { sample_id, _ ->
      tuple(sample_id, file("${params.outdir_base}/04_tissue_mask/${sample_id}/${sample_id}_tissue_mask.tif", checkIfExists: true))
    }
  }

  if (run_cell_assignment || run_cytoplasm || run_gigatime || run_marker_quantification || run_uni2 || run_kodama || run_clustering || run_cluster_mask || run_grow_tissue || run_cluster_geojson) {
    def objects_assigned_ch = Channel.empty()
    if (run_cell_assignment) {
      def assign_input_ch = objects_for_assignment_ch
        .join(shift_json_ch)
        .join(roi_geojson_ch)
        .map { sample_id, objects_csv, shift_json, roi_geojson ->
          tuple(sample_id, objects_csv, roi_geojson, shift_json)
        }
      MAP_CELLS_TO_ROI_POLYGONS(assign_input_ch)
      objects_assigned_ch = MAP_CELLS_TO_ROI_POLYGONS.out.objects_assigned
    } else {
      objects_assigned_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/07_cell_assignments/${sample_id}/${sample_id}_objects_assigned.csv", checkIfExists: true))
      }
    }

    def cyto_mask_ch = Channel.empty()
    def cyto_mask_full_ch = Channel.empty()
    def nuclei_mask_for_quant_ch = run_stardist
      ? labels_tif_ch
      : image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/labels.tif", checkIfExists: true))
      }
    if (run_cytoplasm) {
      def expand_primary_ch = labels_tif_ch
        .join(crop_roi_ch)
        .map { sample_id, labels_tif, preview_background_tif ->
          tuple(sample_id, labels_tif, 'labels_cyto', preview_background_tif.toString())
        }
      EXPAND_LABELS_TO_CYTOPLASM_PRIMARY(expand_primary_ch)

      if ((params.expand_full_labels as boolean) && !(run_uni2 && use_roi_crop_for_uni2)) {
        def expand_full_ch = labels_full_ch.map { sample_id, labels_full_path ->
          tuple(sample_id, labels_full_path, 'labels_full_cyto', '')
        }
        EXPAND_LABELS_TO_CYTOPLASM_FULL(expand_full_ch)
        cyto_mask_full_ch = EXPAND_LABELS_TO_CYTOPLASM_FULL.out.expanded_labels
          .filter { sample_id, expanded_mask, label_kind -> label_kind == 'labels_full_cyto' }
          .map { sample_id, expanded_mask, label_kind -> tuple(sample_id, expanded_mask) }
      } else if (run_uni2 && !use_roi_crop_for_uni2 && (include_uni2_cyto || include_uni2_inner_square)) {
        error "UNI-2 cyto/inner-square embeddings require expand_full_labels=true to generate *_labels_full_cyto.tif."
      }

      cyto_mask_ch = EXPAND_LABELS_TO_CYTOPLASM_PRIMARY.out.expanded_labels
        .filter { sample_id, expanded_mask, label_kind -> label_kind == 'labels_cyto' }
        .map { sample_id, expanded_mask, label_kind -> tuple(sample_id, expanded_mask) }
    } else if (run_uni2) {
      cyto_mask_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/08_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif", checkIfExists: true))
      }
      if (!use_roi_crop_for_uni2) {
        cyto_mask_full_ch = image_input_ch.map { sample_id, _ ->
          tuple(sample_id, file("${params.outdir_base}/08_cytoplasm/${sample_id}/${sample_id}_labels_full_cyto.tif", checkIfExists: true))
        }
      }
    }

    def cyto_mask_for_quant_ch = run_cytoplasm
      ? cyto_mask_ch
      : image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/08_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif", checkIfExists: true))
      }

    if (run_gigatime) {
      def gigatime_input_ch = crop_roi_for_masks_ch
        .join(shift_json_for_masks_ch)
        .join(nuclei_mask_for_quant_ch)
        .join(cyto_mask_for_quant_ch)
        .map { sample_id, crop_roi_tif, shift_json, nuclei_mask_tif, cyto_mask_tif ->
          tuple(sample_id, crop_roi_tif, shift_json, nuclei_mask_tif, cyto_mask_tif)
        }
      RUN_GIGATIME_ON_CROP(gigatime_input_ch)
      gigatime_image_ch = RUN_GIGATIME_ON_CROP.out.gigatime_dir
      gigatime_quant_dir_ch = RUN_GIGATIME_ON_CROP.out.quant_dir
    }

    if (run_marker_quantification && !run_gigatime) {
      def nuclei_quant_input_ch = gigatime_image_ch
        .join(nuclei_mask_for_quant_ch)
        .map { sample_id, gigatime_input, nuclei_mask_tif ->
          tuple(sample_id, gigatime_input, nuclei_mask_tif, 'nuclei')
        }
      def cyto_quant_input_ch = gigatime_image_ch
        .join(cyto_mask_for_quant_ch)
        .map { sample_id, gigatime_input, cyto_mask_tif ->
          tuple(sample_id, gigatime_input, cyto_mask_tif, 'cyto')
        }

      QUANTIFY_GIGATIME_INTENSITY(nuclei_quant_input_ch.mix(cyto_quant_input_ch))
    }

    def tile_embeddings_ch = Channel.empty()
    def nuclei_embeddings_ch = Channel.empty()
    def cyto_embeddings_ch = Channel.empty()
    def inner_square_embeddings_ch = Channel.empty()
    def placeholder_embeddings_dir = file("${projectDir}/resources/empty_embeddings_placeholder", checkIfExists: true)
    def placeholder_embeddings_ch = image_input_ch.map { sample_id, _ ->
      tuple(sample_id, placeholder_embeddings_dir)
    }
    if (run_uni2) {
      def uni2_image_ch = use_roi_crop_for_uni2 ? crop_roi_ch : ome_tif_ch
      def uni2_label_mask_ch = use_roi_crop_for_uni2 ? labels_tif_ch : labels_full_ch
      def uni2_cyto_mask_source_ch = use_roi_crop_for_uni2 ? cyto_mask_ch : cyto_mask_full_ch
      def canFuseTileInnerSquare = fuse_tile_inner_square_uni2 &&
        include_uni2_inner_square &&
        (params.uni2_encoder?.toString()?.toLowerCase() == 'uni2-h')

      def uni2_input_ch = Channel.empty()
      def haveSeparateUni2Inputs = false
      if (canFuseTileInnerSquare) {
        def uni2_shared_input_ch = uni2_image_ch
          .join(uni2_label_mask_ch)
          .map { sample_id, image_tif, labels_tif ->
            tuple(sample_id, image_tif, labels_tif)
          }
        EXTRACT_UNI2_EMBEDDINGS_SHARED(uni2_shared_input_ch)
        tile_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS_SHARED.out.tile_embeddings_dir
          .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
        inner_square_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS_SHARED.out.inner_square_embeddings_dir
          .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
      } else {
        def uni2_tile_input_ch = uni2_image_ch
          .join(uni2_label_mask_ch)
          .map { sample_id, image_tif, labels_tif ->
            tuple(sample_id, image_tif, labels_tif, 'tile', false, 'none', 255)
          }
        uni2_input_ch = uni2_tile_input_ch
        haveSeparateUni2Inputs = true
      }
      if (include_uni2_cyto) {
        def uni2_cyto_input_ch = uni2_image_ch
          .join(uni2_cyto_mask_source_ch)
          .map { sample_id, image_tif, cyto_mask_tif ->
            tuple(sample_id, image_tif, cyto_mask_tif, 'cyto', true, 'label', 255)
          }
        uni2_input_ch = haveSeparateUni2Inputs ? uni2_input_ch.mix(uni2_cyto_input_ch) : uni2_cyto_input_ch
        haveSeparateUni2Inputs = true
      }
      if (include_uni2_inner_square && !canFuseTileInnerSquare) {
        def uni2_inner_square_input_ch = uni2_image_ch
          .join(uni2_cyto_mask_source_ch)
          .map { sample_id, image_tif, cyto_mask_tif ->
            tuple(sample_id, image_tif, cyto_mask_tif, 'inner_square', true, 'inner_square', 255)
          }
        uni2_input_ch = haveSeparateUni2Inputs ? uni2_input_ch.mix(uni2_inner_square_input_ch) : uni2_inner_square_input_ch
        haveSeparateUni2Inputs = true
      }
      if (include_uni2_nuclei) {
        def uni2_nuclei_input_ch = uni2_image_ch
          .join(uni2_label_mask_ch)
          .map { sample_id, image_tif, labels_tif ->
            tuple(sample_id, image_tif, labels_tif, 'nuclei', true, 'label', 255)
          }
        uni2_input_ch = haveSeparateUni2Inputs ? uni2_input_ch.mix(uni2_nuclei_input_ch) : uni2_nuclei_input_ch
        haveSeparateUni2Inputs = true
      }

      if (haveSeparateUni2Inputs) {
        EXTRACT_UNI2_EMBEDDINGS(uni2_input_ch)
        if (!canFuseTileInnerSquare) {
          tile_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
            .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'tile' }
            .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
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
    } else if (run_kodama) {
      tile_embeddings_ch = image_input_ch.map { sample_id, _ ->
        def embedding_dir = kodama_requested_modes.contains('tile')
          ? file("${params.outdir_base}/09_embeddings/${sample_id}/embeddings_${sample_id}_tile", checkIfExists: true)
          : placeholder_embeddings_dir
        tuple(sample_id, embedding_dir)
      }
      nuclei_embeddings_ch = image_input_ch.map { sample_id, _ ->
        def embedding_dir = kodama_requested_modes.contains('nuclei')
          ? file("${params.outdir_base}/09_embeddings/${sample_id}/embeddings_${sample_id}_nuclei", checkIfExists: true)
          : placeholder_embeddings_dir
        tuple(sample_id, embedding_dir)
      }
      cyto_embeddings_ch = image_input_ch.map { sample_id, _ ->
        def embedding_dir = kodama_requested_modes.contains('cyto')
          ? file("${params.outdir_base}/09_embeddings/${sample_id}/embeddings_${sample_id}_cyto", checkIfExists: true)
          : placeholder_embeddings_dir
        tuple(sample_id, embedding_dir)
      }
      inner_square_embeddings_ch = image_input_ch.map { sample_id, _ ->
        def embedding_dir = kodama_requested_modes.contains('inner_square')
          ? file("${params.outdir_base}/09_embeddings/${sample_id}/embeddings_${sample_id}_inner_square", checkIfExists: true)
          : placeholder_embeddings_dir
        tuple(sample_id, embedding_dir)
      }
    }

    if (run_kodama) {
      if (!kodama_requested_modes.contains('tile')) {
        tile_embeddings_ch = placeholder_embeddings_ch
      }
      if (!kodama_requested_modes.contains('nuclei')) {
        nuclei_embeddings_ch = placeholder_embeddings_ch
      }
      if (!kodama_requested_modes.contains('cyto')) {
        cyto_embeddings_ch = placeholder_embeddings_ch
      }
      if (!kodama_requested_modes.contains('inner_square')) {
        inner_square_embeddings_ch = placeholder_embeddings_ch
      }
    }

    def kodama_dir_ch = Channel.empty()
    if (run_kodama) {
      def embedding_quad_ch = tile_embeddings_ch
        .join(nuclei_embeddings_ch)
        .join(cyto_embeddings_ch)
        .join(inner_square_embeddings_ch)
        .map { sample_id, tile_embeddings_dir, nuclei_embeddings_dir, cyto_embeddings_dir, inner_square_embeddings_dir ->
          tuple(sample_id, tile_embeddings_dir, nuclei_embeddings_dir, cyto_embeddings_dir, inner_square_embeddings_dir)
        }

      def kodama_input_ch = embedding_quad_ch
        .join(objects_assigned_ch)
        .map { sample_id, tile_embeddings_dir, nuclei_embeddings_dir, cyto_embeddings_dir, inner_square_embeddings_dir, objects_assigned ->
          tuple(sample_id, tile_embeddings_dir, cyto_embeddings_dir, inner_square_embeddings_dir, nuclei_embeddings_dir, objects_assigned)
        }
      RUN_KODAMA_ANALYSIS(kodama_input_ch)
      kodama_dir_ch = RUN_KODAMA_ANALYSIS.out.kodama_dir
    } else if (run_clustering || run_cluster_mask || run_grow_tissue || run_cluster_geojson) {
      kodama_dir_ch = image_input_ch.map { sample_id, _ ->
        tuple(sample_id, file("${params.outdir_base}/10_kodama/${sample_id}/kodama_output", checkIfExists: true))
      }
    }

    def cluster_primary_variant = (params.cluster_primary_variant ?: 'standard').toString().trim()
    def cluster_secondary_variant = (params.cluster_secondary_variant ?: '').toString().trim()
    def cluster_secondary_profile = (params.cluster_secondary_profile ?: 'fine').toString().trim().toLowerCase()
    if (!cluster_primary_variant) cluster_primary_variant = 'standard'
    def cluster_resolution_value = (params.cluster_resolution ?: 'auto').toString().trim()
    if (!cluster_resolution_value) cluster_resolution_value = 'auto'
    def cluster_variant_defs = [
      [variant: cluster_primary_variant, profile: 'standard', resolution: cluster_resolution_value]
    ]
    if (cluster_secondary_variant && !(cluster_secondary_variant.toLowerCase() in ['none', 'false', 'off', '0'])) {
      if (cluster_primary_variant == cluster_secondary_variant) {
        error "cluster_primary_variant and cluster_secondary_variant must be different."
      }
      cluster_variant_defs << [variant: cluster_secondary_variant, profile: cluster_secondary_profile ?: 'fine', resolution: cluster_resolution_value]
    }
    def cluster_variant_request_ch = image_input_ch.flatMap { sample_id, _ ->
      cluster_variant_defs.collect { spec ->
        def sample_key = "${sample_id}::${spec.variant}"
        tuple(sample_id, sample_key, spec.variant, spec.profile, spec.resolution)
      }
    }

    def cluster_csv_ch = Channel.empty()
    def cluster_kodama_png_ch = Channel.empty()
    if (run_clustering) {
      def clustering_base_ch = kodama_dir_ch
        .join(objects_assigned_ch)
        .map { sample_id, kodama_dir, objects_assigned_csv ->
          tuple(sample_id, kodama_dir, objects_assigned_csv)
        }
      def clustering_input_ch = clustering_base_ch
        .flatMap { sample_id, kodama_dir, objects_assigned_csv ->
          cluster_variant_defs.collect { spec ->
            def sample_key = "${sample_id}::${spec.variant}"
            tuple(sample_key, sample_id, spec.variant, spec.profile, cluster_resolution_value, kodama_dir, objects_assigned_csv)
          }
        }
      RUN_RCODE_CLUSTERING(clustering_input_ch)
      cluster_csv_ch = RUN_RCODE_CLUSTERING.out.cluster_csv
      cluster_kodama_png_ch = RUN_RCODE_CLUSTERING.out.membership_png
    } else if (run_cluster_mask || run_grow_tissue || run_cluster_geojson) {
      cluster_csv_ch = image_input_ch.flatMap { sample_id, _ ->
        cluster_variant_defs.collect { spec ->
          def sample_key = "${sample_id}::${spec.variant}"
          tuple(sample_key, sample_id, spec.variant, file("${params.outdir_base}/11_clustering/${sample_id}/${sample_id}_${spec.variant}_cluster.csv", checkIfExists: true))
        }
      }
      cluster_kodama_png_ch = image_input_ch.flatMap { sample_id, _ ->
        cluster_variant_defs.collect { spec ->
          def sample_key = "${sample_id}::${spec.variant}"
          tuple(sample_key, sample_id, spec.variant, file("${params.outdir_base}/11_clustering/${sample_id}/${sample_id}_${spec.variant}_cluster_kodama_membership.png", checkIfExists: true))
        }
      }
    }

    def labels_for_cluster_ch = Channel.empty()
    if (run_cluster_mask || run_grow_tissue || run_cluster_geojson) {
      labels_for_cluster_ch = run_cytoplasm
        ? cyto_mask_ch
        : image_input_ch.map { sample_id, _ ->
          tuple(sample_id, file("${params.outdir_base}/08_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif", checkIfExists: true))
        }
    }

    def cluster_mask_ch = Channel.empty()
    if (run_cluster_mask) {
      def preview_image_for_cluster_mask_ch = run_stardist
        ? crop_roi_ch
        : image_input_ch.map { sample_id, _ ->
          tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/crop_roi.tif", checkIfExists: true))
        }

      def labels_for_cluster_variant_ch = labels_for_cluster_ch
        .flatMap { sample_id, labels_tif ->
          cluster_variant_defs.collect { spec ->
            def sample_key = "${sample_id}::${spec.variant}"
            tuple(sample_key, labels_tif)
          }
        }

      def preview_image_for_cluster_variant_ch = preview_image_for_cluster_mask_ch
        .flatMap { sample_id, preview_tif ->
          cluster_variant_defs.collect { spec ->
            def sample_key = "${sample_id}::${spec.variant}"
            tuple(sample_key, preview_tif)
          }
        }

      def cluster_mask_input_ch = cluster_csv_ch
        .join(labels_for_cluster_variant_ch)
        .join(preview_image_for_cluster_variant_ch)
        .map { sample_key, sample_id, cluster_variant, cluster_csv, labels_tif, preview_tif ->
          tuple(sample_key, sample_id, cluster_variant, labels_tif, cluster_csv, preview_tif)
        }
      LABELS_TO_CLUSTER_MASK(cluster_mask_input_ch)
      cluster_mask_ch = LABELS_TO_CLUSTER_MASK.out.cluster_mask
    } else if (run_grow_tissue || run_cluster_geojson) {
      cluster_mask_ch = image_input_ch.flatMap { sample_id, _ ->
        cluster_variant_defs.collect { spec ->
          def sample_key = "${sample_id}::${spec.variant}"
          tuple(sample_key, sample_id, spec.variant, file("${params.outdir_base}/12_cluster_mask/${sample_id}/${sample_id}_${spec.variant}_cluster_mask.tif", checkIfExists: true))
        }
      }
    }

    def image_for_growth_variant_ch = Channel.empty()
    def tissue_mask_variant_ch = Channel.empty()
    if (run_grow_tissue || run_cluster_geojson) {
      def image_for_growth_ch = run_stardist
        ? crop_roi_ch
        : image_input_ch.map { sample_id, _ ->
          tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/crop_roi.tif", checkIfExists: true))
        }

      image_for_growth_variant_ch = image_for_growth_ch
        .flatMap { sample_id, image_tif ->
          cluster_variant_defs.collect { spec ->
            def sample_key = "${sample_id}::${spec.variant}"
            tuple(sample_key, image_tif)
          }
        }

      tissue_mask_variant_ch = tissue_mask_ch
        .flatMap { sample_id, tissue_mask_tif ->
          cluster_variant_defs.collect { spec ->
            def sample_key = "${sample_id}::${spec.variant}"
            tuple(sample_key, tissue_mask_tif)
          }
        }
    }

    def grown_mask_ch = Channel.empty()
    if (run_grow_tissue) {
      def grow_input_ch = cluster_mask_ch
        .join(image_for_growth_variant_ch)
        .join(tissue_mask_variant_ch)
        .map { sample_key, sample_id, cluster_variant, cluster_mask_tif, image_tif, tissue_mask_tif ->
          tuple(sample_key, sample_id, cluster_variant, image_tif, cluster_mask_tif, tissue_mask_tif)
        }
      GROW_TO_TISSUE(grow_input_ch)
      grown_mask_ch = GROW_TO_TISSUE.out.grown_mask
    } else if (run_cluster_geojson) {
      grown_mask_ch = image_input_ch.flatMap { sample_id, _ ->
        cluster_variant_defs.collect { spec ->
          def sample_key = "${sample_id}::${spec.variant}"
          tuple(sample_key, sample_id, spec.variant, file("${params.outdir_base}/13_grown_tissue/${sample_id}/${sample_id}_${spec.variant}_grown_mask.ome.tif", checkIfExists: true))
        }
      }
    }

    def refined_mask_ch = grown_mask_ch
    def grown_refine_method = (params.grown_tissue_refine_method ?: 'medsam_border_refine').toString()
    if (run_cluster_geojson && grown_refine_method == 'medsam_border_refine') {
      def cluster_mask_file_ch = cluster_mask_ch.map { sample_key, sample_id, cluster_variant, cluster_mask_tif ->
        tuple(sample_key, cluster_mask_tif)
      }
      def cluster_kodama_png_file_ch = cluster_kodama_png_ch.map { sample_key, sample_id, cluster_variant, membership_png ->
        tuple(sample_key, membership_png)
      }
      def refine_input_ch = grown_mask_ch
        .join(image_for_growth_variant_ch)
        .join(cluster_mask_file_ch)
        .join(cluster_kodama_png_file_ch)
        .map { sample_key, sample_id, cluster_variant, grown_mask_tif, image_tif, cluster_mask_tif, kodama_membership_png ->
          tuple(sample_key, sample_id, cluster_variant, image_tif, cluster_mask_tif, grown_mask_tif, kodama_membership_png)
        }
      REFINE_GROWN_TISSUE_MEDSAM(refine_input_ch)
      refined_mask_ch = REFINE_GROWN_TISSUE_MEDSAM.out.refined_mask
    } else if (run_cluster_geojson && !(grown_refine_method in ['none', ''])) {
      error "Unsupported grown_tissue_refine_method: ${grown_refine_method}"
    }

    if (run_cluster_geojson) {
      MASK_TO_GEOJSON(refined_mask_ch)
    }
  }
}

workflow.onComplete {
  def outdir = params.outdir_base ?: 'results'
  def executionDir = new File("${outdir}/00_execution")
  executionDir.mkdirs()
  def reporterScript = new File("${baseDir}/bin/write_pipeline_execution_reports.py").canonicalPath
  def cmd = [
    'python3',
    reporterScript,
    '--outdir', new File(outdir).canonicalPath,
    '--run-name', workflow.runName,
    '--success', workflow.success.toString(),
    '--start-point', (params._resolved_start_point ?: 'convert').toString(),
    '--end-point', (params._resolved_end_point ?: 'cluster_geojson').toString(),
    '--image-input', (params.image_input ?: '').toString(),
    '--roi-geojson', (params.roi_geojson ?: '').toString()
  ]
  try {
    def pb = new ProcessBuilder(cmd)
    pb.redirectErrorStream(true)
    def proc = pb.start()
    def output = proc.inputStream.getText('UTF-8').trim()
    def rc = proc.waitFor()
    if (output) {
      println output
    }
    if (rc != 0) {
      println "WARN: execution report writer exited with code ${rc}"
    }
  } catch (Throwable t) {
    println "WARN: failed to write execution reports: ${t.message}"
  }
  if (workflow.success) {
    println "PIPELINE COMPLETED SUCCESSFULLY"
    println "Stage window: ${params._resolved_start_point ?: 'convert'} -> ${params._resolved_end_point ?: 'cluster_geojson'}"
    if ((params._resolved_end_point ?: '') in ['clustering', 'cluster_mask', 'grow_tissue', 'cluster_geojson']) {
      println "Cluster GeoJSON output dir: ${params.outdir_base}/15_cluster_geojson"
    }
  }
}
