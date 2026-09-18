import groovy.io.FileType
import groovy.json.JsonOutput
import groovy.json.JsonSlurper

nextflow.enable.dsl = 2

include { PREPARE_INPUT_OMETIFF } from './modules/prepare_input_ometiff'
include { RUN_GRANDQC_ARTIFACT_ANALYSIS } from './modules/run_grandqc_artifact_analysis'
include { PREPARE_ROI_GEOJSON } from './modules/prepare_roi_geojson'
include { PREPARE_STARDIST_AUTO_ROI } from './modules/prepare_stardist_auto_roi'
include { PREPARE_ANALYSIS_CROP } from './modules/prepare_analysis_crop'
include { CROP_GRANDQC_CLEAN_MASK } from './modules/crop_grandqc_clean_mask'
include { RUN_STARDIST_ROI_SEGMENTATION } from './modules/run_stardist_roi_segmentation'
include { RUN_HOVERNET_MONUSAC } from './modules/run_hovernet_monusac'
include { RUN_CELLVITPP } from './modules/run_cellvitpp'
include { BUILD_CELL_CONSENSUS } from './modules/build_cell_consensus'
include { DETECT_TMA_SPOTS } from './modules/detect_tma_spots'
include { RUN_GIGATIME_ON_CROP } from './modules/run_gigatime_on_crop'
include { EXPORT_GIGATIME_OMETIFF } from './modules/export_gigatime_ometiff'
include { ROI_GEOJSON_TO_MASK } from './modules/roi_geojson_to_mask'
include { MAP_CELLS_TO_ROI_POLYGONS } from './modules/map_cells_to_roi_polygons'
include { EXPAND_LABELS_TO_CYTOPLASM as EXPAND_LABELS_TO_CYTOPLASM_PRIMARY } from './modules/expand_labels_to_cytoplasm'
include { EXPAND_LABELS_TO_CYTOPLASM as EXPAND_LABELS_TO_CYTOPLASM_FULL } from './modules/expand_labels_to_cytoplasm'
include { QUANTIFY_GIGATIME_INTENSITY } from './modules/quantify_gigatime_intensity'
include { EXTRACT_UNI2_EMBEDDINGS } from './modules/extract_uni2_embeddings'
include { EXTRACT_UNI2_EMBEDDINGS_SHARED } from './modules/extract_uni2_embeddings_shared'
include { RUN_KODAMA_ANALYSIS } from './modules/run_kodama_analysis'
include { RUN_GIGATIME_KODAMA } from './modules/run_gigatime_kodama'
include { RUN_RCODE_CLUSTERING } from './modules/run_rcode_clustering'
include { ASSESS_CLUSTER_INTERPRETATION } from './modules/assess_cluster_interpretation'
include { GROW_TO_TISSUE } from './modules/grow_to_tissue'
include { PREPARE_UNI2_SPATIAL_GRID } from './subworkflows/prepare_uni2_spatial_grid'
include { RUN_AUXILIARY_CELL_ROUTE } from './subworkflows/run_auxiliary_cell_route'
include { BUILD_CLUSTER_MASK_OUTPUT } from './subworkflows/build_cluster_mask_output'
include { POST_GROW_SPATIAL_OUTPUTS } from './subworkflows/post_grow_spatial_outputs'
include { POST_CLUSTER_PATHOFM } from './subworkflows/post_cluster_pathofm'
include { RUN_CELL_PROFILE_ATLAS } from './subworkflows/run_cell_profile_atlas'
include { RUN_TISSUE_REGION_ATLAS } from './subworkflows/run_tissue_region_atlas'
include { EXTRACT_PRIMARY_UNI2 } from './subworkflows/extract_primary_uni2'
include { RUN_PATHSEGMENTOR_INFERENCE } from './subworkflows/run_pathsegmentor_inference'
include { ANNOTATE_PATHSEGMENTOR_EVIDENCE } from './subworkflows/annotate_pathsegmentor_evidence'
include { RUN_PATHSEGMENTOR_REFINEMENT } from './subworkflows/run_pathsegmentor_refinement'

workflow {
if (params.cell_measured_assays && !(params.cell_profiles_enable && params.cell_profiles_spatialdata))
  error 'cell_measured_assays requires cell_profiles_enable=true and cell_profiles_spatialdata=true'
if (params.cell_reference_atlas && !params.cell_profiles_enable)
  error 'cell_reference_atlas requires cell_profiles_enable=true'
PipelineHelpers.validateCohortOptions(params)
if (params.region_reference_atlas && !params.tissue_hierarchy_enable)
  error 'region_reference_atlas requires tissue_hierarchy_enable=true'
def run_full_pipeline = params.run_full_pipeline as boolean
def active_profiles = (workflow.profile ?: '')
.split(',')
.collect { it.trim().toLowerCase() }
.findAll { it }

def runtime_profiles = active_profiles.intersect(['singularity', 'docker'])
if (runtime_profiles.size() > 1) {
error "Select only one runtime profile: use either '-profile singularity' or '-profile docker' (not both)."
}

def normalize_arch = { raw -> HostRuntime.normalizeArch(raw) }
def command_output = { String cmd -> HostRuntime.commandOutput(cmd) }
def command_succeeds = { String cmd -> HostRuntime.commandSucceeds(cmd) }
def parse_positive_int = { raw, fallback -> HostRuntime.positiveInt(raw, fallback) }

def paramOr = { String key, def fallback ->
params.containsKey(key) ? params[key] : fallback
}

def scheduler_cpu_raw = [
System.getenv('SLURM_CPUS_PER_TASK'),
System.getenv('NSLOTS'),
System.getenv('PBS_NP'),
System.getenv('LSB_DJOB_NUMPROC')
].find { it && it.toString().trim() ==~ /\d+/ }
def host_cpus = scheduler_cpu_raw
? parse_positive_int(scheduler_cpu_raw, Runtime.runtime.availableProcessors())
: Math.max(1, Runtime.runtime.availableProcessors())
def raw_max_cpus = paramOr('max_cpus', 'auto')
def raw_max_cpus_text = raw_max_cpus == null ? 'auto' : raw_max_cpus.toString().trim().toLowerCase()
def cpu_reserve_raw = paramOr('hardware_cpu_reserve', 'auto')
def cpu_reserve_text = cpu_reserve_raw == null ? 'auto' : cpu_reserve_raw.toString().trim().toLowerCase()
def cpu_reserve = scheduler_cpu_raw ? 0 : ((cpu_reserve_text in ['', 'auto']) ? (host_cpus >= 24 ? 2 : (host_cpus > 4 ? 1 : 0)) : Math.max(0, parse_positive_int(cpu_reserve_raw, 0)))
def usable_host_cpus = Math.max(1, host_cpus - cpu_reserve)
def configured_max_cpus = (raw_max_cpus_text in ['', '0', 'auto'])
? usable_host_cpus
: Math.max(1, Math.min(parse_positive_int(raw_max_cpus, usable_host_cpus), usable_host_cpus))
params.max_cpus = configured_max_cpus

def host_memory_gb = HostRuntime.memoryGb()
def requested_memory_reserve_gb = 6
try {
requested_memory_reserve_gb = Math.max(1, Math.ceil((paramOr('hardware_min_free_system_gb', 6.0) as double)) as int)
} catch (Throwable ignored) {}
def memory_reserve_gb = Math.min(requested_memory_reserve_gb, Math.max(1, Math.floor(host_memory_gb * 0.15d) as int))
def usable_host_memory_gb = Math.max(2, host_memory_gb - memory_reserve_gb)
def raw_max_memory = paramOr('max_memory_gb', 'auto')
def raw_max_memory_text = raw_max_memory == null ? 'auto' : raw_max_memory.toString().trim().toLowerCase()
def configured_max_memory_gb = (raw_max_memory_text in ['', '0', 'auto'])
? usable_host_memory_gb
: parse_positive_int(raw_max_memory, usable_host_memory_gb)
def effective_max_memory_gb = Math.max(2, Math.min(configured_max_memory_gb, usable_host_memory_gb))
if (raw_max_memory_text in ['', '0', 'auto']) {
println "Runtime resource auto-detection: max_cpus=${configured_max_cpus}/${host_cpus}, max_memory_gb=${effective_max_memory_gb}/${host_memory_gb} (reserved CPUs=${cpu_reserve}, RAM=${memory_reserve_gb} GB)."
} else if (effective_max_memory_gb != configured_max_memory_gb) {
println "WARN: Reducing max_memory_gb from ${configured_max_memory_gb} to ${effective_max_memory_gb} to preserve ${memory_reserve_gb} GB for the host."
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
params._resolved_compute_device = resolved_compute_device

def gpu_inventory = []
if (detected_nvidia) {
def gpu_query = command_output('nvidia-smi --query-gpu=index,memory.total,memory.free,name --format=csv,noheader,nounits 2>/dev/null')
gpu_query.readLines().each { line ->
  def fields = line.split(',', 4).collect { it.trim() }
  if (fields.size() >= 4 && fields[0] ==~ /\d+/ && fields[1] ==~ /\d+/ && fields[2] ==~ /\d+/) {
    gpu_inventory << [index: fields[0].toInteger(), total_memory_gb: fields[1].toDouble() / 1024.0d, free_memory_gb: fields[2].toDouble() / 1024.0d, name: fields[3]]
  }
}
}
def largest_gpu_vram_gb = gpu_inventory ? (gpu_inventory.collect { it.total_memory_gb as double }.max() as double) : 0.0d
def hardware_plan = HardwarePolicy.resolve(params, configured_max_cpus, effective_max_memory_gb, resolved_compute_device == 'gpu', largest_gpu_vram_gb)
def runtime_plan = TaskRuntime.create(hardware_plan, resolved_compute_device)
hardware_plan.updates.each { key, value -> params[key.toString()] = value }
params.hardware_profile = hardware_plan.profile
params._resolved_hardware_profile = hardware_plan.profile
params._detected_host_cpus = host_cpus
params._detected_host_memory_gb = host_memory_gb
params._detected_gpu_count = gpu_inventory.size()
params._detected_gpu_vram_gb = largest_gpu_vram_gb

def hardware_plan_document = [
schema_version: 1,
detected: [architecture: detected_arch, host_cpus: host_cpus, host_memory_gb: host_memory_gb, gpus: gpu_inventory],
reserved: [cpus: cpu_reserve, memory_gb: memory_reserve_gb],
policy: hardware_plan,
task_runtime_plan: runtime_plan
]
println "Hardware policy: enabled=${hardware_plan.enabled}, profile=${hardware_plan.profile}, CPU budget=${hardware_plan.cpu_budget}, RAM budget=${hardware_plan.memory_budget_gb} GB, GPUs=${gpu_inventory.size()}, max GPU VRAM=${String.format('%.1f', largest_gpu_vram_gb)} GB."
println "Hardware stage plan: " + hardware_plan.stages.collect { name, resource -> "${name}=${resource.cpus}c/${resource.memory_gb}GB" }.join(', ')
try {
  def outputRoot = (params.outdir_base ?: '').toString()
  if (outputRoot && !outputRoot.contains('://')) {
    def executionDir = new File(outputRoot, '00_execution')
    executionDir.mkdirs()
    def hardwarePlanFile = new File(executionDir, 'hardware_plan.json')
    hardwarePlanFile.text = JsonOutput.prettyPrint(JsonOutput.toJson(hardware_plan_document)) + System.lineSeparator()
    params._hardware_plan_file = hardwarePlanFile.absolutePath
  }
} catch (Throwable exc) {
  log.warn "Could not write hardware_plan.json: ${exc.message}"
}

def resolved_singularity_image = (paramOr("singularity_image", "") ?: "").toString().trim()
def resolved_docker_image = (paramOr("docker_image", "") ?: "").toString().trim()

def stage_order = PipelineInputs.STAGES
def stage_index = stage_order.withIndex().collectEntries { stage_name, idx -> [(stage_name): idx] }

def normalize_stage = { raw_value, fallback_value ->
PipelineInputs.normalizeStage(raw_value, fallback_value)
}

def titan_requested = (((params.titan_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
def pathofmpred_requested = (((params.pathofmpred_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
if (pathofmpred_requested) titan_requested = true
def default_end_point = run_full_pipeline
  ? (pathofmpred_requested ? 'pathofmpred' : (titan_requested ? 'titan' : 'cluster_geojson'))
  : 'tissue_mask'
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

def requireStageOutput = { String stage_name, source_ch ->
source_ch
  .ifEmpty { error "Stage '${stage_name}' emitted no outputs; check joins." }
  .view { value -> "[OK] Stage output contract satisfied: ${stage_name}" }
}

def gigatime_enabled = (((params.gigatime_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
def marker_quantification_enabled = (((params.marker_quantification_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
def gigatime_output_format = (params.gigatime_output_format ?: 'none').toString().trim().toLowerCase()
def gigatime_export_ometiff_enabled = (((params.gigatime_export_ometiff ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
if (gigatime_export_ometiff_enabled && gigatime_output_format != 'zarr') {
error 'gigatime_export_ometiff=true requires gigatime_output_format=zarr; use ome_tiff for a direct dense TIFF or none for integrated tables only.'
}
def gigatime_kodama_enabled = ((((params.gigatime_kodama_enable == null) ? true : params.gigatime_kodama_enable).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
def grandqc_enabled = (((params.grandqc_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
def pathsegmentor_enabled = (((params.pathsegmentor_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
def pathsegmentor_refine_enabled = (((params.pathsegmentor_guided_refine_enable ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
if (pathsegmentor_refine_enabled && !pathsegmentor_enabled) {
error 'pathsegmentor_guided_refine_enable requires pathsegmentor_enable=true'
}
def tma_enabled = ((((params.tma_enable == null) ? true : params.tma_enable).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
def legacy_consensus_enabled = ((((params.cell_consensus_enable == null) ? true : params.cell_consensus_enable).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
def cell_detection_mode = (params.cell_detection_mode ?: (legacy_consensus_enabled ? 'consensus' : 'stardist')).toString().trim().toLowerCase()
if (!(cell_detection_mode in ['consensus', 'stardist'])) {
error "Invalid --cell_detection_mode '${params.cell_detection_mode}'. Use consensus or stardist."
}
def cell_consensus_enabled = cell_detection_mode == 'consensus'
params._resolved_cell_detection_mode = cell_detection_mode
if (cell_consensus_enabled && resolved_compute_device != 'gpu') {
error "cell_detection_mode=consensus requires GPU execution. Use --compute_device gpu, or explicitly choose --cell_detection_mode stardist; the pipeline will not change the scientific cell set based on hardware."
}
if (!grandqc_enabled && stage_index[end_point] >= stage_index['grandqc']) {
error "GrandQC is a mandatory upstream stage. Remove --grandqc_enable false or restrict the run to --end_point convert."
}
def uni2_reuse_existing = (((params.uni2_reuse_existing ?: false).toString().trim().toLowerCase()) in ['true', '1', 'yes', 'y', 'on'])
def uni2_sampling_mode = (params.uni2_sampling_mode ?: 'grid').toString().trim().toLowerCase()
if (uni2_sampling_mode in ['cell', 'cell_centered', 'cell-centred', 'cell-centered']) uni2_sampling_mode = 'cells'
if (!(uni2_sampling_mode in ['cells', 'grid', 'both'])) {
error "Invalid --uni2_sampling_mode '${params.uni2_sampling_mode}'. Use cells, grid, or both."
}
def uni2_grid_mode = uni2_sampling_mode in ['grid', 'both']
if (params.tissue_hierarchy_enable && !uni2_grid_mode)
  error 'tissue_hierarchy_enable requires the grid or both UNI-2 sampling route'
def uni2_cell_auxiliary_mode = uni2_sampling_mode == 'both'
params._resolved_uni2_sampling_mode = uni2_sampling_mode
def analysis_contract = ScientificContract.resolve(
  params.analysis_intent,
  uni2_sampling_mode,
  cell_detection_mode,
  gigatime_enabled,
  titan_requested,
  pathofmpred_requested,
  start_point,
  end_point,
)
def analysis_intent = analysis_contract.analysis_intent
params._resolved_analysis_intent = analysis_intent
def validation_readiness = StudyEvidence.resolve(
  params.study_manifest,
  params.evidence_gate_mode,
  analysis_intent,
)
analysis_contract.validation_readiness = validation_readiness
try {
  params._analysis_contract_file = StudyEvidence.writeReports(analysis_contract, validation_readiness, params.outdir_base)
} catch (Throwable exc) {
  log.warn "Could not write scientific contract reports: ${exc.message}"
}
println "Scientific analysis contract: intent=${analysis_intent}, observation_unit=${analysis_contract.observation_unit}, UNI-2 route=${uni2_sampling_mode}, cell detection=${cell_detection_mode}."
if (!validation_readiness.gate_passed) {
  def message = StudyEvidence.gateMessage(validation_readiness)
  if (validation_readiness.gate_mode == 'fail') error message
  if (validation_readiness.gate_mode == 'warn') log.warn message
}
def run_convert = should_run_stage('convert')
def run_grandqc = should_run_stage('grandqc')
def run_stardist = should_run_stage('stardist')
def run_cell_consensus = should_run_stage('cell_consensus') && cell_consensus_enabled
def run_tma = should_run_stage('tma') && tma_enabled
def run_gigatime = should_run_stage('gigatime') && gigatime_enabled
def run_tissue_mask = should_run_stage('tissue_mask')
def run_pathsegmentor = should_run_stage('pathsegmentor') && pathsegmentor_enabled
def run_cell_assignment = should_run_stage('cell_assignment')
def run_cytoplasm = should_run_stage('cytoplasm')
def run_marker_quantification = should_run_stage('marker_quantification') && gigatime_enabled && marker_quantification_enabled
def run_grid_tiles = should_run_stage('grid_tiles') && uni2_grid_mode
def run_uni2 = should_run_stage('uni2') && !uni2_reuse_existing
def run_kodama = should_run_stage('kodama')
def run_gigatime_kodama = run_kodama && gigatime_enabled && gigatime_kodama_enabled
if (run_gigatime_kodama && !(params.gigatime_integrated_quantification as boolean)) {
error "GigaTIME KODAMA requires --gigatime_integrated_quantification true."
}
def run_clustering = should_run_stage('clustering')
if ((params.cluster_target_clusters as int) > 0 && !((params.cluster_forced_count_sensitivity_acknowledged ?: false) as boolean)) {
error "A forced cluster count is sensitivity-only. Set --cluster_forced_count_sensitivity_acknowledged true or restore --cluster_target_clusters 0."
}
def run_cluster_mask = should_run_stage('cluster_mask')
def grow_tissue_requested = should_run_stage('grow_tissue')
def run_grow_tissue = grow_tissue_requested && !uni2_grid_mode
def run_auxiliary_cell_grow_tissue = grow_tissue_requested && uni2_cell_auxiliary_mode
def run_medsam_refine = should_run_stage('medsam_refine')
def run_pathsegmentor_refine = should_run_stage('medsam_refine') && pathsegmentor_refine_enabled
def run_cluster_geojson = should_run_stage('cluster_geojson')
def run_neoplastic_section = should_run_stage('neoplastic_section') && titan_requested
def run_titan = should_run_stage('titan') && titan_requested
def run_pathofmpred = should_run_stage('pathofmpred') && pathofmpred_requested
if (grow_tissue_requested && uni2_grid_mode) {
println uni2_cell_auxiliary_mode
  ? "Both UNI-2 routes: the dense grid mask bypasses grow_tissue, while the sparse cell-centred mask runs grow_tissue before MedSAM."
  : "Grid UNI-2 route: grow_tissue is not applicable because adjacent grid cores already form a dense cluster mask; downstream MedSAM, when requested, receives the grid mask directly."
}
if (should_run_stage('neoplastic_section') && !titan_requested) {
error "The neoplastic_section stage requires --titan_enable true or --pathofmpred_enable true."
}
if (should_run_stage('titan') && !titan_requested) {
error "The titan stage requires --titan_enable true or --pathofmpred_enable true."
}
if (should_run_stage('pathofmpred') && !pathofmpred_requested) {
error "The pathofmpred stage requires --pathofmpred_enable true."
}
if ((run_titan || run_pathofmpred) && resolved_compute_device != 'gpu') {
error "TITAN/PathoFMPred execution requires --compute_device gpu."
}
if (run_pathsegmentor && resolved_compute_device != 'gpu') {
error 'PathSegmentor execution requires --compute_device gpu.'
}
if (run_pathsegmentor) {
  ['pathsegmentor_repo', 'pathsegmentor_config', 'pathsegmentor_checkpoint'].each { key ->
    if (!(params[key] ?: '').toString().trim()) error "PathSegmentor requires --${key}"
  }
}
if (run_pathofmpred && !(params.pathofmpred_cancer ?: '').toString().trim()) {
error "PathoFMPred requires --pathofmpred_cancer with the intended TCGA cancer code (for example BRCA)."
}
if (run_neoplastic_section && !cell_consensus_enabled) {
error "Neoplastic-section selection requires --cell_detection_mode consensus."
}
def kodama_requested_modes = PipelineHelpers.resolveKodamaModes(params.kodama_embedding_mode)
def include_uni2_nuclei = params.uni2_include_nuclei == null ? kodama_requested_modes.contains('nuclei') : (params.uni2_include_nuclei as boolean)
def include_uni2_cyto = params.uni2_include_cyto == null ? kodama_requested_modes.contains('cyto') : (params.uni2_include_cyto as boolean)
def include_uni2_inner_square = params.uni2_include_inner_square == null ? kodama_requested_modes.contains('inner_square') : (params.uni2_include_inner_square as boolean)
def use_roi_crop_for_uni2 = params.uni2_use_roi_crop == null ? true : (params.uni2_use_roi_crop as boolean)
def fuse_tile_inner_square_uni2 = params.uni2_fuse_tile_inner_square == null ? true : (params.uni2_fuse_tile_inner_square as boolean)
if (uni2_grid_mode) {
if (!use_roi_crop_for_uni2) error "Grid sampling currently requires --uni2_use_roi_crop true."
if (!fuse_tile_inner_square_uni2 || !include_uni2_inner_square) {
  error "Grid sampling requires fused tile+inner-square UNI2."
}
if (include_uni2_nuclei || include_uni2_cyto || kodama_requested_modes.any { it in ['nuclei', 'cyto'] }) {
  error "Grid sampling supports only tile and inner_square modes."
}
def supported_grid_encoders = ['uni2-h', 'virchow', 'virchow2', 'phikon-v2'] as Set
if (!supported_grid_encoders.contains(params.uni2_encoder?.toString()?.toLowerCase())) {
  error "Grid sampling requires a registered foundation encoder: ${supported_grid_encoders.sort().join(', ')}."
}
}
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

println "Runtime: arch=${detected_arch}, device=${resolved_compute_device}, image_mode=${runtime_image_mode}."
println "Pipeline stage window: ${start_point} -> ${end_point}"

def supported_image_suffixes = PipelineInputs.IMAGE_SUFFIXES
def detectImageSuffix = { String fileName -> PipelineInputs.imageSuffix(fileName) }
def imageSuffixPriority = { String suffix -> PipelineInputs.imageSuffixPriority(suffix) }
def deriveSampleId = { imageFile -> PipelineInputs.sampleId(imageFile) }
def extractCziRegionLabel = { String rawName -> PipelineInputs.cziRegionLabel(rawName) }
def extractCziRegionOrder = { String rawName -> PipelineInputs.cziRegionOrder(rawName) }

def buildSampleId = { imageFile, String regionLabel = '' ->
def baseId = deriveSampleId(imageFile)
def rawId = regionLabel ? "${baseId}__${regionLabel}" : baseId
def safeId = rawId.replaceAll(/[^A-Za-z0-9._-]+/, '_').replaceAll(/^_+|_+$/, '')
safeId ?: 'sample'
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
  error "No supported images found in --folder_input ${input_dir}."
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
  def input_support = file(PipelineInputs.inputSupport(image_file, baseDir), checkIfExists: true)
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
            roi_file.bytes.encodeBase64().toString(),
            input_support
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
  sample_rows << tuple(buildSampleId(image_file), file(image_file.absolutePath, checkIfExists: true), '', roi_hint_name, roi_hint_b64, input_support)
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
def input_support = file(PipelineInputs.inputSupport(image_file, baseDir), checkIfExists: true)
sample_rows << tuple(sample_id, image_file, input_region, roi_hint_name, roi_hint_b64, input_support)
}

if (!sample_rows) {
error "No input samples were resolved."
}
println "Resolved input samples (${sample_rows.size()}): ${sample_rows.collect { it[0] }.join(', ')}"

StoragePreflight.run(params, sample_rows.collect { it[1].toString() }, baseDir, workflow.workDir, workflow.profile)

Channel
.fromList(sample_rows)
.set { input_spec_ch }

def convert_input_ch = input_spec_ch
.map { sample_id, image_file, input_region, roi_hint_name, roi_hint_b64, input_support ->
  tuple(sample_id, image_file, input_region ?: '', input_support)
}

def image_input_ch = input_spec_ch
.map { sample_id, image_file, input_region, roi_hint_name, roi_hint_b64, input_support ->
  tuple(sample_id, image_file)
}

def roi_hint_ch = input_spec_ch
.map { sample_id, image_file, input_region, roi_hint_name, roi_hint_b64, input_support ->
  tuple(sample_id, roi_hint_name ?: '', roi_hint_b64 ?: '')
}
def roi_provided_ch = roi_hint_ch
.map { sample_id, roi_hint_name, roi_hint_b64 ->
  tuple(sample_id, ((roi_hint_b64 ?: '').toString().trim() ? true : false))
}

def ome_tif_ch
def converted_resolution_report_ch
if (run_convert) {
PREPARE_INPUT_OMETIFF(convert_input_ch)
ome_tif_ch = requireStageOutput('convert', PREPARE_INPUT_OMETIFF.out.ome_tif)
converted_resolution_report_ch = requireStageOutput('convert', PREPARE_INPUT_OMETIFF.out.converted_resolution_report)
} else {
ome_tif_ch = convert_input_ch.map { sample_id, image_file, input_region, input_support ->
  def is_ome = image_file.name.toLowerCase().endsWith('.ome.tif')
  def ome_tif = is_ome
    ? image_file
    : file("${params.outdir_base}/01_input/${sample_id}/${sample_id}.ome.tif", checkIfExists: true)
  tuple(sample_id, ome_tif)
}
converted_resolution_report_ch = convert_input_ch.map { sample_id, image_file, input_region, input_support ->
  tuple(sample_id, file("${params.outdir_base}/01_input/${sample_id}/${sample_id}.converted_resolution.json", checkIfExists: true))
}
}

def grandqc_dir_ch = Channel.empty()
def grandqc_clean_tissue_mask_ch = Channel.empty()
if (run_grandqc) {
RUN_GRANDQC_ARTIFACT_ANALYSIS(ome_tif_ch, runtime_plan)
grandqc_dir_ch = RUN_GRANDQC_ARTIFACT_ANALYSIS.out.grandqc_dir
grandqc_clean_tissue_mask_ch = requireStageOutput('grandqc', RUN_GRANDQC_ARTIFACT_ANALYSIS.out.clean_tissue_mask)
} else if (stage_index[end_point] > stage_index['grandqc']) {
grandqc_clean_tissue_mask_ch = image_input_ch.map { sample_id, _image_input ->
  tuple(sample_id, file("${params.outdir_base}/02_grandqc/${sample_id}/grandqc_${sample_id}/${sample_id}_grandqc_clean_tissue_mask.tif", checkIfExists: true))
}
}

def roi_geojson_ch = Channel.empty()
if (run_stardist) {
def roi_prepare_input_ch = ome_tif_ch
  .join(grandqc_clean_tissue_mask_ch)
  .join(roi_hint_ch)
  .join(converted_resolution_report_ch)
  .map { sample_id, ome_tif, _clean_tissue_mask, roi_hint_name, roi_hint_b64, image_qc_report ->
    tuple(sample_id, ome_tif, image_qc_report, roi_hint_name ?: '', roi_hint_b64 ?: '')
  }
PREPARE_ROI_GEOJSON(roi_prepare_input_ch)
roi_geojson_ch = PREPARE_ROI_GEOJSON.out.roi_geojson
} else if (run_cell_assignment) {
roi_geojson_ch = image_input_ch.map { sample_id, _image_input ->
  tuple(sample_id, file("${params.outdir_base}/06_roi/${sample_id}/${sample_id}.roi.geojson", checkIfExists: true))
}
}

def stardist_roi_geojson_ch = roi_geojson_ch

def need_stardist_outputs = run_stardist || run_cell_consensus || run_tma || run_tissue_mask || pathsegmentor_enabled || run_cell_assignment || run_cytoplasm || run_gigatime || run_marker_quantification || run_grid_tiles || run_uni2 || run_cluster_mask || run_grow_tissue || run_auxiliary_cell_grow_tissue || run_medsam_refine || run_neoplastic_section || params.cell_profiles_enable || params.tissue_hierarchy_enable

def crop_roi_ch = Channel.empty()
def labels_tif_ch = Channel.empty()
def labels_full_ch = Channel.empty()
def objects_csv_ch = Channel.empty()
def roi_crop_geojson_ch = Channel.empty()
def shift_json_ch = Channel.empty()
def tissue_mask_ch = Channel.empty()
def pathsegmentor_bundle_ch = Channel.empty()

if (need_stardist_outputs) {
def require_labels_full = (!use_roi_crop_for_uni2 && run_uni2) || ((run_cytoplasm && (params.expand_full_labels as boolean)) && !use_roi_crop_for_uni2)
if (run_stardist) {
  def crop_input_ch = ome_tif_ch
    .join(stardist_roi_geojson_ch)
    .join(grandqc_clean_tissue_mask_ch)
    .join(converted_resolution_report_ch)
    .map { sample_id, ome_tif, stardist_roi_geojson, clean_tissue_mask, resolution_json ->
      tuple(sample_id, ome_tif, stardist_roi_geojson, clean_tissue_mask, resolution_json)
    }
  PREPARE_ANALYSIS_CROP(crop_input_ch)
  crop_roi_ch = requireStageOutput('analysis_crop', PREPARE_ANALYSIS_CROP.out.crop_roi)
  roi_crop_geojson_ch = PREPARE_ANALYSIS_CROP.out.roi_crop_geojson
  shift_json_ch = PREPARE_ANALYSIS_CROP.out.shift_json
} else {
  crop_roi_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/prepared_crop/crop_roi.tif", checkIfExists: true))
  }
  labels_tif_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/labels.tif", checkIfExists: true))
  }
  objects_csv_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/stardist_out/objects.csv", checkIfExists: true))
  }
  roi_crop_geojson_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/prepared_crop/roi_all_crop.geojson", checkIfExists: true))
  }
  shift_json_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/prepared_crop/shift.json", checkIfExists: true))
  }
  labels_full_ch = require_labels_full
    ? image_input_ch.map { sample_id, _image_input ->
      tuple(sample_id, resolveLabelsFullArtifact(sample_id))
    }
    : Channel.empty()
}

if (run_stardist || run_tissue_mask) {
  def crop_clean_mask_input_ch = grandqc_clean_tissue_mask_ch
    .join(shift_json_ch)
    .join(roi_crop_geojson_ch)
    .map { sample_id, clean_tissue_mask, shift_json, roi_crop_geojson -> tuple(sample_id, clean_tissue_mask, shift_json, roi_crop_geojson) }
  CROP_GRANDQC_CLEAN_MASK(crop_clean_mask_input_ch)
  tissue_mask_ch = requireStageOutput('tissue_mask', CROP_GRANDQC_CLEAN_MASK.out.tissue_mask)
} else {
  tissue_mask_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/04_tissue_mask/${sample_id}/${sample_id}_tissue_mask.tif", checkIfExists: true))
  }
}

if (pathsegmentor_enabled && stage_index[end_point] >= stage_index['pathsegmentor']) {
  def pathsegmentor_input_ch = crop_roi_ch
    .join(tissue_mask_ch, failOnMismatch: true, failOnDuplicate: true)
    .join(converted_resolution_report_ch, failOnMismatch: true, failOnDuplicate: true)
    .map { sample_id, image_tif, tissue_mask_tif, resolution_json ->
      tuple(sample_id, image_tif, tissue_mask_tif, resolution_json)
    }
  RUN_PATHSEGMENTOR_INFERENCE(pathsegmentor_input_ch, run_pathsegmentor, runtime_plan)
  pathsegmentor_bundle_ch = run_pathsegmentor
    ? requireStageOutput('pathsegmentor', RUN_PATHSEGMENTOR_INFERENCE.out.bundle)
    : RUN_PATHSEGMENTOR_INFERENCE.out.bundle
}

if (run_stardist) {
  params._resolved_stardist_write_full_labels = require_labels_full
  params._resolved_stardist_full_format = params.full_format ?: 'tif'
  params._resolved_allow_huge_tif = require_labels_full ? (params.allow_huge_tif as boolean) : false
  def stardist_input_ch = crop_roi_ch
    .join(roi_crop_geojson_ch)
    .join(shift_json_ch)
    .join(tissue_mask_ch)
    .map { sample_id, crop_tif, roi_crop_geojson, source_shift_json, clean_tissue_mask ->
      tuple(sample_id, crop_tif, roi_crop_geojson, source_shift_json, clean_tissue_mask)
    }
  RUN_STARDIST_ROI_SEGMENTATION(stardist_input_ch, runtime_plan)
  labels_tif_ch = requireStageOutput('stardist', RUN_STARDIST_ROI_SEGMENTATION.out.labels_tif)
  labels_full_ch = RUN_STARDIST_ROI_SEGMENTATION.out.labels_full
  objects_csv_ch = requireStageOutput('stardist_objects', RUN_STARDIST_ROI_SEGMENTATION.out.objects_csv)
}
}

def cell_profile_cellvit_ch = Channel.empty()
if (cell_consensus_enabled) {
if (run_cell_consensus) {
  def hovernet_input_ch = crop_roi_ch
    .join(shift_json_ch)
    .join(tissue_mask_ch)
    .map { sample_id, crop_tif, shift_json, clean_tissue_mask -> tuple(sample_id, crop_tif, shift_json, clean_tissue_mask) }
  RUN_HOVERNET_MONUSAC(hovernet_input_ch, runtime_plan)
  def cellvit_input_ch = crop_roi_ch
    .join(shift_json_ch, failOnDuplicate: true, failOnMismatch: true)
    .join(tissue_mask_ch, failOnDuplicate: true, failOnMismatch: true)
    .join(converted_resolution_report_ch, failOnDuplicate: true, failOnMismatch: true)
    .map { sample_id, crop_tif, shift_json, clean_tissue_mask, resolution -> tuple(sample_id, crop_tif, shift_json, clean_tissue_mask, resolution) }
  RUN_CELLVITPP(cellvit_input_ch, runtime_plan)
  cell_profile_cellvit_ch = RUN_CELLVITPP.out.cellvit_dir

  def consensus_input_ch = objects_csv_ch
    .join(RUN_HOVERNET_MONUSAC.out.cells_json)
    .join(RUN_CELLVITPP.out.cells_json)
    .join(crop_roi_ch)
    .join(shift_json_ch)
    .map { sample_id, stardist_objects, hovernet_cells, cellvit_cells, crop_tif, shift_json ->
      tuple(sample_id, stardist_objects, hovernet_cells, cellvit_cells, crop_tif, shift_json)
    }
  BUILD_CELL_CONSENSUS(consensus_input_ch)
  labels_tif_ch = requireStageOutput('cell_consensus', BUILD_CELL_CONSENSUS.out.labels_tif)
  objects_csv_ch = requireStageOutput('cell_consensus_objects', BUILD_CELL_CONSENSUS.out.objects_csv)
} else if (stage_index[end_point] >= stage_index['cell_consensus']) {
  labels_tif_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03d_cell_consensus/${sample_id}/consensus_${sample_id}/labels.tif", checkIfExists: true))
  }
  objects_csv_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03d_cell_consensus/${sample_id}/consensus_${sample_id}/objects.csv", checkIfExists: true))
  }
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
objects_for_assignment_ch = requireStageOutput('tma', DETECT_TMA_SPOTS.out.objects_tma_assigned)
} else if (tma_outputs_available && run_cell_assignment) {
objects_for_assignment_ch = image_input_ch.map { sample_id, _image_input ->
  tuple(sample_id, file("${params.outdir_base}/04_TMA/${sample_id}/tma_${sample_id}/${sample_id}_objects_tma_assigned.csv", checkIfExists: true))
}
}

def crop_roi_for_masks_ch = (run_gigatime || run_cell_assignment)
? (run_stardist
  ? crop_roi_ch
  : image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/prepared_crop/crop_roi.tif", checkIfExists: true))
  })
: Channel.empty()
def roi_crop_geojson_for_masks_ch = run_cell_assignment
? (run_stardist
  ? roi_crop_geojson_ch
  : image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/prepared_crop/roi_all_crop.geojson", checkIfExists: true))
  })
: Channel.empty()
def shift_json_for_masks_ch = run_gigatime
? (run_stardist
  ? shift_json_ch
  : image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/prepared_crop/shift.json", checkIfExists: true))
  })
: Channel.empty()
def gigatime_image_ch = Channel.empty()
def gigatime_quant_dir_ch = Channel.empty()

if (!run_gigatime && (run_marker_quantification || run_gigatime_kodama)) {
  gigatime_image_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/05_gigatime/${sample_id}/gigatime_${sample_id}", checkIfExists: true))
  }
  gigatime_quant_dir_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/05_gigatime/${sample_id}/quantification_${sample_id}", checkIfExists: true))
  }
}
if (run_cell_assignment) {
def roi_mask_input_ch = roi_crop_geojson_for_masks_ch
  .join(crop_roi_for_masks_ch)
  .join(roi_provided_ch)
  .filter { sample_id, roi_crop_geojson, crop_roi_tif, roi_provided -> roi_provided }
  .map { sample_id, roi_crop_geojson, crop_roi_tif, roi_provided ->
    tuple(sample_id, roi_crop_geojson, crop_roi_tif)
  }
ROI_GEOJSON_TO_MASK(roi_mask_input_ch)
}

def final_cluster_geojson_ch = Channel.empty()
if (run_cell_assignment || run_cytoplasm || run_gigatime || run_marker_quantification || run_grid_tiles || run_uni2 || run_kodama || run_clustering || run_cluster_mask || run_grow_tissue || run_auxiliary_cell_grow_tissue || run_medsam_refine || run_cluster_geojson || params.cell_profiles_enable || params.tissue_hierarchy_enable) {
def objects_assigned_ch = Channel.empty()
if (run_cell_assignment) {
  def assign_input_ch = objects_for_assignment_ch
    .join(shift_json_ch)
    .join(roi_geojson_ch)
    .map { sample_id, objects_csv, shift_json, roi_geojson ->
      tuple(sample_id, objects_csv, roi_geojson, shift_json)
    }
  MAP_CELLS_TO_ROI_POLYGONS(assign_input_ch)
  objects_assigned_ch = requireStageOutput('cell_assignment', MAP_CELLS_TO_ROI_POLYGONS.out.objects_assigned)
} else if ((run_kodama || run_clustering) && (!uni2_grid_mode || uni2_cell_auxiliary_mode)) {
  objects_assigned_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/07_cell_assignments/${sample_id}/${sample_id}_objects_assigned.csv", checkIfExists: true))
  }
}

def cyto_mask_ch = Channel.empty()
def cyto_mask_full_ch = Channel.empty()
def nuclei_mask_for_quant_ch = labels_tif_ch
if (run_cytoplasm) {
  def expand_primary_ch = labels_tif_ch
    .join(crop_roi_ch)
    .join(shift_json_ch).join(converted_resolution_report_ch).join(tissue_mask_ch)
    .map { sample_id, labels_tif, preview_background_tif, shift, resolution, tissue ->
      tuple(sample_id, labels_tif, 'labels_cyto', preview_background_tif.toString(), shift, resolution, tissue, 'crop')
    }
  EXPAND_LABELS_TO_CYTOPLASM_PRIMARY(expand_primary_ch)

  if ((params.expand_full_labels as boolean) && !(run_uni2 && use_roi_crop_for_uni2)) {
    def expand_full_ch = labels_full_ch.join(shift_json_ch).join(converted_resolution_report_ch).join(grandqc_clean_tissue_mask_ch).map { sample_id, labels_full_path, shift, resolution, tissue ->
      tuple(sample_id, labels_full_path, 'labels_full_cyto', '', shift, resolution, tissue, 'original')
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
  cyto_mask_ch = requireStageOutput('cytoplasm', cyto_mask_ch)
} else if (run_uni2 && !uni2_grid_mode) {
  cyto_mask_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/08_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif", checkIfExists: true))
  }
  if (!use_roi_crop_for_uni2) {
    cyto_mask_full_ch = image_input_ch.map { sample_id, _image_input ->
      tuple(sample_id, file("${params.outdir_base}/08_cytoplasm/${sample_id}/${sample_id}_labels_full_cyto.tif", checkIfExists: true))
    }
  }
}

def cyto_mask_for_quant_ch = Channel.empty()
def ring_mask_for_quant_ch = image_input_ch.map { id, _image -> tuple(id, file("${projectDir}/resources/empty_embeddings_placeholder", checkIfExists: true)) }
def marker_quant_output_ch = Channel.empty()
if (run_gigatime || run_marker_quantification) {
  if ((params.expand_um as double) >= 0) {
    ring_mask_for_quant_ch = run_cytoplasm ? EXPAND_LABELS_TO_CYTOPLASM_PRIMARY.out.ring_labels.map { id, mask, kind -> tuple(id, mask) }
      : image_input_ch.map { id, _image -> tuple(id, file("${params.outdir_base}/08_cytoplasm/${id}/${id}_labels_cyto_compartments/labels_perinuclear_ring.tif", checkIfExists: true)) }
  }
  cyto_mask_for_quant_ch = run_cytoplasm
    ? cyto_mask_ch
    : image_input_ch.map { sample_id, _image_input ->
      tuple(sample_id, file("${params.outdir_base}/08_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif", checkIfExists: true))
    }
}

if (run_gigatime) {
  def gigatime_input_ch = crop_roi_for_masks_ch
    .join(shift_json_for_masks_ch)
    .join(nuclei_mask_for_quant_ch)
    .join(cyto_mask_for_quant_ch)
    .join(tissue_mask_ch)
    .join(ring_mask_for_quant_ch)
  RUN_GIGATIME_ON_CROP(gigatime_input_ch, runtime_plan)
  gigatime_image_ch = requireStageOutput('gigatime', RUN_GIGATIME_ON_CROP.out.gigatime_dir)
  gigatime_quant_dir_ch = RUN_GIGATIME_ON_CROP.out.quant_dir
  if (run_marker_quantification) {
    marker_quant_output_ch = requireStageOutput('marker_quantification', gigatime_quant_dir_ch)
  }
  if (gigatime_export_ometiff_enabled) {
    EXPORT_GIGATIME_OMETIFF(gigatime_image_ch)
  }
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

  def quantInputs = nuclei_quant_input_ch.mix(cyto_quant_input_ch)
  if ((params.expand_um as double) >= 0) {
    quantInputs = quantInputs.mix(gigatime_image_ch.join(ring_mask_for_quant_ch).map { id, image, mask -> tuple(id, image, mask, 'ring') })
  }
  QUANTIFY_GIGATIME_INTENSITY(quantInputs)
  marker_quant_output_ch = requireStageOutput('marker_quantification', QUANTIFY_GIGATIME_INTENSITY.out.quant_csv)
}

def grid_objects_ch = Channel.empty()
def grid_metadata_ch = Channel.empty()
def grid_artifacts_needed = uni2_grid_mode && (run_grid_tiles || run_uni2 || run_kodama || run_clustering || run_cluster_mask || params.tissue_hierarchy_enable)
PREPARE_UNI2_SPATIAL_GRID(image_input_ch, crop_roi_ch, tissue_mask_ch, converted_resolution_report_ch, run_grid_tiles, grid_artifacts_needed)
grid_objects_ch = PREPARE_UNI2_SPATIAL_GRID.out.grid_objects
grid_metadata_ch = PREPARE_UNI2_SPATIAL_GRID.out.grid_metadata
if (run_grid_tiles) {
  grid_metadata_ch = requireStageOutput('grid_tiles', grid_metadata_ch)
}

def analysis_objects_ch = uni2_grid_mode ? grid_objects_ch : objects_assigned_ch

def tile_embeddings_ch = Channel.empty()
def nuclei_embeddings_ch = Channel.empty()
def cyto_embeddings_ch = Channel.empty()
def inner_square_embeddings_ch = Channel.empty()
def placeholder_embeddings_dir = file("${projectDir}/resources/empty_embeddings_placeholder", checkIfExists: true)
def placeholder_observations_file = file("${projectDir}/resources/empty_uni2_observations.csv", checkIfExists: true)
def placeholder_resolution_file = file("${projectDir}/resources/empty_uni2_resolution.json", checkIfExists: true)
def placeholder_embeddings_ch = image_input_ch.map { sample_id, _image_input ->
  tuple(sample_id, placeholder_embeddings_dir)
}
def published_gigatime_quant_dir_ch = (!run_gigatime && !run_marker_quantification && !run_gigatime_kodama && (gigatime_enabled || marker_quantification_enabled))
  ? image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/05_gigatime/${sample_id}/quantification_${sample_id}", checkIfExists: true))
  }
  : placeholder_embeddings_ch
if (run_uni2) {
  EXTRACT_PRIMARY_UNI2(crop_roi_ch, ome_tif_ch, labels_tif_ch, labels_full_ch, tissue_mask_ch,
    cyto_mask_ch, cyto_mask_full_ch, grid_objects_ch, grid_metadata_ch, converted_resolution_report_ch,
    [use_crop: use_roi_crop_for_uni2, grid: uni2_grid_mode, fuse: fuse_tile_inner_square_uni2,
     inner: include_uni2_inner_square, cyto: include_uni2_cyto, nuclei: include_uni2_nuclei], runtime_plan)
  tile_embeddings_ch = requireStageOutput('uni2', EXTRACT_PRIMARY_UNI2.out.tile)
  inner_square_embeddings_ch = EXTRACT_PRIMARY_UNI2.out.local
  nuclei_embeddings_ch = EXTRACT_PRIMARY_UNI2.out.nuclei
  cyto_embeddings_ch = EXTRACT_PRIMARY_UNI2.out.cyto
} else if (run_kodama) {
  tile_embeddings_ch = image_input_ch.map { sample_id, _image_input ->
    def embedding_dir = kodama_requested_modes.contains('tile')
      ? file("${params.outdir_base}/09_embeddings/${sample_id}/embeddings_${sample_id}_tile", checkIfExists: true)
      : placeholder_embeddings_dir
    tuple(sample_id, embedding_dir)
  }
  nuclei_embeddings_ch = image_input_ch.map { sample_id, _image_input ->
    def embedding_dir = kodama_requested_modes.contains('nuclei')
      ? file("${params.outdir_base}/09_embeddings/${sample_id}/embeddings_${sample_id}_nuclei", checkIfExists: true)
      : placeholder_embeddings_dir
    tuple(sample_id, embedding_dir)
  }
  cyto_embeddings_ch = image_input_ch.map { sample_id, _image_input ->
    def embedding_dir = kodama_requested_modes.contains('cyto')
      ? file("${params.outdir_base}/09_embeddings/${sample_id}/embeddings_${sample_id}_cyto", checkIfExists: true)
      : placeholder_embeddings_dir
    tuple(sample_id, embedding_dir)
  }
  inner_square_embeddings_ch = image_input_ch.map { sample_id, _image_input ->
    def embedding_dir = kodama_requested_modes.contains('inner_square')
      ? file("${params.outdir_base}/09_embeddings/${sample_id}/embeddings_${sample_id}_inner_square", checkIfExists: true)
      : placeholder_embeddings_dir
    tuple(sample_id, embedding_dir)
  }
}

// Retain actual extraction outputs before KODAMA applies its own mode selection.
def profile_primary_tile_ch = tile_embeddings_ch
def profile_primary_local_ch = inner_square_embeddings_ch
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
    .join(analysis_objects_ch)
    .map { sample_id, tile_embeddings_dir, nuclei_embeddings_dir, cyto_embeddings_dir, inner_square_embeddings_dir, analysis_objects ->
      tuple(sample_id, tile_embeddings_dir, cyto_embeddings_dir, inner_square_embeddings_dir, nuclei_embeddings_dir, analysis_objects)
    }
  RUN_KODAMA_ANALYSIS(kodama_input_ch, runtime_plan)
  kodama_dir_ch = requireStageOutput('kodama', RUN_KODAMA_ANALYSIS.out.kodama_dir)
  if (run_gigatime_kodama) {
    RUN_GIGATIME_KODAMA(gigatime_quant_dir_ch, runtime_plan)
  }
} else if (run_clustering) {
  kodama_dir_ch = image_input_ch.map { sample_id, _image_input ->
    tuple(sample_id, file("${params.outdir_base}/10_kodama/${sample_id}/kodama_output", checkIfExists: true))
  }
}

def profile_aux_tile_ch = Channel.empty()
def profile_aux_local_ch = Channel.empty()
if (uni2_cell_auxiliary_mode) {
  def auxiliary_flags = [
    run_uni2: run_uni2, run_kodama: run_kodama, run_clustering: run_clustering,
    run_cluster_mask: run_cluster_mask, run_grow_tissue: run_auxiliary_cell_grow_tissue,
    run_medsam_refine: run_medsam_refine, run_cluster_geojson: run_cluster_geojson,
    run_cytoplasm: run_cytoplasm,
    use_current_markers: run_gigatime || run_marker_quantification || run_gigatime_kodama,
    markers_configured: gigatime_enabled || marker_quantification_enabled,
  ]
  RUN_AUXILIARY_CELL_ROUTE(
    image_input_ch, crop_roi_ch, labels_tif_ch, objects_assigned_ch, tile_embeddings_ch,
    kodama_dir_ch, grid_objects_ch, tissue_mask_ch, shift_json_ch, converted_resolution_report_ch, cyto_mask_ch,
    gigatime_quant_dir_ch, auxiliary_flags, runtime_plan,
  )
  profile_aux_tile_ch = RUN_AUXILIARY_CELL_ROUTE.out.tile_embeddings
  profile_aux_local_ch = RUN_AUXILIARY_CELL_ROUTE.out.inner_embeddings
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
def cluster_csv_ch = Channel.empty()
def cluster_kodama_png_ch = Channel.empty()
if (run_clustering) {
  def clustering_base_ch = kodama_dir_ch
    .join(analysis_objects_ch)
    .map { sample_id, kodama_dir, analysis_objects_csv ->
      tuple(sample_id, kodama_dir, analysis_objects_csv)
    }
  def clustering_input_ch = clustering_base_ch
    .flatMap { sample_id, kodama_dir, objects_assigned_csv ->
      cluster_variant_defs.collect { spec ->
        def sample_key = "${sample_id}::${spec.variant}"
        tuple(sample_key, sample_id, spec.variant, spec.profile, cluster_resolution_value, kodama_dir, objects_assigned_csv)
      }
    }
  RUN_RCODE_CLUSTERING(clustering_input_ch)
  cluster_csv_ch = requireStageOutput('clustering', RUN_RCODE_CLUSTERING.out.cluster_csv)
  cluster_kodama_png_ch = RUN_RCODE_CLUSTERING.out.membership_png

  def marker_quant_for_cluster_assessment_ch = (run_gigatime || run_marker_quantification || run_gigatime_kodama)
    ? gigatime_quant_dir_ch
    : ((gigatime_enabled || marker_quantification_enabled)
      ? published_gigatime_quant_dir_ch
      : image_input_ch.map { sample_id, _image -> tuple(sample_id, placeholder_embeddings_dir) })
  def cluster_assessment_context_ch = analysis_objects_ch
    .join(marker_quant_for_cluster_assessment_ch)
    .flatMap { sample_id, analysis_objects_csv, marker_quant_dir ->
      cluster_variant_defs.collect { spec ->
        tuple("${sample_id}::${spec.variant}", analysis_objects_csv, marker_quant_dir)
      }
    }
  def cluster_assessment_input_ch = cluster_csv_ch
    .join(cluster_assessment_context_ch)
    .map { sample_key, sample_id, cluster_variant, cluster_csv, analysis_objects_csv, marker_quant_dir ->
      tuple(sample_key, sample_id, cluster_variant, cluster_csv, analysis_objects_csv, marker_quant_dir)
    }
  ASSESS_CLUSTER_INTERPRETATION(cluster_assessment_input_ch)
} else if (run_cluster_mask || run_grow_tissue || run_medsam_refine) {
  cluster_csv_ch = image_input_ch.flatMap { sample_id, _image_input ->
    cluster_variant_defs.collect { spec ->
      def sample_key = "${sample_id}::${spec.variant}"
      tuple(sample_key, sample_id, spec.variant, file("${params.outdir_base}/11_clustering/${sample_id}/${sample_id}_${spec.variant}_cluster.csv", checkIfExists: true))
    }
  }
  cluster_kodama_png_ch = image_input_ch.flatMap { sample_id, _image_input ->
    cluster_variant_defs.collect { spec ->
      def sample_key = "${sample_id}::${spec.variant}"
      tuple(sample_key, sample_id, spec.variant, file("${params.outdir_base}/11_clustering/${sample_id}/${sample_id}_${spec.variant}_cluster_kodama_membership.png", checkIfExists: true))
    }
  }
}

def pathsegmentor_annotations_ch = Channel.empty()
def have_cluster_outputs = run_clustering || run_cluster_mask || run_grow_tissue || run_medsam_refine || run_cluster_geojson
if (pathsegmentor_enabled && have_cluster_outputs) {
  ANNOTATE_PATHSEGMENTOR_EVIDENCE(pathsegmentor_bundle_ch, analysis_objects_ch, objects_assigned_ch,
    cluster_csv_ch, cluster_variant_defs, uni2_grid_mode, placeholder_observations_file)
  pathsegmentor_annotations_ch = ANNOTATE_PATHSEGMENTOR_EVIDENCE.out.annotations
}

def labels_for_cluster_ch = Channel.empty()
if ((run_cluster_mask || run_grow_tissue) && !uni2_grid_mode) {
  labels_for_cluster_ch = run_cytoplasm
    ? cyto_mask_ch
    : image_input_ch.map { sample_id, _image_input ->
      tuple(sample_id, file("${params.outdir_base}/08_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif", checkIfExists: true))
    }
}

def cluster_mask_ch = Channel.empty()
def cluster_uncertainty_ch = Channel.empty()
if (run_cluster_mask) {
  BUILD_CLUSTER_MASK_OUTPUT(image_input_ch, crop_roi_ch, cluster_csv_ch, labels_for_cluster_ch, grid_objects_ch, grid_metadata_ch, run_stardist, uni2_grid_mode)
  cluster_mask_ch = requireStageOutput('cluster_mask', BUILD_CLUSTER_MASK_OUTPUT.out.cluster_mask)
  cluster_uncertainty_ch = BUILD_CLUSTER_MASK_OUTPUT.out.uncertainty_mask
} else if (run_grow_tissue || run_medsam_refine || (run_cluster_geojson && uni2_grid_mode)) {
  cluster_mask_ch = image_input_ch.flatMap { sample_id, _image_input ->
    cluster_variant_defs.collect { spec ->
      def sample_key = "${sample_id}::${spec.variant}"
      tuple(sample_key, sample_id, spec.variant, file("${params.outdir_base}/12_cluster_mask/${sample_id}/${sample_id}_${spec.variant}_cluster_mask.tif", checkIfExists: true))
    }
  }
  cluster_uncertainty_ch = cluster_mask_ch.map { key, id, variant, mask ->
    tuple(key, id, variant, file("${params.outdir_base}/12_cluster_mask/${id}/${id}_${variant}_cluster_uncertainty_mask.tif", checkIfExists: true))
  }
}

def image_for_growth_variant_ch = Channel.empty()
def tissue_mask_variant_ch = Channel.empty()
def resolution_for_growth_variant_ch = Channel.empty()
if (run_grow_tissue || run_medsam_refine) {
  def image_for_growth_ch = run_stardist
    ? crop_roi_ch
    : image_input_ch.map { sample_id, _image_input ->
      tuple(sample_id, file("${params.outdir_base}/03_stardist/${sample_id}/prepared_crop/crop_roi.tif", checkIfExists: true))
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

  resolution_for_growth_variant_ch = shift_json_ch
    .flatMap { sample_id, resolution_json ->
      cluster_variant_defs.collect { spec ->
        def sample_key = "${sample_id}::${spec.variant}"
        tuple(sample_key, resolution_json)
      }
    }
}

def spatial_baseline_mask_ch = Channel.empty()
def grown_refine_method = (params.grown_tissue_refine_method ?: 'medsam_border_refine').toString().trim().toLowerCase()
if (run_grow_tissue) {
  def grow_input_ch = cluster_mask_ch
    .join(image_for_growth_variant_ch)
    .join(tissue_mask_variant_ch)
    .join(resolution_for_growth_variant_ch)
    .map { sample_key, sample_id, cluster_variant, cluster_mask_tif, image_tif, tissue_mask_tif, resolution_json ->
      tuple(sample_key, sample_id, cluster_variant, image_tif, cluster_mask_tif, tissue_mask_tif, resolution_json)
    }
  GROW_TO_TISSUE(grow_input_ch)
  spatial_baseline_mask_ch = requireStageOutput('grow_tissue', GROW_TO_TISSUE.out.grown_mask)
} else if (uni2_grid_mode && (run_medsam_refine || run_cluster_geojson)) {
  spatial_baseline_mask_ch = cluster_mask_ch
} else if (run_medsam_refine || (run_cluster_geojson && grown_refine_method in ['none', ''])) {
  spatial_baseline_mask_ch = image_input_ch.flatMap { sample_id, _image_input ->
    cluster_variant_defs.collect { spec ->
      def sample_key = "${sample_id}::${spec.variant}"
      tuple(sample_key, sample_id, spec.variant, file("${params.outdir_base}/13_grown_tissue/${sample_id}/${sample_id}_${spec.variant}_grown_mask.ome.tif", checkIfExists: true))
    }
  }
}

def pathsegmentor_refined_mask_ch = Channel.empty()
if (run_pathsegmentor_refine) {
  RUN_PATHSEGMENTOR_REFINEMENT(pathsegmentor_bundle_ch, spatial_baseline_mask_ch,
    tissue_mask_variant_ch, cluster_variant_defs)
  pathsegmentor_refined_mask_ch = RUN_PATHSEGMENTOR_REFINEMENT.out.refined_mask
}

def profile_final_domain_ch = Channel.empty()
def profile_final_uncertainty_ch = Channel.empty()
if (run_medsam_refine || run_cluster_geojson) {
  POST_GROW_SPATIAL_OUTPUTS(
    image_input_ch,
    spatial_baseline_mask_ch,
    image_for_growth_variant_ch,
    cluster_mask_ch,
    cluster_uncertainty_ch,
    tissue_mask_variant_ch,
    cluster_kodama_png_ch,
    resolution_for_growth_variant_ch,
    run_medsam_refine,
    run_cluster_geojson,
    runtime_plan,
  )
  if (run_medsam_refine) {
    requireStageOutput('medsam_refine', POST_GROW_SPATIAL_OUTPUTS.out.refined_masks)
  }
  final_cluster_geojson_ch = run_cluster_geojson
    ? requireStageOutput('cluster_geojson', POST_GROW_SPATIAL_OUTPUTS.out.cluster_geojson)
    : POST_GROW_SPATIAL_OUTPUTS.out.cluster_geojson
  profile_final_domain_ch = POST_GROW_SPATIAL_OUTPUTS.out.final_masks
  profile_final_uncertainty_ch = POST_GROW_SPATIAL_OUTPUTS.out.final_uncertainty
}

def cell_profile_hierarchy_ch = Channel.empty()
def cell_profile_region_reference_ch = Channel.empty()
if (params.tissue_hierarchy_enable) {
  RUN_TISSUE_REGION_ATLAS(image_input_ch, crop_roi_ch, grid_objects_ch, grid_metadata_ch, tissue_mask_ch,
    shift_json_ch, converted_resolution_report_ch, profile_final_domain_ch, profile_final_uncertainty_ch,
    run_medsam_refine || run_cluster_geojson, runtime_plan)
  cell_profile_hierarchy_ch = RUN_TISSUE_REGION_ATLAS.out.hierarchy
  cell_profile_region_reference_ch = RUN_TISSUE_REGION_ATLAS.out.reference_mapping_bundles
}

if (params.cell_profiles_enable) {
  RUN_CELL_PROFILE_ATLAS(image_input_ch, crop_roi_ch, labels_tif_ch, objects_csv_ch,
    shift_json_ch, converted_resolution_report_ch, tissue_mask_ch, profile_final_domain_ch, cluster_mask_ch,
    profile_final_uncertainty_ch, cluster_uncertainty_ch,
    run_cytoplasm ? EXPAND_LABELS_TO_CYTOPLASM_PRIMARY.out.compartments : Channel.empty(), final_cluster_geojson_ch,
    run_gigatime ? gigatime_quant_dir_ch : (run_marker_quantification ? QUANTIFY_GIGATIME_INTENSITY.out.profile_bundle : Channel.empty()),
    cell_profile_cellvit_ch, profile_primary_tile_ch, profile_primary_local_ch, profile_aux_tile_ch, profile_aux_local_ch, cell_profile_hierarchy_ch, cell_profile_region_reference_ch,
    [grid: uni2_grid_mode, both: uni2_cell_auxiliary_mode, run_uni2: run_uni2,
     uni2: params.cell_profiles_uni2_enable, markers: params.cell_profiles_markers_enable,
     cellvit: params.cellvit_export_embeddings, consensus: cell_consensus_enabled, run_consensus: run_cell_consensus,
     run_gigatime: run_gigatime, run_marker_quantification: run_marker_quantification,
     have_final_domains: run_medsam_refine || run_cluster_geojson, run_cluster_mask: run_cluster_mask,
     run_cytoplasm: run_cytoplasm, run_cluster_geojson: run_cluster_geojson, hierarchy: params.tissue_hierarchy_enable,
     primary_variant: cluster_primary_variant], runtime_plan)
}

}

if (run_neoplastic_section || run_titan || run_pathofmpred) {
  POST_CLUSTER_PATHOFM(
    image_input_ch,
    final_cluster_geojson_ch,
    objects_csv_ch,
    crop_roi_ch,
    shift_json_ch,
    runtime_plan,
  )
  if (run_neoplastic_section) requireStageOutput('neoplastic_section', POST_CLUSTER_PATHOFM.out.selected_sections)
  if (run_titan) requireStageOutput('titan', POST_CLUSTER_PATHOFM.out.titan_embeddings)
  if (run_pathofmpred) requireStageOutput('pathofmpred', POST_CLUSTER_PATHOFM.out.predictions)
}
}

workflow.onComplete {
def outdir = params.outdir_base ?: 'results'
def executionDir = new File("${outdir}/00_execution")
executionDir.mkdirs()
def reporterScript = new File("${baseDir}/bin/write_pipeline_execution_reports.py").canonicalPath
def analysisContractFile = new File(executionDir, 'analysis_contract.json')
def analysisContractReport = [:]
try {
  if (analysisContractFile.exists()) {
    analysisContractReport = (Map) new JsonSlurper().parse(analysisContractFile)
  }
} catch (Throwable t) {
  println "WARN: failed to read analysis contract for execution report: ${t.message}"
}
def scientificRouteReport = (analysisContractReport.scientific_route instanceof Map)
  ? analysisContractReport.scientific_route
  : [:]
def stageWindowReport = (analysisContractReport.stage_window instanceof Map)
  ? analysisContractReport.stage_window
  : [:]
def reportStartPoint = (stageWindowReport.start ?: params.start_point ?: 'convert').toString()
def reportEndPoint = (stageWindowReport.end ?: params.end_point ?: 'cluster_geojson').toString()
def reportCellDetectionMode = (scientificRouteReport.cell_detection_mode ?: params.cell_detection_mode ?: 'not_applicable').toString()
def reportAnalysisIntent = (analysisContractReport.analysis_intent ?: params.analysis_intent ?: 'exploratory').toString()
def reportUni2SamplingMode = (scientificRouteReport.uni2_sampling_mode ?: params.uni2_sampling_mode ?: 'cells').toString()
def cmd = [
'python3',
reporterScript,
'--outdir', new File(outdir).canonicalPath,
'--run-name', workflow.runName,
'--success', workflow.success.toString(),
'--start-point', reportStartPoint,
'--end-point', reportEndPoint,
'--image-input', (params.image_input ?: '').toString(),
'--roi-geojson', (params.roi_geojson ?: '').toString(),
'--cell-detection-mode', reportCellDetectionMode,
'--analysis-intent', reportAnalysisIntent,
'--uni2-sampling-mode', reportUni2SamplingMode,
'--analysis-contract', analysisContractFile.exists() ? analysisContractFile.canonicalPath : ''
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
println "Stage window: ${reportStartPoint} -> ${reportEndPoint}"
if (reportEndPoint in ['cluster_geojson', 'neoplastic_section', 'titan', 'pathofmpred']) {
  println "Cluster GeoJSON output dir: ${params.outdir_base}/15_cluster_geojson"
}
}
}
