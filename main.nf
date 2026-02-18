import groovy.io.FileType
import groovy.json.JsonOutput

nextflow.enable.dsl = 2

include { PREPARE_INPUT_OMETIFF } from './modules/prepare_input_ometiff'
include { PREPARE_ROI_GEOJSON } from './modules/prepare_roi_geojson'
include { RUN_STARDIST_ROI_SEGMENTATION } from './modules/run_stardist_roi_segmentation'
include { BUILD_TISSUE_MASK } from './modules/build_tissue_mask'
include { MAP_CELLS_TO_ROI_POLYGONS } from './modules/map_cells_to_roi_polygons'
include { EXPAND_LABELS_TO_CYTOPLASM as EXPAND_LABELS_TO_CYTOPLASM_PRIMARY } from './modules/expand_labels_to_cytoplasm'
include { EXPAND_LABELS_TO_CYTOPLASM as EXPAND_LABELS_TO_CYTOPLASM_FULL } from './modules/expand_labels_to_cytoplasm'
include { EXTRACT_UNI2_EMBEDDINGS } from './modules/extract_uni2_embeddings'
include { RUN_KODAMA_ANALYSIS } from './modules/run_kodama_analysis'
include { RUN_RCODE_CLUSTERING } from './modules/run_rcode_clustering'
include { LABELS_TO_CLUSTER_MASK } from './modules/labels_to_cluster_mask'
include { GROW_TO_TISSUE } from './modules/grow_to_tissue'
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

    // Keep user-facing defaults lightweight (1 CPU / 8 GB), then auto-cap to host capacity.
    // This avoids immediate scheduler failures on low-core environments.
    def configured_max_cpus = parse_positive_int(paramOr('max_cpus', 1), 1)
    def host_available_cpus = Math.max(1, Runtime.runtime.availableProcessors() as int)
    def effective_max_cpus = Math.max(1, Math.min(configured_max_cpus, host_available_cpus))
    if (effective_max_cpus != configured_max_cpus) {
        println "WARN: Reducing max_cpus from ${configured_max_cpus} to ${effective_max_cpus} (host available CPUs: ${host_available_cpus})."
    }
    params.max_cpus = effective_max_cpus

    def detect_host_memory_gb = {
        def candidates = [
            command_output('awk \'/MemTotal/ {printf "%d", int($2/1024/1024)}\' /proc/meminfo'),
            command_output('free -g | awk \'/^Mem:/ {print $2}\''),
            command_output('sysctl -n hw.memsize 2>/dev/null | awk \'{printf "%d", int($1/1024/1024/1024)}\'')
        ].findAll { it && it ==~ /\\d+/ }
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

    def nvidia_visible = (System.getenv('NVIDIA_VISIBLE_DEVICES') ?: '').toString().trim().toLowerCase()
    def cuda_visible = (System.getenv('CUDA_VISIBLE_DEVICES') ?: '').toString().trim().toLowerCase()
    def is_positive = { String value -> value && !(value in ['none', 'void', 'no', 'false', '-1']) }
    def detected_nvidia = is_positive(nvidia_visible) || is_positive(cuda_visible)
    if (!detected_nvidia) {
        detected_nvidia = command_succeeds('command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1')
    }

    def resolved_compute_device = requested_compute_device == 'auto'
        ? (((detected_arch in ['amd64', 'arm64']) && detected_nvidia) ? 'gpu' : 'cpu')
        : requested_compute_device
    if (resolved_compute_device == 'gpu' && !(detected_arch in ['amd64', 'arm64'])) {
        error "compute_device='gpu' is supported only on amd64/x86_64 or arm64/aarch64. Detected architecture '${detected_arch}'."
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
    def container_gpu_tag_amd64 = (paramOr('container_gpu_tag_amd64', container_gpu_tag) ?: container_gpu_tag).toString().trim()
    if (!container_gpu_tag_amd64) {
        container_gpu_tag_amd64 = '0.2.0-gpu'
    }
    def container_gpu_tag_arm64 = (paramOr('container_gpu_tag_arm64', '0.2.0-gpu-arm64') ?: '0.2.0-gpu-arm64').toString().trim()
    if (!container_gpu_tag_arm64) {
        container_gpu_tag_arm64 = '0.2.0-gpu-arm64'
    }

    def selected_cpu_tag = detected_arch == 'amd64' ? container_cpu_tag_amd64 : container_cpu_tag_arm64
    def selected_gpu_tag = detected_arch == 'amd64' ? container_gpu_tag_amd64 : container_gpu_tag_arm64
    def selected_container_tag = resolved_compute_device == 'gpu' ? selected_gpu_tag : selected_cpu_tag
    def auto_docker_image = "${container_repo}:${selected_container_tag}"
    def auto_docker_singularity_image = "docker://${auto_docker_image}"

    def singularity_image_source = (paramOr('singularity_image_source', 'auto') ?: 'auto').toString().trim().toLowerCase()
    if (!(singularity_image_source in ['auto', 'release', 'docker'])) {
        singularity_image_source = 'auto'
    }
    def singularity_release_repo = (paramOr('singularity_release_repo', 'tkcaccia/CellPhenotyper') ?: 'tkcaccia/CellPhenotyper').toString().trim()
    def singularity_release_tag = (paramOr('singularity_release_tag', 'v0.2.0') ?: 'v0.2.0').toString().trim()
    def singularity_cpu_asset_amd64 = (paramOr('singularity_cpu_asset_amd64', 'cellphenotyper-0.2.0-amd64.sif') ?: 'cellphenotyper-0.2.0-amd64.sif').toString().trim()
    def singularity_cpu_asset_arm64 = (paramOr('singularity_cpu_asset_arm64', 'cellphenotyper-0.2.0-arm64.sif') ?: 'cellphenotyper-0.2.0-arm64.sif').toString().trim()
    def singularity_gpu_asset_amd64 = (paramOr('singularity_gpu_asset_amd64', 'cellphenotyper-0.2.0-gpu-amd64.sif') ?: 'cellphenotyper-0.2.0-gpu-amd64.sif').toString().trim()
    def singularity_gpu_asset_arm64 = (paramOr('singularity_gpu_asset_arm64', 'cellphenotyper-0.2.0-gpu-arm64.sif') ?: 'cellphenotyper-0.2.0-gpu-arm64.sif').toString().trim()
    def singularity_local_dir = (paramOr('singularity_local_dir', '') ?: '').toString().trim()

    def selected_release_asset = resolved_compute_device == 'gpu'
        ? (detected_arch == 'amd64' ? singularity_gpu_asset_amd64 : singularity_gpu_asset_arm64)
        : (detected_arch == 'amd64' ? singularity_cpu_asset_amd64 : singularity_cpu_asset_arm64)
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
    if (runtime_profiles.contains('singularity') && singularity_image_source in ['auto', 'release'] && auto_local_singularity_image) {
        auto_singularity_image = auto_local_singularity_image
        auto_singularity_origin = 'local'
    } else if (runtime_profiles.contains('singularity') && singularity_image_source in ['auto', 'release'] && auto_release_singularity_image && release_asset_reachable) {
        auto_singularity_image = auto_release_singularity_image
        auto_singularity_origin = 'release'
    } else if (runtime_profiles.contains('singularity') && singularity_image_source in ['auto', 'release']) {
        auto_singularity_image = auto_docker_singularity_image
        auto_singularity_origin = 'docker'
        if (singularity_image_source == 'release') {
            println "WARN: Release-hosted Singularity image is not reachable (${auto_release_singularity_image}); falling back to ${auto_docker_singularity_image}"
        }
    }

    def raw_singularity_image = (paramOr('singularity_image', '') ?: '').toString().trim()
    def resolved_singularity_image = raw_singularity_image
    if (runtime_image_mode == 'auto' || !raw_singularity_image) {
        resolved_singularity_image = auto_singularity_image
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
        resolved_docker_image = auto_docker_image
    }

    // Propagate resolved runtime context so module/config logic uses the same values.
    params.compute_device = resolved_compute_device
    params.host_arch = detected_arch
    params.container_repo = container_repo
    params.container_cpu_tag_amd64 = container_cpu_tag_amd64
    params.container_cpu_tag_arm64 = container_cpu_tag_arm64
    params.container_gpu_tag_amd64 = container_gpu_tag_amd64
    params.container_gpu_tag_arm64 = container_gpu_tag_arm64
    params.singularity_gpu_asset_arm64 = singularity_gpu_asset_arm64

    if (runtime_profiles.contains('singularity')) {
        def singularity_image = runtime_image_mode == 'manual'
            ? (paramOr('singularity_image', '') ?: '').toString().trim()
            : resolved_singularity_image
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
        stardist        : 'stardist',
        startdist       : 'stardist',
        tissue_mask     : 'tissue_mask',
        mask_tissue     : 'tissue_mask',
        tissue_geojson  : 'tissue_mask',
        geojson         : 'cluster_geojson',
        cell_assignment : 'cell_assignment',
        assign          : 'cell_assignment',
        cytoplasm       : 'cytoplasm',
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
    def stage_order = ['convert', 'stardist', 'tissue_mask', 'cell_assignment', 'cytoplasm', 'uni2', 'kodama', 'clustering', 'cluster_mask', 'grow_tissue', 'cluster_geojson']
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

    def should_run_stage = { stage_name ->
        def idx = stage_index[stage_name]
        idx >= stage_index[start_point] && idx <= stage_index[end_point]
    }

    def run_convert = should_run_stage('convert')
    def run_stardist = should_run_stage('stardist')
    def run_tissue_mask = should_run_stage('tissue_mask')
    def run_cell_assignment = should_run_stage('cell_assignment')
    def run_cytoplasm = should_run_stage('cytoplasm')
    def run_uni2 = should_run_stage('uni2')
    def run_kodama = should_run_stage('kodama')
    def run_clustering = should_run_stage('clustering')
    def run_cluster_mask = should_run_stage('cluster_mask')
    def run_grow_tissue = should_run_stage('grow_tissue')
    def run_cluster_geojson = should_run_stage('cluster_geojson')
    def include_uni2_nuclei = params.uni2_include_nuclei == null ? true : (params.uni2_include_nuclei as boolean)
    def include_uni2_cyto = params.uni2_include_cyto == null ? true : (params.uni2_include_cyto as boolean)
    def include_uni2_inner_square = params.uni2_include_inner_square == null ? true : (params.uni2_include_inner_square as boolean)
    if (run_kodama && run_uni2 && (!include_uni2_nuclei || !include_uni2_cyto || !include_uni2_inner_square)) {
        error "KODAMA stage requires all embedding families from UNI-2. Set uni2_include_nuclei=true, uni2_include_cyto=true and uni2_include_inner_square=true."
    }

    println "Runtime auto-selection: runtime_image_mode=${runtime_image_mode}, requested_arch=${requested_arch_raw ?: 'auto'}, detected_arch=${detected_arch}, arch_candidates=${detected_arch_candidates.join(',')}, requested_compute_device=${requested_compute_device}, resolved_compute_device=${resolved_compute_device}, singularity_image_source=${singularity_image_source}, singularity_origin=${auto_singularity_origin}, singularity_asset=${selected_release_asset}, singularity_image=${resolved_singularity_image}, docker_image=${resolved_docker_image}"
    println "Pipeline stage window: ${start_point} -> ${end_point}"

    def supported_image_suffixes = [
        [suffix: '.ome.tif',  priority: 80],
        [suffix: '.ome.tiff', priority: 75],
        [suffix: '.btf',      priority: 70],
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

    def deriveSampleId = { File imageFile ->
        def name = imageFile.name
        def suffix = detectImageSuffix(name)
        if (suffix) {
            return name.substring(0, name.length() - suffix.length())
        }
        def dot = name.lastIndexOf('.')
        dot > 0 ? name.substring(0, dot) : name
    }

    def folder_input = (params.folder_input ?: '').toString().trim()
    def image_input_param = (params.image_input ?: '').toString().trim()
    def roi_geojson_param = (params.roi_geojson ?: '').toString().trim()
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
            def roi_candidate = new File(input_dir, "${sample_id}.geojson")
            def roi_hint = roi_candidate.exists() ? roi_candidate.absolutePath : ''
            sample_rows << tuple(sample_id, file(image_file.absolutePath, checkIfExists: true), roi_hint)
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
        def sample_id = deriveSampleId(image_file)
        def roi_hint = ''
        if (roi_geojson_param) {
            roi_hint = file(roi_geojson_param, checkIfExists: true).absolutePath
        } else {
            def roi_candidate = new File(image_file.parentFile, "${sample_id}.geojson")
            roi_hint = roi_candidate.exists() ? roi_candidate.absolutePath : ''
        }
        sample_rows << tuple(sample_id, image_file, roi_hint)
    }

    if (!sample_rows) {
        error "No input samples were resolved."
    }
    println "Resolved input samples (${sample_rows.size()}): ${sample_rows.collect { it[0] }.join(', ')}"

    Channel
        .fromList(sample_rows)
        .set { input_spec_ch }

    def image_input_ch = input_spec_ch
        .map { sample_id, image_file, roi_hint ->
            tuple(sample_id, image_file)
        }

    def roi_hint_ch = input_spec_ch
        .map { sample_id, image_file, roi_hint ->
            tuple(sample_id, roi_hint)
        }

    def ome_tif_ch
    if (run_convert) {
        PREPARE_INPUT_OMETIFF(image_input_ch)
        ome_tif_ch = PREPARE_INPUT_OMETIFF.out.ome_tif
    } else {
        ome_tif_ch = image_input_ch.map { sample_id, image_file ->
            def is_ome = image_file.name.toLowerCase().endsWith('.ome.tif')
            def ome_tif = is_ome
                ? image_file
                : file("${params.outdir_base}/01_input/${sample_id}/${sample_id}.ome.tif", checkIfExists: true)
            tuple(sample_id, ome_tif)
        }
    }

    def roi_geojson_ch = Channel.empty()
    if (run_stardist) {
        def roi_prepare_input_ch = ome_tif_ch
            .join(roi_hint_ch)
            .map { sample_id, ome_tif, roi_hint ->
                tuple(sample_id, ome_tif, roi_hint ?: '')
            }
        PREPARE_ROI_GEOJSON(roi_prepare_input_ch)
        roi_geojson_ch = PREPARE_ROI_GEOJSON.out.roi_geojson
    } else if (run_cell_assignment) {
        roi_geojson_ch = image_input_ch.map { sample_id, _ ->
            tuple(sample_id, file("${params.outdir_base}/04_roi/${sample_id}/${sample_id}.roi.geojson", checkIfExists: true))
        }
    }

    def need_stardist_outputs = run_stardist || (!tissue_mask_from_input && run_tissue_mask) || run_cell_assignment || run_cytoplasm || run_uni2

    def crop_roi_ch = Channel.empty()
    def labels_tif_ch = Channel.empty()
    def labels_full_tif_ch = Channel.empty()
    def objects_csv_ch = Channel.empty()
    def shift_json_ch = Channel.empty()

    if (need_stardist_outputs) {
        if (run_stardist) {
            def stardist_input_ch = ome_tif_ch
                .join(roi_geojson_ch)
                .map { sample_id, ome_tif, roi_geojson ->
                    tuple(sample_id, ome_tif, roi_geojson)
                }
            RUN_STARDIST_ROI_SEGMENTATION(stardist_input_ch)
            crop_roi_ch = RUN_STARDIST_ROI_SEGMENTATION.out.crop_roi
            labels_tif_ch = RUN_STARDIST_ROI_SEGMENTATION.out.labels_tif
            labels_full_tif_ch = RUN_STARDIST_ROI_SEGMENTATION.out.labels_full_tif
            objects_csv_ch = RUN_STARDIST_ROI_SEGMENTATION.out.objects_csv
            shift_json_ch = RUN_STARDIST_ROI_SEGMENTATION.out.shift_json
        } else {
            def require_labels_full = run_uni2 || (run_cytoplasm && (params.expand_full_labels as boolean))
            crop_roi_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/02_stardist/${sample_id}/stardist_out/crop_roi.tif", checkIfExists: true))
            }
            labels_tif_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/02_stardist/${sample_id}/stardist_out/labels.tif", checkIfExists: true))
            }
            objects_csv_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/02_stardist/${sample_id}/stardist_out/objects.csv", checkIfExists: true))
            }
            shift_json_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/02_stardist/${sample_id}/stardist_out/shift.json", checkIfExists: true))
            }
            labels_full_tif_ch = require_labels_full
                ? image_input_ch.map { sample_id, _ ->
                    tuple(sample_id, file("${params.outdir_base}/02_stardist/${sample_id}/stardist_out/labels_full.tif", checkIfExists: true))
                }
                : Channel.empty()
        }
    }

    def tissue_mask_ch = Channel.empty()
    if (run_tissue_mask) {
        def tissue_mask_input_ch = tissue_mask_from_input ? ome_tif_ch : crop_roi_ch
        BUILD_TISSUE_MASK(tissue_mask_input_ch)
        tissue_mask_ch = BUILD_TISSUE_MASK.out.tissue_mask
    }

    if (run_cell_assignment || run_cytoplasm || run_uni2 || run_kodama || run_clustering || run_cluster_mask || run_grow_tissue || run_cluster_geojson) {
        def objects_assigned_ch = Channel.empty()
        if (run_cell_assignment) {
            def assign_input_ch = objects_csv_ch
                .join(shift_json_ch)
                .join(roi_geojson_ch)
                .map { sample_id, objects_csv, shift_json, roi_geojson ->
                    tuple(sample_id, objects_csv, roi_geojson, shift_json)
                }
            MAP_CELLS_TO_ROI_POLYGONS(assign_input_ch)
            objects_assigned_ch = MAP_CELLS_TO_ROI_POLYGONS.out.objects_assigned
        } else {
            objects_assigned_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/05_cell_assignments/${sample_id}/${sample_id}_objects_assigned.csv", checkIfExists: true))
            }
        }

        def cyto_mask_ch = Channel.empty()
        def cyto_mask_full_ch = Channel.empty()
        if (run_cytoplasm) {
            def expand_primary_ch = labels_tif_ch
                .join(crop_roi_ch)
                .map { sample_id, labels_tif, preview_background_tif ->
                    tuple(sample_id, labels_tif, 'labels_cyto', preview_background_tif.toString())
                }
            EXPAND_LABELS_TO_CYTOPLASM_PRIMARY(expand_primary_ch)

            if (params.expand_full_labels as boolean) {
                def expand_full_ch = labels_full_tif_ch.map { sample_id, labels_full_tif ->
                    tuple(sample_id, labels_full_tif, 'labels_full_cyto', '')
                }
                EXPAND_LABELS_TO_CYTOPLASM_FULL(expand_full_ch)
                cyto_mask_full_ch = EXPAND_LABELS_TO_CYTOPLASM_FULL.out.expanded_labels
                    .filter { sample_id, expanded_mask, label_kind -> label_kind == 'labels_full_cyto' }
                    .map { sample_id, expanded_mask, label_kind -> tuple(sample_id, expanded_mask) }
            } else if (run_uni2 && (include_uni2_cyto || include_uni2_inner_square)) {
                error "UNI-2 cyto/inner-square embeddings require expand_full_labels=true to generate *_labels_full_cyto.tif."
            }

            cyto_mask_ch = EXPAND_LABELS_TO_CYTOPLASM_PRIMARY.out.expanded_labels
                .filter { sample_id, expanded_mask, label_kind -> label_kind == 'labels_cyto' }
                .map { sample_id, expanded_mask, label_kind -> tuple(sample_id, expanded_mask) }
        } else if (run_uni2) {
            cyto_mask_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/06_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif", checkIfExists: true))
            }
            cyto_mask_full_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/06_cytoplasm/${sample_id}/${sample_id}_labels_full_cyto.tif", checkIfExists: true))
            }
        }

        def tile_embeddings_ch = Channel.empty()
        def nuclei_embeddings_ch = Channel.empty()
        def cyto_embeddings_ch = Channel.empty()
        def inner_square_embeddings_ch = Channel.empty()
        if (run_uni2) {
            def uni2_tile_input_ch = ome_tif_ch
                .join(labels_full_tif_ch)
                .map { sample_id, ome_tif, labels_full_tif ->
                    tuple(sample_id, ome_tif, labels_full_tif, 'tile', false, 'none', 255)
                }

            def uni2_input_ch = uni2_tile_input_ch
            if (include_uni2_cyto) {
                def uni2_cyto_input_ch = ome_tif_ch
                    .join(cyto_mask_full_ch)
                    .map { sample_id, ome_tif, cyto_mask_tif ->
                        tuple(sample_id, ome_tif, cyto_mask_tif, 'cyto', true, 'label', 255)
                    }
                uni2_input_ch = uni2_input_ch.mix(uni2_cyto_input_ch)
            }
            if (include_uni2_inner_square) {
                def uni2_inner_square_input_ch = ome_tif_ch
                    .join(cyto_mask_full_ch)
                    .map { sample_id, ome_tif, cyto_mask_tif ->
                        tuple(sample_id, ome_tif, cyto_mask_tif, 'inner_square', true, 'inner_square', 255)
                    }
                uni2_input_ch = uni2_input_ch.mix(uni2_inner_square_input_ch)
            }
            if (include_uni2_nuclei) {
                // Keep nuclei embeddings in the same full-image coordinate system as tile/cyto/inner_square
                def uni2_nuclei_input_ch = ome_tif_ch
                    .join(labels_full_tif_ch)
                    .map { sample_id, ome_tif, labels_full_tif ->
                        tuple(sample_id, ome_tif, labels_full_tif, 'nuclei', true, 'label', 255)
                    }
                uni2_input_ch = uni2_input_ch.mix(uni2_nuclei_input_ch)
            }

            EXTRACT_UNI2_EMBEDDINGS(uni2_input_ch)
            tile_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
                .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'tile' }
                .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
            nuclei_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
                .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'nuclei' }
                .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
            cyto_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
                .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'cyto' }
                .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
            inner_square_embeddings_ch = EXTRACT_UNI2_EMBEDDINGS.out.embeddings_dir
                .filter { sample_id, embedding_mode, embeddings_dir -> embedding_mode == 'inner_square' }
                .map { sample_id, embedding_mode, embeddings_dir -> tuple(sample_id, embeddings_dir) }
        } else if (run_kodama) {
            tile_embeddings_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/07_embeddings/${sample_id}/embeddings_${sample_id}_tile", checkIfExists: true))
            }
            nuclei_embeddings_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/07_embeddings/${sample_id}/embeddings_${sample_id}_nuclei", checkIfExists: true))
            }
            cyto_embeddings_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/07_embeddings/${sample_id}/embeddings_${sample_id}_cyto", checkIfExists: true))
            }
            inner_square_embeddings_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/07_embeddings/${sample_id}/embeddings_${sample_id}_inner_square", checkIfExists: true))
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
                tuple(sample_id, file("${params.outdir_base}/08_kodama/${sample_id}/kodama_output", checkIfExists: true))
            }
        }

        def cluster_csv_ch = Channel.empty()
        if (run_clustering) {
            def clustering_input_ch = kodama_dir_ch
                .join(objects_assigned_ch)
                .map { sample_id, kodama_dir, objects_assigned_csv ->
                    tuple(sample_id, kodama_dir, objects_assigned_csv)
                }
            RUN_RCODE_CLUSTERING(clustering_input_ch)
            cluster_csv_ch = RUN_RCODE_CLUSTERING.out.cluster_csv
        } else if (run_cluster_mask || run_grow_tissue || run_cluster_geojson) {
            cluster_csv_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/09_clustering/${sample_id}/${sample_id}_cluster.csv", checkIfExists: true))
            }
        }

        def labels_for_cluster_ch = Channel.empty()
        if (run_cluster_mask || run_grow_tissue || run_cluster_geojson) {
            labels_for_cluster_ch = run_cytoplasm
                ? cyto_mask_ch
                : image_input_ch.map { sample_id, _ ->
                    tuple(sample_id, file("${params.outdir_base}/06_cytoplasm/${sample_id}/${sample_id}_labels_cyto.tif", checkIfExists: true))
                }
        }

        def cluster_mask_ch = Channel.empty()
        if (run_cluster_mask) {
            def preview_image_for_cluster_mask_ch = run_stardist
                ? crop_roi_ch
                : image_input_ch.map { sample_id, _ ->
                    tuple(sample_id, file("${params.outdir_base}/02_stardist/${sample_id}/stardist_out/crop_roi.tif", checkIfExists: true))
                }

            def cluster_mask_input_ch = labels_for_cluster_ch
                .join(cluster_csv_ch)
                .join(preview_image_for_cluster_mask_ch)
                .map { sample_id, labels_tif, cluster_csv, preview_tif ->
                    tuple(sample_id, labels_tif, cluster_csv, preview_tif)
                }
            LABELS_TO_CLUSTER_MASK(cluster_mask_input_ch)
            cluster_mask_ch = LABELS_TO_CLUSTER_MASK.out.cluster_mask
        } else if (run_grow_tissue || run_cluster_geojson) {
            cluster_mask_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/10_cluster_mask/${sample_id}/${sample_id}_cluster_mask.tif", checkIfExists: true))
            }
        }

        def grown_mask_ch = Channel.empty()
        if (run_grow_tissue) {
            def image_for_growth_ch = run_stardist
                ? crop_roi_ch
                : image_input_ch.map { sample_id, _ ->
                    tuple(sample_id, file("${params.outdir_base}/02_stardist/${sample_id}/stardist_out/crop_roi.tif", checkIfExists: true))
                }
            def tissue_mask_for_growth_ch = run_tissue_mask
                ? tissue_mask_ch
                : image_input_ch.map { sample_id, _ ->
                    tuple(sample_id, file("${params.outdir_base}/03_tissue_mask/${sample_id}/${sample_id}_tissue_mask.tif", checkIfExists: true))
                }

            def grow_input_ch = image_for_growth_ch
                .join(cluster_mask_ch)
                .join(tissue_mask_for_growth_ch)
                .map { sample_id, image_tif, cluster_mask_tif, tissue_mask_tif ->
                    tuple(sample_id, image_tif, cluster_mask_tif, tissue_mask_tif)
                }
            GROW_TO_TISSUE(grow_input_ch)
            grown_mask_ch = GROW_TO_TISSUE.out.grown_mask
        } else if (run_cluster_geojson) {
            grown_mask_ch = image_input_ch.map { sample_id, _ ->
                tuple(sample_id, file("${params.outdir_base}/11_grown_tissue/${sample_id}/${sample_id}_grown_mask.ome.tif", checkIfExists: true))
            }
        }

        if (run_cluster_geojson) {
            MASK_TO_GEOJSON(grown_mask_ch)
        }
    }
}

workflow.onComplete {
    def outdir = params.outdir_base ?: 'results'
    def executionDir = new File("${outdir}/00_execution")
    executionDir.mkdirs()

    def stageDefs = [
        [folder: '01_input',          title: 'Input Conversion',        expected: ['.ome.tif']],
        [folder: '02_stardist',       title: 'StarDist Segmentation',   expected: ['labels.tif', 'objects.csv']],
        [folder: '03_tissue_mask',    title: 'Tissue Mask',             expected: ['_tissue_mask.tif']],
        [folder: '04_roi',            title: 'ROI GeoJSON',             expected: ['.roi.geojson']],
        [folder: '05_cell_assignments', title: 'Cell Assignment',       expected: ['_objects_assigned.csv']],
        [folder: '06_cytoplasm',      title: 'Cytoplasm Expansion',     expected: ['_labels_cyto.tif']],
        [folder: '07_embeddings',     title: 'UNI-2 Embeddings',        expected: ['embeddings_']],
        [folder: '08_kodama',         title: 'KODAMA',                  expected: ['kodama_output']],
        [folder: '08_kodama_logs',    title: 'KODAMA Logs',             expected: ['.Rout']],
        [folder: '09_clustering',     title: 'Clustering',              expected: ['_cluster.csv']],
        [folder: '09_clustering_logs', title: 'Clustering Logs',        expected: ['.Rout']],
        [folder: '10_cluster_mask',   title: 'Cluster Mask',            expected: ['_cluster_mask.tif']],
        [folder: '11_grown_tissue',   title: 'Grown Tissue',            expected: ['_grown_mask.ome.tif']],
        [folder: '12_cluster_geojson', title: 'Cluster GeoJSON',        expected: ['.geojson']],
        [folder: '00_execution',      title: 'Execution Metadata',      expected: ['trace.tsv', 'timeline.html', 'dag.html']]
    ]

    def bytesToHuman = { long bytes ->
        if (bytes < 1024L) return "${bytes} B"
        def units = ['KB', 'MB', 'GB', 'TB']
        double value = bytes
        int idx = -1
        while (value >= 1024.0d && idx < units.size() - 1) {
            value /= 1024.0d
            idx++
        }
        return String.format('%.2f %s', value, units[idx])
    }

    def parseDurationSeconds = { String raw ->
        if (!raw) return 0.0d
        def value = raw.trim()
        def m = (value =~ /(?i)^([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h|d)$/)
        if (!m.matches()) return 0.0d
        def n = m.group(1) as double
        switch (m.group(2).toLowerCase()) {
            case 'ms': return n / 1000.0d
            case 's' : return n
            case 'm' : return n * 60.0d
            case 'h' : return n * 3600.0d
            case 'd' : return n * 86400.0d
            default  : return 0.0d
        }
    }

    def parseMemoryBytes = { String raw ->
        if (!raw) return 0L
        def value = raw.trim()
        def m = (value =~ /(?i)^([0-9]+(?:\.[0-9]+)?)\s*([kmgtp]?i?b)?$/)
        if (!m.matches()) return 0L
        def n = m.group(1) as double
        def unit = (m.group(2) ?: 'b').toLowerCase()
        def factor = 1.0d
        switch (unit) {
            case 'kb':
            case 'kib': factor = 1024.0d; break
            case 'mb':
            case 'mib': factor = 1024.0d * 1024.0d; break
            case 'gb':
            case 'gib': factor = 1024.0d * 1024.0d * 1024.0d; break
            case 'tb':
            case 'tib': factor = 1024.0d * 1024.0d * 1024.0d * 1024.0d; break
            default: factor = 1.0d
        }
        return (long) (n * factor)
    }

    def stageSummaries = []
    stageDefs.each { def stage ->
        def dir = new File(outdir, stage.folder)
        def fileCount = 0
        def totalBytes = 0L
        def relFiles = []
        if (dir.exists()) {
            dir.eachFileRecurse(FileType.FILES) { f ->
                fileCount++
                totalBytes += f.length()
                relFiles << f.path
            }
        }
        relFiles = relFiles.sort()
        def keyFiles = relFiles.findAll { p ->
            stage.expected.any { needle -> p.contains(needle) }
        }.take(5)

        stageSummaries << [
            folder      : stage.folder,
            title       : stage.title,
            present     : dir.exists(),
            file_count  : fileCount,
            size_bytes  : totalBytes,
            size_human  : bytesToHuman(totalBytes),
            key_files   : keyFiles
        ]
    }

    def traceSummaryRows = []
    def traceFile = new File(executionDir, 'trace.tsv')
    if (traceFile.exists()) {
        def lines = traceFile.readLines().findAll { it?.trim() }
        if (lines.size() > 1) {
            def header = lines[0].split('\t', -1)
            def idxName = header.findIndexOf { it == 'name' }
            def idxRealtime = header.findIndexOf { it == 'realtime' }
            def idxPeakRss = header.findIndexOf { it == 'peak_rss' }
            def idxStatus = header.findIndexOf { it == 'status' }

            def grouped = [:].withDefault { [tasks: 0, realtime_s: 0.0d, peak_bytes: 0L, failed: 0] }
            lines.drop(1).each { row ->
                def cols = row.split('\t', -1)
                if (idxName < 0 || idxName >= cols.size()) return
                def processName = cols[idxName]
                grouped[processName].tasks += 1
                if (idxRealtime >= 0 && idxRealtime < cols.size()) {
                    grouped[processName].realtime_s += parseDurationSeconds(cols[idxRealtime])
                }
                if (idxPeakRss >= 0 && idxPeakRss < cols.size()) {
                    grouped[processName].peak_bytes = Math.max(grouped[processName].peak_bytes, parseMemoryBytes(cols[idxPeakRss]))
                }
                if (idxStatus >= 0 && idxStatus < cols.size() && cols[idxStatus] != 'COMPLETED') {
                    grouped[processName].failed += 1
                }
            }

            traceSummaryRows = grouped.collect { k, v ->
                [
                    process     : k,
                    tasks       : v.tasks,
                    realtime_s  : v.realtime_s,
                    peak_bytes  : v.peak_bytes,
                    failed      : v.failed
                ]
            }.sort { a, b -> b.realtime_s <=> a.realtime_s }
        }
    }

    def manifest = new File(executionDir, 'outputs_manifest.txt')
    def lines = []
    lines << "CellPhenotyper Output Manifest"
    lines << "Run name: ${workflow.runName}"
    lines << "Success: ${workflow.success}"
    lines << "Stage window: ${params._resolved_start_point ?: 'convert'} -> ${params._resolved_end_point ?: 'cluster_geojson'}"
    lines << ""
    lines << "Stage folders under: ${outdir}"
    stageSummaries.each { s ->
        def status = s.present ? 'PRESENT' : 'MISSING'
        lines << "${status}\t${new File(outdir, s.folder).path}\tfiles=${s.file_count}\tsize=${s.size_human}"
    }
    manifest.text = lines.join('\n') + '\n'

    def finalReportMd = new File(executionDir, 'final_report.md')
    def reportLines = []
    reportLines << "# CellPhenotyper Final Report"
    reportLines << ""
    reportLines << "- Run name: `${workflow.runName}`"
    reportLines << "- Success: `${workflow.success}`"
    reportLines << "- Stage window: `${params._resolved_start_point ?: 'convert'} -> ${params._resolved_end_point ?: 'cluster_geojson'}`"
    reportLines << "- Output root: `${outdir}`"
    reportLines << ""
    reportLines << "## Stage Outputs"
    reportLines << ""
    reportLines << "| Folder | Stage | Status | Files | Size |"
    reportLines << "|---|---|---:|---:|---:|"
    stageSummaries.each { s ->
        reportLines << "| `${s.folder}` | ${s.title} | ${s.present ? 'PRESENT' : 'MISSING'} | ${s.file_count} | ${s.size_human} |"
    }
    reportLines << ""
    reportLines << "## Key Result Files"
    reportLines << ""
    stageSummaries.each { s ->
        if (s.key_files && !s.key_files.isEmpty()) {
            reportLines << "- `${s.folder}`"
            s.key_files.each { path ->
                reportLines << "  - `${path}`"
            }
        }
    }
    reportLines << ""
    reportLines << "## Runtime And Memory (From trace.tsv)"
    reportLines << ""
    if (traceSummaryRows && !traceSummaryRows.isEmpty()) {
        reportLines << "| Process | Tasks | Total Realtime (s) | Peak RSS | Failed Tasks |"
        reportLines << "|---|---:|---:|---:|---:|"
        traceSummaryRows.each { t ->
            reportLines << "| `${t.process}` | ${t.tasks} | ${String.format('%.1f', t.realtime_s)} | ${bytesToHuman(t.peak_bytes)} | ${t.failed} |"
        }
    } else {
        reportLines << "Trace file not available."
    }
    reportLines << ""
    finalReportMd.text = reportLines.join('\n') + '\n'

    def finalReportJson = new File(executionDir, 'final_report.json')
    def jsonPayload = [
        run_name      : workflow.runName,
        success       : workflow.success,
        stage_window  : [
            start: (params._resolved_start_point ?: 'convert'),
            end  : (params._resolved_end_point ?: 'cluster_geojson')
        ],
        output_root   : outdir,
        stages        : stageSummaries.collect { s ->
            [
                folder      : s.folder,
                stage       : s.title,
                present     : s.present,
                file_count  : s.file_count,
                size_bytes  : s.size_bytes,
                key_files   : s.key_files
            ]
        },
        process_trace_summary: traceSummaryRows
    ]
    finalReportJson.text = JsonOutput.prettyPrint(JsonOutput.toJson(jsonPayload)) + '\n'

    if (workflow.success) {
        println "PIPELINE COMPLETED SUCCESSFULLY"
        println "Stage window: ${params._resolved_start_point ?: 'convert'} -> ${params._resolved_end_point ?: 'cluster_geojson'}"
        println "Execution reports dir: ${outdir}/00_execution"
        println "Output manifest: ${outdir}/00_execution/outputs_manifest.txt"
        println "Final report: ${outdir}/00_execution/final_report.md"
        println "Final report JSON: ${outdir}/00_execution/final_report.json"
        if ((params._resolved_end_point ?: '') in ['clustering', 'cluster_mask', 'grow_tissue', 'cluster_geojson']) {
            println "Cluster GeoJSON output dir: ${params.outdir_base}/12_cluster_geojson"
        }
    } else {
        println "Execution reports dir: ${outdir}/00_execution"
        println "Output manifest: ${outdir}/00_execution/outputs_manifest.txt"
        println "Final report: ${outdir}/00_execution/final_report.md"
        println "Final report JSON: ${outdir}/00_execution/final_report.json"
    }
}
