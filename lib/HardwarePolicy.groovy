class HardwarePolicy {
    private static int positiveInt(def value, int fallback) {
        try {
            int parsed = value == null ? fallback : value.toString().trim().toInteger()
            parsed > 0 ? parsed : fallback
        } catch (Throwable ignored) {
            fallback
        }
    }

    private static double positiveDouble(def value, double fallback) {
        try {
            double parsed = value == null ? fallback : value.toString().trim().toDouble()
            parsed > 0.0d ? parsed : fallback
        } catch (Throwable ignored) {
            fallback
        }
    }

    private static int atlasPositiveInt(def value, String key) {
        try {
            int parsed = value.toString().trim().toInteger()
            if (parsed > 0) return parsed
        } catch (Exception ignored) {}
        throw new IllegalArgumentException("${key} must be a positive integer; received '${value}'")
    }

    /** Nextflow GB/MB are binary units. Preserve sub-GB caps without rounding up. */
    private static BigDecimal atlasMemoryGb(def value, String key, boolean unitBearing) {
        if (!unitBearing) return new BigDecimal(atlasPositiveInt(value, key))
        def matched = value.toString().trim() =~ /^([0-9]+(?:\.[0-9]+)?)\s+(GB|MB)$/
        if (!matched.matches())
            throw new IllegalArgumentException("${key} requires a positive quantity with explicit MB or GB units; received '${value}'")
        BigDecimal amount = new BigDecimal(matched.group(1))
        if (amount.signum() <= 0)
            throw new IllegalArgumentException("${key} must be positive; received '${value}'")
        BigDecimal gb = matched.group(2) == 'MB' ? amount / new BigDecimal(1024) : amount
        // A request smaller than one byte cannot be represented by Nextflow.
        if (gb * new BigDecimal(1073741824) < BigDecimal.ONE)
            throw new IllegalArgumentException("${key} must request at least one byte")
        gb
    }

    private static boolean boolValue(def value, boolean fallback = false) {
        if (value == null) return fallback
        if (value instanceof Boolean) return value
        def normalized = value.toString().trim().toLowerCase()
        if (normalized in ['true', '1', 'yes', 'y', 'on']) return true
        if (normalized in ['false', '0', 'no', 'n', 'off']) return false
        fallback
    }

    private static def getValue(def values, String key, def fallback = null) {
        try {
            if (values.containsKey(key) && values[key] != null) return values[key]
        } catch (Throwable ignored) {}
        fallback
    }

    private static String resolveProfile(def values, int cpus, int memoryGb, boolean gpu, double gpuVramGb) {
        def requested = (getValue(values, 'hardware_profile', 'auto') ?: 'auto').toString().trim().toLowerCase()
        if (!(requested in ['auto', 'conservative', 'balanced', 'aggressive'])) requested = 'auto'
        if (requested != 'auto') return requested
        if (cpus <= 4 || memoryGb <= 12 || (gpu && gpuVramGb > 0.0d && gpuVramGb < 10.0d)) return 'conservative'
        if (cpus >= 24 && memoryGb >= 64 && (!gpu || gpuVramGb >= 24.0d)) return 'aggressive'
        'balanced'
    }

    /**
     * Resolve resource requests for every pipeline stage from one hardware envelope.
     * Existing stage values remain hard ceilings, so explicit site constraints are
     * preserved while smaller machines receive safe allocations automatically.
     */
    static Map resolve(def values, int cpuBudget, int memoryBudgetGb, boolean gpu, double gpuVramGb) {
        cpuBudget = Math.max(1, cpuBudget)
        memoryBudgetGb = Math.max(2, memoryBudgetGb)
        boolean enabled = boolValue(getValue(values, 'hardware_auto', true), true)
        String profile = resolveProfile(values, cpuBudget, memoryBudgetGb, gpu, gpuVramGb)
        double cpuScale = profile == 'conservative' ? 0.75d : (profile == 'aggressive' ? 1.25d : 1.0d)
        double memoryScale = profile == 'conservative' ? 0.85d : (profile == 'aggressive' ? 1.10d : 1.0d)

        // cpuRatio controls useful parallelism; maxUseful prevents serial code from
        // reserving cores that are better used by another ready process.
        def specs = [
            convert:              [cpu: 'convert_cpus', memory: 'convert_memory_gb', cpuRatio: 0.50d, maxUseful: 8,  minCpu: 2, memRatio: 0.18d, minMem: 4],
            grandqc:              [cpu: 'grandqc_cpus', memory: 'grandqc_memory_gb', cpuRatio: 0.50d, maxUseful: 8,  minCpu: 2, memRatio: 0.25d, minMem: 6],
            grandqc_crop_mask:    [cpu: null, memory: 'grandqc_crop_mask_memory_gb', cpuRatio: 0.0d, maxUseful: 1, minCpu: 1, memRatio: 0.06d, minMem: 2],
            prepare_crop:         [cpu: 'prepare_crop_cpus', memory: 'prepare_crop_memory_gb', cpuRatio: 0.25d, maxUseful: 4, minCpu: 1, memRatio: 0.15d, minMem: 4],
            stardist_auto_roi:    [cpu: 'stardist_auto_roi_cpus', memory: 'stardist_auto_roi_memory_gb', cpuRatio: 0.25d, maxUseful: 4, minCpu: 1, memRatio: 0.12d, minMem: 4],
            stardist:             [cpu: 'stardist_cpus', memory: 'stardist_memory_gb', cpuRatio: 0.75d, maxUseful: 16, minCpu: 2, memRatio: 0.40d, minMem: 8],
            hovernet:             [cpu: 'hovernet_cpus', memory: 'hovernet_memory_gb', cpuRatio: 0.50d, maxUseful: 8, minCpu: 2, memRatio: 0.32d, minMem: 8],
            cellvit:              [cpu: 'cellvit_cpus', memory: 'cellvit_memory_gb', cpuRatio: 0.75d, maxUseful: 16, minCpu: 4, memRatio: 0.40d, minMem: 10],
            cell_consensus:       [cpu: 'cell_consensus_cpus', memory: 'cell_consensus_memory_gb', cpuRatio: 0.20d, maxUseful: 2, minCpu: 1, memRatio: 0.22d, minMem: 6],
            tma:                  [cpu: 'tma_cpus', memory: 'tma_memory_gb', cpuRatio: 0.15d, maxUseful: 2, minCpu: 1, memRatio: 0.10d, minMem: 3],
            gigatime:             [cpu: 'gigatime_cpus', memory: 'gigatime_memory_gb', cpuRatio: 0.50d, maxUseful: 8, minCpu: 2, memRatio: 0.28d, minMem: 8],
            gigatime_export:      [cpu: 'gigatime_export_cpus', memory: 'gigatime_export_memory_gb', cpuRatio: 0.40d, maxUseful: 8, minCpu: 2, memRatio: 0.18d, minMem: 4],
            marker_quantification:[cpu: 'marker_quantification_cpus', memory: 'marker_quantification_memory_gb', cpuRatio: 0.20d, maxUseful: 2, minCpu: 1, memRatio: 0.25d, minMem: 6],
            input_roi_mask:       [cpu: 'input_roi_mask_cpus', memory: 'input_roi_mask_memory_gb', cpuRatio: 0.20d, maxUseful: 2, minCpu: 1, memRatio: 0.10d, minMem: 3],
            assign:               [cpu: 'assign_cpus', memory: 'assign_memory_gb', cpuRatio: 0.75d, maxUseful: 12, minCpu: 2, memRatio: 0.28d, minMem: 6],
            expand:               [cpu: 'expand_cpus', memory: 'expand_memory_gb', cpuRatio: 0.20d, maxUseful: 2, minCpu: 1, memRatio: 0.18d, minMem: 4],
            uni2_grid:            [cpu: 'uni2_grid_cpus', memory: 'uni2_grid_memory_gb', cpuRatio: 0.20d, maxUseful: 2, minCpu: 1, memRatio: 0.10d, minMem: 3],
            uni2:                 [cpu: 'uni2_cpus', memory: 'uni2_memory_gb', cpuRatio: 0.75d, maxUseful: 16, minCpu: 2, memRatio: 0.42d, minMem: 8],
            kodama:               [cpu: 'r_cpus', memory: 'r_memory_gb', cpuRatio: 0.75d, maxUseful: 16, minCpu: 2, memRatio: 0.50d, minMem: 8],
            uni2_route_compare:   [cpu: 'uni2_route_compare_cpus', memory: 'uni2_route_compare_memory_gb', cpuRatio: 0.25d, maxUseful: 4, minCpu: 1, memRatio: 0.18d, minMem: 4],
            clustering:           [cpu: 'cluster_cpus', memory: 'cluster_memory_gb', cpuRatio: 0.35d, maxUseful: 4, minCpu: 1, memRatio: 0.42d, minMem: 8],
            cluster_assessment:   [cpu: 'cluster_assessment_cpus', memory: 'cluster_assessment_memory_gb', cpuRatio: 0.25d, maxUseful: 4, minCpu: 1, memRatio: 0.18d, minMem: 4],
            cluster_mask:         [cpu: 'cluster_mask_cpus', memory: 'cluster_mask_memory_gb', cpuRatio: 0.20d, maxUseful: 2, minCpu: 1, memRatio: 0.20d, minMem: 4],
            grow:                 [cpu: 'grow_cpus', memory: 'grow_memory_gb', cpuRatio: 0.65d, maxUseful: 12, minCpu: 2, memRatio: 0.42d, minMem: 8],
            medsam_refine:        [cpu: 'medsam_refine_cpus', memory: 'medsam_refine_memory_gb', cpuRatio: 0.50d, maxUseful: 12, minCpu: 2, memRatio: 0.38d, minMem: 8],
            cluster_geojson:      [cpu: 'cluster_geojson_cpus', memory: 'cluster_geojson_memory_gb', cpuRatio: 0.20d, maxUseful: 2, minCpu: 1, memRatio: 0.22d, minMem: 4],
            neoplastic_section:   [cpu: 'neoplastic_section_cpus', memory: 'neoplastic_section_memory_gb', cpuRatio: 0.25d, maxUseful: 4, minCpu: 1, memRatio: 0.15d, minMem: 4],
            titan:                [cpu: 'titan_cpus', memory: 'titan_memory_gb', cpuRatio: 0.50d, maxUseful: 8, minCpu: 2, memRatio: 0.30d, minMem: 8],
            pathofmpred:          [cpu: 'pathofmpred_cpus', memory: 'pathofmpred_memory_gb', cpuRatio: 0.20d, maxUseful: 2, minCpu: 1, memRatio: 0.15d, minMem: 4],
            cell_profiles:       [cpu: 'cell_profiles_cpus', memory: 'cell_profiles_memory_gb', atlas: true, cpuRatio: 0.35d, maxUseful: 4, minCpu: 1, memRatio: 0.30d, minMem: 4],
            cohort_niches:       [cpu: 'cell_profiles_cpus', memory: 'cell_profiles_memory_gb', atlas: true, cpuRatio: 0.35d, maxUseful: 4, minCpu: 1, memRatio: 0.30d, minMem: 4],
            cell_tissue_links:   [cpu: 'cell_profiles_cpus', memory: 'cell_profiles_memory_gb', atlas: true, cpuRatio: 0.20d, maxUseful: 2, minCpu: 1, memRatio: 0.20d, minMem: 2],
            reference_mapping:  [cpu: 'cell_profiles_cpus', memory: 'cell_profiles_memory_gb', atlas: true, cpuRatio: 0.25d, maxUseful: 4, minCpu: 1, memRatio: 0.20d, minMem: 2],
            region_reference_mapping: [cpu: 'cell_profiles_cpus', memory: 'cell_profiles_memory_gb', atlas: true, cpuRatio: 0.25d, maxUseful: 4, minCpu: 1, memRatio: 0.20d, minMem: 2],
            spatialdata:         [cpu: 'cell_profiles_cpus', memory: 'cell_profiles_memory_gb', atlas: true, cpuRatio: 0.35d, maxUseful: 4, minCpu: 1, memRatio: 0.30d, minMem: 4],
            hierarchy_features:  [cpu: 'tissue_hierarchy_cpus', memory: 'tissue_hierarchy_memory', atlas: true, memoryUnits: true, cpuRatio: 0.50d, maxUseful: 8, minCpu: 1, memRatio: 0.30d, minMem: 4],
            hierarchy_discovery: [cpu: 'tissue_hierarchy_cpus', memory: 'tissue_hierarchy_memory', atlas: true, memoryUnits: true, cpuRatio: 0.50d, maxUseful: 8, minCpu: 1, memRatio: 0.35d, minMem: 4]
        ]

        def updates = [:]
        def stages = [:]
        specs.each { String stage, Map spec ->
            int cpuCap = spec.cpu ? (spec.atlas
                ? atlasPositiveInt(getValue(values, spec.cpu as String, cpuBudget), spec.cpu as String)
                : positiveInt(getValue(values, spec.cpu as String, cpuBudget), cpuBudget)) : 1
            def memoryCap = spec.atlas
                ? atlasMemoryGb(getValue(values, spec.memory as String, spec.memoryUnits ? "${memoryBudgetGb} GB" : memoryBudgetGb), spec.memory as String, spec.memoryUnits == true)
                : positiveInt(getValue(values, spec.memory as String, memoryBudgetGb), memoryBudgetGb)
            int cpus
            def memoryGb
            if (enabled) {
                int cpuTarget = Math.max(spec.minCpu as int, Math.ceil(cpuBudget * (spec.cpuRatio as double) * cpuScale) as int)
                cpus = Math.max(1, Math.min(cpuBudget, Math.min(cpuCap, Math.min(spec.maxUseful as int, cpuTarget))))
                int memoryTarget = Math.max(spec.minMem as int, Math.ceil(memoryBudgetGb * (spec.memRatio as double) * memoryScale) as int)
                memoryGb = spec.atlas
                    ? (memoryCap as BigDecimal).min(new BigDecimal(memoryBudgetGb)).min(new BigDecimal(memoryTarget))
                    : Math.max(1, Math.min(memoryBudgetGb, Math.min(memoryCap as int, memoryTarget)))
            } else {
                cpus = Math.max(1, Math.min(cpuBudget, cpuCap))
                memoryGb = spec.atlas ? (memoryCap as BigDecimal).min(new BigDecimal(memoryBudgetGb))
                    : Math.max(1, Math.min(memoryBudgetGb, memoryCap as int))
            }
            // Several atlas stages share user-facing caps. Keep the largest
            // resolved allocation in settings, never let iteration order reduce
            // a sibling stage; directives consume the per-stage map directly.
            if (spec.cpu) updates[spec.cpu as String] = Math.max((updates[spec.cpu as String] ?: 0) as int, cpus)
            if (spec.memoryUnits) {
                def previous = updates[spec.memory as String]
                BigDecimal maximum = previous ? atlasMemoryGb(previous, spec.memory as String, true).max(memoryGb as BigDecimal) : memoryGb as BigDecimal
                updates[spec.memory as String] = "${maximum.stripTrailingZeros().toPlainString()} GB".toString()
            } else {
                updates[spec.memory as String] = updates.containsKey(spec.memory as String)
                    ? (updates[spec.memory as String] as BigDecimal).max(memoryGb as BigDecimal).intValueExact()
                    : memoryGb
            }
            stages[stage] = [cpus: cpus, memory_gb: memoryGb]
        }

        int kodamaRequested = positiveInt(getValue(values, 'kodama_n_cores', 0), 0)
        updates.kodama_n_cores = kodamaRequested > 0
            ? Math.max(1, Math.min(stages.kodama.cpus as int, kodamaRequested))
            : Math.max(1, stages.kodama.cpus as int)
        updates.uni2_torch_threads = Math.max(1, Math.min(stages.uni2.cpus as int, positiveInt(getValue(values, 'uni2_torch_threads', stages.uni2.cpus), stages.uni2.cpus as int)))
        updates.grow_max_workers = Math.max(1, Math.min(stages.grow.cpus as int, positiveInt(getValue(values, 'grow_max_workers', stages.grow.cpus), stages.grow.cpus as int)))

        int medsamRequested = positiveInt(getValue(values, 'medsam_refine_max_workers', 0), 0)
        int medsamTarget = profile == 'aggressive' && gpuVramGb >= 24.0d ? 4 : (profile == 'conservative' ? 1 : 2)
        updates.medsam_refine_max_workers = medsamRequested > 0
            ? Math.max(1, Math.min(stages.medsam_refine.cpus as int, medsamRequested))
            : Math.max(1, Math.min(stages.medsam_refine.cpus as int, medsamTarget))

        int rayWorkers = positiveInt(getValue(values, 'cellvit_ray_workers', 1), 1)
        int rayWorkerCpus = positiveInt(getValue(values, 'cellvit_ray_worker_cpus', 0), 0)
        updates.cellvit_ray_workers = Math.max(1, Math.min(rayWorkers, stages.cellvit.cpus as int))
        updates.cellvit_ray_worker_cpus = rayWorkerCpus > 0
            ? Math.max(1, Math.min(rayWorkerCpus, stages.cellvit.cpus as int))
            : Math.max(1, Math.floor((stages.cellvit.cpus as int) / (updates.cellvit_ray_workers as int)) as int)

        [
            enabled: enabled,
            profile: profile,
            cpu_budget: cpuBudget,
            memory_budget_gb: memoryBudgetGb,
            gpu_available: gpu,
            gpu_vram_gb: Math.round(gpuVramGb * 100.0d) / 100.0d,
            updates: updates,
            stages: stages,
            runtime: [
                kodama_n_cores: updates.kodama_n_cores,
                uni2_torch_threads: updates.uni2_torch_threads,
                grow_max_workers: updates.grow_max_workers,
                medsam_refine_max_workers: updates.medsam_refine_max_workers,
                cellvit_ray_workers: updates.cellvit_ray_workers,
                cellvit_ray_worker_cpus: updates.cellvit_ray_worker_cpus
            ]
        ]
    }
}
