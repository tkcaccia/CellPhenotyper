/** Explicit, serializable execution policy passed as a value input to tasks.
 *
 * Never consult late params mutations or repeat device detection inside a task.
 * The input becomes part of Nextflow cache identity. Scientific parameters and
 * specimen data do not belong in this map.
 */
class TaskRuntime {
    private static BigDecimal positive(def value, String name) {
        try {
            BigDecimal parsed = new BigDecimal(value.toString())
            if (parsed <= 0) throw new IllegalArgumentException()
            return parsed
        } catch (Throwable ignored) {
            throw new IllegalArgumentException("Runtime plan requires finite positive ${name}")
        }
    }

    private static int positiveInt(def value, String name) {
        BigDecimal parsed = positive(value, name)
        try { return parsed.intValueExact() }
        catch (Throwable ignored) { throw new IllegalArgumentException("Runtime plan requires integer ${name}") }
    }

    private static BigDecimal memoryGb(def value, String name) {
        BigDecimal amount = positive(value, name)
        BigDecimal bytes = amount * new BigDecimal(1073741824)
        if (bytes < BigDecimal.ONE || bytes > new BigDecimal(Long.MAX_VALUE))
            throw new IllegalArgumentException("Runtime plan ${name} must be representable as 1..Long.MAX_VALUE bytes")
        return amount
    }

    private static void validate(def plan) {
        if (!(plan instanceof Map) || plan.schema_version != 1)
            throw new IllegalArgumentException('Missing or unsupported explicit runtime plan (schema_version=1 required)')
        if (!(plan.compute_device in ['cpu', 'gpu']))
            throw new IllegalArgumentException('Runtime plan compute_device must already be resolved to cpu or gpu')
        if (!(plan.profile in ['conservative', 'balanced', 'aggressive']))
            throw new IllegalArgumentException('Runtime plan profile must already be resolved')
        positiveInt(plan.cpu_budget, 'cpu_budget')
        memoryGb(plan.memory_budget_gb, 'memory_budget_gb')
        if (!(plan.stages instanceof Map) || !(plan.settings instanceof Map))
            throw new IllegalArgumentException('Runtime plan requires stages and settings maps')
    }

    static Map create(Map hardware, String computeDevice) {
        def plan = [schema_version: 1, compute_device: computeDevice,
                    profile: hardware.profile, cpu_budget: hardware.cpu_budget,
                    memory_budget_gb: hardware.memory_budget_gb,
                    stages: hardware.stages.collectEntries { key, value ->
                        [(key.toString()): [cpus: value.cpus, memory_gb: value.memory_gb]]
                    }, settings: new LinkedHashMap(hardware.updates ?: [:])]
        validate(plan)
        plan.stages.each { name, ignored -> cpus(plan, name); memory(plan, name) }
        return plan
    }

    private static Map stage(def plan, String name) {
        validate(plan)
        def value = plan.stages[name]
        if (!(value instanceof Map))
            throw new IllegalArgumentException("Runtime plan has no resource allocation for stage '${name}'")
        return value
    }

    static int cpus(def plan, String name) {
        def allocation = stage(plan, name)
        return Math.min(positiveInt(allocation.cpus, "${name}.cpus"), positiveInt(plan.cpu_budget, 'cpu_budget'))
    }

    static String memory(def plan, String name) {
        def allocation = stage(plan, name)
        BigDecimal amount = memoryGb(allocation.memory_gb, "${name}.memory_gb")
            .min(memoryGb(plan.memory_budget_gb, 'memory_budget_gb'))
        // Preserve sub-GB and decimal user caps without rounding allocations up.
        return "${amount.stripTrailingZeros().toPlainString()} GB"
    }

    static String device(def plan) { validate(plan); return plan.compute_device }
    static String profile(def plan) { validate(plan); return plan.profile }
    static def setting(def plan, String name, def fallback) {
        validate(plan)
        return plan.settings.containsKey(name) && plan.settings[name] != null ? plan.settings[name] : fallback
    }

    /** Model-free artifact entry point, bounded by configured executor/profile caps. */
    static Map forArtifacts(def values) {
        int budget = Math.min(Runtime.runtime.availableProcessors(),
            Math.min(positiveInt(values._executor_max_cpus, '_executor_max_cpus'),
                     positiveInt(values.cell_profiles_cpus, 'cell_profiles_cpus')))
        int memoryBudget = Math.min(HostRuntime.memoryGb(),
            Math.min(positiveInt(values._executor_max_memory_gb, '_executor_max_memory_gb'),
                     positiveInt(values.cell_profiles_memory_gb, 'cell_profiles_memory_gb')))
        def hardware = HardwarePolicy.resolve(values, budget, memoryBudget, false, 0.0d)
        // HardwarePolicy's full-inference envelope has a 2-GB floor. Artifact
        // jobs may have a smaller explicit/host cap; never increase that cap.
        hardware.memory_budget_gb = memoryBudget
        hardware.stages.each { name, allocation ->
            allocation.memory_gb = (allocation.memory_gb as BigDecimal).min(new BigDecimal(memoryBudget))
        }
        hardware.updates.each { name, value ->
            if (name.endsWith('_memory_gb'))
                hardware.updates[name] = (value as BigDecimal).min(new BigDecimal(memoryBudget))
        }
        BigDecimal hierarchyMemory = hardware.stages.hierarchy_features.memory_gb
            .max(hardware.stages.hierarchy_discovery.memory_gb)
        hardware.updates.tissue_hierarchy_memory = "${hierarchyMemory.stripTrailingZeros().toPlainString()} GB".toString()
        return create(hardware, 'cpu')
    }
}
