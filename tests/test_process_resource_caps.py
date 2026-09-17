from pathlib import Path
import re

import pytest


ROOT = Path(__file__).parents[1]


def test_process_directives_never_cast_user_facing_auto_limits() -> None:
    offenders = []
    for module in sorted((ROOT / "modules").glob("*.nf")):
        source = module.read_text(encoding="utf-8")
        if "params.max_cpus as int" in source or "params.max_memory_gb as " in source:
            offenders.append(module.name)
    assert offenders == []


def resource_directive(source: str, name: str) -> str:
    header = re.split(r"(?m)^\s*input\s*:", source, maxsplit=1)[0]
    header = re.sub(r"(?m)^\s*//.*$", "", header)
    matched = re.search(
        rf"(?ms)^\s*{name}\b(.*?)(?=^\s*(?:cpus|memory|time|tag|label|cache|publishDir)\b|\Z)", header
    )
    assert matched, f"Missing {name} directive"
    return matched.group(1).strip()


def allocation_kind(source: str) -> str:
    """Check both directives, not a cap/helper mention elsewhere in the script."""
    cpu = resource_directive(source, "cpus")
    memory = resource_directive(source, "memory")
    if "TaskRuntime." in cpu + memory:
        cpu_call = re.search(r"TaskRuntime\.cpus\(\s*(\w+)\s*,\s*'([^']+)'\s*\)", cpu)
        memory_call = re.search(r"TaskRuntime\.memory\(\s*(\w+)\s*,\s*'([^']+)'\s*\)", memory)
        assert cpu_call and memory_call, "Both resources must use validated TaskRuntime allocation helpers"
        assert cpu_call.groups() == memory_call.groups(), "CPU/memory must use the same plan and stage"
        plan, _ = cpu_call.groups()
        inputs = re.split(r"(?m)^\s*input\s*:", source, maxsplit=1)[1]
        inputs = re.split(r"(?m)^\s*output\s*:", inputs, maxsplit=1)[0]
        assert re.search(rf"\bval\s*(?:\(\s*{plan}\s*\)|\s+{plan}\b)", inputs), "Runtime plan must be a value input"
        return "explicit_runtime"
    if "_executor_max_cpus" in cpu or "_executor_max_memory_gb" in memory:
        # A fixed one-core serial process needs no CPU-budget clamp.
        assert cpu == "1" or ("params._executor_max_cpus" in cpu and "Math.min" in cpu)
        assert "params._executor_max_memory_gb" in memory and "Math.min" in memory
        return "numeric_executor"
    assert cpu == "1" and memory == "'2 GB'", "Unrecognized or uncapped resource allocation"
    return "fixed_serial"


def test_all_resource_capped_modules_use_validated_caps() -> None:
    config = (ROOT / "nextflow.config").read_text(encoding="utf-8")
    assert "params._executor_max_cpus = configured_executor_cpus" in config
    assert "params._executor_max_memory_gb = configured_executor_memory_gb" in config

    capped_modules = 0
    for module in sorted((ROOT / "modules").glob("*.nf")):
        source = module.read_text(encoding="utf-8")
        kind = allocation_kind(source)
        if kind != "fixed_serial":
            capped_modules += 1
            assert "params.max_cpus as int" not in source
            assert "params.max_memory_gb as int" not in source
            assert "params.max_memory_gb as double" not in source
        else:
            # Preserve the existing ROI-preparation exception, not an exemption
            # for new modules to bypass the executor/runtime-plan contracts.
            assert module.name == "prepare_roi_geojson.nf"
    assert capped_modules >= 30


@pytest.mark.parametrize("mutation", ["missing_memory", "different_stage", "missing_input", "comment_only"])
def test_runtime_census_rejects_incomplete_or_cosmetic_migration(mutation):
    source = """process P {
    cpus { TaskRuntime.cpus(runtime_plan, 'probe') }
    memory { TaskRuntime.memory(runtime_plan, 'probe') }
    input:
    val(runtime_plan)
    output:
    path 'out'
}
"""
    if mutation == "missing_memory":
        source = source.replace("TaskRuntime.memory(runtime_plan, 'probe')", "'64 GB'")
    elif mutation == "different_stage":
        source = source.replace("TaskRuntime.memory(runtime_plan, 'probe')", "TaskRuntime.memory(runtime_plan, 'other')")
    elif mutation == "missing_input":
        source = source.replace("val(runtime_plan)", "val(unrelated)")
    else:
        source = source.replace("cpus { TaskRuntime.cpus(runtime_plan, 'probe') }", "// TaskRuntime.cpus(runtime_plan, 'probe')\n    cpus 64")
        source = source.replace("memory { TaskRuntime.memory(runtime_plan, 'probe') }", "// TaskRuntime.memory(runtime_plan, 'probe')\n    memory '64 GB'")
    with pytest.raises(AssertionError):
        allocation_kind(source)
