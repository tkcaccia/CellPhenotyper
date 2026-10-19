from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_process_directives_never_cast_user_facing_auto_limits() -> None:
    offenders = []
    for module in sorted((ROOT / "modules").glob("*.nf")):
        source = module.read_text(encoding="utf-8")
        if "params.max_cpus as int" in source or "params.max_memory_gb as " in source:
            offenders.append(module.name)
    assert offenders == []


def test_all_resource_capped_modules_use_numeric_executor_limits() -> None:
    config = (ROOT / "nextflow.config").read_text(encoding="utf-8")
    assert "params._executor_max_cpus = configured_executor_cpus" in config
    assert "params._executor_max_memory_gb = configured_executor_memory_gb" in config

    capped_modules = 0
    for module in sorted((ROOT / "modules").glob("*.nf")):
        source = module.read_text(encoding="utf-8")
        if "_executor_max_cpus" in source or "_executor_max_memory_gb" in source:
            capped_modules += 1
            assert "params.max_cpus as int" not in source
            assert "params.max_memory_gb as int" not in source
            assert "params.max_memory_gb as double" not in source
    assert capped_modules >= 30
