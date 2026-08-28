from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r_processes_do_not_inherit_host_user_libraries() -> None:
    modules = [
        ROOT / "modules" / "run_kodama_analysis.nf",
        ROOT / "modules" / "run_rcode_clustering.nf",
        ROOT / "modules" / "run_pathofmpred.nf",
    ]
    for module in modules:
        source = module.read_text(encoding="utf-8")
        assert "R_LIBS_USER" in source, module
        assert "R_ENVIRON_USER=/dev/null" in source, module
        assert "R_PROFILE_USER=/dev/null" in source, module


def test_container_recipes_default_to_an_empty_r_user_library() -> None:
    recipes = [
        ROOT / "docker" / "Dockerfile.full.cpu",
        ROOT / "docker" / "Dockerfile.full.gpu",
        ROOT / "docker" / "Dockerfile.runtime-update.gpu",
        ROOT / "docker" / "Dockerfile.source-refresh.gpu",
        ROOT / "singularity" / "cellphenotyper_full_cpu.def",
        ROOT / "singularity" / "cellphenotyper_full_gpu.def",
    ]
    for recipe in recipes:
        source = recipe.read_text(encoding="utf-8")
        assert "R_LIBS_USER=/opt/cellphenotyper/empty-r-library" in source, recipe
        assert "R_ENVIRON_USER=/dev/null" in source, recipe
        assert "R_PROFILE_USER=/dev/null" in source, recipe
