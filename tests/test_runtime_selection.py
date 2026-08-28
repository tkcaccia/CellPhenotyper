from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "nextflow.config").read_text(encoding="utf-8")
PARAMETERS = (ROOT / "pipeline_paramers.yml").read_text(encoding="utf-8")


def test_verified_gpu_runtime_is_the_default() -> None:
    assert "default_container_gpu_repo = 'ghcr.io/tkcaccia/cellphenotyper-runtime'" in CONFIG
    assert "default_container_gpu_tag = '2.7-gpu-amd64'" in CONFIG
    assert "container_gpu_repo: ghcr.io/tkcaccia/cellphenotyper-runtime" in PARAMETERS
    assert "container_gpu_tag: 2.7-gpu-amd64" in PARAMETERS


def test_singularity_gpu_roles_use_the_verified_oci_runtime() -> None:
    assert "singularity_gpu_image_source   = 'docker'" in CONFIG
    assert "singularity_gpu_image_source: docker" in PARAMETERS
    assert CONFIG.count('"docker://${gpuRepo}:${gpuTag}"') >= 2


def test_docker_gpu_roles_use_the_separate_gpu_repository() -> None:
    assert CONFIG.count('"${gpuRepo}:${gpuTag}"') >= 2
