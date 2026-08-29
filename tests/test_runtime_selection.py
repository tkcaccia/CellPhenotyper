from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "nextflow.config").read_text(encoding="utf-8")
PARAMETERS = (ROOT / "pipeline_paramers.yml").read_text(encoding="utf-8")
MAIN = (ROOT / "main.nf").read_text(encoding="utf-8")

GPU_MODULES = (
    "extract_uni2_embeddings.nf",
    "extract_uni2_embeddings_shared.nf",
    "run_gigatime_on_crop.nf",
    "run_grandqc_artifact_analysis.nf",
    "run_stardist_roi_segmentation.nf",
    "refine_grown_tissue_medsam.nf",
)


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
    assert 'def auto_gpu_docker_image = "${container_gpu_repo}:${container_gpu_tag}"' in MAIN


def test_auto_device_is_pipeline_wide_default() -> None:
    assert "compute_device                 = 'auto'" in CONFIG
    assert "compute_device: auto" in PARAMETERS
    assert "uni2_device_auto" not in CONFIG
    assert "uni2_device_auto" not in PARAMETERS
    assert "def configured_gpu_runtime" in CONFIG
    assert "singularityBaseRunOptions = configured_gpu_runtime ? '--nv' : ''" in CONFIG
    assert "dockerGpuOptions = configured_gpu_runtime ? '--gpus all --shm-size=3g' : ''" in CONFIG


def test_gpu_modules_consume_the_resolved_device() -> None:
    for module_name in GPU_MODULES:
        module = (ROOT / "modules" / module_name).read_text(encoding="utf-8")
        assert "params._resolved_compute_device ?: params.compute_device" in module, module_name


def test_medsam_auto_device_and_retry_follow_pipeline_policy() -> None:
    module = (ROOT / "modules" / "refine_grown_tissue_medsam.nf").read_text(encoding="utf-8")
    assert "medsam_device                  = 'auto'" in CONFIG
    assert "medsam_device: auto" in PARAMETERS
    assert "label 'compute_heavy'" in module
    assert "requestedMedsamDevice == 'auto'" in module
