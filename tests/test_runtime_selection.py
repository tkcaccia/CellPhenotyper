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


def test_published_cpu_runtimes_are_the_defaults() -> None:
    assert "default_container_cpu_tag_amd64 = '2.2-amd64'" in CONFIG
    assert "default_container_cpu_tag_arm64 = '0.2.0'" in CONFIG
    assert "container_cpu_tag_amd64: 2.2-amd64" in PARAMETERS
    assert "container_cpu_tag_arm64: 0.2.0" in PARAMETERS
    assert "2.6-amd64" not in CONFIG
    assert "2.6-arm64" not in CONFIG
    assert "2.6-amd64" not in PARAMETERS
    assert "2.6-arm64" not in PARAMETERS


def test_gpu_release_refresh_uses_a_published_base_image() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish-runtime-release.yml").read_text()
    dockerfile = (ROOT / "docker" / "Dockerfile.runtime-update.gpu").read_text()
    assert "BASE_IMAGE=${{ env.IMAGE_REPO }}:2.2-gpu-amd64" in workflow
    assert "ARG BASE_IMAGE=ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64" in dockerfile
    assert "2.3-gpu-amd64" not in workflow
    assert "2.3-gpu-amd64" not in dockerfile


def test_sif_publisher_requires_an_explicit_version() -> None:
    publisher = (ROOT / "singularity" / "publish_sif_release_asset.sh").read_text()
    assert 'VERSION=""' in publisher
    assert "Missing required --version <semver>." in publisher
    assert 'VERSION="2.3"' not in publisher


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


def test_params_file_keeps_the_selected_landmark_clustering_defaults() -> None:
    assert "kodama_landmarks: 1000" in PARAMETERS
    assert "cluster_snn_k: 50" in PARAMETERS
    assert "cluster_algorithm: leiden" in PARAMETERS
    assert "cluster_target_clusters: 0" in PARAMETERS
    assert "cluster_landmark_cells: 10000" in PARAMETERS
    assert "cluster_landmark_assign_k: 50" in PARAMETERS
    assert "cluster_landmark_sample_strategy: knn_inverse_distance" in PARAMETERS
    assert "cluster_landmark_density_power: 2.0" in PARAMETERS
    assert "cluster_resolution: 0.3" in PARAMETERS


def test_target_cluster_count_is_wired_to_clustering() -> None:
    module = (ROOT / "modules" / "run_rcode_clustering.nf").read_text()
    script = (ROOT / "bin" / "Rcode_Clustering.R").read_text()
    assert "--target-clusters ${params.cluster_target_clusters}" in module
    assert 'flag == "--target-clusters"' in script
    assert "collapse_clusters_to_target" in script
    assert "nearest_centroid_merge_in_kodama_space" in script


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
