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
    assert "default_container_gpu_tag = '2.8-gpu-amd64'" in CONFIG
    assert "container_gpu_repo: ghcr.io/tkcaccia/cellphenotyper-runtime" in PARAMETERS
    assert "container_gpu_tag: 2.8-gpu-amd64" in PARAMETERS


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
    assert "singularity_gpu_image_source   = 'oras'" in CONFIG
    assert "singularity_gpu_image_source: oras" in PARAMETERS
    assert CONFIG.count('"docker://${gpuRepo}:${gpuTag}"') >= 2


def test_docker_gpu_roles_use_the_separate_gpu_repository() -> None:
    assert CONFIG.count('"${gpuRepo}:${gpuTag}"') >= 2
    assert "default_container_gpu_repo = 'ghcr.io/tkcaccia/cellphenotyper-runtime'" in CONFIG


def test_auto_device_is_pipeline_wide_default() -> None:
    assert "compute_device                 = 'auto'" in CONFIG
    assert "compute_device: auto" in PARAMETERS
    assert "uni2_device_auto" not in CONFIG
    assert "uni2_device_auto" not in PARAMETERS
    assert "def configured_gpu_runtime" in CONFIG
    assert "gpu_task_container_options.call(params, task.process.toString(), explicitPlan, 'singularity')" in CONFIG
    assert "gpu_task_container_options.call(params, task.process.toString(), explicitPlan, 'docker')" in CONFIG
    assert "docker.runOptions = dockerExtraOptions" in CONFIG


def test_singularity_never_binds_a_missing_optional_cache_directory() -> None:
    assert "def nearest_existing_bind_target" in CONFIG
    assert "while (target != null && !target.exists())" in CONFIG
    assert "def target = nearest_existing_bind_target(file)" in CONFIG
    assert "target.exists() ? target.getCanonicalPath() : target.getAbsolutePath()" not in CONFIG


def test_params_file_keeps_the_selected_landmark_clustering_defaults() -> None:
    assert "kodama_landmarks: 10000" in PARAMETERS
    assert "cluster_snn_k: 50" in PARAMETERS
    assert "cluster_algorithm: leiden" in PARAMETERS
    assert "cluster_target_clusters: 0" in PARAMETERS
    assert "cluster_forced_count_sensitivity_acknowledged: false" in PARAMETERS
    assert "cluster_landmark_cells: 10000" in PARAMETERS
    assert "cluster_landmark_assign_k: 50" in PARAMETERS
    assert "cluster_landmark_sample_strategy: knn_inverse_distance" in PARAMETERS
    assert "cluster_landmark_density_power: 2.0" in PARAMETERS
    assert "cluster_resolution: auto" in PARAMETERS


def test_relative_gpu_lock_directory_is_shared_across_nextflow_sessions() -> None:
    config = (ROOT / "nextflow.config").read_text(encoding="utf-8")
    assert "new File(baseDir.toString(), configuredGpuLockDir).canonicalPath" in config
    assert "cellphenotyper_acquire_gpu_slot '${quotedGpuLockDir}'" in config


def test_target_cluster_count_is_wired_to_clustering() -> None:
    module = (ROOT / "modules" / "run_rcode_clustering.nf").read_text()
    script = (ROOT / "bin" / "Rcode_Clustering.R").read_text()
    assert "--target-clusters ${params.cluster_target_clusters}" in module
    assert 'flag == "--target-clusters"' in script
    assert "collapse_clusters_to_target" in script
    assert 'paste0("nearest_centroid_merge_in_", cluster_representation, "_space")' in script
    assert "cluster_forced_count_sensitivity_acknowledged" in module
    assert "sensitivity_forced_cluster_count" in script
    assert "SENSITIVITY ONLY" in script
    assert "A forced cluster count is sensitivity-only" in MAIN


def test_gpu_modules_consume_the_resolved_device() -> None:
    explicit_runtime_modules = {
        "extract_uni2_embeddings.nf",
        "extract_uni2_embeddings_shared.nf",
        "run_gigatime_on_crop.nf",
        "run_cellvitpp.nf",
        "run_grandqc_artifact_analysis.nf",
        "run_stardist_roi_segmentation.nf",
        "run_hovernet_monusac.nf",
        "run_grandqc_artifact_analysis.nf",
        "run_stardist_roi_segmentation.nf",
        "run_hovernet_monusac.nf",
        "refine_grown_tissue_medsam.nf",
        "extract_titan_section_embedding.nf",
    }
    for module_name in set(GPU_MODULES) | explicit_runtime_modules:
        module = (ROOT / "modules" / module_name).read_text(encoding="utf-8")
        if module_name in explicit_runtime_modules:
            assert "TaskRuntime.device(runtime_plan)" in module, module_name
            assert "val(runtime_plan)" in module, module_name
        else:
            assert "params._resolved_compute_device ?: params.compute_device" in module, module_name


def test_medsam_auto_device_and_retry_follow_pipeline_policy() -> None:
    module = (ROOT / "modules" / "refine_grown_tissue_medsam.nf").read_text(encoding="utf-8")
    assert "medsam_device                  = 'auto'" in CONFIG
    assert "medsam_device: auto" in PARAMETERS
    assert "label 'compute_heavy'" in module
    assert "requestedMedsamDevice == 'auto'" in module


def test_stardist_is_default_and_consensus_does_not_change_with_hardware() -> None:
    assert "cell_detection_mode           = 'stardist'" in CONFIG
    assert "cell_detection_mode: stardist" in PARAMETERS
    assert "cell_detection_mode=consensus requires GPU execution" in MAIN
    assert "falling back to StarDist-only" not in MAIN


def test_gpu_scheduler_has_memory_aware_admission_for_all_gpu_stages() -> None:
    helper = (ROOT / "bin" / "acquire_gpu_slot.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in helper
    assert "memory.total,memory.free" in helper
    assert "gpu_memory_token" in CONFIG
    assert "withLabel: gpu_capable" in CONFIG
    for role in ("grandqc", "stardist", "hovernet", "cellvit", "gigatime", "uni2", "medsam", "kodama", "titan"):
        assert f"{role}_gpu_memory_gb" in CONFIG
