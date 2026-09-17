"""Evaluate merged task directives without launching a container or GPU lease.

Each included process stops deliberately after reading its real Nextflow config
directives. Both container engines are explicitly disabled in the final config.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "DIRECTIVE_AUDIT_ONLY_NO_TASK_EXECUTION"


def configured_task(tmp_path, profile, *, requested="auto", device="gpu", visible=True,
                    process="EXTRACT_UNI2_EMBEDDINGS", explicit_plan=True, arch="amd64",
                    extra_params=None, custom_before=False, extra_env=None):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    input_name = "runtime_plan" if explicit_plan else "payload"
    (tmp_path / "probe.nf").write_text(f"""process PROBE {{
  label 'gpu_capable'
  input:
  val({input_name})
  script:
  println 'DIRECTIVE_AUDIT=' + groovy.json.JsonOutput.toJson([
    process: task.process.toString(),
    container: task.container,
    container_options: task.containerOptions,
    before_script: task.beforeScript,
    included_device_param: params.compute_device,
    included_resolved_param: params._resolved_compute_device
  ])
  throw new IllegalArgumentException('{SENTINEL}')
}}
""")
    # Included params remain stale while the concrete task input is passed after
    # the workflow resolves its execution plan.
    (tmp_path / "main.nf").write_text(f"""nextflow.enable.dsl=2
include {{ PROBE as {process} }} from './probe.nf'
workflow {{
  params._resolved_compute_device = '{device}'
  {process}([schema_version: 1, compute_device: '{device}'])
}}
""")
    override = """profiles {
  docker { docker.enabled = false }
  singularity { singularity.enabled = false }
}
report.enabled = false
trace.enabled = false
timeline.enabled = false
dag.enabled = false
"""
    if custom_before:
        override += f"process {{ withName: '{process}' {{ beforeScript = 'echo TASK_SPECIFIC_SETUP' }} }}\n"
    (tmp_path / "engine_off.config").write_text(override)
    params = {"compute_device": requested, "host_arch": arch, "enable_gpu_on_arm64": True,
              "gpu_scheduler_enable": True, "singularity_image_source": "docker",
              "singularity_gpu_image_source": "docker"}
    params.update(extra_params or {})
    (tmp_path / "params.json").write_text(json.dumps(params))
    env = dict(os.environ, NXF_OFFLINE="true", NXF_OPTS="-Xms64m -Xmx512m")
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if visible:
        env["NVIDIA_VISIBLE_DEVICES"] = "all"
    env.update(extra_env or {})
    command = [nextflow, "-log", str(tmp_path / "nextflow.log"),
               "-c", str(ROOT / "nextflow.config"), "-c", str(tmp_path / "engine_off.config"),
               "run", str(tmp_path / "main.nf"),
               "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false",
               "-work-dir", str(tmp_path / "work")]
    if profile != "local":
        command += ["-profile", profile]
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=40)
    output = result.stdout + result.stderr
    assert result.returncode != 0, "Probe must stop during script generation"
    assert not list((tmp_path / "work").glob("*/*/.command.run")), "No task shell may be generated/launched"
    assert "Failed to pull" not in output and "Pulling Singularity image" not in output
    evidence = [json.loads(line[len("DIRECTIVE_AUDIT="):]) for line in output.splitlines()
                if line.startswith("DIRECTIVE_AUDIT=")]
    return evidence[0] if evidence else None, output


def assert_selection(evidence, profile, device):
    assert evidence is not None
    expected = ("ghcr.io/tkcaccia/cellphenotyper-runtime:2.7-gpu-amd64" if device == "gpu"
                else "ghcr.io/tkcaccia/cellphenotyper:2.2-amd64")
    if profile == "singularity":
        expected = "docker://" + expected
    assert evidence["container"] == expected
    lease = evidence["before_script"] or ""
    assert ("cellphenotyper_acquire_gpu_slot" in lease) == (device == "gpu")
    options = evidence["container_options"] or ""
    assert (("--nv" if profile == "singularity" else "--gpus all") in options) == (device == "gpu")


@pytest.mark.parametrize("profile", ["docker", "singularity"])
@pytest.mark.parametrize("device,visible", [("gpu", True), ("cpu", False)])
def test_auto_device_container_and_lease_survive_profile_merge(tmp_path, profile, device, visible):
    evidence, output = configured_task(tmp_path, profile, device=device, visible=visible)
    assert SENTINEL in output
    assert_selection(evidence, profile, device)
    assert evidence["included_device_param"] == "auto"
    assert evidence["included_resolved_param"] == ""


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_explicit_cpu_plan_does_not_lease_visible_gpu_or_use_gpu_container(tmp_path, profile):
    evidence, output = configured_task(tmp_path, profile, requested="gpu", device="cpu")
    assert SENTINEL in output
    assert_selection(evidence, profile, "cpu")


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_gpu_request_without_visibility_environment_still_has_admission(tmp_path, profile):
    evidence, output = configured_task(tmp_path, profile, requested="gpu", visible=False)
    assert SENTINEL in output
    assert_selection(evidence, profile, "gpu")


@pytest.mark.parametrize("profile", ["docker", "singularity"])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_legacy_tasks_retain_config_resolved_device_policy(tmp_path, profile, device):
    evidence, output = configured_task(tmp_path, profile, requested=device, device=device,
                                       process="RUN_GRANDQC_ARTIFACT_ANALYSIS", explicit_plan=False)
    assert SENTINEL in output
    assert_selection(evidence, profile, device)


@pytest.mark.parametrize("profile", ["docker", "singularity"])
@pytest.mark.parametrize("hierarchy_device", ["cpu", "mps", "cuda"])
def test_hierarchy_device_remains_explicit_and_cpu_mps_skip_cuda_lease(tmp_path, profile, hierarchy_device):
    global_device = "cpu" if hierarchy_device == "cuda" else "gpu"
    evidence, output = configured_task(tmp_path, profile, requested=global_device, device=global_device,
                                       process="PREPARE_HIERARCHY_FEATURES",
                                       extra_params={"tissue_hierarchy_device": hierarchy_device})
    if hierarchy_device == "mps":
        assert evidence is None and "native-only" in output
        return
    assert SENTINEL in output
    assert_selection(evidence, profile, "gpu" if hierarchy_device == "cuda" else "cpu")


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_unsupported_arm64_gpu_image_selection_fails_closed(tmp_path, profile):
    evidence, output = configured_task(tmp_path, profile, arch="arm64")
    assert evidence is None
    assert "No default arm64 GPU" in output and "--gpu_container_image" in output


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_arm64_gpu_explicit_compatible_image_is_honored(tmp_path, profile):
    selected_image = "example.invalid/explicit-arm64-gpu:audit-only"
    evidence, output = configured_task(tmp_path, profile, arch="arm64",
                                       extra_params={"gpu_container_image": selected_image})
    assert SENTINEL in output
    assert evidence["container"] == selected_image
    assert "cellphenotyper_acquire_gpu_slot" in evidence["before_script"]


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_task_specific_before_script_override_is_preserved(tmp_path, profile):
    evidence, output = configured_task(tmp_path, profile, custom_before=True)
    assert SENTINEL in output
    assert evidence["before_script"] == "echo TASK_SPECIFIC_SETUP"


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_explicit_scheduler_disable_is_preserved(tmp_path, profile):
    evidence, output = configured_task(tmp_path, profile, extra_params={"gpu_scheduler_enable": False})
    assert SENTINEL in output
    assert not evidence["before_script"]


@pytest.mark.parametrize("profile", ["docker", "singularity"])
@pytest.mark.parametrize("process", ["RUN_KODAMA_ANALYSIS", "RUN_KODAMA_CELL_AUXILIARY", "RUN_GIGATIME_KODAMA"])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_kodama_backend_and_alias_override_opposing_global_device(tmp_path, profile, process, backend):
    global_device = "cpu" if backend == "cuda" else "gpu"
    evidence, output = configured_task(tmp_path, profile, requested=global_device, device=global_device,
                                       process=process, extra_params={"kodama_backend": backend,
                                       "kodama_gpu_memory_gb": 7, "gpu_default_task_memory_gb": 2})
    assert SENTINEL in output
    assert_selection(evidence, profile, "gpu" if backend == "cuda" else "cpu")
    if backend == "cuda":
        assert "'7.0'" in evidence["before_script"]


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_auxiliary_uni2_alias_requests_uni2_memory_admission(tmp_path, profile):
    evidence, output = configured_task(tmp_path, profile, process="EXTRACT_UNI2_CELL_AUXILIARY",
                                       extra_params={"uni2_gpu_memory_gb": 11, "gpu_default_task_memory_gb": 2})
    assert SENTINEL in output
    assert_selection(evidence, profile, "gpu")
    assert "'11.0'" in evidence["before_script"]


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_malformed_actual_runtime_map_is_not_treated_as_missing_legacy_input(tmp_path, profile):
    evidence, output = configured_task(tmp_path, profile, device="not-resolved")
    assert evidence is None
    assert "Invalid concrete runtime_plan device" in output


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_auxiliary_medsam_alias_uses_medsam_gpu_memory_admission(tmp_path, profile):
    evidence, output = configured_task(tmp_path, profile, process="REFINE_CELL_AUXILIARY_MEDSAM",
                                       extra_params={"medsam_gpu_memory_gb": 12, "gpu_default_task_memory_gb": 2})
    assert SENTINEL in output
    assert_selection(evidence, profile, "gpu")
    assert "'12.0'" in evidence["before_script"]


def test_marker_kodama_declares_gpu_capable_for_explicit_gpu_backend():
    module = (ROOT / "modules/run_gigatime_kodama.nf").read_text()
    assert "label 'gpu_capable'" in module
    assert '--backend "${params.kodama_backend}"' in module


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_titan_absolute_model_bind_is_composed_with_gpu_options(tmp_path, profile):
    model = str(tmp_path / "model directory")
    evidence, output = configured_task(tmp_path, profile, process="EXTRACT_TITAN_SECTION_EMBEDDING",
                                       extra_params={"titan_model": model})
    assert SENTINEL in output
    assert_selection(evidence, profile, "gpu")
    bind = "-B" if profile == "singularity" else "-v"
    assert f"{bind} '{model}:{model}:ro'" in evidence["container_options"]


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_hovernet_site_options_are_composed_with_gpu_options(tmp_path, profile):
    extra = "--audit-hovernet-option preserved"
    evidence, output = configured_task(tmp_path, profile, process="RUN_HOVERNET_MONUSAC",
                                       extra_env={"CELLPHENOTYPER_HOVERNET_CONTAINER_OPTIONS": extra})
    assert SENTINEL in output
    assert_selection(evidence, profile, "gpu")
    assert extra in evidence["container_options"]


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_legacy_manual_image_mode_is_honored_for_gpu_labeled_task(tmp_path, profile):
    image = "example.invalid/explicit-manual-image:audit-only"
    image_param = "singularity_image" if profile == "singularity" else "docker_image"
    evidence, output = configured_task(tmp_path, profile, extra_params={
        "runtime_image_mode": "manual", image_param: image})
    assert SENTINEL in output
    assert evidence["container"] == image


@pytest.mark.parametrize("engine", ["docker", "singularity"])
@pytest.mark.parametrize("scheduler,visibility", [(True, None), (False, None), (False, "")])
def test_generated_wrapper_passes_host_lease_selection_to_model_free_engine(tmp_path, engine, scheduler, visibility):
    """Fake engine + fake lease only: no container, inference or lock acquisition."""
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    import sys
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    record_path = tmp_path / "engine_record.json"
    fake_engine = fake_bin / engine
    fake_engine.write_text(f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
if '--version' in sys.argv:
    print('singularity version 3.10.0')
elif ('run' if '{engine}' == 'docker' else 'exec') in sys.argv[1:]:
    Path({str(record_path)!r}).write_text(json.dumps({{
        'args': sys.argv[1:], 'executed_container': False,
        'environment': {{key: os.environ.get(key) for key in [
            'CUDA_VISIBLE_DEVICES', 'CELLPHENOTYPER_GPU_INDEX',
                'SINGULARITYENV_CUDA_VISIBLE_DEVICES', 'SINGULARITYENV_CELLPHENOTYPER_GPU_INDEX']}}
    }}))
""")
    fake_engine.chmod(0o755)
    lease_log = tmp_path / "fake_lease_argument.txt"
    fake_lease = fake_bin / "acquire_gpu_slot.sh"
    fake_lease.write_text(f"""# Model-free recorder: deliberately does not acquire locks or inspect GPUs.
cellphenotyper_acquire_gpu_slot() {{
  printf '%s\\n' "$1" > '{lease_log}'
  export CUDA_VISIBLE_DEVICES=GPU-22222222-2222-2222-2222-222222222222
  export CELLPHENOTYPER_GPU_INDEX=2
}}
""")
    (tmp_path / "main.nf").write_text("""nextflow.enable.dsl=2
process PREPARE_HIERARCHY_FEATURES {
  label 'gpu_capable'
  input:
  val(runtime_plan)
  output:
  val 'no_inference'
  script:
  'exit 0'
}
workflow { PREPARE_HIERARCHY_FEATURES([schema_version: 1, compute_device: 'cpu']) }
""")
    fake_image = tmp_path / "not_a_container.sif"
    fake_image.write_text("Not a container; model-free command wrapper fixture.\n")
    lock_root = tmp_path / "shared_host_gpu_locks"
    params = {"compute_device": "cpu", "host_arch": "amd64", "tissue_hierarchy_device": "cuda",
              "gpu_scheduler_enable": scheduler, "gpu_lock_dir": str(lock_root),
              "gpu_container_image": str(fake_image) if engine == "singularity" else "audit-only:not-a-container",
              "docker_extra_run_options": "--label audit-site-option=retained", "image_input": str(fake_image),
              "singularity_image_source": "docker", "singularity_gpu_image_source": "docker"}
    (tmp_path / "params.json").write_text(json.dumps(params))
    (tmp_path / "quiet.config").write_text("report.enabled=false\ntrace.enabled=false\ntimeline.enabled=false\ndag.enabled=false\n")
    env = dict(os.environ, NXF_OFFLINE="true", NXF_OPTS="-Xms64m -Xmx512m",
               PATH=str(fake_bin) + os.pathsep + os.environ["PATH"])
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    if visibility is not None:
        env["CUDA_VISIBLE_DEVICES"] = visibility
    result = subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"),
        "-c", str(ROOT / "nextflow.config"), "-c", str(tmp_path / "quiet.config"),
        "run", str(tmp_path / "main.nf"), "-profile", engine, "-params-file", str(tmp_path / "params.json"),
        "-ansi-log", "false", "-work-dir", str(tmp_path / "work")], cwd=tmp_path, env=env,
        text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    if scheduler:
        assert lease_log.read_text().strip() == str(lock_root)
    else:
        assert not lease_log.exists()
    assert not lock_root.exists(), "Fake lease must not create/acquire any GPU lock"
    record = json.loads(record_path.read_text())
    assert record["executed_container"] is False
    args = record["args"]
    selected_visibility = "GPU-22222222-2222-2222-2222-222222222222" if scheduler else visibility
    if engine == "docker":
        assert record["environment"]["CUDA_VISIBLE_DEVICES"] == selected_visibility
        assert record["environment"]["CELLPHENOTYPER_GPU_INDEX"] == ("2" if scheduler else None)
        assert "--gpus" in args and args[args.index("--gpus") + 1] == "all"
        assert "--env" in args and "CUDA_VISIBLE_DEVICES" in args and "CELLPHENOTYPER_GPU_INDEX" in args
        assert "audit-site-option=retained" in args
    else:
        assert "--nv" in args
        forwarded = record["environment"]["SINGULARITYENV_CUDA_VISIBLE_DEVICES"]
        explicit_env_args = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--env"]
        for value in explicit_env_args:
            if value.startswith("CUDA_VISIBLE_DEVICES="):
                forwarded = value.split("=", 1)[1]
        assert forwarded == selected_visibility
        assert record["environment"]["SINGULARITYENV_CELLPHENOTYPER_GPU_INDEX"] == ("2" if scheduler else None)
        assert "-B" in args, "Existing global source-directory binds must remain"
    wrappers = list((tmp_path / "work").glob("*/*/.command.run"))
    assert len(wrappers) == 1
    wrapper = wrappers[0].read_text()
    if scheduler:
        assert wrapper.index("cellphenotyper_acquire_gpu_slot") < wrapper.index("nxf_launch |"), "Host lease must precede container launch"
    else:
        assert "cellphenotyper_acquire_gpu_slot" not in wrapper


@pytest.mark.parametrize("profile", ["docker", "singularity"])
@pytest.mark.parametrize("process,parameter,selected", [
    ("REFINE_GROWN_TISSUE_MEDSAM", "medsam_device", "cpu"),
    ("REFINE_CELL_AUXILIARY_MEDSAM", "medsam_device", "cuda:0"),
    ("RUN_GRANDQC_ARTIFACT_ANALYSIS", "grandqc_device", "cpu"),
    ("RUN_GRANDQC_ARTIFACT_ANALYSIS", "grandqc_device", "cuda"),
])
def test_explicit_stage_device_override_controls_image_exposure_and_lease(tmp_path, profile, process, parameter, selected):
    expected = "gpu" if selected.startswith("cuda") else "cpu"
    opposite = "cpu" if expected == "gpu" else "gpu"
    evidence, output = configured_task(tmp_path, profile, requested=opposite, device=opposite,
                                       process=process, extra_params={parameter: selected})
    assert SENTINEL in output
    assert_selection(evidence, profile, expected)


@pytest.mark.parametrize("profile", ["docker", "singularity"])
@pytest.mark.parametrize("process,parameter,selected", [
    ("REFINE_CELL_AUXILIARY_MEDSAM", "medsam_device", "mps"),
    ("RUN_GRANDQC_ARTIFACT_ANALYSIS", "grandqc_device", "mps"),
    ("RUN_GIGATIME_KODAMA", "kodama_backend", "metal"),
])
def test_non_cuda_accelerators_fail_clearly_in_linux_container_profiles(tmp_path, profile, process, parameter, selected):
    evidence, output = configured_task(tmp_path, profile, process=process, extra_params={parameter: selected})
    assert evidence is None and "native-only" in output


@pytest.mark.parametrize("process,parameter,selected", [
    ("PREPARE_HIERARCHY_FEATURES", "tissue_hierarchy_device", "mps"),
    ("REFINE_GROWN_TISSUE_MEDSAM", "medsam_device", "mps"),
    ("RUN_GRANDQC_ARTIFACT_ANALYSIS", "grandqc_device", "mps"),
    ("RUN_GIGATIME_KODAMA", "kodama_backend", "metal"),
])
def test_native_non_cuda_accelerator_selection_does_not_acquire_cuda_lease(tmp_path, process, parameter, selected):
    evidence, output = configured_task(tmp_path, "local", process=process, extra_params={parameter: selected})
    assert SENTINEL in output
    assert evidence is not None and evidence["container"] is None
    assert "cellphenotyper_acquire_gpu_slot" not in evidence["before_script"]


@pytest.mark.parametrize("process,extra_params", [
    ("RUN_CELLVITPP", {"cellvit_gpu": 1}),
    ("RUN_HOVERNET_MONUSAC", {"hovernet_gpu": "1"}),
    ("EXTRACT_TITAN_SECTION_EMBEDDING", {"titan_gpu": 1}),
    ("RUN_GIGATIME_KODAMA", {"kodama_backend": "cuda", "kodama_gpu_device": 1}),
    ("REFINE_GROWN_TISSUE_MEDSAM", {"medsam_device": "cuda:1"}),
    ("REFINE_CELL_AUXILIARY_MEDSAM", {"medsam_device": "cuda:2"}),
])
def test_nonzero_gpu_ordinal_fails_before_dynamic_single_device_admission(tmp_path, process, extra_params):
    evidence, output = configured_task(tmp_path, "docker", process=process, extra_params=extra_params)
    assert evidence is None
    assert "dynamic GPU admission exposes one leased device as ordinal 0" in output


@pytest.mark.parametrize("profile", ["docker", "singularity"])
def test_manually_managed_gpu_ordinal_is_not_rewritten_when_scheduler_disabled(tmp_path, profile):
    evidence, output = configured_task(tmp_path, profile, process="REFINE_GROWN_TISSUE_MEDSAM", extra_params={
        "medsam_device": "cuda:2", "gpu_scheduler_enable": False})
    assert SENTINEL in output
    assert evidence is not None and "gpu" in evidence["container"]
    assert not evidence["before_script"]


def test_kodama_rejects_invalid_gpu_backend_spelling_instead_of_misrouting(tmp_path):
    evidence, output = configured_task(tmp_path, "docker", process="RUN_GIGATIME_KODAMA",
                                       extra_params={"kodama_backend": "gpu"})
    assert evidence is None and "kodama_backend must be cpu, cuda, or metal" in output


def test_native_cpu_stardist_cannot_autodetect_a_visible_cuda_gpu(tmp_path):
    evidence, output = configured_task(tmp_path, "local", requested="cpu", device="cpu",
                                       process="RUN_STARDIST_ROI_SEGMENTATION", explicit_plan=False)
    assert SENTINEL in output
    assert "export CUDA_VISIBLE_DEVICES=''" in evidence["before_script"]
    assert "cellphenotyper_acquire_gpu_slot" not in evidence["before_script"]


@pytest.mark.parametrize("profile", ["docker", "singularity"])
@pytest.mark.parametrize("arm_gpu", [False, True])
def test_stardist_explicit_gpu_plan_keeps_arm64_fallback_in_all_config_directives(tmp_path, profile, arm_gpu):
    manual = "example.invalid/compatible-arm64-gpu:audit-only"
    evidence, output = configured_task(tmp_path, profile, process="RUN_STARDIST_ROI_SEGMENTATION", arch="arm64",
        extra_params={"enable_stardist_gpu_on_arm64": arm_gpu, "gpu_container_image": manual})
    assert SENTINEL in output and evidence is not None
    cpu = "ghcr.io/tkcaccia/cellphenotyper:0.2.0"
    if profile == "singularity":
        cpu = "docker://" + cpu
    assert evidence["container"] == (manual if arm_gpu else cpu)
    assert ("cellphenotyper_acquire_gpu_slot" in (evidence["before_script"] or "")) == arm_gpu
    assert (("--nv" if profile == "singularity" else "--gpus all") in (evidence["container_options"] or "")) == arm_gpu
