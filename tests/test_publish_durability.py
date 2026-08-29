from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_default_publish_mode_keeps_results_independent_of_work_cache() -> None:
    config = (ROOT / "nextflow.config").read_text(encoding="utf-8")
    parameters = (ROOT / "pipeline_paramers.yml").read_text(encoding="utf-8")

    assert "publish_dir_mode               = 'copy'" in config
    assert "publish_dir_mode: copy" in parameters
    assert "publish_dir_mode: rellink" not in parameters
