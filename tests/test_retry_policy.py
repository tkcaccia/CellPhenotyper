import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETRY_LABELS = {"io_heavy", "compute_medium", "compute_heavy"}


def test_every_process_module_has_retry_policy_or_explicit_terminal_lifecycle() -> None:
    missing = []
    for module_path in sorted((ROOT / "modules").glob("*.nf")):
        source = module_path.read_text(encoding="utf-8")
        processes = re.findall(r"^process\s+(\w+)\s*\{", source, flags=re.MULTILINE)
        labels = set(re.findall(r"^\s*label\s+'([^']+)'", source, flags=re.MULTILINE))
        if processes and labels.isdisjoint(RETRY_LABELS):
            # Full-slide GPU lifecycle failures must never be retried by the
            # workflow engine; the campaign owns its versioned recovery.
            terminal = "errorStrategy 'terminate'" in source and re.search(r"maxRetries\s+0\b", source)
            if not terminal:
                missing.extend(f"{module_path.name}:{process}" for process in processes)

    assert not missing, f"Processes without a retry policy label: {', '.join(missing)}"
