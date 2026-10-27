"""Test-only isolation for independent source-pinned Bora integrations."""

import pytest


@pytest.fixture
def isolated_bora_source_imports():
    # Independent source-pinned integrations must not reuse the contextual
    # package imported by a preceding test. Production still rejects a mixed
    # installation; isolate imports here instead of weakening that guard.
    import sys
    saved = {name: value for name, value in sys.modules.items()
             if name == "bora" or name.startswith("bora.")}
    saved_path = sys.path[:]
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "bora" or name.startswith("bora."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
