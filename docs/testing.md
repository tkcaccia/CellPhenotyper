# Local validation

Install the pinned CPU test environment and run:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q bin tests
python bin/validate_pipeline_params.py --params pipeline_paramers.yml
python -m pytest -q -rs
```

The tests do not run model inference or submit a remote campaign. Nextflow
checks use bounded local synthetic or stub processes. Install Nextflow to run
those checks; otherwise their skip reasons are reported. R- and model-specific
checks similarly report unavailable optional runtimes.

## Optional linked Bora integration

The contextual full-slide integration test accepts `BORA_SOURCE_DIR`, pointing
to a Bora checkout containing `src/bora/kodama_bora.py`. The conventional
fallback is a sibling `Bora` checkout. Install that checkout's CPU dependencies
before running its linked checks.

The older continuous and expert-semantic diagnostic adapters have separate
source contracts. Their integration tests accept `BORA_CONTINUOUS_SOURCE_DIR`
and `BORA_SEMANTIC_SOURCE_ROOT` respectively; the latter points to a checkout's
`src` directory. These historical modules are not supplied by the accepted
contextual Bora v0.1 package. Missing optional source is reported as a skip,
not as evidence of a passing integration. When supplied, the tests verify
source identities, tamper rejection and incomplete-run/export behavior.

The detector-choice checks in `tests/test_cell_detection_choice.py` exercise the
actual Nextflow settings resolver, schema, production module parsing and stub
routing. They assert that the chosen CellViT++ or StarDist engine runs alone,
that invalid combinations stop before submission, and that CPU-only execution
cannot silently substitute a different engine.
