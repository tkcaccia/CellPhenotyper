#!/bin/bash
# Build-time operation only; never called by a pipeline task.
set -euo pipefail
atlas_source="${1:-/opt/cellphenotyper}"
atlas_prefix=/opt/micromamba/envs/cellphenotyper-atlas
native_prefix=/opt/micromamba/envs/kodama-r
native_rscript="$native_prefix/bin/Rscript"
if [[ ! -f "$atlas_source/requirements-atlas.txt" || ! -f "$atlas_source/docker/verify_atlas_runtime.py" ]]; then
    echo 'Atlas requirements and verifier must be copied into the build context first.' >&2
    exit 66
fi
if [[ -e "$atlas_prefix" ]]; then
    echo "Refusing to replace existing runtime $atlas_prefix; use a base without the atlas overlay." >&2
    exit 73
fi
if [[ ! -f "$native_rscript" || ! -x "$native_rscript" ]]; then
    echo "The atlas overlay requires the existing native KODAMA runtime at $native_rscript; it will not install or replace KODAMA." >&2
    exit 78
fi
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
export MAMBA_ROOT_PREFIX=/opt/micromamba
micromamba create -y --prefix "$atlas_prefix" -c conda-forge --strict-channel-priority python=3.12 pip
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$atlas_prefix/bin/python" -E -s -m pip install \
    --no-cache-dir --only-binary=:all: -r "$atlas_source/requirements-atlas.txt"
# Add only absent graph-consumer dependencies in the native environment. The
# native KODAMA package is deliberately neither installed nor upgraded here.
# Existing incompatible dependencies fail the functional check, not silently
# replace the requested native runtime. sha256sum is supplied by Linux coreutils.
env -u LD_PRELOAD -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH -u R_HOME -u R_LIBS \
    LD_LIBRARY_PATH="$native_prefix/lib" R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null \
    R_LIBS_USER="$native_prefix/lib/R/library" R_LIBS_SITE="$native_prefix/lib/R/library" \
    "$native_rscript" --vanilla - <<'RSCRIPT'
stopifnot(getRversion() >= "4.6.0")
if (!identical(normalizePath(.Library), normalizePath(Sys.getenv("R_LIBS_USER"))))
  stop("Selected Rscript does not use the expected native library")
root <- normalizePath(find.package("KODAMA"), mustWork=TRUE)
if (!identical(root, normalizePath(file.path(.Library, "KODAMA"), mustWork=TRUE)))
  stop("The selected native environment does not own the loaded KODAMA package")
identity <- function() {
  files <- sort(list.files(root, recursive=TRUE, full.names=TRUE, all.files=TRUE))
  files <- files[!dir.exists(files)]
  if (!nzchar(Sys.which("sha256sum"))) stop("Linux build requires sha256sum")
  result <- system2("sha256sum", shQuote(files), stdout=TRUE)
  if (!is.null(attr(result,"status")) || length(result) != length(files))
    stop("Cannot bind existing native KODAMA bytes")
  result
}
before <- identity()
if (!all(c("KODAMA.matrix","KODAMA.graph.materialize") %in% getNamespaceExports("KODAMA")))
  stop("Existing KODAMA lacks the required native graph handle API")
required <- c("Matrix","igraph","digest","jsonlite")
installed <- rownames(installed.packages(lib.loc=.Library))
missing <- setdiff(required, installed)
if (length(missing)) install.packages(missing, lib=.Library, dependencies=NA,
  repos="https://cloud.r-project.org", Ncpus=1L)
if (!identical(before, identity())) stop("KODAMA package bytes changed while adding graph dependencies")
for (package in required) if (!requireNamespace(package, quietly=TRUE))
  stop("Missing or incompatible native graph dependency: ", package)
cat("Existing KODAMA bytes preserved; native graph dependencies available\n")
RSCRIPT
install -m 0755 "$atlas_source/docker/cellphenotyper-atlas-python" /usr/local/bin/cellphenotyper-atlas-python
install -m 0755 "$atlas_source/docker/cellphenotyper-hierarchy-python" /usr/local/bin/cellphenotyper-hierarchy-python
/usr/local/bin/cellphenotyper-atlas-python -m pip check
/usr/local/bin/cellphenotyper-atlas-python "$atlas_source/docker/verify_atlas_runtime.py" \
    --source-root "$atlas_source" --requirements "$atlas_source/requirements-atlas.txt" --strict-isolation \
    --report "$atlas_source/atlas_runtime_build.json"
/usr/local/bin/cellphenotyper-atlas-python "$atlas_source/docker/verify_atlas_runtime.py" \
    --component hierarchy-native --source-root "$atlas_source" --strict-isolation \
    --native-rscript "$native_rscript" --native-r-library "$native_prefix/lib/R/library" \
    --report "$atlas_source/atlas_native_hierarchy_runtime_build.json"
model_python="${CELLPHENOTYPER_ML_PYTHON:-/opt/micromamba/envs/stardist/bin/python}"
"$model_python" -E -s "$atlas_source/docker/verify_atlas_runtime.py" \
    --component model --source-root "$atlas_source" --report "$atlas_source/atlas_model_runtime_build.json"
