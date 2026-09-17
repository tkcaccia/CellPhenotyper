#!/usr/bin/env bash
# Source inside an atlas task before invoking Python. Nextflow's byte-content
# cache key cannot stop Python from accepting an old timestamp/size-based .pyc.
# A fresh private prefix prevents reads of those shared caches; disabling writes
# also avoids duplicating installed-library bytecode in every task directory.
# Do not install/replace traps or delete shared/user caches. The empty directory
# deliberately remains in this task's work directory.
# The non-PYTHON marker lets isolated -E wrappers distinguish this exact task
# contract from unrelated user Python settings and forward explicit CLI flags.

if ! cellphenotyper_source_pycache_dir="$(mktemp -d "${PWD}/.cellphenotyper-pycache.XXXXXX")"; then
    printf '%s\n' '[ERROR] Cannot create the task-private Python bytecode-cache prefix.' >&2
    return 1 2>/dev/null || exit 1
fi
if [[ -z "${cellphenotyper_source_pycache_dir}" || ! -d "${cellphenotyper_source_pycache_dir}" ]]; then
    printf '%s\n' '[ERROR] mktemp did not create the task-private Python bytecode-cache directory.' >&2
    return 1 2>/dev/null || exit 1
fi
export PYTHONPYCACHEPREFIX="${cellphenotyper_source_pycache_dir}"
export PYTHONDONTWRITEBYTECODE=1
export CELLPHENOTYPER_SOURCE_PYCACHEPREFIX="${cellphenotyper_source_pycache_dir}"
unset cellphenotyper_source_pycache_dir
