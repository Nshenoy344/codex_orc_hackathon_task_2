#!/usr/bin/env bash
# Local child-process limits only: no kernel, NIC, scheduler or robot settings.
lego_thread_limit="${LEGO_ANALYSIS_THREADS:-1}"
if ! [[ "$lego_thread_limit" =~ ^[1-4]$ ]]; then
    echo 'LEGO_ANALYSIS_THREADS must be an integer from1 through4 (default1).' >&2
    return 2 2>/dev/null || exit 2
fi
export LEGO_ANALYSIS_THREADS="$lego_thread_limit"
export OPENBLAS_NUM_THREADS="$lego_thread_limit"
export OMP_NUM_THREADS="$lego_thread_limit"
export MKL_NUM_THREADS="$lego_thread_limit"
export NUMEXPR_NUM_THREADS="$lego_thread_limit"
export BLIS_NUM_THREADS="$lego_thread_limit"
export VECLIB_MAXIMUM_THREADS="$lego_thread_limit"
export OMP_WAIT_POLICY=PASSIVE
