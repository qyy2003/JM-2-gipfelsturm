#!/bin/bash
#
# Submit the DP/DDP/FSDP comparison matrix.
#
# Usage:
#   ./experiments/submit_dp_backend_sweep.sh smoke
#   ./experiments/submit_dp_backend_sweep.sh debug
#   ./experiments/submit_dp_backend_sweep.sh good
#   ./experiments/submit_dp_backend_sweep.sh all
#
# Optional overrides:
#   BACKENDS="megatron ddp fsdp" GOOD_MODEL=1.5b GOOD_STEPS=200 GOOD_NODES=1 ./experiments/submit_dp_backend_sweep.sh good
#   FP8=false MBS_OVERRIDE=2 ./experiments/submit_dp_backend_sweep.sh smoke

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PHASE=${1:-smoke}
BACKENDS=${BACKENDS:-"megatron ddp fsdp"}

submit_matrix() {
    local model=$1
    local steps=$2
    local nodes=$3
    local walltime=$4

    for backend in $BACKENDS; do
        echo "[submit] model=$model steps=$steps nodes=$nodes walltime=$walltime backend=$backend"
        WALLTIME="$walltime" ./launch.sh throughput "$model" "$steps" "$nodes" "$backend"
    done
}

case "$PHASE" in
    smoke)
        submit_matrix "${SMOKE_MODEL:-125m}" "${SMOKE_STEPS:-20}" "${SMOKE_NODES:-1}" "${SMOKE_WALLTIME:-00:05:00}"
        ;;
    debug)
        submit_matrix "${DEBUG_MODEL:-125m}" "${DEBUG_STEPS:-50}" "${DEBUG_NODES:-1}" "${DEBUG_WALLTIME:-00:15:00}"
        ;;
    good)
        submit_matrix "${GOOD_MODEL:-1.5b}" "${GOOD_STEPS:-200}" "${GOOD_NODES:-1}" "${GOOD_WALLTIME:-00:30:00}"
        ;;
    all)
        submit_matrix "${SMOKE_MODEL:-125m}" "${SMOKE_STEPS:-20}" "${SMOKE_NODES:-1}" "${SMOKE_WALLTIME:-00:10:00}"
        submit_matrix "${DEBUG_MODEL:-125m}" "${DEBUG_STEPS:-50}" "${DEBUG_NODES:-1}" "${DEBUG_WALLTIME:-00:15:00}"
        submit_matrix "${GOOD_MODEL:-1.5b}" "${GOOD_STEPS:-200}" "${GOOD_NODES:-1}" "${GOOD_WALLTIME:-00:30:00}"
        ;;
    *)
        echo "Unknown phase: $PHASE. Choose: smoke, debug, good, all." >&2
        exit 1
        ;;
esac
