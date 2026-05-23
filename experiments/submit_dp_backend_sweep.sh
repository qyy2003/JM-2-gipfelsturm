#!/bin/bash
#
# Submit the DP/DDP/FSDP comparison matrix.
#
# Usage:
#   ./experiments/submit_dp_backend_sweep.sh smoke
#   ./experiments/submit_dp_backend_sweep.sh good
#   ./experiments/submit_dp_backend_sweep.sh all
#
# Optional overrides:
#   BACKENDS="megatron ddp fsdp" GOOD_MODEL=1.5b GOOD_STEPS=50 GOOD_NODES=1 ./experiments/submit_dp_backend_sweep.sh good
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

    for backend in $BACKENDS; do
        echo "[submit] model=$model steps=$steps nodes=$nodes backend=$backend"
        ./launch.sh throughput "$model" "$steps" "$nodes" "$backend"
    done
}

case "$PHASE" in
    smoke)
        submit_matrix "${SMOKE_MODEL:-125m}" "${SMOKE_STEPS:-10}" "${SMOKE_NODES:-1}"
        ;;
    good)
        submit_matrix "${GOOD_MODEL:-1.5b}" "${GOOD_STEPS:-50}" "${GOOD_NODES:-1}"
        ;;
    all)
        submit_matrix "${SMOKE_MODEL:-125m}" "${SMOKE_STEPS:-10}" "${SMOKE_NODES:-1}"
        submit_matrix "${GOOD_MODEL:-1.5b}" "${GOOD_STEPS:-50}" "${GOOD_NODES:-1}"
        ;;
    *)
        echo "Unknown phase: $PHASE. Choose: smoke, good, all." >&2
        exit 1
        ;;
esac
