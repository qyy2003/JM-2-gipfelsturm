#!/bin/bash
#
# Submit DeepSpeed ZeRO comparison runs.
#
# Usage:
#   ./experiments/submit_deepspeed_sweep.sh smoke
#   ./experiments/submit_deepspeed_sweep.sh debug
#   ./experiments/submit_deepspeed_sweep.sh good
#   ./experiments/submit_deepspeed_sweep.sh all
#
# Optional overrides:
#   ZERO_STAGES="1 2 3" GOOD_MODEL=1.5b GOOD_STEPS=200 ./experiments/submit_deepspeed_sweep.sh good
#   MBS_OVERRIDE=2 SEQ_LEN=2048 ./experiments/submit_deepspeed_sweep.sh smoke

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/config.sh"

PHASE=${1:-smoke}
ZERO_STAGES=${ZERO_STAGES:-"1 2 3"}

model_default_mbs() {
    case "$1" in
        125m) echo 16 ;;
        350m) echo 8 ;;
        760m) echo 4 ;;
        1.5b) echo 4 ;;
        3b) echo 4 ;;
        8b) echo 2 ;;
        *) echo "Unknown model size: $1" >&2; exit 1 ;;
    esac
}

submit_one() {
    local model=$1
    local steps=$2
    local nodes=$3
    local walltime=$4
    local zero_stage=$5

    local mbs=${MBS_OVERRIDE:-$(model_default_mbs "$model")}
    local gbs=${GBS:-256}
    local seq_len=${SEQ_LEN:-4096}
    local exp_name="throughput-${model}-deepspeed-zero${zero_stage}-${steps}s-${nodes}n-${gbs}gbs-${mbs}mbs-seq${seq_len}-bf16"
    local job_name="gipfel-${exp_name}"
    local script="logs/${job_name}.sbatch"

    mkdir -p logs

    cat > "$script" << HEADER
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${walltime}
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

echo "START TIME: \$(date)"

WORKDIR=${WORKDIR}
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/${exp_name}
mkdir -p "\$LOG_DIR"

cd "\$WORKDIR"
export PYTHONPATH=\$WORKDIR:\$PYTHONPATH
export OMP_NUM_THREADS=\$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
MASTER_ADDR=\$(hostname)
MASTER_PORT=25679

TORCHRUN_ARGS=(
    --nproc-per-node \$SLURM_GPUS_PER_NODE
    --nnodes \$SLURM_NNODES
    --rdzv_endpoint \$MASTER_ADDR:\$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun \${TORCHRUN_ARGS[@]} \$WORKDIR/experiments/deepspeed_gpt_train.py \\
    --model-size ${model} \\
    --zero-stage ${zero_stage} \\
    --train-iters ${steps} \\
    --seq-len ${seq_len} \\
    --micro-batch-size ${mbs} \\
    --global-batch-size ${gbs} \\
    --wandb-project gipfelsturm \\
    --wandb-exp-name ${exp_name}-\$SLURM_JOB_ID \\
    --wandb-save-dir \$LOG_DIR"

if [ -z "\${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
    echo "[\$(date)] WANDB disabled."
else
    echo "[\$(date)] WANDB enabled."
fi

echo "CMD: \$TRAINING_CMD"
set +e
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task \$SLURM_CPUS_PER_TASK --wait 60 bash -c "cd \$WORKDIR && numactl --membind=0-3 \$TRAINING_CMD"
SRUN_RC=\$?
set -e

echo "END TIME: \$(date)"
echo "srun exit code: \$SRUN_RC"
exit \$SRUN_RC
HEADER

    chmod +x "$script"
    echo "[submit] model=$model zero=$zero_stage steps=$steps nodes=$nodes walltime=$walltime mbs=$mbs gbs=$gbs seq_len=$seq_len"
    sbatch "$script"
}

submit_matrix() {
    local model=$1
    local steps=$2
    local nodes=$3
    local walltime=$4

    for zero_stage in $ZERO_STAGES; do
        submit_one "$model" "$steps" "$nodes" "$walltime" "$zero_stage"
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
        submit_matrix "${SMOKE_MODEL:-125m}" "${SMOKE_STEPS:-20}" "${SMOKE_NODES:-1}" "${SMOKE_WALLTIME:-00:05:00}"
        submit_matrix "${DEBUG_MODEL:-125m}" "${DEBUG_STEPS:-50}" "${DEBUG_NODES:-1}" "${DEBUG_WALLTIME:-00:15:00}"
        submit_matrix "${GOOD_MODEL:-1.5b}" "${GOOD_STEPS:-200}" "${GOOD_NODES:-1}" "${GOOD_WALLTIME:-00:30:00}"
        ;;
    *)
        echo "Unknown phase: $PHASE. Choose: smoke, debug, good, all." >&2
        exit 1
        ;;
esac
