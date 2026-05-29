#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:05:00
#SBATCH --job-name=gipfel-throughput-125m-deepspeed-zero1-20s-1n-256gbs-16mbs-seq4096-bf16
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

echo "START TIME: $(date)"

WORKDIR=/users/course_00323/scratch/JM-2-gipfelsturm
LOG_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/throughput-125m-deepspeed-zero1-20s-1n-256gbs-16mbs-seq4096-bf16
mkdir -p "$LOG_DIR"

cd "$WORKDIR"
export PYTHONPATH=$WORKDIR:$PYTHONPATH
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
MASTER_ADDR=$(hostname)
MASTER_PORT=25679

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $WORKDIR/experiments/deepspeed_gpt_train.py \
    --model-size 125m \
    --zero-stage 1 \
    --train-iters 20 \
    --seq-len 4096 \
    --micro-batch-size 16 \
    --global-batch-size 256 \
    --wandb-project gipfelsturm \
    --wandb-exp-name throughput-125m-deepspeed-zero1-20s-1n-256gbs-16mbs-seq4096-bf16-$SLURM_JOB_ID \
    --wandb-save-dir $LOG_DIR"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
else
    echo "[$(date)] WANDB enabled."
fi

echo "CMD: $TRAINING_CMD"
set +e
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "cd $WORKDIR && numactl --membind=0-3 $TRAINING_CMD"
SRUN_RC=$?
set -e

echo "END TIME: $(date)"
echo "srun exit code: $SRUN_RC"
exit $SRUN_RC
