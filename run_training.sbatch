#!/bin/bash
#SBATCH --job-name=megatron-gpt
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
 
set -eo pipefail

mkdir -p logs

# ── Edit these ───────────────────────────────────────────────────────────────
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_LM_DIR="$WORKDIR/Megatron-LM"
CHECKPOINT_DIR="/iopsstor/scratch/cscs/$USER/checkpoints/megatron-gpt-8b"
TENSORBOARD_DIR="/iopsstor/scratch/cscs/$USER/logs/megatron-gpt-8b/tensorboard"
DATASET_CACHE_DIR="/iopsstor/scratch/cscs/$USER/cache/megatron"
VOCAB_FILE="$WORKDIR/data/gpt2-vocab.json"
MERGE_FILE="$WORKDIR/data/gpt2-merges.txt"
DATA_PREFIX="/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small"
MAX_STEPS=5000
# ─────────────────────────────────────────────────────────────────────────────

echo "START TIME: $(date)"
echo "[sbatch] job=${SLURM_JOB_ID}  nodes=${SLURM_NODELIST}"

mkdir -p "$CHECKPOINT_DIR" "$TENSORBOARD_DIR" "$DATASET_CACHE_DIR"

# Apply patches to Megatron-LM (idempotent via flock)
flock "$MEGATRON_LM_DIR/.git-lock" bash -c \
    "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export PYTHONPATH="$MEGATRON_LM_DIR:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="/iopsstor/scratch/cscs/$USER/cache/.triton"
export TORCHINDUCTOR_CACHE_DIR="/iopsstor/scratch/cscs/$USER/cache/.inductor"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE))

# ── Model: GPT-8B ──────────────────────────────────────────────────────────
NETWORK_SIZE_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 4
    --max-position-embeddings 4096
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length 4096
)

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 256
    --train-iters $MAX_STEPS
    --log-interval 1
    --eval-interval 1000
    --eval-iters 10
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
    --save "$CHECKPOINT_DIR"
    --load "$CHECKPOINT_DIR"
    --save-interval 500
    --recompute-activations

)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 1e-4
    --lr-decay-style cosine
    --lr-warmup-iters 200
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)


LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --tensorboard-dir "$TENSORBOARD_DIR"
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --wandb-project "lsaie-ss26-gipfelsturm"
    --wandb-exp-name "megatron-gpt-8b"
)

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file "$VOCAB_FILE"
    --merge-file "$MERGE_FILE"
)

DATA_ARGS=(
    --data-path "$DATA_PREFIX"
    --data-cache-path "$DATASET_CACHE_DIR"
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT"
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

echo "CMD: $TRAINING_CMD"

srun -lu --mpi=pmix --network=disable_rdzv_get \
    --environment="$WORKDIR/alps3.toml" \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 \
    bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
