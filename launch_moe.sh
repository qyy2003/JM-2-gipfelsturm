#!/bin/bash
#
# Usage: ./launch_moe.sh <mode> [steps] [nodes]
#
# Modes:     throughput  (50 steps, no logging)
#            train       (N steps, with W&B, Tensorboard, and auto-resubmit)
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 8 (32 GPUs total, EP=8, DP=4)
#
# Model:     8B MoE — 32 layers, hidden=4096, 32 heads, 8 experts (top-2),
#            per-expert FFN=7168, activated params ~= 8B dense
#
# Examples:  ./launch_moe.sh throughput
#            ./launch_moe.sh throughput 50 1
#            ./launch_moe.sh train 5000
#            ./launch_moe.sh train 3000 8

set -euo pipefail

MODE=${1:?Usage: ./launch_moe.sh <mode> [steps] [nodes]}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${2:-50}
        NODES=${3:-8}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=false
        SAVE_BLOCK=""
        ;;
    train)
        TRAINING_STEPS=${2:?Usage: ./launch_moe.sh train <steps> [nodes]}
        NODES=${3:-8}
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        # Megatron native checkpoint + graceful exit on SIGTERM
        SAVE_BLOCK="
    --save \$CHECKPOINT_DIR
    --load \$CHECKPOINT_DIR
    --save-interval 500
    --exit-signal-handler"
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

# Fixed 8B MoE architecture
NUM_LAYERS=32
HIDDEN=4096
FFN=14336      # base FFN (attention projections use hidden size; MoE layers use moe-ffn-hidden-size)
HEADS=32
KV_HEADS=8
MBS=2
GBS=256
SEQ_LEN=4096

# MoE: 8 experts, top-2 routing, per-expert FFN=7168 (half of dense)
# Total params ~22B; activated params per token ~= 8B dense
NUM_EXPERTS=8
MOE_TOP_K=2
MOE_FFN_HIDDEN=7168
EXPERT_MP=8    # EP=8 per group; 32 GPUs / EP=8 = DP=4

JOB_NAME="gipfel-${MODE}-8b-moe-${TRAINING_STEPS}s-${NODES}n"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=lsaie-ss26
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY'

echo "START TIME: $(date)"

################ Paths ################
WORKDIR=/users/course_00270/scratch/project/lsaie-ss26-gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache
BODY

cat >> "$SCRIPT" << CONFIGS

MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-8b-moe-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CHECKPOINT_DIR=\$LOG_DIR/checkpoints
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR $CHECKPOINT_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-router-topk ${MOE_TOP_K}
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN}
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --expert-model-parallel-size ${EXPERT_MP}
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
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
    --lr 3e-4
    --lr-decay-style cosine
    --min-lr 3e-5
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)

SAVE_ARGS=(${SAVE_BLOCK}
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

# EP=8, TP=1, PP=1 → DP=32/8=4 across 8 nodes (32 GPUs)
DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${SAVE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

EXIT_CODE=$?

# Auto-resubmit: Megatron's --exit-signal-handler catches SIGTERM (sent by SLURM
# 120s before wall-time), saves a checkpoint, and exits 0. We resubmit if the
# saved iteration is less than the target training steps.
if [ $EXIT_CODE -eq 0 ] && [ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    SAVED_ITER=$(cat "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
    if [ "${SAVED_ITER}" -lt "${TRAINING_STEPS}" ] 2>/dev/null; then
        echo "[$(date)] Graceful exit at step ${SAVED_ITER}/${TRAINING_STEPS}. Resubmitting..."
        sbatch "$0"
    else
        echo "[$(date)] Training complete at step ${SAVED_ITER}."
    fi
fi

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
