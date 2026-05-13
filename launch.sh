#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes]
#
# Modes:     throughput  (50 steps, with W&B)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 8b 50 1
#            ./launch.sh train 760m 5000
#            ./launch.sh train 1.5b 3000 8

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}

# HH:MM:SS -> minutes (rounds seconds up).
hms_to_mins() {
    local h m s
    IFS=: read -r h m s <<< "$1"
    echo $((10#$h * 60 + 10#$m + (10#$s + 59) / 60))
}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA="
    --log-timers-to-tensorboard"
        WANDB=true
        SAVE_INTERVAL=0
        RESUBMIT=false
        MAX_RESUBMITS=0
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${4:-4}
        TIME=00:30:00
        # TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        SAVE_INTERVAL=500
        RESUBMIT=false
        MAX_RESUBMITS=10
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

# Megatron exits cleanly ~5 min before SLURM kills the job, leaving room
# to write the final checkpoint and let the script resubmit.
# EXIT_DURATION_MINS=$(( $(hms_to_mins "$TIME") - 5 ))
EXIT_DURATION_MINS=$(( $(hms_to_mins "$TIME") - 2 ))

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

GBS=128 #256
SEQ_LEN=4096
ATTENTION_BACKEND=flash

# Tag attention backend in EXP_NAME. For "flash", distinguish FA3 (installed via
# build_fa3.sbatch) from the container's bundled FA2 by checking the install dir.
FA3_PREFIX="/iopsstor/scratch/cscs/$USER/gipfelsturm/fa3"
if [ "$ATTENTION_BACKEND" = "flash" ] && [ -d "$FA3_PREFIX/flash_attn_3" ]; then
    ATTN_TAG="fa3"
elif [ "$ATTENTION_BACKEND" = "flash" ]; then
    ATTN_TAG="fa2"
else
    ATTN_TAG="$ATTENTION_BACKEND"
fi

EXP_NAME="${MODE}-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n-${GBS}gbs-${MBS}mbs-${ATTN_TAG}"
JOB_NAME="gipfel-${EXP_NAME}"

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
#SBATCH --account=${SBATCH_ACCOUNT}
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
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: \$(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${EXP_NAME}
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CKPT_DIR=\$LOG_DIR/checkpoints
CORES_DIR=\$LOG_DIR/cores

# Resubmit knobs (consumed by the resubmit footer below).
RESUBMIT=${RESUBMIT}
MAX_RESUBMITS=${MAX_RESUBMITS}
RESUBMIT_COUNT=\${RESUBMIT_COUNT:-0}
echo "RESUBMIT_COUNT=\$RESUBMIT_COUNT (max=\$MAX_RESUBMITS)"
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $CKPT_DIR $CORES_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
# core_pattern on this cluster is a bare 'core_%h_%p' (no path), so dumps
# land in the crashing process's cwd. The cgroup/container blocks the actual
# write, leaving 0-byte litter scattered through the workdir; suppress instead.
ulimit -c 0
export PYTHONPATH=$MEGATRON_LM_DIR:/iopsstor/scratch/cscs/$USER/gipfelsturm/fa3:$PYTHONPATH
# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
# A crashed prior run can leave Triton cache entries with no 'cubin' artifact,
# which breaks the next compile with KeyError: "Unknown key: 'cubin'". Wipe on
# fresh chain start; chained resubmits exit cleanly so their cache is safe.
if [ "${RESUBMIT_COUNT:-0}" -eq 0 ]; then
    echo "[cache] fresh chain — wiping Triton/Inductor caches"
    rm -rf "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
fi
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

# Write a probe script that runs inside the main srun's container (rank 0 only).
# Cannot use a separate srun step here — pyxis/enroot rejects spawning a second
# container in the same job step.
PROBE_SCRIPT=$LOG_DIR/probe_versions.py
cat > $PROBE_SCRIPT << 'PROBE_EOF'
try:
    import importlib.metadata as md
except ImportError:
    md = None

try:
    import transformer_engine
    print(f'[probe][te] TransformerEngine: v{transformer_engine.__version__}')
except Exception as e:
    print(f'[probe][te] not found: {e}')

found = False
if md is not None:
    for pkg in ['flash-attn', 'flash-attn-3', 'flashattn-hopper']:
        try:
            print(f'[probe][attn] {pkg}: v{md.version(pkg)}')
            found = True
        except md.PackageNotFoundError:
            pass

for mod in ['flash_attn_3', 'flashattn_hopper', 'flash_attn']:
    try:
        m = __import__(mod)
        print(f'[probe][attn] import {mod}: v{getattr(m, "__version__", "unknown")}')
        found = True
    except ImportError:
        pass

if not found:
    print('[probe][attn] No FlashAttention python package found')
PROBE_EOF


SETUP

cat >> "$SCRIPT" << TRANSFORMER_ENGINE_BLOCK
TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --attention-backend ${ATTENTION_BACKEND}
)
TRANSFORMER_ENGINE_BLOCK

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
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
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

cat >> "$SCRIPT" << CHECKPOINT_ARGS
SAVE_INTERVAL=${SAVE_INTERVAL}
EXIT_DURATION_MINS=${EXIT_DURATION_MINS}
CHECKPOINT_ARGS

cat >> "$SCRIPT" << 'CHECKPOINT_ARGS_BODY'
CHECKPOINT_ARGS=()
if [ "$SAVE_INTERVAL" -gt 0 ]; then
    CHECKPOINT_ARGS=(
        --save "$CKPT_DIR"
        --load "$CKPT_DIR"
        --save-interval "$SAVE_INTERVAL"
        --ckpt-format torch_dist
        --exit-duration-in-mins "$EXIT_DURATION_MINS"
    )
fi
CHECKPOINT_ARGS_BODY

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
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "cd $CORES_DIR && if [ \$SLURM_PROCID -eq 0 ]; then python3 $PROBE_SCRIPT || echo '[probe] failed (continuing)'; fi && numactl --membind=0-3 $TRAINING_CMD"
SRUN_RC=$?

echo "END TIME: $(date)"
echo "srun exit code: $SRUN_RC"

# Auto-resubmit until TRAINING_STEPS reached or MAX_RESUBMITS hit.
# Only resubmits on clean Megatron exit (rc=0); a crash leaves the chain
# broken so a real failure does not trigger a runaway loop.
if [ "$RESUBMIT" = "true" ] && [ "$SRUN_RC" -eq 0 ]; then
    LATEST=0
    if [ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
        LATEST=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt")
    fi
    NEXT_COUNT=$((RESUBMIT_COUNT + 1))
    echo "[resubmit] latest_iter=$LATEST target=$TRAINING_STEPS attempt=$NEXT_COUNT/$MAX_RESUBMITS"
    if [ "$LATEST" -ge "$TRAINING_STEPS" ]; then
        echo "[resubmit] training complete, not resubmitting"
    elif [ "$NEXT_COUNT" -gt "$MAX_RESUBMITS" ]; then
        echo "[resubmit] hit MAX_RESUBMITS=$MAX_RESUBMITS, not resubmitting"
    elif [ "$LATEST" -eq 0 ]; then
        # Megatron exited cleanly without saving any checkpoint. Resubmitting
        # would just rerun from scratch and likely repeat the same exit.
        echo "[resubmit] no checkpoint written this run, not resubmitting (manual retry needed)"
    else
        echo "[resubmit] sbatch $0 with RESUBMIT_COUNT=$NEXT_COUNT"
        # --chdir pins WorkDir to $WORKDIR so #SBATCH --output=logs/%x-%j.log
        # resolves under the project tree, not wherever cwd happens to be when
        # this footer runs (the script does `cd $MEGATRON_LM_DIR` earlier).
        sbatch --chdir="$WORKDIR" --export=ALL,RESUBMIT_COUNT=$NEXT_COUNT "$0"
    fi
fi

exit $SRUN_RC
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
