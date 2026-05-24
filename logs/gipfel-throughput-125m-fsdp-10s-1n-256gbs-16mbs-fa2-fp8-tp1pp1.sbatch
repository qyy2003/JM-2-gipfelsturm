#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:30:00
#SBATCH --job-name=gipfel-throughput-125m-fsdp-10s-1n-256gbs-16mbs-fa2-fp8-tp1pp1
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue

echo "START TIME: \$(date)"

################ Configs ################
WORKDIR=/users/course_00323/scratch/JM-2-gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache

# Training config
MBS=16
GBS=256
SEQ_LEN=4096
TRAINING_STEPS=10
MODEL_SIZE=125m
TP=1
PP=1
DP_BACKEND=fsdp
ATTENTION_BACKEND=flash
FP8=true
CKPT_FORMAT=fsdp_dtensor

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=throughput-125m-fsdp-10s-1n-256gbs-16mbs-fa2-fp8-tp1pp1
LOG_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/$EXP_NAME
TENSORBOARD_DIR=$LOG_DIR/tensorboard
CKPT_DIR=$LOG_DIR/checkpoints
CORES_DIR=$LOG_DIR/cores

# Resubmit knobs (consumed by the resubmit footer below).
RESUBMIT=false
MAX_RESUBMITS=0
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
echo "RESUBMIT_COUNT=$RESUBMIT_COUNT (max=$MAX_RESUBMITS)"

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

try:
    import deepspeed
    print(f'[probe][deepspeed] v{deepspeed.__version__}')
except Exception as e:
    print(f'[probe][deepspeed] not found: {e}')
PROBE_EOF


TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --attention-backend flash
)
NETWORK_SIZE_ARGS=(
    --num-layers 12
    --hidden-size 768
    --ffn-hidden-size 2048
    --num-attention-heads 12
    --group-query-attention
    --num-query-groups 4
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length $SEQ_LEN
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAINING_STEPS
    --log-interval 1
    --eval-interval 10
    --eval-iters 0
    # --cross-entropy-loss-fusion-impl te
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
    --lr-warmup-iters 10
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
    --fp8-format hybrid
)

COMMON_DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
)

case "$DP_BACKEND" in
    megatron)
        DISTRIBUTED_ARGS=(
            "${COMMON_DISTRIBUTED_ARGS[@]}"
            --use-distributed-optimizer
            --overlap-grad-reduce
            --overlap-param-gather
        )
        ;;
    ddp)
        DISTRIBUTED_ARGS=(
            "${COMMON_DISTRIBUTED_ARGS[@]}"
        )
        ;;
    fsdp)
        DISTRIBUTED_ARGS=(
            "${COMMON_DISTRIBUTED_ARGS[@]}"
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads_params
        )
        ;;
    *)
        echo "Unsupported DP_BACKEND=$DP_BACKEND" >&2
        exit 2
        ;;
esac

echo "Experiment backend: $DP_BACKEND"
echo "Experiment shape: model=$MODEL_SIZE steps=$TRAINING_STEPS nodes=$SLURM_NNODES gpus_per_node=$SLURM_GPUS_PER_NODE mbs=$MBS gbs=$GBS seq_len=$SEQ_LEN tp=$TP pp=$PP fp8=$FP8 attention=$ATTENTION_BACKEND"

LOGGING_ARGS=(
    --log-throughput
    --log-progress

    --log-timers-to-tensorboard
)
SAVE_INTERVAL=0
EXIT_DURATION_MINS=28
CHECKPOINT_ARGS=()
if [ "$SAVE_INTERVAL" -gt 0 ]; then
    CHECKPOINT_ARGS=(
        --save "$CKPT_DIR"
        --load "$CKPT_DIR"
        --save-interval "$SAVE_INTERVAL"
        --ckpt-format "$CKPT_FORMAT"
        --exit-duration-in-mins "$EXIT_DURATION_MINS"
    )
fi

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
fi

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
