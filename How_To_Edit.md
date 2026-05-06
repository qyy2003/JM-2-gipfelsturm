# How to Edit Megatron-LM

`Megatron-LM/` is reset and re-patched on every job ([launch.sh:164](launch.sh#L164)).
Direct edits are wiped — patches are the canonical home for changes.

Two helper scripts at the project root:

- `./code_restore.sh` — clean + apply all patches
- `./code_store.sh`   — save WIP as a patch

## Workflow

```bash
./code_restore.sh                       # set up Megatron-LM
cd Megatron-LM && <edit> && cd ..
./code_store.sh                         # checkpoint -> patches/1000-cache.patch
./launch.sh throughput 760m 50 4        # run job
./code_store.sh -N my-feature           # done iterating: promote -> 0NNN-my-feature.patch
git add patches/0NNN-my-feature.patch && git commit && git push
```

## Patch slots

| File                       | Purpose                       | In git? |
|----------------------------|-------------------------------|---------|
| `patches/1000-cache.patch` | private WIP, overwritten      | no      |
| `patches/0NNN-name.patch`  | shared, named, reviewed       | yes     |

## Script options

```bash
./code_restore.sh           # refuses if dirty
./code_restore.sh -f        # nuke dirty edits, then apply

./code_store.sh             # save WIP -> 1000-cache.patch
./code_store.sh -N name     # promote cache -> next free 0NNN-name.patch
```

## Don'ts

- Don't `git diff > patches/…` by hand — `code_store.sh` subtracts other patches' content for you.
- Don't edit `1000-cache.patch` directly — restore, edit source, re-store.
- Don't commit the cache — already gitignored.

## Where to apply techniques

### Challenge 2 — throughput (mostly flags in launch.sh)

| Technique | Where | Change |
|---|---|---|
| Tensor / pipeline parallel | [launch.sh:246-247](launch.sh#L246-L247) | `--tensor-model-parallel-size 4`, `--pipeline-model-parallel-size 4` |
| Sequence parallel | `DISTRIBUTED_ARGS` | add `--sequence-parallel` |
| Micro-batch size | [launch.sh:57-78](launch.sh#L57-L78) | grid-search `MBS` |
| Activation recompute | `TRAINING_ARGS` | `--recompute-granularity selective`/`full` |
| FP8 | `MIXED_PRECISION_ARGS` | `--fp8-format hybrid` |
| Custom kernel | new patch | `Megatron-LM/megatron/core/fusions/` |

Full flag list: `python Megatron-LM/megatron/training/arguments.py --help`.

### Challenge 1 — loss (flags + patches)

| Technique | Where |
|---|---|
| LR schedule / warmup | [launch.sh:227-230](launch.sh#L227-L230) — flags |
| Optimizer | `--optimizer` flag, or patch `megatron/core/optimizer/` |
| Architecture | [launch.sh:55-84](launch.sh#L55-L84) `MODEL_SIZE` block |
| Token budget | `TRAINING_STEPS × GBS × SEQ_LEN` |
| Data ordering | `DATA_ARGS` flags + `megatron/core/datasets/` |
| Custom loss / reg | patch `pretrain_gpt.py` `loss_func` |

## Profiling (nsys)

```bash
./launch.sh profile 760m            # 20 steps, captures iters 10-15, all ranks
./launch.sh profile 760m 30 4       # override [steps] [nodes]
```

Output: `$LOG_DIR/nsys/$SLURM_JOB_ID/node{0..N-1}.nsys-rep` — one file per node, each containing all 4 local GPUs (nsys follows torchrun's children).

| Step | Command |
|---|---|
| Text summary | `nsys stats node0.nsys-rep` |
| Kernel / memcpy / NCCL breakdown | `nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nccl_sum node0.nsys-rep` |
| Export to SQLite for pandas | `nsys export --type=sqlite -o node0.sqlite node0.nsys-rep` |
| Open timeline GUI | download `.nsys-rep`, open in Nsight Systems desktop |

Useful SQLite tables: `CUPTI_ACTIVITY_KIND_KERNEL` (compute), `CUPTI_ACTIVITY_KIND_MEMCPY` (H↔D/D↔D), `NVTX_EVENTS` (Megatron's forward/backward/comm ranges).

Knobs in [launch.sh `profile` mode](launch.sh#L62-L77): `PROFILE_STEP_START/END` (default 10-15), `TRAINING_STEPS` (default 20). Profile range must include only steady-state steps — first few iters are compile-heavy and will skew totals.
