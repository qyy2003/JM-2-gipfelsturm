# DP Backend Comparison

Goal: compare data-parallel strategies for throughput under the same model,
sequence length, precision, attention backend, global batch size, and node count.

## Backends

| Backend | Launcher value | Meaning |
|---|---|---|
| Strong baseline | `megatron` | Megatron distributed optimizer with gradient-reduce and parameter-gather overlap |
| DDP ablation | `ddp` | Replicated data parallelism without Megatron optimizer sharding |
| FSDP | `fsdp` | Megatron FSDP with `optim_grads_params` sharding |

DeepSpeed is intentionally not submitted by this launcher yet. Megatron-LM does
not expose a DeepSpeed ZeRO path in this repo setup, so that needs a separate
harness after we verify `deepspeed` is available in the Alps container.

## First Runs

Smoke tests, one node, 20 steps, 10-minute allocation:

```bash
./experiments/submit_dp_backend_sweep.sh smoke
```

Debug throughput tests, one node, 50 steps, 15-minute allocation:

```bash
./experiments/submit_dp_backend_sweep.sh debug
```

Main throughput comparison, one node, 1.5B model, 200 steps, 30-minute allocation:

```bash
./experiments/submit_dp_backend_sweep.sh good
```

Submit both:

```bash
./experiments/submit_dp_backend_sweep.sh all
```

## Useful Overrides

```bash
GOOD_MODEL=760m GOOD_STEPS=200 GOOD_NODES=2 ./experiments/submit_dp_backend_sweep.sh good
GOOD_WALLTIME=00:20:00 ./experiments/submit_dp_backend_sweep.sh good
MBS_OVERRIDE=2 BACKENDS="megatron fsdp" ./experiments/submit_dp_backend_sweep.sh good
FP8=false ./experiments/submit_dp_backend_sweep.sh smoke
```

## Metrics To Compare

- W&B run name contains backend, model, steps, nodes, GBS, MBS, attention, FP8, TP, and PP.
- Primary metric: `tokens/sec/GPU`.
- Secondary metrics: train loss, step time, memory, OOM/failure status.
- Use the generated `logs/gipfel-*.sbatch` files as exact run artifacts.
