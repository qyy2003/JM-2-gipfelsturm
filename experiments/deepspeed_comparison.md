# DeepSpeed ZeRO Comparison

This harness compares DeepSpeed ZeRO-1/2/3 against the Megatron runs without
modifying Megatron-LM. It uses the same model-size presets, sequence length,
MBS/GBS accounting, BF16 precision, SLURM shape, and W&B project naming, but it
uses synthetic token batches instead of the Megatron indexed dataset.

That makes it a systems benchmark for ZeRO behavior, not a perfectly identical
training recipe.

## Run Smoke

```bash
./experiments/submit_deepspeed_sweep.sh smoke
```

This submits:

```text
125m, ZeRO-1/2/3, 20 steps, 1 node, 5 min
```

To run one stage:

```bash
ZERO_STAGES=2 ./experiments/submit_deepspeed_sweep.sh smoke
```

## Run Main Comparison

```bash
./experiments/submit_deepspeed_sweep.sh good
```

This submits:

```text
1.5b, ZeRO-1/2/3, 200 steps, 1 node, 30 min
```

Useful overrides:

```bash
ZERO_STAGES="1 2" GOOD_MODEL=760m ./experiments/submit_deepspeed_sweep.sh good
MBS_OVERRIDE=2 ZERO_STAGES=3 ./experiments/submit_deepspeed_sweep.sh good
SEQ_LEN=2048 ./experiments/submit_deepspeed_sweep.sh smoke
```

## Metrics

- Primary: `tokens/sec/GPU`
- Secondary: `lm loss`, `iteration-time`, `max_memory_gb`, success/failure
- W&B run names contain model, ZeRO stage, steps, nodes, GBS, MBS, sequence length, and BF16.
