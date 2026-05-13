# FlashAttention 3

The default container ships only FlashAttention 2 kernels. To use FA3 on Hopper
(H100/H200), build it once into a host-mounted prefix, then point `launch.sh` at
the `flash` backend.

## 1. Build (once per user)

```bash
sbatch build_fa3.sbatch
```

This compiles FA3 inside the training container and installs the artifacts to
`$WORKDIR/../gipfelsturm/fa3` (~1.5 GB). Takes ~30-45 min cold. The output is
tied to the container's CUDA / PyTorch / GPU arch, so don't share the prefix
across users with different setups — rebuild instead.

Check `logs/build-fa3-*.log` for `[fa3] import OK: v3.x.x` at the end.

## 2. Enable in training

In `launch.sh`:

```bash
ATTENTION_BACKEND=flash
```

`flash` tells TransformerEngine to set `NVTE_FLASH_ATTN=1`, which picks up the
FA3 kernels at runtime. Other valid values: `fused` (TE's cuDNN fused attn),
`unfused` (PyTorch fallback), `auto` (TE chooses).

To verify FA3 is actually loaded, check the `[probe][attn]` lines at the top of
the training log — you should see `import flash_attn_3: v3.x.x`.
