#!/usr/bin/env python3
"""Small DeepSpeed ZeRO throughput benchmark for Gipfelsturm.

This intentionally lives outside Megatron-LM. It matches the launcher model
presets, sequence length, MBS/GBS accounting, BF16 precision, and W&B logging,
while using synthetic GPT-2-token batches to isolate DeepSpeed ZeRO behavior.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import deepspeed


VOCAB_SIZE = 50257


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    default_mbs: int


MODEL_CONFIGS = {
    "125m": ModelConfig(12, 768, 2048, 12, 4, 16),
    "350m": ModelConfig(24, 1024, 2816, 16, 4, 8),
    "760m": ModelConfig(24, 1536, 4096, 16, 4, 4),
    "1.5b": ModelConfig(48, 1600, 4352, 20, 4, 4),
    "3b": ModelConfig(32, 3072, 8192, 24, 8, 4),
    "8b": ModelConfig(32, 4096, 14336, 32, 8, 2),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight).to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, seq_len: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = self.cos[:, :, : x.shape[-2], :]
        sin = self.sin[:, :, : x.shape[-2], :]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, seq_len: int) -> None:
        super().__init__()
        assert hidden % heads == 0
        assert heads % kv_heads == 0
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        if self.kv_heads != self.heads:
            repeats = self.heads // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.heads * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, ffn: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden, ffn, bias=False)
        self.w3 = nn.Linear(hidden, ffn, bias=False)
        self.w2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, seq_len: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden)
        self.attn = CausalSelfAttention(cfg.hidden, cfg.heads, cfg.kv_heads, seq_len)
        self.mlp_norm = RMSNorm(cfg.hidden)
        self.mlp = SwiGLU(cfg.hidden, cfg.ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, seq_len: int, checkpoint_activations: bool) -> None:
        super().__init__()
        self.checkpoint_activations = checkpoint_activations
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.hidden)
        self.layers = nn.ModuleList([Block(cfg, seq_len) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.hidden)
        self.lm_head = nn.Linear(cfg.hidden, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for layer in self.layers:
            if self.checkpoint_activations and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.lm_head(self.norm(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS.keys(), required=True)
    parser.add_argument("--zero-stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--wandb-project", default="gipfelsturm")
    parser.add_argument("--wandb-exp-name", default=None)
    parser.add_argument("--wandb-save-dir", default=None)
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    return parser.parse_args()


def build_ds_config(args: argparse.Namespace, grad_accum_steps: int) -> dict:
    zero = {
        "stage": args.zero_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
    }
    if args.zero_stage == 3:
        zero["stage3_gather_16bit_weights_on_model_save"] = False

    return {
        "bf16": {"enabled": True},
        "zero_optimization": zero,
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "steps_per_print": args.log_interval,
        "wall_clock_breakdown": False,
    }


def maybe_init_wandb(args: argparse.Namespace, rank: int, config: dict):
    if rank != 0 or not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb

    if args.wandb_save_dir:
        os.makedirs(args.wandb_save_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_exp_name,
        dir=args.wandb_save_dir,
        config=config,
    )


def main() -> None:
    args = parse_args()
    cfg = MODEL_CONFIGS[args.model_size]
    if args.micro_batch_size is None:
        args.micro_batch_size = cfg.default_mbs

    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    denom = args.micro_batch_size * world_size
    if args.global_batch_size % denom != 0:
        raise ValueError(f"GBS={args.global_batch_size} must be divisible by MBS*world={denom}")
    grad_accum_steps = args.global_batch_size // denom

    torch.manual_seed(42 + rank)
    model = GPT(cfg, args.seq_len, checkpoint_activations=not args.no_activation_checkpointing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    ds_config = build_ds_config(args, grad_accum_steps)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    engine.train()

    run_config = {
        "backend": f"deepspeed_zero{args.zero_stage}",
        "model_size": args.model_size,
        "seq_len": args.seq_len,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "world_size": world_size,
        "bf16": True,
        "synthetic_data": True,
        "activation_checkpointing": not args.no_activation_checkpointing,
    }
    wandb_run = maybe_init_wandb(args, rank, run_config)

    tokens = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(args.micro_batch_size, args.seq_len),
        device=device,
        dtype=torch.long,
    )
    torch.cuda.synchronize()

    for iteration in range(1, args.train_iters + 1):
        step_start = time.perf_counter()
        total_loss = 0.0
        for _ in range(grad_accum_steps):
            logits = engine(tokens)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB_SIZE),
                tokens[:, 1:].contiguous().view(-1),
            )
            engine.backward(loss)
            total_loss += float(loss.detach())
        engine.step()
        torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start

        tokens_per_step = args.global_batch_size * args.seq_len
        toks_per_sec_gpu = tokens_per_step / step_time / world_size
        avg_loss = total_loss / grad_accum_steps
        max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

        if rank == 0 and iteration % args.log_interval == 0:
            print(
                f"iteration {iteration:6d}/{args.train_iters:6d} | "
                f"elapsed time per iteration (ms): {step_time * 1000:.1f} | "
                f"tokens/sec/GPU: {toks_per_sec_gpu:.0f} | "
                f"lm loss: {avg_loss:.6E} | "
                f"max memory (GB): {max_mem_gb:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "iteration": iteration,
                        "iteration-time": step_time,
                        "tokens/sec/GPU": toks_per_sec_gpu,
                        "lm loss": avg_loss,
                        "max_memory_gb": max_mem_gb,
                    },
                    step=iteration,
                )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
