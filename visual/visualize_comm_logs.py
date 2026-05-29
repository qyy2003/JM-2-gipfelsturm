#!/usr/bin/env python3
"""Visualize per-rank PP / TP / DP communication logs.

Caveats baked into this script:
  * wall_start_us is each process's monotonic clock — NOT a shared epoch.
    Cross-rank timelines must normalize per rank, not globally.
  * Some logged "steps" are eval-only (no backward passes); those skew
    aggregates if mixed with training steps, so we separate them.
  * Logging is asymmetric across PP stages: stage-0 ranks log TP-AR + the
    full per-layer DP-AG/DP-RS bucket stream; stage-1 ranks log neither
    TP-AR nor the per-layer DP traffic (just one DP-AG/DP-RS per step).
    Don't read absence as silence — it's a logging gap.

Reads every comm_rank*.jsonl in --logs-dir, then emits PNGs into --out-dir:
  timeline_step<STEP>.pdf      per-rank Gantt for one full training step
                               (color = event; TP/PP/DP all overlaid)
  per_step_breakdown.pdf       stacked bars of mean per-rank comm time / step
  bandwidth_distribution.pdf   boxplot of effective GB/s by event
  rank_event_heatmap.pdf       median wait per (rank, event) in µs
  topology.pdf                 8 nodes x 4 GPUs schematic with TP / DP / PP groups
  comm_matrix.pdf              3 panels (TP / PP / DP) of 32x32 GB sent;
                               traffic attributed via each event's `peers` list
  pair_latency.pdf             per-PP-pair median latency, surfaces stragglers
  tp_dp_latency.pdf            per-rank median latency for TP-AR / DP-AG / DP-RS

If node*.sqlite (exported from the .nsys-rep files) are passed via --nsys-sqlite,
also emits the REAL compute/communication overlap from the GPU kernel timeline
(the comm logs alone can't show overlap — they serialize each collective):
  nsys_overlap_summary.pdf     per-rank step-time accounting (compute / hidden
                               comm / exposed comm / idle) + exposed comm split
                               by collective (TP-AR / DP-RS / DP-AG / PP)
  nsys_overlap_timeline.pdf    per-step Gantt of compute vs comm CUDA streams

Usage:
  python visualize_comm_logs.py [--logs-dir DIR] [--out-dir DIR] [--step N]
                                [--nsys-sqlite node0.sqlite node4.sqlite]
                                [--skip-comm-logs]
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PP_EVENTS = {
    "PP-fwd-send", "PP-fwd-recv",
    "PP-bwd-send", "PP-bwd-recv",
    "PP-fwd-send+bwd-recv", "PP-bwd-send+fwd-recv",
}
TP_EVENTS = {"TP-AR"}
DP_EVENTS = {"DP-AG", "DP-RS"}


def event_category(name: str) -> str:
    if name in TP_EVENTS:
        return "TP"
    if name in PP_EVENTS:
        return "PP"
    if name in DP_EVENTS:
        return "DP"
    return "?"


# Fixed per-event colors so all plots agree.  Greens=TP, blues/reds=PP, purples=DP.
EVENT_COLORS = {
    "TP-AR":                "#2ca02c",
    "PP-fwd-send":          "#1f77b4",
    "PP-fwd-recv":          "#7faed6",
    "PP-bwd-send":          "#d62728",
    "PP-bwd-recv":          "#e88a8a",
    "PP-fwd-send+bwd-recv": "#17becf",
    "PP-bwd-send+fwd-recv": "#ff7f0e",
    "DP-AG":                "#9467bd",
    "DP-RS":                "#6a3d9a",
}


def load_logs(logs_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    files = sorted(logs_dir.glob("comm_rank*.jsonl"))
    if not files:
        raise SystemExit(f"no comm_rank*.jsonl files in {logs_dir}")
    for p in files:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("meta"):
                    continue  # rank metadata header line
                rows.append(obj)
    df = pd.DataFrame(rows)
    df["bytes_total"] = df["send_bytes"] + df["recv_bytes"]
    dur_s = df["wall_dur_us"].clip(lower=1) / 1e6
    df["GB_per_s"] = (df["bytes_total"] / 1e9) / dur_s
    df["category"] = df["name"].map(event_category)
    return df


def classify_steps(df: pd.DataFrame) -> tuple[set[int], set[int]]:
    """Return (training_steps, eval_only_steps).

    Classification looks only at PP events (TP/DP logging is asymmetric across
    stages and would otherwise leak into the "eval" bucket). A training step
    has the full 6-event PP set; everything else lands in eval_only.
    """
    pp_only = df[df["name"].isin(PP_EVENTS)]
    by_step = pp_only.groupby("step")["name"].agg(set)
    training, eval_only = set(), set()
    for step, names in by_step.items():
        if PP_EVENTS.issubset(names):
            training.add(int(step))
        else:
            eval_only.add(int(step))
    # steps that have ONLY non-PP events (e.g. cleanup steps with just DP-AG)
    all_steps = set(int(s) for s in df["step"].unique())
    eval_only |= all_steps - training - eval_only
    return training, eval_only


def pick_step(df: pd.DataFrame, training: set[int]) -> int:
    # the busiest training step (most events) — skip step 0 if there are others
    candidates = training - {0} if training - {0} else training
    counts = df[df["step"].isin(candidates)].groupby("step").size()
    return int(counts.idxmax())


def plot_timeline(df: pd.DataFrame, step: int, out_path: Path) -> None:
    """Per-rank Gantt for one step — TP, PP, and DP events overlaid.

    Each rank's row is normalized to ITS OWN earliest event in this step
    because wall_start_us is per-process monotonic time, not a shared clock.
    Bars use vectorized broken_barh per rank so dense TP/DP traces stay fast.
    """
    sub = df[df["step"] == step].copy()
    if sub.empty:
        print(f"no events for step {step}; skipping timeline")
        return
    sub["rank_t0"] = sub.groupby("rank")["wall_start_us"].transform("min")
    sub["start_ms"] = (sub["wall_start_us"] - sub["rank_t0"]) / 1000.0
    sub["dur_ms"] = (sub["wall_dur_us"] / 1000.0).clip(lower=0.02)

    names = [n for n in EVENT_COLORS if n in set(sub["name"].unique())]
    ranks = sorted(sub["rank"].unique())

    fig, ax = plt.subplots(figsize=(14, max(6, 0.32 * len(ranks))))
    for r in ranks:
        rsub = sub[sub["rank"] == r]
        for name in names:
            ev = rsub[rsub["name"] == name]
            if ev.empty:
                continue
            bars = list(zip(ev["start_ms"].values, ev["dur_ms"].values))
            ax.broken_barh(bars, (r - 0.4, 0.8),
                           facecolors=EVENT_COLORS[name], edgecolors="none")
    ax.set_xlabel("ms since this rank's first event in the step")
    ax.set_ylabel("rank")
    ax.set_yticks(ranks)
    ax.invert_yaxis()
    ax.axhline(15.5, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    xmax = ax.get_xlim()[1]
    ax.text(xmax * 0.99, 7.5, "PP stage 0", ha="right", va="center",
            fontsize=9, alpha=0.6)
    ax.text(xmax * 0.99, 23.5, "PP stage 1", ha="right", va="center",
            fontsize=9, alpha=0.6)
    ax.set_title(f"Communication timeline @ training step {step}  "
                 f"(per-rank normalized — clocks are not shared across ranks)\n"
                 f"green=TP  blue/red=PP  purple=DP   "
                 f"stage-1 ranks show no TP-AR / per-layer DP — logging gap, not silence")
    handles = [mpatches.Patch(color=EVENT_COLORS[n], label=n) for n in names]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_per_step_breakdown(df: pd.DataFrame, training: set[int],
                            eval_only: set[int], out_path: Path) -> None:
    """Stacked bars of mean per-rank comm time (ms) per step.

    Each event's total ms is divided by the number of ranks that *actually
    logged* that event — TP-AR and the per-layer DP traffic only appear on
    stage-0 ranks, so dividing by the full world_size would understate them.
    Step 0 (NCCL init) is shown as an annotation, not in the bars.
    """
    train_df = df[df["step"].isin(training)]

    def per_step_table(sub: pd.DataFrame) -> pd.DataFrame | None:
        if sub.empty:
            return None
        # per-event divisor: how many distinct ranks ever logged this event
        rank_per_event = sub.groupby("name")["rank"].nunique()
        per_step = (sub.groupby(["step", "name"])["wall_dur_us"].sum()
                    .unstack(fill_value=0) / 1000.0).sort_index()
        for col in per_step.columns:
            per_step[col] = per_step[col] / max(int(rank_per_event[col]), 1)
        return per_step

    per_step = per_step_table(train_df)
    if per_step is None:
        print("no training steps to plot per-step breakdown")
        return
    eval_per_step = per_step_table(df[df["step"].isin(eval_only)])

    main = per_step.drop(index=0, errors="ignore")  # exclude warm-up outlier
    warmup = per_step.loc[[0]] if 0 in per_step.index else None

    fig, ax = plt.subplots(figsize=(13, 6))
    # stack by category order so each color band is contiguous and readable
    name_order = [n for n in EVENT_COLORS if n in per_step.columns]

    bottoms = np.zeros(len(main.index))
    x = main.index.values
    for name in name_order:
        vals = main[name].values
        ax.bar(x, vals, bottom=bottoms, color=EVENT_COLORS[name], width=1.0,
               label=name, edgecolor="none")
        bottoms += vals

    if eval_per_step is not None and not eval_per_step.empty:
        ev_bottoms = np.zeros(len(eval_per_step.index))
        for name in name_order:
            if name in eval_per_step.columns:
                vals = eval_per_step[name].values
                ax.bar(eval_per_step.index.values, vals, bottom=ev_bottoms,
                       color=EVENT_COLORS[name], width=1.0, hatch="//",
                       edgecolor="black", linewidth=0.3)
                ev_bottoms += vals
        ax.bar([], [], color="white", hatch="//", edgecolor="black",
               label="non-training step")

    ax.set_xlabel("step")
    ax.set_ylabel("mean per-logging-rank comm time (ms)")
    ax.set_title("Per-step communication-time breakdown — TP + PP + DP\n"
                 "(each event averaged over ranks that logged it; "
                 "TP/DP per-layer traffic is stage-0 only)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    if warmup is not None:
        warm_total = float(warmup.sum(axis=1).iloc[0])
        ax.text(0.01, 0.98,
                f"step 0 omitted from bars: {warm_total:,.0f} ms "
                f"(NCCL init / first collective)",
                transform=ax.transAxes, fontsize=9, color="crimson",
                va="top", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_bandwidth(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["bytes_total"] > 0]
    names = sorted(sub["name"].unique())
    data = [sub.loc[sub["name"] == n, "GB_per_s"].values for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=names, showfliers=False, patch_artist=True)
    cmap = plt.get_cmap("tab10")
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(cmap(i % 10))
        box.set_alpha(0.6)
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title("Per-event effective bandwidth — (send_bytes + recv_bytes) / wall_dur")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# topology  (PP=2, TP=2, DP=8; 32 GPUs across 8 nodes, 4 GPUs/node)
# ---------------------------------------------------------------------------
PP_SIZE = 2
TP_SIZE = 2
DP_SIZE = 8
GPUS_PER_NODE = 4

# Megatron default rank ordering: tp - cp - dp - pp  (tp innermost)
# So with TP=2, DP=8, PP=2:
#   tp_pos = rank % TP
#   dp_pos = (rank // TP) % DP
#   pp_pos = rank // (TP * DP)
# Physical placement: ranks fill nodes in order, 4 per node.

def rank_coords(r: int) -> dict:
    return {
        "tp": r % TP_SIZE,
        "dp": (r // TP_SIZE) % DP_SIZE,
        "pp": r // (TP_SIZE * DP_SIZE),
        "node": r // GPUS_PER_NODE,
        "node_slot": r % GPUS_PER_NODE,
    }


def plot_topology(out_path: Path) -> None:
    """Draw the 32-GPU layout: 8 nodes x 4 GPUs, TP intra-node, PP across stages."""
    fig, ax = plt.subplots(figsize=(15, 7))
    # limits hug the drawn content (boxes span x:0.5..16.4, y:0.5..8.7;
    # PP labels at x=-0.8) so the page isn't mostly blank background.
    ax.set_xlim(-1.3, 16.7); ax.set_ylim(-0.1, 9.0)
    ax.set_aspect("equal"); ax.axis("off")

    dp_cmap = plt.get_cmap("tab10")
    dp_color = {d: dp_cmap(d) for d in range(DP_SIZE)}

    # node positions: 4 per stage row
    def node_xy(node_idx: int) -> tuple[float, float]:
        # nodes 0..3 = stage 0 (top), nodes 4..7 = stage 1 (bottom)
        col = node_idx % 4
        row = node_idx // 4  # 0 = top, 1 = bottom
        x = col * 4.3 + 0.5
        y = 5.5 if row == 0 else 0.5
        return x, y

    # node + GPU geometry: box is 3.0 wide, 3.2 tall (extra room at top for label)
    BOX_W, BOX_H = 3.0, 3.2
    GPU_R = 0.42
    GPU_DX = (0.75, 2.25)  # centers of the two GPU columns (inside the box)
    GPU_DY = (0.7, 2.0)    # centers of the two GPU rows

    # rank position inside its node (2x2 grid of GPUs)
    def gpu_xy(r: int) -> tuple[float, float]:
        c = rank_coords(r)
        nx, ny = node_xy(c["node"])
        slot = c["node_slot"]
        return nx + GPU_DX[slot % 2], ny + GPU_DY[slot // 2]

    # draw nodes
    for n in range(8):
        x, y = node_xy(n)
        rect = mpatches.FancyBboxPatch(
            (x, y), BOX_W, BOX_H,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.5, edgecolor="black",
            facecolor="#f0f0f0" if n < 4 else "#e8eef8",
        )
        ax.add_patch(rect)
        # node label sits inside the box, above the GPUs
        ax.text(x + BOX_W / 2, y + BOX_H - 0.25, f"node {n}",
                ha="center", va="center", fontsize=10, fontweight="bold")

    # draw GPUs
    for r in range(32):
        gx, gy = gpu_xy(r)
        c = rank_coords(r)
        circle = mpatches.Circle((gx, gy), GPU_R,
                                 facecolor=dp_color[c["dp"]],
                                 edgecolor="black", linewidth=1.0)
        ax.add_patch(circle)
        ax.text(gx, gy, str(r), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    # TP edges (intra-node NVLink): rank r <-> r+1 when r is even and in same node
    tp_drawn = set()
    for r in range(32):
        partner = r ^ 1  # toggle tp_pos
        if (r, partner) in tp_drawn or (partner, r) in tp_drawn:
            continue
        if rank_coords(r)["node"] != rank_coords(partner)["node"]:
            continue
        a = gpu_xy(r); b = gpu_xy(partner)
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color="green", linewidth=2.0, alpha=0.6, zorder=1)
        tp_drawn.add((r, partner))

    # PP edges (inter-node, across stages): rank r <-> r+16
    for r in range(16):
        peer = r + 16
        a = gpu_xy(r); b = gpu_xy(peer)
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color="crimson", linewidth=1.0, alpha=0.5,
                linestyle="--", zorder=0)

    # PP stage labels
    ax.text(-0.8, 5.5 + BOX_H / 2, "PP stage 0", rotation=90, fontsize=12,
            ha="center", va="center", color="#444", fontweight="bold")
    ax.text(-0.8, 0.5 + BOX_H / 2, "PP stage 1", rotation=90, fontsize=12,
            ha="center", va="center", color="#444", fontweight="bold")

    # legend
    legend = [
        mpatches.Patch(color="green", label="TP link (NVLink, intra-node, logged)"),
        mpatches.Patch(color="crimson", label="PP link (inter-node, logged)"),
    ]
    legend += [mpatches.Patch(color=dp_color[d], label=f"DP pos {d}")
               for d in range(DP_SIZE)]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=5, fontsize=9, frameon=False)

    fig.suptitle(
        "32 GPU topology — PP=2  TP=2  DP=8  (4 GPUs/node, 8 nodes)\n"
        "Green = TP (NVLink, intra-node).  Red dashed = PP (inter-node Slingshot).  "
        "Same circle color = same DP position → DP group of 8 across nodes.\n"
        "All three (TP-AR, PP send/recv, DP-AG/RS) are now logged — "
        "see other plots for traffic volume and latency.",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {out_path}")


def _build_category_matrix(df: pd.DataFrame, n: int = 32) -> np.ndarray:
    """Return an n x n matrix of GB sent from row -> col.

    For point-to-point events (PP) the bytes go to the one non-self peer.
    For collectives (TP-AR, DP-AG, DP-RS) we attribute send_bytes evenly
    across the other peers in the group. This is "logical" attribution,
    not the wire-level pattern (a ring all-reduce doesn't really fan out
    uniformly), but it makes the structure of the group visible.
    """
    mat = np.zeros((n, n))
    for _, row in df.iterrows():
        send = row["send_bytes"]
        if send <= 0:
            continue
        sender = int(row["rank"])
        peers = row.get("peers") or []
        others = [int(p) for p in peers if int(p) != sender]
        if not others:
            continue
        per_peer = send / len(others)
        for p in others:
            mat[sender, p] += per_peer
    return mat


def plot_comm_matrix(df: pd.DataFrame, out_path: Path) -> None:
    """Three 32x32 matrices side-by-side: TP, PP, DP GB sent (row -> col).

    Each event's `peers` list determines who counts as a receiver. PP events
    have exactly one non-self peer so they're exact; TP/DP are collectives
    so the bytes are spread uniformly across the group (see helper docstring).
    """
    n = 32
    cats = [
        ("TP", "TP-AR all-reduce  (peers within node)",     "Greens"),
        ("PP", "PP send/recv  (stage 0 <-> stage 1)",        "Blues"),
        ("DP", "DP-AG / DP-RS  (across DP group of 8)",      "Purples"),
    ]
    mats = {c: _build_category_matrix(df[df["category"] == c], n) for c, _, _ in cats}

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, (cat, title, cmap_name) in zip(axes, cats):
        gb = mats[cat] / 1e9
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#222")
        masked = np.ma.masked_where(gb == 0, gb)
        im = ax.imshow(masked, cmap=cmap, aspect="equal")
        ax.set_xlabel("receiver rank")
        ax.set_ylabel("sender rank")
        ax.set_xticks(range(0, n, 2))
        ax.set_yticks(range(0, n, 2))
        ax.axhline(15.5, color="white", linewidth=0.6, alpha=0.5)
        ax.axvline(15.5, color="white", linewidth=0.6, alpha=0.5)
        ax.set_title(f"{cat}: {title}\n total = {gb.sum():.1f} GB",
                     fontsize=10)
        fig.colorbar(im, ax=ax, label="GB", fraction=0.045)

    fig.suptitle(
        "Communication matrices — sender (row) -> receiver (col).  "
        "Collective bytes (TP / DP) are spread evenly across group members.\n"
        "Empty rows in TP and the per-layer DP panels = stage-1 ranks, "
        "which don't log those events (logging gap, not absence of traffic).",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_pair_latency(df: pd.DataFrame, out_path: Path) -> None:
    """Per-PP-pair median latency for the fused steady-state events."""
    # use the busy steady-state events as the comparison point
    fused0 = df[df["name"] == "PP-fwd-send+bwd-recv"]  # emitted by stage 0
    fused1 = df[df["name"] == "PP-bwd-send+fwd-recv"]  # emitted by stage 1
    pairs = sorted(set(int(r) for r in fused0["rank"].unique()))

    rows = []
    for r in pairs:
        s0 = fused0.loc[fused0["rank"] == r, "wall_dur_us"]
        s1 = fused1.loc[fused1["rank"] == r + 16, "wall_dur_us"]
        rows.append({
            "pair": f"{r}<->{r+16}",
            "stage0_median_us": float(s0.median()),
            "stage1_median_us": float(s1.median()),
            "stage0_p95_us": float(s0.quantile(0.95)),
            "stage1_p95_us": float(s1.quantile(0.95)),
        })

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(rows))
    w = 0.4
    s0_med = [r["stage0_median_us"] for r in rows]
    s1_med = [r["stage1_median_us"] for r in rows]
    s0_p95 = [r["stage0_p95_us"] for r in rows]
    s1_p95 = [r["stage1_p95_us"] for r in rows]
    ax.bar(x - w/2, s0_med, w, label="stage 0 view (fwd-send+bwd-recv) median",
           color="#4477aa")
    ax.bar(x + w/2, s1_med, w, label="stage 1 view (bwd-send+fwd-recv) median",
           color="#cc6677")
    ax.plot(x - w/2, s0_p95, "o", color="#224466", label="stage 0 p95")
    ax.plot(x + w/2, s1_p95, "o", color="#aa3344", label="stage 1 p95")
    ax.set_xticks(x)
    ax.set_xticklabels([r["pair"] for r in rows], rotation=45, ha="right",
                       fontsize=8)
    ax.set_ylabel("wall_dur (µs)")
    ax.set_title("Per-PP-pair latency for the fused 1F1B exchange  "
                 "(both endpoints should see similar wait — divergence = straggler)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_tp_dp_latency(df: pd.DataFrame, out_path: Path) -> None:
    """Per-rank median + p95 latency for the collective events (TP-AR, DP-AG, DP-RS).

    These come out as one row per event type with one bar per rank, so the
    stage-0 / stage-1 logging asymmetry is plainly visible: stage-1 bars are
    empty for TP-AR and dwarfed for the per-layer DP traffic.
    """
    events = ["TP-AR", "DP-AG", "DP-RS"]
    fig, axes = plt.subplots(len(events), 1, figsize=(14, 9), sharex=True)
    ranks = list(range(32))
    for ax, name in zip(axes, events):
        sub = df[df["name"] == name]
        med = [float(sub.loc[sub["rank"] == r, "wall_dur_us"].median())
               if (sub["rank"] == r).any() else 0.0 for r in ranks]
        p95 = [float(sub.loc[sub["rank"] == r, "wall_dur_us"].quantile(0.95))
               if (sub["rank"] == r).any() else 0.0 for r in ranks]
        ax.bar(ranks, med, color=EVENT_COLORS[name], label=f"{name} median")
        ax.plot(ranks, p95, "o", color="black", markersize=3.5, label="p95")
        ax.set_ylabel("wall_dur (µs)")
        ax.set_title(f"{name}  —  per-rank latency  "
                     f"(n events on stage-0 ranks ~ {int((sub['rank'] < 16).sum() / max(1,16))}/rank, "
                     f"stage-1 ~ {int((sub['rank'] >= 16).sum() / max(1,16))}/rank)")
        ax.axvline(15.5, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        if med and max(med) > 0:
            ax.set_yscale("log")
    axes[-1].set_xlabel("rank   (0–15 = PP stage 0,  16–31 = PP stage 1)")
    axes[-1].set_xticks(range(0, 32, 2))
    fig.suptitle("TP & DP collective latency per rank "
                 "(log scale — empty bars = event not logged on that rank)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    pivot = (df.groupby(["rank", "name"])["wall_dur_us"]
               .median()
               .unstack())  # keep µs
    fig, ax = plt.subplots(figsize=(8, max(6, 0.28 * len(pivot.index))))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#222222")
    masked = np.ma.masked_invalid(pivot.values)
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_xlabel("event")
    ax.set_ylabel("rank")
    ax.set_title("Median wait per (rank, event) — µs   (gray = N/A for that PP stage)")
    fig.colorbar(im, ax=ax, label="µs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# nsys GPU-kernel compute / communication OVERLAP analysis
# ---------------------------------------------------------------------------
# Why this needs nsys and not the comm logs:
#   The comm logger brackets each collective with CUDA events and reads the
#   elapsed time — which forces a synchronize, serializing the call. So the
#   comm logs measure every collective *in isolation* and structurally CANNOT
#   show overlap.  The nsys kernel timeline can: NCCL kernels and compute
#   kernels run on different CUDA streams of the SAME GPU, so we define
#       overlap = wall time where a comm kernel AND a compute kernel are
#                 simultaneously resident on the device.
#   Exposed comm (comm time with no compute running) is what actually extends
#   the step; hidden comm is free.
#
# Input: sqlite exported from each per-node .nsys-rep, e.g.
#     nsys export --type sqlite -o node0.sqlite node0.nsys-rep
#   The reader's nsys version must be >= the writer's, so run the export
#   inside the same container that produced the report (see nsys_work/).
#
# Rank mapping: each rank's process owns one GPU, so deviceId 0..3 within a
#   node map to that node's 4 ranks.  node<N> holds ranks N*4 .. N*4+3
#   (node0 -> 0..3 = PP stage 0, node4 -> 16..19 = PP stage 1).
#
# NCCL kernel name -> logical category:
#   AllReduce -> TP-AR   ReduceScatter -> DP-RS   AllGather -> DP-AG
#   SendRecv  -> PP      (Broadcast/etc -> other-comm)
#   With the distributed optimizer, gradients reduce-scatter and params
#   all-gather; TP uses all-reduce; PP uses point-to-point send/recv.  TP-AR
#   is launched inline on the compute stream (so it can't overlap), while the
#   DP and PP collectives sit on their own streams (so they *can*).
# ===========================================================================

COMM_CATS = ["TP-AR", "DP-RS", "DP-AG", "PP", "Bcast", "other-comm"]

# GPU-time accounting segments (one step = sum of these four)
SEG_COLORS = {
    "compute": "#4c78a8",   # compute running, no comm
    "hidden":  "#54a24b",   # comm overlapped with compute (free)
    "exposed": "#e45756",   # comm running with NO compute (stalls the step)
    "idle":    "#d9d9d9",   # GPU idle (bubble / dataloader / sync)
}
# per-category colors (match the comm-log palette where possible)
CAT_COLORS = {
    "TP-AR": "#2ca02c", "DP-RS": "#6a3d9a", "DP-AG": "#9467bd",
    "PP": "#1f77b4", "Bcast": "#ff7f0e", "other-comm": "#999999",
    "compute": "#bbbbbb",
}

# Human-readable notes shown in the printed summary.
CAT_NOTES = {
    "TP-AR": "tensor-parallel all-reduce (intra-node NVLink)",
    "DP-RS": "distributed-optimizer gradient reduce-scatter (cross-node)",
    "DP-AG": "distributed-optimizer param all-gather (cross-node)",
    "PP":    "pipeline send/recv (cross-node)",
    "Bcast": "per-microbatch input broadcast — spin-wait for the TP-source "
             "rank's dataloader, NOT interconnect bandwidth",
    "other-comm": "misc collectives",
}


def _kernel_category(name: str) -> str:
    low = name.lower()
    if "ccl" not in low:
        return "compute"
    if "allreduce" in low:
        return "TP-AR"
    if "reducescatter" in low:
        return "DP-RS"
    if "allgather" in low:
        return "DP-AG"
    if "sendrecv" in low:
        return "PP"
    if "broadcast" in low:
        return "Bcast"
    return "other-comm"


def _union_len(ivs: list[tuple[int, int]], w0: int, w1: int) -> int:
    """Total length of the union of intervals, clipped to [w0, w1]."""
    pts = []
    for s, e in ivs:
        if s < w0:
            s = w0
        if e > w1:
            e = w1
        if e > s:
            pts.append((s, 1))
            pts.append((e, -1))
    pts.sort()
    depth = last = tot = 0
    for t, v in pts:
        if depth > 0:
            tot += t - last
        depth += v
        last = t
    return tot


def _overlap_len(a: list[tuple[int, int]], b: list[tuple[int, int]],
                 w0: int, w1: int) -> int:
    """Total time where the union of `a` and the union of `b` are both active."""
    ev = []
    for s, e in a:
        s = max(s, w0); e = min(e, w1)
        if e > s:
            ev.append((s, 0, 1)); ev.append((e, 0, -1))
    for s, e in b:
        s = max(s, w0); e = min(e, w1)
        if e > s:
            ev.append((s, 1, 1)); ev.append((e, 1, -1))
    ev.sort()
    da = db = last = tot = 0
    for t, k, v in ev:
        if da > 0 and db > 0:
            tot += t - last
        if k == 0:
            da += v
        else:
            db += v
        last = t
    return tot


def _node_base_from_name(path: Path) -> int:
    """node<N>.sqlite -> N*4 (first global rank on that node)."""
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) * 4 if digits else 0


def load_nsys(sqlite_path: Path) -> tuple[dict[int, dict[str, list]], list[tuple[int, int]]]:
    """Return (per_rank_categories, step_windows).

    per_rank_categories[rank][category] = list of (start_ns, end_ns) kernel intervals.
    step_windows = [(w0, w1), ...] in ns, one per steady-state training iteration,
    taken between consecutive Optimizer.step NVTX markers from a single thread.
    """
    base = _node_base_from_name(sqlite_path)
    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()
    strings = dict(cur.execute("select id, value from StringIds"))

    pid_dev = dict(cur.execute(
        "select distinct globalPid, deviceId from CUPTI_ACTIVITY_KIND_KERNEL"))

    per_rank: dict[int, dict[str, list]] = {}
    for pid, dev in pid_dev.items():
        rank = base + int(dev)
        cats = {c: [] for c in ["compute", *COMM_CATS]}
        for sn, st, en in cur.execute(
                "select shortName, start, end from CUPTI_ACTIVITY_KIND_KERNEL "
                "where globalPid=?", (pid,)):
            cats[_kernel_category(strings.get(sn, ""))].append((st, en))
        per_rank[rank] = cats

    windows: list[tuple[int, int]] = []
    row = cur.execute("select globalTid from NVTX_EVENTS "
                      "where text like 'Optimizer.step%' limit 1").fetchone()
    if row is not None:
        opt = sorted(s for (s,) in cur.execute(
            "select start from NVTX_EVENTS "
            "where text like 'Optimizer.step%' and globalTid=?", (row[0],)))
        windows = [(opt[i], opt[i + 1]) for i in range(len(opt) - 1)]
    con.close()
    return per_rank, windows


def rank_overlap_metrics(cats: dict[str, list],
                         windows: list[tuple[int, int]]) -> dict[str, float]:
    """Per-step (ms) overlap accounting for one rank, averaged over windows."""
    acc: dict[str, float] = collections.defaultdict(float)
    for w0, w1 in windows:
        allcomm = [iv for c in COMM_CATS for iv in cats[c]]
        comp = _union_len(cats["compute"], w0, w1)
        comm = _union_len(allcomm, w0, w1)
        both = _overlap_len(cats["compute"], allcomm, w0, w1)
        acc["W"] += (w1 - w0)
        acc["compute"] += comp
        acc["comm"] += comm
        acc["both"] += both
        for c in COMM_CATS:
            acc[c + "_busy"] += _union_len(cats[c], w0, w1)
            acc[c + "_hidden"] += _overlap_len(cats[c], cats["compute"], w0, w1)
    n = max(len(windows), 1)
    return {k: v / n / 1e6 for k, v in acc.items()}  # ns -> ms/step


def plot_overlap_summary(metrics: dict[int, dict[str, float]], out_path: Path) -> None:
    """Per-rank GPU-time accounting + exposed-comm-by-category breakdown."""
    ranks = sorted(metrics)
    x = np.arange(len(ranks))

    compute_only = np.array([metrics[r]["compute"] - metrics[r]["both"] for r in ranks])
    hidden = np.array([metrics[r]["both"] for r in ranks])
    exposed = np.array([metrics[r]["comm"] - metrics[r]["both"] for r in ranks])
    W = np.array([metrics[r]["W"] for r in ranks])
    idle = W - (compute_only + hidden + exposed)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11))

    # ---- panel 1: where each step's wall time goes (ms) ----
    bottoms = np.zeros(len(ranks))
    for label, vals, key in [
        ("compute (no comm)", compute_only, "compute"),
        ("comm hidden behind compute (free)", hidden, "hidden"),
        ("comm EXPOSED (stalls step)", exposed, "exposed"),
        ("GPU idle (bubble / dataloader)", idle, "idle"),
    ]:
        ax1.bar(x, vals, bottom=bottoms, width=0.8, color=SEG_COLORS[key],
                label=label, edgecolor="white", linewidth=0.4)
        bottoms += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranks)
    ax1.set_xlabel("global rank   (0–3 = node0 / PP stage 0,  16–19 = node4 / PP stage 1)")
    ax1.set_ylabel("time per training step (ms)")
    tot_comm = float((hidden + exposed).mean())
    hid_pct = 100 * hidden.sum() / max((hidden + exposed).sum(), 1e-9)
    ax1.set_title(
        "GPU time accounting per step (from nsys kernel timeline)\n"
        f"mean step {W.mean():.0f} ms — compute {100*compute_only.mean()/W.mean():.0f}%  "
        f"| comm {100*tot_comm/W.mean():.0f}% (only {hid_pct:.0f}% of it hidden)  "
        f"| exposed comm {100*exposed.mean()/W.mean():.0f}%  | idle {100*idle.mean()/W.mean():.0f}%",
        fontsize=11)
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.axvline(3.5, color="black", lw=0.6, ls="--", alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # ---- panel 2: exposed comm split by collective type ----
    bottoms = np.zeros(len(ranks))
    for c in COMM_CATS:
        exp_c = np.array([metrics[r][c + "_busy"] - metrics[r][c + "_hidden"]
                          for r in ranks])
        if exp_c.sum() < 0.5:
            continue
        ax2.bar(x, exp_c, bottom=bottoms, width=0.8, color=CAT_COLORS[c],
                label=c, edgecolor="white", linewidth=0.4)
        bottoms += exp_c
    ax2.set_xticks(x)
    ax2.set_xticklabels(ranks)
    ax2.set_xlabel("global rank")
    ax2.set_ylabel("EXPOSED comm per step (ms)")
    ax2.set_title("What the exposed communication actually is  "
                  "(AllReduce=TP, ReduceScatter/AllGather=DP, SendRecv=PP)\n"
                  "tall bars = the collective that most directly costs you wall-clock time",
                  fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.axvline(3.5, color="black", lw=0.6, ls="--", alpha=0.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_overlap_timeline(per_rank: dict[int, dict[str, list]],
                          rank_windows: dict[int, list[tuple[int, int]]],
                          ranks: list[int], out_path: Path,
                          win_idx: int | None = None) -> None:
    """Gantt of one full step for representative ranks: compute vs comm lanes.

    Exposed comm is visible as comm bars (TP/DP/PP lanes) with NO grey compute
    bar above them at the same x — most clearly in the reduce-scatter tail at
    the end of the step.  Each rank uses ITS OWN report's step windows (nsys
    timestamps are per-node clocks and do not align across reports).
    """
    lanes = ["compute", "TP-AR", "DP-RS", "DP-AG", "PP", "Bcast"]

    fig, axes = plt.subplots(len(ranks), 1, figsize=(15, 3.0 * len(ranks)),
                             squeeze=False)
    for ax, rank in zip(axes[:, 0], ranks):
        windows = rank_windows.get(rank, [])
        if not windows:
            continue
        wi = win_idx if win_idx is not None else len(windows) // 2
        wi = max(0, min(wi, len(windows) - 1))
        w0, w1 = windows[wi]
        cats = per_rank[rank]
        for y, lane in enumerate(lanes):
            bars = []
            for s, e in cats.get(lane, []):
                if e <= w0 or s >= w1:
                    continue
                s = max(s, w0); e = min(e, w1)
                bars.append(((s - w0) / 1e6, max((e - s) / 1e6, 0.05)))
            if bars:
                ax.broken_barh(bars, (y - 0.4, 0.8),
                               facecolors=CAT_COLORS[lane], edgecolors="none")
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels(lanes)
        ax.invert_yaxis()
        ax.set_xlim(0, (w1 - w0) / 1e6)
        ax.set_xlabel("ms within the step")
        ax.set_title(f"rank {rank}  ({'PP stage 0' if rank < 16 else 'PP stage 1'})  "
                     f"— one full step ({(w1 - w0) / 1e6:.0f} ms).  "
                     f"comm bar with no 'compute' bar at the same time = exposed",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Compute / communication overlap on the GPU timeline  "
                 "(rows are CUDA-stream groups; bars are kernels)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def run_nsys_overlap(sqlite_paths: list[Path], out_dir: Path) -> None:
    """Load every node sqlite, compute per-rank overlap, emit plots + a summary.

    nsys timestamps are per-node monotonic clocks, so each report's ranks are
    analyzed against THAT report's own Optimizer.step windows.
    """
    per_rank: dict[int, dict[str, list]] = {}
    rank_windows: dict[int, list[tuple[int, int]]] = {}
    for p in sqlite_paths:
        pr, win = load_nsys(p)
        per_rank.update(pr)
        for r in pr:
            rank_windows[r] = win
        print(f"  {p.name}: ranks {sorted(pr)}  windows={len(win)}")
    if not per_rank:
        print("no nsys kernel data loaded; skipping overlap analysis")
        return
    if not any(rank_windows.values()):
        print("no Optimizer.step markers; cannot define per-step windows")
        return

    metrics = {r: rank_overlap_metrics(cats, rank_windows[r])
               for r, cats in per_rank.items() if rank_windows.get(r)}

    # textual summary, averaged over PP stage-0 ranks (the busy stage)
    stage0 = [r for r in metrics if r < 16] or list(metrics)
    m = {k: float(np.mean([metrics[r][k] for r in stage0])) for k in metrics[stage0[0]]}
    W = m["W"]
    comm = m["comm"]
    both = m["both"]
    print("\n=== nsys compute/comm overlap — PP stage-0 mean (ms/step) ===")
    print(f"  step wall (GPU window) : {W:8.0f}")
    print(f"  compute busy           : {m['compute']:8.0f}  ({100*m['compute']/W:4.0f}%)")
    print(f"  comm busy (any NCCL)   : {comm:8.0f}  ({100*comm/W:4.0f}%)")
    print(f"    hidden behind compute: {both:8.0f}  ({100*both/comm:4.0f}% of comm)")
    print(f"    EXPOSED (costs wall) : {comm-both:8.0f}  ({100*(comm-both)/W:4.0f}% of step)")
    print(f"  GPU idle (bubble/load) : {W-(m['compute']+comm-both):8.0f}  "
          f"({100*(W-(m['compute']+comm-both))/W:4.0f}%)")
    print("  exposed comm by collective (ms/step, % hidden):")
    for c in COMM_CATS:
        busy = m[c + "_busy"]; hid = m[c + "_hidden"]
        if busy < 0.5:
            continue
        print(f"    {c:11s}: busy {busy:7.0f}  exposed {busy-hid:7.0f}  "
              f"({100*hid/busy:3.0f}% hidden)  — {CAT_NOTES.get(c, '')}")

    plot_overlap_summary(metrics, out_dir / "nsys_overlap_summary.pdf")
    rep = [r for r in (0, 16) if r in per_rank] or sorted(per_rank)[:2]
    plot_overlap_timeline(per_rank, rank_windows, rep,
                          out_dir / "nsys_overlap_timeline.pdf")


def main() -> None:
    here = Path(__file__).resolve().parent
    default_logs = Path(
        "/iopsstor/scratch/cscs/course_00293/gipfelsturm/gipfelsturm/profile-8b-8n/comm_logs"
        "/20260527-171554-2406105"
    )
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_nsys = sorted((here / "nsys_work").glob("node*.sqlite"))
    ap.add_argument("--logs-dir", type=Path, default=default_logs)
    ap.add_argument("--out-dir", type=Path, default=here / "plots",
                    help="default: <script-dir>/plots")
    ap.add_argument("--step", type=int, default=None,
                    help="step for timeline plot (default: busiest training step)")
    ap.add_argument("--nsys-sqlite", type=Path, nargs="*", default=default_nsys,
                    help="node*.sqlite exported from the .nsys-rep files; enables "
                         "the compute/comm overlap plots (default: nsys_work/node*.sqlite)")
    ap.add_argument("--skip-comm-logs", action="store_true",
                    help="only run the nsys overlap analysis")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_comm_logs:
        if args.nsys_sqlite:
            print(f"nsys overlap from: {[str(p) for p in args.nsys_sqlite]}")
            run_nsys_overlap(list(args.nsys_sqlite), out_dir)
        print(f"done -> {out_dir}")
        return

    df = load_logs(args.logs_dir)
    training, eval_only = classify_steps(df)
    print(f"loaded {len(df):,} events  |  {df['rank'].nunique()} ranks")
    print(f"  training steps (all 6 events): {len(training)}  "
          f"({min(training) if training else '-'}..{max(training) if training else '-'})")
    print(f"  eval-only steps (fwd only):    {len(eval_only)}  -> {sorted(eval_only)}")

    step = args.step if args.step is not None else pick_step(df, training)
    print(f"timeline step = {step}")

    plot_timeline(df, step, out_dir / f"timeline_step{step}.pdf")
    plot_per_step_breakdown(df, training, eval_only, out_dir / "per_step_breakdown.pdf")
    plot_bandwidth(df, out_dir / "bandwidth_distribution.pdf")
    plot_heatmap(df, out_dir / "rank_event_heatmap.pdf")
    plot_topology(out_dir / "topology.pdf")
    plot_comm_matrix(df, out_dir / "comm_matrix.pdf")
    plot_pair_latency(df, out_dir / "pair_latency.pdf")
    plot_tp_dp_latency(df, out_dir / "tp_dp_latency.pdf")

    if args.nsys_sqlite:
        print(f"\nnsys overlap from: {[str(p) for p in args.nsys_sqlite]}")
        run_nsys_overlap(list(args.nsys_sqlite), out_dir)

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
