"""Experiment 08: Track A Fixed-Speed Topological Exploration Benchmark.

Evaluates 3 Candidates across 3 Procedural Random Mazes (Seeds 42, 100, 777):
- A1: Myopic scan_z + Direct Corner Cut
- A2_G2: Myopic scan_z + G² Quintic Spline A*
- A3_Lidar_G2: 6m Chest LiDAR Ray-Cast Mapping + G² Quintic Spline A*

Under strictly locked Track A speed controls:
- speed = 0.60 m/s, min_speed = 0.30 m/s, max_turn = 1.00 rad/s
- max_acc = 1.40 m/s², max_ang_acc = 3.00 rad/s², turn_early = 0.60 m
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from g1_maze_lab.terrains.maze import generate_maze, maze_to_ascii

PROJECT_ROOT = "/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze"
EXP_DIR = os.path.join(PROJECT_ROOT, "artifacts", "experiment_08_track_a_lidar_exploration")
RUN_DIR = os.path.join(EXP_DIR, "run")
CHECKPOINT = "pretrained/g1_rough/model_pretrained.pt"

SEEDS = [42, 100, 777]

CANDIDATES = [
    {
        "id": "a1",
        "name": "A1 (Myopic Direct Cut)",
        "label": "A1 (Myopic Cut)",
        "color": "#f59e0b",  # Amber
        "env": {
            "BENCHMARK_UPGRADED": "1",
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_TURN_EARLY": "0.60",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_USE_ASTAR": "0",
            "MAZE_LIDAR_LOOKAHEAD": "0",
        }
    },
    {
        "id": "a2_g2",
        "name": "A2_G2 (Myopic G² Spline)",
        "label": "A2_G2 (Myopic G²)",
        "color": "#3b82f6",  # Blue
        "env": {
            "BENCHMARK_UPGRADED": "1",
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_TURN_EARLY": "0.60",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_USE_ASTAR": "1",
            "MAZE_CURVE_MODE": "g2_quintic",
            "MAZE_R_FILLET": "0.65",
            "MAZE_LOOKAHEAD": "0.50",
            "MAZE_LIDAR_LOOKAHEAD": "0",
        }
    },
    {
        "id": "a3_lidar_g2",
        "name": "A3_Lidar_G2 (6m LiDAR + G² Spline)",
        "label": "A3_Lidar_G2 (LiDAR + G²)",
        "color": "#10b981",  # Emerald Green
        "env": {
            "BENCHMARK_UPGRADED": "1",
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_TURN_EARLY": "0.60",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_USE_ASTAR": "1",
            "MAZE_CURVE_MODE": "g2_quintic",
            "MAZE_R_FILLET": "0.65",
            "MAZE_LOOKAHEAD": "0.50",
            "MAZE_LIDAR_LOOKAHEAD": "1",
        }
    }
]

def run_all_benchmarks(overwrite=False):
    os.makedirs(RUN_DIR, exist_ok=True)
    
    cmd_base = [
        "uv", "run", "python", "scripts/benchmark_autopilot.py",
        "--task", "G1-Maze-Challenge",
        "--num_envs", "1",
        "--checkpoint", CHECKPOINT
    ]
    
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        w = generate_maze(5, rng, open_ratio=0.15)
        layout_str = maze_to_ascii(w)
        
        for cand in CANDIDATES:
            out_name = f"seed{seed}_{cand['id']}.npz"
            out_path = os.path.join(RUN_DIR, out_name)
            
            if os.path.exists(out_path) and not overwrite:
                print(f"[SKIP] {out_name} already exists. Use --overwrite to rerun.")
                continue
                
            print(f"\n{'='*75}")
            print(f"RUNNING SEED {seed} | CANDIDATE: {cand['name']}")
            print(f"Target npz: {out_path}")
            print(f"{'='*75}")
            
            run_env = os.environ.copy()
            run_env.update(cand["env"])
            run_env["BENCHMARK_OUTPUT"] = out_path
            run_env["BENCHMARK_MODE_NAME"] = f"seed{seed}_{cand['id']}"
            run_env["MAZE_LAYOUT_OVERRIDE"] = layout_str
            
            res = subprocess.run(cmd_base, cwd=os.path.join(PROJECT_ROOT, "g1_maze_lab"), env=run_env)
            if res.returncode != 0:
                print(f"[ERROR] Run failed for Seed {seed} {cand['id']} with code {res.returncode}")
                sys.exit(res.returncode)

def analyze_and_plot():
    results = {}
    
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        w = generate_maze(5, rng, open_ratio=0.15)
        results[seed] = {"maze_walls": w, "runs": {}}
        
        for cand in CANDIDATES:
            out_name = f"seed{seed}_{cand['id']}.npz"
            out_path = os.path.join(RUN_DIR, out_name)
            if not os.path.exists(out_path):
                print(f"[WARN] Missing {out_path}")
                continue
            
            data = np.load(out_path)
            t = data["t"]
            pos = data["pos"]
            lin_vel = data["lin_vel"]
            finish_time = float(data["finish_time"]) if not np.isnan(data["finish_time"]) else float(t[-1])
            
            diffs = np.diff(pos, axis=0)
            path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
            speed_2d = np.linalg.norm(lin_vel, axis=1)
            avg_speed = float(np.mean(speed_2d))
            peak_speed = float(np.max(speed_2d))
            
            cells = data["cell"]
            unique_cells = []
            for c in cells:
                ct = tuple(c)
                if not unique_cells or unique_cells[-1] != ct:
                    unique_cells.append(ct)
            cells_walked = len(unique_cells)
            
            results[seed]["runs"][cand["id"]] = {
                "id": cand["id"],
                "name": cand["name"],
                "label": cand["label"],
                "color": cand["color"],
                "t": t,
                "pos": pos,
                "finish_time": finish_time,
                "path_len": path_len,
                "avg_speed": avg_speed,
                "peak_speed": peak_speed,
                "cells_walked": cells_walked,
                "unique_cells": unique_cells,
                "walls": data["walls"],
            }

    # Print Quantitative Markdown Table
    print("\n" + "="*120)
    print("EXPERIMENT 08: TRACK A FIXED-SPEED TOPOLOGICAL EXPLORATION BENCHMARK RESULTS")
    print("="*120)
    print(f"{'Seed':<6} | {'Candidate':<32} | {'Cells':<6} | {'Goal Time':<10} | {'Path Len':<9} | {'Avg Spd':<8} | {'Peak Spd':<9} | {'Status':<8}")
    print("-" * 120)
    
    for seed in SEEDS:
        for cand in CANDIDATES:
            cid = cand["id"]
            if cid in results[seed]["runs"]:
                r = results[seed]["runs"][cid]
                print(f"Seed {seed:<2} | {r['name']:<32} | {r['cells_walked']:<6} | {r['finish_time']:<8.2f} s | {r['path_len']:<7.2f} m | {r['avg_speed']:<6.3f} | {r['peak_speed']:<7.3f} | SUCCESS")
        print("-" * 120)

    # Plot 2x3 high-resolution comparison figure
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=200)
    
    c = 2.0
    n = 5
    half = c / 2.0
    
    # Row 1: 2D Trajectories for Seeds 42, 100, 777
    for col_idx, seed in enumerate(SEEDS):
        ax = axes[0, col_idx]
        w = results[seed]["maze_walls"]
        
        # Draw cells & walls
        for i in range(n):
            for j in range(n):
                cx, cy = i * c, j * c
                rect = patches.Rectangle(
                    (cx - half, cy - half), c, c,
                    linewidth=0.5, edgecolor="#27272a", facecolor="#18181b", zorder=1
                )
                ax.add_patch(rect)
                
                # West & South walls
                if w[i, j, 2]:
                    ax.plot([cx - half, cx - half], [cy - half, cy + half], color="#a1a1aa", lw=3.0, zorder=2)
                if w[i, j, 3]:
                    ax.plot([cx - half, cx + half], [cy - half, cy - half], color="#a1a1aa", lw=3.0, zorder=2)
                # Outer East & North walls
                if i == n - 1 and w[i, j, 0]:
                    ax.plot([cx + half, cx + half], [cy - half, cy + half], color="#a1a1aa", lw=3.0, zorder=2)
                if j == n - 1 and w[i, j, 1]:
                    ax.plot([cx - half, cx + half], [cy + half, cy + half], color="#a1a1aa", lw=3.0, zorder=2)

        # Highlight Start and Goal
        s_rect = patches.Rectangle((-half, -half), c, c, facecolor="#064e3b", alpha=0.6, zorder=1)
        g_rect = patches.Rectangle(((n-1)*c - half, (n-1)*c - half), c, c, facecolor="#7f1d1d", alpha=0.6, zorder=1)
        ax.add_patch(s_rect)
        ax.add_patch(g_rect)
        ax.text(0.0, 0.0, "START (0,0)", color="#34d399", ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)
        ax.text((n-1)*c, (n-1)*c, "GOAL (4,4)", color="#f87171", ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)

        # Plot candidate trajectories
        for cand in CANDIDATES:
            cid = cand["id"]
            if cid in results[seed]["runs"]:
                r = results[seed]["runs"][cid]
                pos = r["pos"]
                lw = 2.4 if cid == "a3_lidar_g2" else (1.8 if cid == "a2_g2" else 1.4)
                ls = "-" if cid == "a3_lidar_g2" else ("--" if cid == "a2_g2" else ":")
                ax.plot(pos[:, 0], pos[:, 1], color=r["color"], lw=lw, linestyle=ls,
                        label=f"{r['label']} ({r['finish_time']:.1f}s, {r['path_len']:.1f}m)", zorder=4)

        ax.set_xlim(-half - 0.3, (n-1)*c + half + 0.3)
        ax.set_ylim(-half - 0.3, (n-1)*c + half + 0.3)
        ax.set_aspect("equal")
        ax.set_title(f"Random Maze Seed {seed} 2D Trajectories", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("X Position [m]", fontsize=9)
        ax.set_ylabel("Y Position [m]", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.3, color="#52525b")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85, facecolor="#09090b")

    # Row 2: Summary Bar Charts across Seeds
    # Panel 4: Completion Time (s)
    ax_time = axes[1, 0]
    # Panel 5: Path Distance (m)
    ax_dist = axes[1, 1]
    # Panel 6: Cells Walked
    ax_cells = axes[1, 2]

    x_indices = np.arange(len(SEEDS))
    bar_width = 0.25

    for idx_c, cand in enumerate(CANDIDATES):
        cid = cand["id"]
        times = [results[s]["runs"][cid]["finish_time"] for s in SEEDS if cid in results[s]["runs"]]
        dists = [results[s]["runs"][cid]["path_len"] for s in SEEDS if cid in results[s]["runs"]]
        cells = [results[s]["runs"][cid]["cells_walked"] for s in SEEDS if cid in results[s]["runs"]]
        
        offset = (idx_c - 1) * bar_width
        
        # Time bars
        bars_t = ax_time.bar(x_indices + offset, times, width=bar_width, color=cand["color"],
                             label=cand["label"], alpha=0.88, edgecolor="#ffffff", linewidth=0.5)
        for b in bars_t:
            yval = b.get_height()
            ax_time.text(b.get_x() + b.get_width()/2.0, yval + 0.6, f"{yval:.1f}s",
                         ha="center", va="bottom", fontsize=7.5, rotation=0, color="#f4f4f5")

        # Dist bars
        bars_d = ax_dist.bar(x_indices + offset, dists, width=bar_width, color=cand["color"],
                             label=cand["label"], alpha=0.88, edgecolor="#ffffff", linewidth=0.5)
        for b in bars_d:
            yval = b.get_height()
            ax_dist.text(b.get_x() + b.get_width()/2.0, yval + 0.5, f"{yval:.1f}m",
                         ha="center", va="bottom", fontsize=7.5, rotation=0, color="#f4f4f5")

        # Cells bars
        bars_c = ax_cells.bar(x_indices + offset, cells, width=bar_width, color=cand["color"],
                              label=cand["label"], alpha=0.88, edgecolor="#ffffff", linewidth=0.5)
        for b in bars_c:
            yval = b.get_height()
            ax_cells.text(b.get_x() + b.get_width()/2.0, yval + 0.3, f"{int(yval)}",
                          ha="center", va="bottom", fontsize=8, fontweight="bold", color="#f4f4f5")

    # Format Panel 4 (Time)
    ax_time.set_title("Completion Time by Seed (Lower is Better)", fontsize=11, fontweight="bold")
    ax_time.set_xticks(x_indices)
    ax_time.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=9)
    ax_time.set_ylabel("Time to Goal [s]", fontsize=9)
    ax_time.set_ylim(0, 52)
    ax_time.grid(True, axis="y", linestyle=":", alpha=0.3, color="#52525b")
    ax_time.legend(loc="upper left", fontsize=8, framealpha=0.85, facecolor="#09090b")

    # Format Panel 5 (Distance)
    ax_dist.set_title("Path Length by Seed (Lower is Better)", fontsize=11, fontweight="bold")
    ax_dist.set_xticks(x_indices)
    ax_dist.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=9)
    ax_dist.set_ylabel("Total Path Distance [m]", fontsize=9)
    ax_dist.set_ylim(0, 38)
    ax_dist.grid(True, axis="y", linestyle=":", alpha=0.3, color="#52525b")
    ax_dist.legend(loc="upper left", fontsize=8, framealpha=0.85, facecolor="#09090b")

    # Format Panel 6 (Cells)
    ax_cells.set_title("Cells Walked by Seed (Lower is Better)", fontsize=11, fontweight="bold")
    ax_cells.set_xticks(x_indices)
    ax_cells.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=9)
    ax_cells.set_ylabel("Cells Walked [count]", fontsize=9)
    ax_cells.set_ylim(0, 22)
    ax_cells.grid(True, axis="y", linestyle=":", alpha=0.3, color="#52525b")
    ax_cells.legend(loc="upper left", fontsize=8, framealpha=0.85, facecolor="#09090b")

    plt.suptitle("Experiment 08: Track A Fixed-Speed Topological Exploration Benchmark\n(6m Chest LiDAR Multi-Cell Ray-Cast Mapping + Frontier Tie-Breaking vs Myopic Scan_z at Locked 0.60 m/s)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = os.path.join(EXP_DIR, "track_a_lidar_random_mazes.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[FIGURE] Saved benchmark plot to {plot_path}")

    return results

if __name__ == "__main__":
    overwrite = "--overwrite" in sys.argv
    run_all_benchmarks(overwrite=overwrite)
    analyze_and_plot()
