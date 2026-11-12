"""Experiment 09: Track A Fixed-Speed Benchmark with Randomized Start & Goal Coordinates.

Evaluates 4 Candidates across 4 Distinct Maps with Arbitrary Start/Goal Ends:
- Map 1 (challenge_rev): Fixed Challenge 5x5 layout, Start (0, 4) -> Goal (4, 1) (Manhattan dist = 7)
- Map 2 (seed42_rand): Random Maze Seed 42, Start (4, 4) -> Goal (0, 1) (South-West traversal, Manhattan dist = 7)
- Map 3 (seed100_rand): Random Maze Seed 100, Start (4, 0) -> Goal (0, 4) (North-West traversal, Manhattan dist = 8)
- Map 4 (seed777_rand): Random Maze Seed 777, Start (0, 3) -> Goal (4, 0) (South-East traversal, Manhattan dist = 7)

Candidates:
- A0: Manhattan Center (Ablation Baseline)
- A1: Myopic Direct Cut
- A2_G2: Myopic G² Quintic Spline A*
- A3_Lidar_G2: 6m Chest LiDAR Ray-Cast Mapping + G² Quintic Spline A*

Under strictly locked Track A speed controls:
- speed = 0.60 m/s, min_speed = 0.30 m/s, max_turn = 1.00 rad/s
- max_acc = 1.40 m/s², max_ang_acc = 3.00 rad/s²
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from g1_maze_lab.terrains.maze import generate_maze, parse_layout, maze_to_ascii
from g1_maze_lab.tasks.maze.env_cfg import CHALLENGE_LAYOUT

PROJECT_ROOT = "/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze"
EXP_DIR = os.path.join(PROJECT_ROOT, "artifacts", "experiment_09_track_a_random_ends")
RUN_DIR = os.path.join(EXP_DIR, "run")
CHECKPOINT = "pretrained/g1_rough/model_pretrained.pt"

MAPS = [
    {
        "id": "challenge_rev",
        "title": "Map 1: Challenge Non-Corner",
        "subtitle": "Start (0,4) -> Goal (4,1) | Manhattan 7",
        "start": (0, 4),
        "goal": (4, 1),
        "layout_fn": lambda: parse_layout(CHALLENGE_LAYOUT),
        "layout_ascii": CHALLENGE_LAYOUT.strip(),
    },
    {
        "id": "seed42_rand",
        "title": "Map 2: Seed 42 South-West",
        "subtitle": "Start (4,4) -> Goal (0,1) | Manhattan 7",
        "start": (4, 4),
        "goal": (0, 1),
        "layout_fn": lambda: generate_maze(5, np.random.default_rng(42), 0.15),
        "layout_ascii": maze_to_ascii(generate_maze(5, np.random.default_rng(42), 0.15)),
    },
    {
        "id": "seed100_rand",
        "title": "Map 3: Seed 100 North-West",
        "subtitle": "Start (4,0) -> Goal (0,4) | Manhattan 8",
        "start": (4, 0),
        "goal": (0, 4),
        "layout_fn": lambda: generate_maze(5, np.random.default_rng(100), 0.15),
        "layout_ascii": maze_to_ascii(generate_maze(5, np.random.default_rng(100), 0.15)),
    },
    {
        "id": "seed777_rand",
        "title": "Map 4: Seed 777 South-East",
        "subtitle": "Start (0,3) -> Goal (4,0) | Manhattan 7",
        "start": (0, 3),
        "goal": (4, 0),
        "layout_fn": lambda: generate_maze(5, np.random.default_rng(777), 0.15),
        "layout_ascii": maze_to_ascii(generate_maze(5, np.random.default_rng(777), 0.15)),
    },
]

CANDIDATES = [
    {
        "id": "a0",
        "name": "A0 (Manhattan Center)",
        "label": "A0 (Manhattan)",
        "color": "#ef4444",  # Red
        "env": {
            "BENCHMARK_UPGRADED": "0",
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_TURN_EARLY": "0.35",
            "MAZE_DIRECT_CORNER": "0",
            "MAZE_LATERAL_MARGIN": "0",
            "MAZE_USE_ASTAR": "0",
            "MAZE_LIDAR_LOOKAHEAD": "0",
            "MAZE_TIME_LIMIT": "100.0",
        }
    },
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
            "MAZE_TIME_LIMIT": "100.0",
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
            "MAZE_LATERAL_MARGIN": "0",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_USE_ASTAR": "1",
            "MAZE_CURVE_MODE": "g2_quintic",
            "MAZE_R_FILLET": "0.65",
            "MAZE_LOOKAHEAD": "0.50",
            "MAZE_LIDAR_LOOKAHEAD": "0",
            "MAZE_TIME_LIMIT": "100.0",
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
            "MAZE_LATERAL_MARGIN": "0",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_USE_ASTAR": "1",
            "MAZE_CURVE_MODE": "g2_quintic",
            "MAZE_R_FILLET": "0.65",
            "MAZE_LOOKAHEAD": "0.50",
            "MAZE_LIDAR_LOOKAHEAD": "1",
            "MAZE_TIME_LIMIT": "100.0",
        }
    }
]

def run_all_benchmarks(overwrite=False, map_filter=None, cand_filter=None):
    os.makedirs(RUN_DIR, exist_ok=True)
    
    cmd_base = [
        "uv", "run", "python", "scripts/benchmark_autopilot.py",
        "--task", "G1-Maze-Challenge",
        "--num_envs", "1",
        "--checkpoint", CHECKPOINT
    ]
    
    for m in MAPS:
        mid = m["id"]
        if map_filter and mid != map_filter:
            continue
        start_str = f"{m['start'][0]},{m['start'][1]}"
        goal_str = f"{m['goal'][0]},{m['goal'][1]}"
        layout_str = m["layout_ascii"]
        
        for cand in CANDIDATES:
            cid = cand["id"]
            if cand_filter and cid != cand_filter:
                continue
            out_name = f"{mid}_{cid}.npz"
            out_path = os.path.join(RUN_DIR, out_name)
            
            if os.path.exists(out_path) and not overwrite:
                print(f"[SKIP] {out_name} already exists. Use --overwrite to rerun.")
                continue
                
            print(f"\n{'='*80}")
            print(f"RUNNING MAP: {m['title']} | CANDIDATE: {cand['name']}")
            print(f"Start: {m['start']} -> Goal: {m['goal']}")
            print(f"Target npz: {out_path}")
            print(f"{'='*80}")
            
            run_env = os.environ.copy()
            run_env.update(cand["env"])
            run_env["BENCHMARK_OUTPUT"] = out_path
            run_env["BENCHMARK_MODE_NAME"] = f"{mid}_{cand['id']}"
            run_env["MAZE_START_CELL"] = start_str
            run_env["MAZE_GOAL_CELL"] = goal_str
            run_env["MAZE_LAYOUT_OVERRIDE"] = layout_str
            
            res = subprocess.run(cmd_base, cwd=os.path.join(PROJECT_ROOT, "g1_maze_lab"), env=run_env)
            if res.returncode != 0:
                print(f"[ERROR] Run failed for Map {mid} {cand['id']} with code {res.returncode}")
                sys.exit(res.returncode)

def analyze_and_plot():
    results = {}
    
    for m in MAPS:
        mid = m["id"]
        w = m["layout_fn"]()
        results[mid] = {"info": m, "walls": w, "runs": {}}
        
        for cand in CANDIDATES:
            out_name = f"{mid}_{cand['id']}.npz"
            out_path = os.path.join(RUN_DIR, out_name)
            if not os.path.exists(out_path):
                print(f"[WARN] Missing {out_path}")
                continue
            
            data = np.load(out_path)
            t = data["t"]
            pos = data["pos"]
            lin_vel = data["lin_vel"]
            finish_time = float(data["finish_time"]) if not np.isnan(data["finish_time"]) else float(t[-1])
            is_success = not np.isnan(data["finish_time"])
            
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
            
            results[mid]["runs"][cand["id"]] = {
                "id": cand["id"],
                "name": cand["name"],
                "label": cand["label"],
                "color": cand["color"],
                "t": t,
                "pos": pos,
                "finish_time": finish_time,
                "is_success": is_success,
                "path_len": path_len,
                "avg_speed": avg_speed,
                "peak_speed": peak_speed,
                "cells_walked": cells_walked,
                "unique_cells": unique_cells,
                "walls": data["walls"],
            }

    # Print Quantitative Markdown Table
    print("\n" + "="*125)
    print("EXPERIMENT 09: TRACK A FIXED-SPEED BENCHMARK WITH RANDOMIZED START & GOAL RESULTS")
    print("="*125)
    print(f"{'Map':<16} | {'Candidate':<32} | {'Cells':<6} | {'Goal Time':<10} | {'Path Len':<9} | {'Avg Spd':<8} | {'Peak Spd':<9} | {'Status':<8}")
    print("-" * 125)
    
    for m in MAPS:
        mid = m["id"]
        for cand in CANDIDATES:
            cid = cand["id"]
            if cid in results[mid]["runs"]:
                r = results[mid]["runs"][cid]
                status = "SUCCESS" if r["is_success"] else "TIMEOUT"
                print(f"{mid:<16} | {r['name']:<32} | {r['cells_walked']:<6} | {r['finish_time']:<8.2f} s | {r['path_len']:<7.2f} m | {r['avg_speed']:<6.3f} | {r['peak_speed']:<7.3f} | {status}")
        print("-" * 125)

    # Plot 2x4 high-resolution comparison figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 12), dpi=200)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 0.9], hspace=0.28, wspace=0.22)
    
    c = 2.0
    n = 5
    half = c / 2.0
    
    # Row 1: 2D Trajectories for all 4 maps
    for col_idx, m in enumerate(MAPS):
        ax = fig.add_subplot(gs[0, col_idx])
        mid = m["id"]
        w = results[mid]["walls"]
        s_cell = m["start"]
        g_cell = m["goal"]
        
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

        # Highlight Start and Goal with coordinates
        sx, sy = s_cell[0] * c, s_cell[1] * c
        gx, gy = g_cell[0] * c, g_cell[1] * c
        s_rect = patches.Rectangle((sx - half, sy - half), c, c, facecolor="#064e3b", alpha=0.7, zorder=1)
        g_rect = patches.Rectangle((gx - half, gy - half), c, c, facecolor="#7f1d1d", alpha=0.7, zorder=1)
        ax.add_patch(s_rect)
        ax.add_patch(g_rect)
        ax.text(sx, sy, f"START\n({s_cell[0]},{s_cell[1]})", color="#34d399", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=3)
        ax.text(gx, gy, f"GOAL\n({g_cell[0]},{g_cell[1]})", color="#f87171", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=3)

        # Plot candidate trajectories
        for cand in CANDIDATES:
            cid = cand["id"]
            if cid in results[mid]["runs"]:
                r = results[mid]["runs"][cid]
                pos = r["pos"]
                lw = 2.4 if cid == "a3_lidar_g2" else (1.8 if cid == "a2_g2" else 1.2)
                ls = "-" if cid in ("a3_lidar_g2", "a2_g2") else ("--" if cid == "a1" else ":")
                status_str = f"{r['finish_time']:.1f}s" if r["is_success"] else "FAIL"
                ax.plot(pos[:, 0], pos[:, 1], color=r["color"], lw=lw, linestyle=ls,
                        label=f"{r['label']} ({status_str}, {r['path_len']:.1f}m)", zorder=4)

        ax.set_xlim(-half - 0.3, (n-1)*c + half + 0.3)
        ax.set_ylim(-half - 0.3, (n-1)*c + half + 0.3)
        ax.set_aspect("equal")
        ax.set_title(f"{m['title']}\n{m['subtitle']}", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel("X Position [m]", fontsize=8.5)
        ax.set_ylabel("Y Position [m]", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.3, color="#52525b")
        ax.legend(loc="lower right" if col_idx != 1 else "upper left", fontsize=7.0, framealpha=0.88, facecolor="#09090b")

    # Row 2: Bar charts (Panels 1: Completion Time, 2: Path Length, 3: Cells Walked, 4: Average Speed)
    bar_width = 0.18
    x_indices = np.arange(len(MAPS))
    map_labels = [m["id"].replace("_", "\n") for m in MAPS]

    # Panel 5: Time to Goal (s)
    ax_time = fig.add_subplot(gs[1, 0])
    # Panel 6: Path Length (m)
    ax_dist = fig.add_subplot(gs[1, 1])
    # Panel 7: Cells Walked (count)
    ax_cells = fig.add_subplot(gs[1, 2])
    # Panel 8: Average Speed (m/s)
    ax_speed = fig.add_subplot(gs[1, 3])

    for idx_c, cand in enumerate(CANDIDATES):
        cid = cand["id"]
        times = [results[m["id"]]["runs"][cid]["finish_time"] if cid in results[m["id"]]["runs"] else 0.0 for m in MAPS]
        dists = [results[m["id"]]["runs"][cid]["path_len"] if cid in results[m["id"]]["runs"] else 0.0 for m in MAPS]
        cells = [results[m["id"]]["runs"][cid]["cells_walked"] if cid in results[m["id"]]["runs"] else 0 for m in MAPS]
        speeds = [results[m["id"]]["runs"][cid]["avg_speed"] if cid in results[m["id"]]["runs"] else 0.0 for m in MAPS]
        
        offset = (idx_c - 1.5) * bar_width
        
        # Time bars
        bars_t = ax_time.bar(x_indices + offset, times, width=bar_width, color=cand["color"],
                             label=cand["label"], alpha=0.88, edgecolor="#ffffff", linewidth=0.5)
        for b in bars_t:
            yval = b.get_height()
            if yval > 0:
                ax_time.text(b.get_x() + b.get_width()/2.0, yval + 1.0, f"{yval:.1f}s",
                             ha="center", va="bottom", fontsize=6.8, rotation=0, color="#f4f4f5")

        # Dist bars
        bars_d = ax_dist.bar(x_indices + offset, dists, width=bar_width, color=cand["color"],
                             label=cand["label"], alpha=0.88, edgecolor="#ffffff", linewidth=0.5)
        for b in bars_d:
            yval = b.get_height()
            if yval > 0:
                ax_dist.text(b.get_x() + b.get_width()/2.0, yval + 0.6, f"{yval:.1f}m",
                             ha="center", va="bottom", fontsize=6.8, rotation=0, color="#f4f4f5")

        # Cells bars
        bars_c = ax_cells.bar(x_indices + offset, cells, width=bar_width, color=cand["color"],
                              label=cand["label"], alpha=0.88, edgecolor="#ffffff", linewidth=0.5)
        for b in bars_c:
            yval = b.get_height()
            if yval > 0:
                ax_cells.text(b.get_x() + b.get_width()/2.0, yval + 0.4, f"{int(yval)}",
                              ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#f4f4f5")

        # Speed bars
        bars_s = ax_speed.bar(x_indices + offset, speeds, width=bar_width, color=cand["color"],
                              label=cand["label"], alpha=0.88, edgecolor="#ffffff", linewidth=0.5)
        for b in bars_s:
            yval = b.get_height()
            if yval > 0:
                ax_speed.text(b.get_x() + b.get_width()/2.0, yval + 0.015, f"{yval:.2f}",
                              ha="center", va="bottom", fontsize=6.8, rotation=0, color="#f4f4f5")

    # Format Panel 5 (Time)
    ax_time.set_title("Completion Time by Map (Lower is Better)", fontsize=10.5, fontweight="bold")
    ax_time.set_xticks(x_indices)
    ax_time.set_xticklabels(map_labels, fontsize=8.5)
    ax_time.set_ylabel("Time to Goal [s]", fontsize=8.5)
    max_t = max([r["finish_time"] for m in MAPS for r in results[m["id"]]["runs"].values()] + [50.0])
    ax_time.set_ylim(0, max_t * 1.15)
    ax_time.grid(True, axis="y", linestyle=":", alpha=0.3, color="#52525b")
    ax_time.legend(loc="upper left", fontsize=7.5, framealpha=0.85, facecolor="#09090b")

    # Format Panel 6 (Distance)
    ax_dist.set_title("Path Length by Map (Lower is Better)", fontsize=10.5, fontweight="bold")
    ax_dist.set_xticks(x_indices)
    ax_dist.set_xticklabels(map_labels, fontsize=8.5)
    ax_dist.set_ylabel("Total Path Distance [m]", fontsize=8.5)
    max_d = max([r["path_len"] for m in MAPS for r in results[m["id"]]["runs"].values()] + [30.0])
    ax_dist.set_ylim(0, max_d * 1.15)
    ax_dist.grid(True, axis="y", linestyle=":", alpha=0.3, color="#52525b")
    ax_dist.legend(loc="upper left", fontsize=7.5, framealpha=0.85, facecolor="#09090b")

    # Format Panel 7 (Cells)
    ax_cells.set_title("Cells Walked by Map (Lower is Better)", fontsize=10.5, fontweight="bold")
    ax_cells.set_xticks(x_indices)
    ax_cells.set_xticklabels(map_labels, fontsize=8.5)
    ax_cells.set_ylabel("Cells Walked [count]", fontsize=8.5)
    max_c = max([r["cells_walked"] for m in MAPS for r in results[m["id"]]["runs"].values()] + [20])
    ax_cells.set_ylim(0, max_c * 1.2)
    ax_cells.grid(True, axis="y", linestyle=":", alpha=0.3, color="#52525b")
    ax_cells.legend(loc="upper left", fontsize=7.5, framealpha=0.85, facecolor="#09090b")

    # Format Panel 8 (Speed)
    ax_speed.set_title("Average Speed [m/s] (Locked @ 0.60 m/s)", fontsize=10.5, fontweight="bold")
    ax_speed.set_xticks(x_indices)
    ax_speed.set_xticklabels(map_labels, fontsize=8.5)
    ax_speed.set_ylabel("Average 2D Velocity [m/s]", fontsize=8.5)
    ax_speed.set_ylim(0, 0.70)
    ax_speed.axhline(0.60, color="#ef4444", linestyle="--", alpha=0.6, label="Track A Speed (0.60)")
    ax_speed.grid(True, axis="y", linestyle=":", alpha=0.3, color="#52525b")
    ax_speed.legend(loc="lower right", fontsize=7.2, framealpha=0.85, facecolor="#09090b")

    plt.suptitle("Experiment 09: Track A Fixed-Speed Benchmark with Randomized Start & Goal Coordinates\n(Non-Corner Start/Goal Ends across 4 Diverse Quadrants at Strictly Locked 0.60 m/s Speed)",
                 fontsize=13, fontweight="bold", y=0.98)
    
    plot_path = os.path.join(EXP_DIR, "track_a_random_ends_comparison.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[FIGURE] Saved benchmark plot to {plot_path}")

    return results

if __name__ == "__main__":
    overwrite = "--overwrite" in sys.argv
    map_filter = None
    cand_filter = None
    for arg in sys.argv:
        if arg.startswith("--map="):
            map_filter = arg.split("=")[1]
        elif arg.startswith("--cand="):
            cand_filter = arg.split("=")[1]
    run_all_benchmarks(overwrite=overwrite, map_filter=map_filter, cand_filter=cand_filter)
    analyze_and_plot()
