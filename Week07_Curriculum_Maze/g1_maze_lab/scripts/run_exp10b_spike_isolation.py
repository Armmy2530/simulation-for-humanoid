"""Experiment 10b: Isolated Spike & Slalom Corridor Benchmark + Method 1 Upgrade.

Benchmarking ONLY Method 1 (Immediate Replan Trigger + Extended Lead-In/Lead-Out Waypoints)
on a clean 5x5 single-corridor serpentine maze containing ONLY spike beds and 2-bed slaloms
(zero ramps, zero trenches, zero ice, zero rough boxes, zero start forks).

Compares 4 candidates across Course A and Course B in parallel (--num_envs 4):
1. C0_No_Avoid: Centerline baseline (none)
2. C1_Centroid: Naive centroid (centroid)
3. C2_Exp10_Old_Slalom: Unpatched Exp 10 slalom logic (old_slalom)
4. C3_Method1_LeadInOut: Proposed Method 1 with immediate replan + extended waypoints
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.join(PROJECT_ROOT, "artifacts", "experiment_10_track_a_hazard_slalom")
RUN_DIR = os.path.join(EXP_DIR, "run")
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "artifacts", "archive")
CHECKPOINT = os.path.join(PROJECT_ROOT, "g1_maze_lab", "pretrained", "g1_rough", "model_pretrained.pt")

CANDIDATES = [
    {
        "id": "C0_No_Avoid",
        "name": "C0: No Avoidance (Centerline)",
        "color": "#94a3b8",  # slate gray
        "style": "--",
        "env": {
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_R_FILLET": "0.65",
            "MAZE_LOOKAHEAD": "0.50",
            "MAZE_CURVE_MODE": "g2_quintic",
            "MAZE_USE_ASTAR": "1",
            "MAZE_LIDAR_LOOKAHEAD": "1",
            "MAZE_HAZARD_AVOIDANCE": "1",
            "MAZE_SPIKE_AVOID_MODE": "none",
            "MAZE_FORK_ARBITRATOR": "0",
            "MAZE_PERSIST_MAP": "0",
        },
    },
    {
        "id": "C1_Centroid",
        "name": "C1: Centroid Shift",
        "color": "#f59e0b",  # amber
        "style": "-.",
        "env": {
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_R_FILLET": "0.65",
            "MAZE_LOOKAHEAD": "0.50",
            "MAZE_CURVE_MODE": "g2_quintic",
            "MAZE_USE_ASTAR": "1",
            "MAZE_LIDAR_LOOKAHEAD": "1",
            "MAZE_HAZARD_AVOIDANCE": "1",
            "MAZE_SPIKE_AVOID_MODE": "centroid",
            "MAZE_FORK_ARBITRATOR": "0",
            "MAZE_PERSIST_MAP": "0",
        },
    },
    {
        "id": "C2_Exp10_Old_Slalom",
        "name": "C2: Exp 10 Old Slalom",
        "color": "#06b6d4",  # cyan
        "style": ":",
        "env": {
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_R_FILLET": "0.65",
            "MAZE_LOOKAHEAD": "0.50",
            "MAZE_CURVE_MODE": "g2_quintic",
            "MAZE_USE_ASTAR": "1",
            "MAZE_LIDAR_LOOKAHEAD": "1",
            "MAZE_HAZARD_AVOIDANCE": "1",
            "MAZE_SPIKE_AVOID_MODE": "old_slalom",
            "MAZE_FORK_ARBITRATOR": "0",
            "MAZE_PERSIST_MAP": "0",
        },
    },
    {
        "id": "C3_Method1_LeadInOut",
        "name": "C3: Method 1 + Holonomic Vy",
        "color": "#10b981",  # emerald green
        "style": "-",
        "env": {
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_R_FILLET": "0.65",
            "MAZE_LOOKAHEAD": "0.50",
            "MAZE_CURVE_MODE": "g2_quintic",
            "MAZE_USE_ASTAR": "1",
            "MAZE_LIDAR_LOOKAHEAD": "1",
            "MAZE_HAZARD_AVOIDANCE": "1",
            "MAZE_SPIKE_AVOID_MODE": "slalom_s_curve",
            "MAZE_FORK_ARBITRATOR": "0",
            "MAZE_PERSIST_MAP": "0",
        },
    },
]

COURSES = [
    {
        "id": "spike_course_A",
        "title": "Course A (2 Lane Beds + 2 Slaloms)",
        "mode": "A",
        "time_limit": 120.0,
    },
    {
        "id": "spike_course_B",
        "title": "Course B (Mirror-Flipped: 2 Slaloms + 2 Lane Beds)",
        "mode": "B",
        "time_limit": 120.0,
    },
]


def run_benchmarks(overwrite=False, course_filter=None):
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    timings = {}
    for c in COURSES:
        cid = c["id"]
        if course_filter and c["mode"] != course_filter:
            continue
        all_exist = all(
            os.path.exists(os.path.join(RUN_DIR, f"{cid}_{cand['id']}.npz"))
            for cand in CANDIDATES
        )
        if all_exist and not overwrite:
            print(f"[SKIP] All candidates for {cid} exist. Use --overwrite to rerun.")
            continue

        print(f"\n{'='*80}")
        print(f"BENCHMARK: {c['title']} (Parallel --num_envs {len(CANDIDATES)})")
        print(f"Time Limit: {c['time_limit']} s")
        print(f"{'='*80}")

        batch_spec = []
        for cand in CANDIDATES:
            out_path = os.path.join(RUN_DIR, f"{cid}_{cand['id']}.npz")
            batch_spec.append({
                "id": cand["id"],
                "mode_name": f"{cid}_{cand['id']}",
                "output": out_path,
                "env": cand["env"],
            })

        batch_json_path = os.path.join(ARCHIVE_DIR, f"batch_{cid}.json")
        with open(batch_json_path, "w") as f:
            json.dump(batch_spec, f, indent=2)

        cmd = [
            "uv", "run", "python", "scripts/benchmark_autopilot.py",
            "--task", "G1-Maze-SpikeIso",
            "--num_envs", str(len(CANDIDATES)),
            "--checkpoint", CHECKPOINT,
            "--batch-json", batch_json_path,
        ]

        run_env = os.environ.copy()
        run_env["MAZE_SPIKE_ISOLATION_TEST"] = c["mode"]
        run_env["MAZE_TIME_LIMIT"] = str(c["time_limit"])

        t0 = time.time()
        res = subprocess.run(cmd, cwd=os.path.join(PROJECT_ROOT, "g1_maze_lab"), env=run_env)
        elapsed = time.time() - t0
        timings[cid] = elapsed
        print(f"[BENCHMARK] Course {cid} finished in {elapsed:.2f} s wall-clock time (exit code {res.returncode})")

        if res.returncode != 0:
            print(f"[ERROR] Simulation failed for course {cid} with code {res.returncode}")
            sys.exit(res.returncode)

    return timings


def compute_obstacle_metrics(pos, hazards):
    """Compute number of obstacles cleared and minimum clearance to spike beds."""
    spike_entries = hazards.get("spike", [])
    spike_rects = [e[3:] for e in spike_entries]  # list of (x0, y0, x1, y1)

    # 4 obstacle cell centers in order along the path
    # Course A: (2,0), (3,2), (0,3), (2,4)
    # Course B: (2,0), (4,1), (3,2), (2,4)
    hazard_cells = list({(int(e[0]), int(e[1])) for e in spike_entries})
    # Sort hazard cells by appearance along the 16-cell path:
    path_order = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (4, 1), (4, 2),
        (3, 2), (2, 2), (1, 2), (0, 2),
        (0, 3), (0, 4),
        (1, 4), (2, 4), (3, 4), (4, 4),
    ]
    hazard_cells.sort(key=lambda c: path_order.index(c) if c in path_order else 999)

    # Detect reset events (teleport back to near (0, 0))
    dists_from_prev = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    jump_indices = np.where(dists_from_prev > 1.5)[0]

    crash_positions = []
    for ji in jump_indices:
        crash_positions.append(pos[ji])

    # Evaluate obstacle clearance per attempt segment
    jump_indices_with_ends = [0] + list(jump_indices) + [len(pos)]
    attempt_cleared_list = []
    for a_idx in range(len(jump_indices_with_ends) - 1):
        s_idx = jump_indices_with_ends[a_idx]
        e_idx = jump_indices_with_ends[a_idx + 1]
        seg_pos = pos[s_idx:e_idx]
        if len(seg_pos) < 15:
            continue
        crashed = a_idx < len(jump_indices)
        crash_p = pos[jump_indices[a_idx]] if crashed else None

        seg_cleared = 0
        for cell in hazard_cells:
            cx, cy = cell[0] * 2.0, cell[1] * 2.0
            dists = np.linalg.norm(seg_pos - np.array([cx, cy]), axis=1)
            # Reached cell
            if (dists < 0.9).any():
                if crashed and crash_p is not None and abs(crash_p[0] - cx) <= 1.05 and abs(crash_p[1] - cy) <= 1.05:
                    break  # crashed in this cell
                else:
                    seg_cleared += 1
        attempt_cleared_list.append(seg_cleared)

    cleared_count = max(attempt_cleared_list) if attempt_cleared_list else 0

    # Minimum clearance from pelvis to any spike bed
    min_dist_overall = 999.0
    for r in spike_rects:
        x0, y0, x1, y1 = r
        px = pos[:, 0]
        py = pos[:, 1]
        dx = np.maximum(0.0, np.maximum(x0 - px, px - x1))
        dy = np.maximum(0.0, np.maximum(y0 - py, py - y1))
        dists = np.sqrt(dx**2 + dy**2)
        m_d = float(np.min(dists))
        if m_d < min_dist_overall:
            min_dist_overall = m_d

    return cleared_count, min_dist_overall, crash_positions


def analyze_and_plot():
    results = {}

    for c in COURSES:
        cid = c["id"]
        results[cid] = {}
        for cand in CANDIDATES:
            cand_id = cand["id"]
            npz_path = os.path.join(RUN_DIR, f"{cid}_{cand_id}.npz")
            if not os.path.exists(npz_path):
                print(f"[WARN] Missing {npz_path}")
                continue
            data = np.load(npz_path, allow_pickle=True)
            t = data["t"]
            pos = data["pos"]
            finish = float(data["finish_time"]) if "finish_time" in data else float("nan")
            falls = int(data["falls"]) if "falls" in data else 0
            hazards = json.loads(str(data["hazards_json"])) if "hazards_json" in data else (json.loads(str(data["hazards"])) if "hazards" in data else {})
            walls = data["walls"] if "walls" in data else None

            cleared, min_clearance, crash_pts = compute_obstacle_metrics(pos, hazards)
            is_goal = not np.isnan(finish)
            if is_goal:
                cleared = 4

            results[cid][cand_id] = {
                "t": t,
                "pos": pos,
                "finish_time": finish,
                "is_goal": is_goal,
                "falls": falls,
                "cleared": cleared,
                "min_clearance": min_clearance,
                "crash_pts": crash_pts,
                "hazards": hazards,
                "walls": walls,
            }

    # Print Quantitative Table
    print("\n" + "=" * 105)
    print("EXPERIMENT 10b QUANTITATIVE BENCHMARK: ISOLATED SPIKE & SLALOM CORRIDOR (TRACK A SPEED 0.60 m/s)")
    print("=" * 105)
    print(f"{'Course':<16} | {'Candidate':<24} | {'Goal?':<6} | {'Time (s)':<10} | {'Falls/Resets':<12} | {'Cleared':<9} | {'Min Clearance':<14}")
    print("-" * 105)

    for c in COURSES:
        cid = c["id"]
        for cand in CANDIDATES:
            cand_id = cand["id"]
            r = results.get(cid, {}).get(cand_id)
            if not r:
                continue
            goal_str = "YES" if r["is_goal"] else "NO"
            time_str = f"{r['finish_time']:.2f} s" if r["is_goal"] else f">{c['time_limit']:.0f} s (DNF)"
            cleared_str = f"{r['cleared']}/4"
            clear_str = f"{r['min_clearance']:.3f} m"
            print(f"{c['title'][:16]:<16} | {cand['name']:<24} | {goal_str:<6} | {time_str:<10} | {r['falls']:<12} | {cleared_str:<9} | {clear_str:<14}")
        print("-" * 105)

    # Generate Professional Multi-Panel Figure
    fig = plt.figure(figsize=(22, 14), dpi=300, facecolor="#0f172a")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.22)

    # Top Row: Course A (left 1.5 cols) and Course B (right 1.5 cols)
    ax_top_a = fig.add_subplot(gs[0, :2])
    ax_top_b = fig.add_subplot(gs[0, 2])

    def draw_top_down(ax, cid, title):
        ax.set_facecolor("#1e293b")
        c_res = results.get(cid, {})
        if not c_res:
            return

        first_data = list(c_res.values())[0]
        walls = first_data["walls"]
        hazards = first_data["hazards"]
        n = 5
        c_size = 2.0

        # Draw grid and walls
        for i in range(n):
            for j in range(n):
                cx, cy = i * c_size, j * c_size
                # Fill corridor background
                ax.fill([cx - 1.0, cx + 1.0, cx + 1.0, cx - 1.0],
                        [cy - 1.0, cy - 1.0, cy + 1.0, cy + 1.0],
                        color="#0f172a", alpha=0.3, zorder=1)

                if walls is not None:
                    w = walls[i, j]
                    # East
                    if w[0]:
                        ax.plot([cx + 1.0, cx + 1.0], [cy - 1.0, cy + 1.0], color="#64748b", lw=2.5, zorder=3)
                    # North
                    if w[1]:
                        ax.plot([cx - 1.0, cx + 1.0], [cy + 1.0, cy + 1.0], color="#64748b", lw=2.5, zorder=3)
                    # West
                    if w[2]:
                        ax.plot([cx - 1.0, cx - 1.0], [cy - 1.0, cy + 1.0], color="#64748b", lw=2.5, zorder=3)
                    # South
                    if w[3]:
                        ax.plot([cx - 1.0, cx + 1.0], [cy - 1.0, cy - 1.0], color="#64748b", lw=2.5, zorder=3)

        # Draw spike beds
        spike_entries = hazards.get("spike", [])
        for sp in spike_entries:
            ci, cj, axis, x0, y0, x1, y1 = sp
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="#ef4444", edgecolor="#b91c1c", alpha=0.55, hatch="///", zorder=4
            )
            ax.add_patch(rect)

        # Draw Start and Goal
        ax.scatter([0.0], [0.0], s=180, color="#22c55e", edgecolors="#ffffff", lw=2, zorder=6, label="Start (0,0)")
        ax.scatter([8.0], [8.0], s=260, color="#eab308", marker="*", edgecolors="#ffffff", lw=2, zorder=6, label="Goal (4,4)")

        # Draw Candidate Trajectories
        for cand in CANDIDATES:
            cand_id = cand["id"]
            cdat = c_res.get(cand_id)
            if not cdat:
                continue
            pos = cdat["pos"]

            # Break trajectory at jumps/resets for clean plotting
            dists = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            split_idx = np.where(dists > 1.2)[0] + 1
            segments = np.split(pos, split_idx)

            for s_idx, seg in enumerate(segments):
                lbl = cand["name"] if s_idx == 0 else None
                ax.plot(seg[:, 0], seg[:, 1], color=cand["color"], linestyle=cand["style"],
                        lw=2.2 if "Method1" in cand_id else 1.8, alpha=0.9, zorder=5, label=lbl)

            # Mark crashes
            for cp in cdat["crash_pts"]:
                ax.scatter(cp[0], cp[1], s=130, marker="x", color="#dc2626", lw=3.0, zorder=7)

        ax.set_title(title, fontsize=14, fontweight="bold", color="#f8fafc", pad=10)
        ax.set_xlim(-1.3, 9.3)
        ax.set_ylim(-1.3, 9.3)
        ax.set_aspect("equal")
        ax.tick_params(colors="#94a3b8", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, color="#334155", linestyle=":", alpha=0.4)

    draw_top_down(ax_top_a, "spike_course_A", "Course A: 5x5 Serpentine Course (2 Single-Lane Beds + 2 Slaloms)")
    ax_top_a.legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", fontsize=9, labelcolor="#e2e8f0")

    draw_top_down(ax_top_b, "spike_course_B", "Course B: Mirror-Flipped 5x5 Course (2 Slaloms + 2 Lane Beds)")

    # Bottom Row: Zoomed In Panels
    ax_z1 = fig.add_subplot(gs[1, 0])
    ax_z2 = fig.add_subplot(gs[1, 1])
    ax_z3 = fig.add_subplot(gs[1, 2])

    def draw_zoom_cell(ax, cid, cell_xy, title, show_slalom=False):
        ax.set_facecolor("#1e293b")
        c_res = results.get(cid, {})
        if not c_res:
            return

        first_data = list(c_res.values())[0]
        hazards = first_data["hazards"]
        ci, cj = cell_xy
        cx, cy = ci * 2.0, cj * 2.0

        # Draw corridor walls
        ax.axhline(cy + 0.925, color="#64748b", lw=3.0, zorder=3)
        ax.axhline(cy - 0.925, color="#64748b", lw=3.0, zorder=3)

        # Draw spikes inside or touching this cell
        spike_entries = hazards.get("spike", [])
        for sp in spike_entries:
            sci, scj, axis, x0, y0, x1, y1 = sp
            if sci == ci and scj == cj:
                rect = patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor="#ef4444", edgecolor="#b91c1c", alpha=0.6, hatch="////", zorder=4
                )
                ax.add_patch(rect)

        # Plot candidate segments through this cell
        for cand in CANDIDATES:
            cand_id = cand["id"]
            cdat = c_res.get(cand_id)
            if not cdat:
                continue
            pos = cdat["pos"]

            # Filter points near this cell
            in_cell = (pos[:, 0] >= cx - 1.3) & (pos[:, 0] <= cx + 1.3) & (pos[:, 1] >= cy - 1.1) & (pos[:, 1] <= cy + 1.1)
            if not in_cell.any():
                continue
            pts = pos[in_cell]

            # Break at jumps
            dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            split_idx = np.where(dists > 0.4)[0] + 1
            segs = np.split(pts, split_idx)

            for s_idx, seg in enumerate(segs):
                if len(seg) < 2:
                    continue
                ax.plot(seg[:, 0], seg[:, 1], color=cand["color"], linestyle=cand["style"],
                        lw=2.6 if "Method1" in cand_id else 1.8, alpha=0.95, zorder=5)

                if "Method1" in cand_id and s_idx == 0:
                    # Shaded foot-swing envelope (+-0.16m lateral)
                    dx = np.gradient(seg[:, 0])
                    dy = np.gradient(seg[:, 1])
                    ds = np.hypot(dx, dy) + 1e-6
                    nx, ny = -dy / ds * 0.16, dx / ds * 0.16
                    left_b = np.column_stack([seg[:, 0] + nx, seg[:, 1] + ny])
                    right_b = np.column_stack([seg[:, 0] - nx, seg[:, 1] - ny])
                    poly = np.vstack([left_b, right_b[::-1]])
                    poly_patch = patches.Polygon(poly, closed=True, facecolor="#10b981", alpha=0.18, zorder=4)
                    ax.add_patch(poly_patch)

            # Crashes inside this zoom
            for cp in cdat["crash_pts"]:
                if abs(cp[0] - cx) <= 1.2 and abs(cp[1] - cy) <= 1.0:
                    ax.scatter(cp[0], cp[1], s=160, marker="x", color="#dc2626", lw=3.2, zorder=7)

        # Plot Method 1 Waypoints
        if cell_xy == (2, 0):
            # Single-lane bed (Eastbound -> Shift North v = +0.52)
            wps = [(cx - 1.05, cy + 0.52), (cx - 0.60, cy + 0.52), (cx + 0.60, cy + 0.52), (cx + 1.05, cy + 0.52)]
            wp_x, wp_y = zip(*wps)
            ax.plot(wp_x, wp_y, color="#38bdf8", linestyle="--", lw=1.5, alpha=0.85, zorder=6)
            ax.scatter(wp_x, wp_y, s=80, marker="D", facecolor="#38bdf8", edgecolor="#0284c7", lw=1.5, zorder=6, label="M1 Waypoints")
            for idx, (wx, wy) in enumerate(wps):
                tag = f"WP{idx+1} ({[-1.05, -0.60, 0.60, 1.05][idx]:+.2f}m)"
                ax.annotate(tag, (wx, wy), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7.5, color="#38bdf8", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="#0f172a", ec="#38bdf8", alpha=0.8))
        elif cell_xy == (3, 2):
            # 2-Bed Slalom (Westbound -> North entry v=+0.52, South exit v=-0.52)
            wps = [(cx + 1.10, cy + 0.52), (cx + 0.36, cy + 0.52), (cx, cy), (cx - 0.36, cy - 0.52), (cx - 1.10, cy - 0.52)]
            wp_x, wp_y = zip(*wps)
            ax.plot(wp_x, wp_y, color="#38bdf8", linestyle="--", lw=1.5, alpha=0.85, zorder=6)
            ax.scatter(wp_x, wp_y, s=80, marker="D", facecolor="#38bdf8", edgecolor="#0284c7", lw=1.5, zorder=6, label="M1 Waypoints")
            tags = ["WP1 (+1.10m)", "WP2 (+0.36m)", "WP3 (Gap 0.0m)", "WP4 (-0.36m)", "WP5 (-1.10m)"]
            for idx, (wx, wy) in enumerate(wps):
                ax.annotate(tags[idx], (wx, wy), textcoords="offset points", xytext=(0, 7 if wy >= cy else -13),
                            ha="center", fontsize=7.5, color="#38bdf8", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="#0f172a", ec="#38bdf8", alpha=0.8))
        elif cell_xy == (2, 4):
            # 2-Bed Slalom (Eastbound -> South entry v=-0.52, North exit v=+0.52)
            wps = [(cx - 1.10, cy - 0.52), (cx - 0.36, cy - 0.52), (cx, cy), (cx + 0.36, cy + 0.52), (cx + 1.10, cy + 0.52)]
            wp_x, wp_y = zip(*wps)
            ax.plot(wp_x, wp_y, color="#38bdf8", linestyle="--", lw=1.5, alpha=0.85, zorder=6)
            ax.scatter(wp_x, wp_y, s=80, marker="D", facecolor="#38bdf8", edgecolor="#0284c7", lw=1.5, zorder=6, label="M1 Waypoints")
            tags = ["WP1 (-1.10m)", "WP2 (-0.36m)", "WP3 (Gap 0.0m)", "WP4 (+0.36m)", "WP5 (+1.10m)"]
            for idx, (wx, wy) in enumerate(wps):
                ax.annotate(tags[idx], (wx, wy), textcoords="offset points", xytext=(0, 7 if wy >= cy else -13),
                            ha="center", fontsize=7.5, color="#38bdf8", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="#0f172a", ec="#38bdf8", alpha=0.8))

        ax.set_title(title, fontsize=12, fontweight="bold", color="#f8fafc", pad=8)
        ax.set_xlim(cx - 1.25, cx + 1.25)
        ax.set_ylim(cy - 1.05, cy + 1.05)
        ax.set_aspect("equal")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, color="#334155", linestyle=":", alpha=0.4)

    # Zoom 1: Single Lane Bed at (2,0) Eastbound
    draw_zoom_cell(ax_z1, "spike_course_A", (2, 0), "Single-Lane Bed (2,0) [Eastbound -> Shift North]")

    # Zoom 2: 2-Bed Slalom at (3,2) Westbound
    draw_zoom_cell(ax_z2, "spike_course_A", (3, 2), "2-Bed Slalom (3,2) [Westbound -> S-Curve Weave]")

    # Zoom 3: 2-Bed Slalom at (2,4) Eastbound
    draw_zoom_cell(ax_z3, "spike_course_A", (2, 4), "2-Bed Slalom (2,4) [Eastbound -> S-Curve Weave]")

    out_png = os.path.join(EXP_DIR, "spike_isolation_method1_comparison.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\n[PLOT] Comparison plot saved to: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 10b: Spike Isolation Benchmark.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing simulation runs.")
    parser.add_argument("--plot-only", action="store_true", help="Only generate analysis and plots from existing runs.")
    parser.add_argument("--course", choices=["A", "B"], default=None, help="Run only Course A or Course B.")
    args = parser.parse_args()

    if not args.plot_only:
        run_benchmarks(overwrite=args.overwrite, course_filter=args.course)

    analyze_and_plot()


if __name__ == "__main__":
    main()
