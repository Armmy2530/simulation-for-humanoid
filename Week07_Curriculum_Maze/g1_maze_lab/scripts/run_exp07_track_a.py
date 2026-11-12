"""Experiment 07: Track A Fixed-Speed Path Planning Benchmark.

Evaluates 4 configurations under strictly locked speed & acceleration limits:
- A0: Manhattan Center-to-Center (no corner cut)
- A1: Single-Waypoint Direct Corner Cut
- A2_G1: G^1 Quadratic Bezier A*
- A2_G2: G^2 Quintic Hermite Spline + Curvature Feedforward
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = "/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze"
EXP_DIR = os.path.join(PROJECT_ROOT, "artifacts", "experiment_07_track_a_g2_spline")
RUN_DIR = os.path.join(EXP_DIR, "run")
CHECKPOINT = "pretrained/g1_rough/model_pretrained.pt"

CONFIGS = [
    {
        "id": "A0",
        "name": "A0 (Manhattan Center)",
        "output": os.path.join(RUN_DIR, "a0_manhattan_center.npz"),
        "color": "#ef4444",  # Red
        "env": {
            "BENCHMARK_OUTPUT": os.path.join(RUN_DIR, "a0_manhattan_center.npz"),
            "BENCHMARK_MODE_NAME": "A0_Manhattan",
            "BENCHMARK_UPGRADED": "0",
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_TURN_EARLY": "0.35",
            "MAZE_DIRECT_CORNER": "0",
            "MAZE_LATERAL_MARGIN": "0",
            "MAZE_USE_ASTAR": "0",
        }
    },
    {
        "id": "A1",
        "name": "A1 (Direct Corner Cut)",
        "output": os.path.join(RUN_DIR, "a1_direct_corner_cut.npz"),
        "color": "#f59e0b",  # Amber
        "env": {
            "BENCHMARK_OUTPUT": os.path.join(RUN_DIR, "a1_direct_corner_cut.npz"),
            "BENCHMARK_MODE_NAME": "A1_Direct_Corner_Cut",
            "BENCHMARK_UPGRADED": "1",
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_TURN_EARLY": "0.60",
            "MAZE_CORNER_CUT": "0.38",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_USE_ASTAR": "0",
        }
    },
    {
        "id": "A2_G1",
        "name": "A2_G1 (G¹ Bezier A*)",
        "output": os.path.join(RUN_DIR, "a2_g1_bezier_astar.npz"),
        "color": "#3b82f6",  # Blue
        "env": {
            "BENCHMARK_OUTPUT": os.path.join(RUN_DIR, "a2_g1_bezier_astar.npz"),
            "BENCHMARK_MODE_NAME": "A2_G1_Bezier_AStar",
            "BENCHMARK_UPGRADED": "1",
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_TURN_EARLY": "0.60",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_USE_ASTAR": "1",
            "MAZE_CURVE_MODE": "bezier",
            "MAZE_R_FILLET": "0.60",
            "MAZE_LOOKAHEAD": "0.65",
        }
    },
    {
        "id": "A2_G2",
        "name": "A2_G2 (G² Quintic A*)",
        "output": os.path.join(RUN_DIR, "a2_g2_quintic_astar.npz"),
        "color": "#10b981",  # Emerald Green
        "env": {
            "BENCHMARK_OUTPUT": os.path.join(RUN_DIR, "a2_g2_quintic_astar.npz"),
            "BENCHMARK_MODE_NAME": "A2_G2_Quintic_AStar",
            "BENCHMARK_UPGRADED": "1",
            "MAZE_SPEED": "0.60",
            "MAZE_MIN_SPEED": "0.30",
            "MAZE_MAX_TURN": "1.00",
            "MAZE_ACC": "1.40,3.00",
            "MAZE_TURN_EARLY": "0.60",
            "MAZE_DIRECT_CORNER": "1",
            "MAZE_LATERAL_MARGIN": "1",
            "MAZE_USE_ASTAR": "1",
            "MAZE_CURVE_MODE": "g2_quintic",
            "MAZE_R_FILLET": "0.65",
            "MAZE_LOOKAHEAD": "0.50",
        }
    }
]

def run_benchmark(cfg):
    print(f"\n{'='*70}")
    print(f"RUNNING CANDIDATE: {cfg['name']}")
    print(f"Target npz: {cfg['output']}")
    print(f"{'='*70}")
    
    cmd = [
        "uv", "run", "python", "scripts/benchmark_autopilot.py",
        "--task", "G1-Maze-Challenge",
        "--num_envs", "1",
        "--checkpoint", CHECKPOINT
    ]
    
    run_env = os.environ.copy()
    run_env.update(cfg["env"])
    
    res = subprocess.run(cmd, cwd=os.path.join(PROJECT_ROOT, "g1_maze_lab"), env=run_env)
    if res.returncode != 0:
        print(f"Error running candidate {cfg['name']}, exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"Candidate {cfg['name']} completed successfully!")

def analyze_and_plot():
    os.makedirs(RUN_DIR, exist_ok=True)
    import sys
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "g1_maze_lab", "tests"))
    from test_maze_logic import CHALLENGE
    from g1_maze_lab.terrains.maze import parse_layout
    
    true_walls = parse_layout(CHALLENGE)
    c = 2.0
    n = 5
    half = c / 2.0

    # Wall segments for exact distance calculation
    wall_segs = []
    wall_segs.append((np.array([-half, -half]), np.array([n*c - half, -half])))
    wall_segs.append((np.array([-half, n*c - half]), np.array([n*c - half, n*c - half])))
    wall_segs.append((np.array([-half, -half]), np.array([-half, n*c - half])))
    wall_segs.append((np.array([n*c - half, -half]), np.array([n*c - half, n*c - half])))

    for i in range(n):
        for j in range(n):
            cx, cy = i * c, j * c
            if true_walls[i, j, 0] and i < n - 1:
                wall_segs.append((np.array([cx + half, cy - half]), np.array([cx + half, cy + half])))
            if true_walls[i, j, 1] and j < n - 1:
                wall_segs.append((np.array([cx - half, cy + half]), np.array([cx + half, cy + half])))

    def point_to_seg_dist(pts, p1, p2):
        v = p2 - p1
        l2 = np.dot(v, v)
        u = np.clip(np.sum((pts - p1) * v, axis=1) / l2, 0.0, 1.0)
        proj = p1 + np.outer(u, v)
        return np.linalg.norm(pts - proj, axis=1)

    results = []

    for cfg in CONFIGS:
        if not os.path.exists(cfg["output"]):
            print(f"Missing file: {cfg['output']}")
            continue
        data = np.load(cfg["output"])
        t = data["t"]
        pos = data["pos"]
        lin_vel = data["lin_vel"]
        cmd = data["cmd"]
        yaw = data["yaw"]
        walls = data["walls"]
        finish_time = float(data["finish_time"]) if not np.isnan(data["finish_time"]) else t[-1]
        
        body_vel = data["body_vel"] if "body_vel" in data else lin_vel
        ang_vel = data["ang_vel"] if "ang_vel" in data else np.zeros((len(t), 3))
        rpy = data["rpy"] if "rpy" in data else np.zeros((len(t), 3))
        
        diffs = np.diff(pos, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        path_len = float(np.sum(dists))
        
        speed_2d = np.linalg.norm(lin_vel, axis=1)
        avg_speed = float(np.mean(speed_2d))
        peak_speed = float(np.max(speed_2d))
        momentum_ratio = float(avg_speed / 0.60)
        
        yaw_rate = ang_vel[:, 2] if ang_vel.ndim > 1 else np.gradient(yaw, t)
        dt = np.mean(np.diff(t))
        cum_heading_change = float(np.sum(np.abs(yaw_rate) * dt))
        num_stops = int(np.sum(speed_2d < 0.15))
        
        yaw_acc = np.gradient(yaw_rate, t)
        rms_yaw_acc = float(np.sqrt(np.mean(yaw_acc ** 2)))
        
        roll_deg = np.rad2deg(rpy[:, 0]) if rpy.ndim > 1 else np.zeros_like(t)
        pitch_deg = np.rad2deg(rpy[:, 1]) if rpy.ndim > 1 else np.zeros_like(t)
        roll_std = float(np.std(roll_deg))
        pitch_std = float(np.std(pitch_deg))
        
        max_cmd_v = float(np.max(cmd[:, 0]))
        max_cmd_w = float(np.max(np.abs(cmd[:, 2])))

        all_dists = [point_to_seg_dist(pos, p1, p2) for p1, p2 in wall_segs]
        min_clearance = float(np.min(np.min(all_dists, axis=0)))

        results.append({
            "id": cfg["id"],
            "name": cfg["name"],
            "color": cfg["color"],
            "t": t,
            "pos": pos,
            "lin_vel": lin_vel,
            "speed_2d": speed_2d,
            "yaw_rate": yaw_rate,
            "yaw_acc": yaw_acc,
            "walls": walls,
            "finish_time": finish_time,
            "path_len": path_len,
            "avg_speed": avg_speed,
            "peak_speed": peak_speed,
            "momentum_ratio": momentum_ratio,
            "cum_heading_change": cum_heading_change,
            "num_stops": num_stops,
            "rms_yaw_acc": rms_yaw_acc,
            "roll_std": roll_std,
            "pitch_std": pitch_std,
            "max_cmd_v": max_cmd_v,
            "max_cmd_w": max_cmd_w,
            "min_clearance": min_clearance,
            "falls": 0,
        })

    # Print summary table
    print("\n" + "="*110)
    print("TRACK A FIXED-SPEED BENCHMARK RESULTS SUMMARY (SPEED LOCKED: 0.60 m/s)")
    print("="*110)
    header = f"{'Candidate':<24} | {'Goal Time':<10} | {'Path Len':<9} | {'Avg Spd':<8} | {'Peak Spd':<9} | {'RMS α_z':<9} | {'Cum ΔYaw':<9} | {'Min Wall':<9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<24} | {r['finish_time']:<8.2f} s | {r['path_len']:<7.2f} m | {r['avg_speed']:<6.3f} | {r['peak_speed']:<7.3f} | {r['rms_yaw_acc']:<7.2f} | {r['cum_heading_change']:<7.2f} | {r['min_clearance']:<7.3f} m")
    print("="*110)

    # Plot 4-panel figure
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    fig.patch.set_facecolor('#0b0f19')
    for ax in axs.flat:
        ax.set_facecolor('#111827')
        ax.grid(True, linestyle='--', alpha=0.25, color='#4b5563')
        ax.tick_params(colors='#e5e7eb', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#374151')

    # Panel 1: 2D Maze Trajectory
    ax_path = axs[0, 0]
    ax_path.set_title("(1) 2D Maze Trajectories (Track A: Speed Locked at 0.60 m/s)", fontsize=13, fontweight='bold', color='#f9fafb', pad=12)
    ax_path.set_xlabel("X Position [m]", fontsize=11, color='#e5e7eb')
    ax_path.set_ylabel("Y Position [m]", fontsize=11, color='#e5e7eb')

    # Draw exact outer boundaries
    ax_path.plot([-half, n*c - half], [-half, -half], color='#64748b', lw=3.5, solid_capstyle='round')
    ax_path.plot([-half, n*c - half], [n*c - half, n*c - half], color='#64748b', lw=3.5, solid_capstyle='round')
    ax_path.plot([-half, -half], [-half, n*c - half], color='#64748b', lw=3.5, solid_capstyle='round')
    ax_path.plot([n*c - half, n*c - half], [-half, n*c - half], color='#64748b', lw=3.5, solid_capstyle='round')

    # Inner walls
    for i in range(n):
        for j in range(n):
            cx, cy = i * c, j * c
            if true_walls[i, j, 0] and i < n - 1:
                ax_path.plot([cx + half, cx + half], [cy - half, cy + half], color='#64748b', lw=3.5, solid_capstyle='round')
            if true_walls[i, j, 1] and j < n - 1:
                ax_path.plot([cx - half, cx + half], [cy + half, cy + half], color='#64748b', lw=3.5, solid_capstyle='round')

    # Start and Goal patches
    start_patch = patches.Rectangle((-1.0, -1.0), 2.0, 2.0, facecolor='#38bdf8', alpha=0.18, edgecolor='#0284c7', lw=1.5)
    goal_patch = patches.Rectangle((7.0, 7.0), 2.0, 2.0, facecolor='#10b981', alpha=0.18, edgecolor='#059669', lw=1.5)
    ax_path.add_patch(start_patch)
    ax_path.add_patch(goal_patch)
    ax_path.text(0.0, 0.0, "START\n(0,0)", color='#38bdf8', ha='center', va='center', fontweight='bold', fontsize=9)
    ax_path.text(8.0, 8.0, "GOAL\n(4,4)", color='#34d495', ha='center', va='center', fontweight='bold', fontsize=9)

    for r in results:
        lw = 3.0 if "G2" in r["id"] else (2.4 if "G1" in r["id"] else 1.8)
        ls = '-' if "G2" in r["id"] else ('--' if "G1" in r["id"] else '-.')
        ax_path.plot(r["pos"][:, 0], r["pos"][:, 1], label=f"{r['name']}: {r['finish_time']:.2f}s | {r['path_len']:.2f}m", color=r["color"], lw=lw, linestyle=ls, alpha=0.95)
    
    ax_path.legend(loc='lower right', fontsize=9.5, facecolor='#111827', edgecolor='#374151')
    ax_path.set_xlim(-1.8, 9.8)
    ax_path.set_ylim(-1.8, 9.8)
    ax_path.set_aspect('equal')

    # Panel 2: Realized Forward Speed v(t) Profile
    ax_vel = axs[0, 1]
    ax_vel.set_title("(2) Realized Forward Linear Speed v(t) Profile", fontsize=13, fontweight='bold', color='#f9fafb', pad=12)
    ax_vel.set_xlabel("Time [s]", fontsize=11, color='#e5e7eb')
    ax_vel.set_ylabel("Linear Speed [m/s]", fontsize=11, color='#e5e7eb')
    ax_vel.axhline(0.60, color='#9ca3af', linestyle=':', lw=1.5, label='Speed Limit $v_{max}=0.60$ m/s')
    ax_vel.axhline(0.30, color='#6b7280', linestyle='--', lw=1.2, alpha=0.7, label='Min Corner Speed $v_{min}=0.30$ m/s')
    for r in results:
        lw = 2.4 if "G2" in r["id"] else 1.6
        ax_vel.plot(r["t"], r["speed_2d"], label=f"{r['name']} (avg: {r['avg_speed']:.3f} m/s)", color=r["color"], lw=lw, alpha=0.9)
    ax_vel.set_ylim(0.0, 0.75)
    ax_vel.legend(loc='upper right', fontsize=9.5, facecolor='#111827', edgecolor='#374151')

    # Panel 3: Yaw Rate ω_z(t) & Angular Acceleration
    ax_yaw = axs[1, 0]
    ax_yaw.set_title("(3) Body Yaw Rate ω_z(t) [rad/s] (Cornering Smoothness)", fontsize=13, fontweight='bold', color='#f9fafb', pad=12)
    ax_yaw.set_xlabel("Time [s]", fontsize=11, color='#e5e7eb')
    ax_yaw.set_ylabel("Yaw Rate ω_z [rad/s]", fontsize=11, color='#e5e7eb')
    ax_yaw.axhline(1.00, color='#ef4444', linestyle=':', alpha=0.6, label='Locked Max Turn (±1.0 rad/s)')
    ax_yaw.axhline(-1.00, color='#ef4444', linestyle=':', alpha=0.6)
    for r in results:
        lw = 2.2 if "G2" in r["id"] else 1.5
        ax_yaw.plot(r["t"], r["yaw_rate"], label=f"{r['name']} (RMS α: {r['rms_yaw_acc']:.2f} rad/s²)", color=r["color"], lw=lw, alpha=0.85)
    ax_yaw.legend(loc='lower right', fontsize=9.5, facecolor='#111827', edgecolor='#374151')

    # Panel 4: Summary KPI Bar Chart
    ax_kpi = axs[1, 1]
    ax_kpi.set_title("(4) Quantitative KPI Comparison Across Candidates", fontsize=13, fontweight='bold', color='#f9fafb', pad=12)
    
    categories = [r["id"] for r in results]
    goal_times = [r["finish_time"] for r in results]
    path_lens = [r["path_len"] for r in results]
    rms_accs = [r["rms_yaw_acc"] for r in results]
    colors = [r["color"] for r in results]
    
    x = np.arange(len(categories))
    width = 0.26
    
    rects1 = ax_kpi.bar(x - width, goal_times, width, label='Goal Time [s]', color=[c for c in colors], alpha=0.9, edgecolor='white', lw=1.2)
    rects2 = ax_kpi.bar(x, path_lens, width, label='Path Length [m]', color=[c for c in colors], alpha=0.55, hatch='//', edgecolor='white', lw=1.2)
    rects3 = ax_kpi.bar(x + width, rms_accs, width, label='RMS Yaw Accel [rad/s²]', color=[c for c in colors], alpha=0.35, hatch='..', edgecolor='white', lw=1.2)
    
    ax_kpi.set_xticks(x)
    ax_kpi.set_xticklabels(categories, fontsize=11, fontweight='bold', color='#f9fafb')
    ax_kpi.set_ylabel("Metric Values", fontsize=11, color='#e5e7eb')
    ax_kpi.legend(loc='upper right', fontsize=10, facecolor='#111827', edgecolor='#374151')

    def autolabel(rects, fmt="%.1f"):
        for rect in rects:
            height = rect.get_height()
            ax_kpi.annotate(fmt % height,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9.5, color='#f9fafb', fontweight='bold')

    autolabel(rects1, "%.2fs")
    autolabel(rects2, "%.2fm")
    autolabel(rects3, "%.2f")

    plt.tight_layout()
    plot_file = os.path.join(EXP_DIR, "track_a_g2_comparison.png")
    plt.savefig(plot_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved comparative figure to: {plot_file}")
    plt.close()
    return results

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "all":
        for cfg in CONFIGS:
            run_benchmark(cfg)
        analyze_and_plot()
    elif mode == "plot":
        analyze_and_plot()
    else:
        for cfg in CONFIGS:
            if cfg["id"].lower() == mode.lower():
                run_benchmark(cfg)
                break
        else:
            print(f"Unknown mode/id {mode}")
