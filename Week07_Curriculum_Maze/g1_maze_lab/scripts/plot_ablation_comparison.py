"""Generate 4-way ablation comparison plot: Baseline vs Before Upgrade vs Before + Fast Vel vs New A*."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load data
b = np.load("baseline.npz")
ub = np.load("upgraded.npz")
ub_fast = np.load("before_fast_velocity.npz")
ua = np.load("astar_fast.npz")

b_t, b_pos, b_vel = b["t"], b["pos"], b["lin_vel"]
ub_t, ub_pos, ub_vel = ub["t"], ub["pos"], ub["lin_vel"]
ubf_t, ubf_pos, ubf_vel = ub_fast["t"], ub_fast["pos"], ub_fast["lin_vel"]
ua_t, ua_pos, ua_vel = ua["t"], ua["pos"], ua["lin_vel"]

b_speed = np.linalg.norm(b_vel, axis=1)
ub_speed = np.linalg.norm(ub_vel, axis=1)
ubf_speed = np.linalg.norm(ubf_vel, axis=1)
ua_speed = np.linalg.norm(ua_vel, axis=1)

b_len = np.sum(np.linalg.norm(np.diff(b_pos, axis=0), axis=1))
ub_len = np.sum(np.linalg.norm(np.diff(ub_pos, axis=0), axis=1))
ubf_len = np.sum(np.linalg.norm(np.diff(ubf_pos, axis=0), axis=1))
ua_len = np.sum(np.linalg.norm(np.diff(ua_pos, axis=0), axis=1))

b_time = float(b["finish_time"])
ub_time = float(ub["finish_time"])
ubf_time = float(ub_fast["finish_time"])
ua_time = float(ua["finish_time"])

# Plot 4-way comparison figure
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax_path, ax_speed) = plt.subplots(1, 2, figsize=(17, 8.0), dpi=160)
fig.patch.set_facecolor('#0f172a')

for ax in (ax_path, ax_speed):
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#e2e8f0', labelsize=10)
    ax.grid(color='#334155', linestyle='--', linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color('#475569')

# --- 2D Maze Trajectory ---
ax_path.set_title("Ablation: Path Trajectory in G1-Maze-Challenge", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)

# True walls
sys.path.insert(0, 'tests')
from test_maze_logic import CHALLENGE
from g1_maze_lab.terrains.maze import parse_layout
true_walls = parse_layout(CHALLENGE)
c, n, half = 2.0, 5, 1.0

# Boundaries
ax_path.plot([-half, n*c - half], [-half, -half], color='#94a3b8', linewidth=4.0)
ax_path.plot([-half, n*c - half], [n*c - half, n*c - half], color='#94a3b8', linewidth=4.0)
ax_path.plot([-half, -half], [-half, n*c - half], color='#94a3b8', linewidth=4.0)
ax_path.plot([n*c - half, n*c - half], [-half, n*c - half], color='#94a3b8', linewidth=4.0)

for i in range(n):
    for j in range(n):
        cx, cy = i * c, j * c
        if true_walls[i, j, 0] and i < n - 1:
            ax_path.plot([cx + half, cx + half], [cy - half, cy + half], color='#94a3b8', linewidth=4.0)
        if true_walls[i, j, 1] and j < n - 1:
            ax_path.plot([cx - half, cx + half], [cy + half, cy + half], color='#94a3b8', linewidth=4.0)

start_p = patches.Rectangle((-half, -half), c, c, facecolor='#f97316', alpha=0.25, edgecolor='#ea580c', lw=1.5)
goal_p = patches.Rectangle((4*c - half, 4*c - half), c, c, facecolor='#22c55e', alpha=0.25, edgecolor='#16a34a', lw=1.5)
ax_path.add_patch(start_p)
ax_path.add_patch(goal_p)
ax_path.text(0.0, 0.0, "START\n(0,0)", color='#fdba74', ha='center', va='center', fontweight='bold', fontsize=9)
ax_path.text(8.0, 8.0, "GOAL\n(4,4)", color='#86efac', ha='center', va='center', fontweight='bold', fontsize=9)

# 4 Trajectories
ax_path.plot(b_pos[:, 0], b_pos[:, 1], color='#ef4444', linewidth=1.8, linestyle='--', label=f"1. Baseline (41.4s, 20.5m)")
ax_path.plot(ub_pos[:, 0], ub_pos[:, 1], color='#f59e0b', linewidth=2.0, linestyle=':', label=f"2. Before (0.75m/s): 27.6s, 17.9m")
ax_path.plot(ubf_pos[:, 0], ubf_pos[:, 1], color='#a855f7', linewidth=2.8, linestyle='-', label=f"3. Before + Fast Vel (0.88m/s, No A*): 24.3s, 18.1m")
ax_path.plot(ua_pos[:, 0], ua_pos[:, 1], color='#38bdf8', linewidth=2.0, linestyle='-.', label=f"4. With A* Spline (0.88m/s): 24.5s, 18.4m")

ax_path.set_xlim(-1.8, 9.8)
ax_path.set_ylim(-1.8, 9.8)
ax_path.set_xlabel("X Position [m]", color='#cbd5e1', fontsize=10)
ax_path.set_ylabel("Y Position [m]", color='#cbd5e1', fontsize=10)
ax_path.legend(loc="lower right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)

# --- Speed Profile Comparison ---
ax_speed.set_title("Forward Velocity Profile Comparison (Ablation Analysis)", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
ax_speed.plot(b_t, b_speed, color='#ef4444', linewidth=1.4, linestyle='--', alpha=0.7, label="1. Baseline (Stop & Turn at Corners)")
ax_speed.plot(ub_t, ub_speed, color='#f59e0b', linewidth=1.5, linestyle=':', alpha=0.8, label="2. Before Upgrade (0.75 m/s profile)")
ax_speed.plot(ubf_t, ubf_speed, color='#a855f7', linewidth=2.4, linestyle='-', label="3. Before Upgrade + Latest Vel (0.88 m/s, No A*)")
ax_speed.plot(ua_t, ua_speed, color='#38bdf8', linewidth=1.8, linestyle='-.', alpha=0.9, label="4. Upgraded with A* Spline (0.88 m/s)")

ax_speed.axvline(ubf_time, color='#a855f7', linestyle='-', linewidth=1.5, label=f"Before + Fast Vel Finish ({ubf_time:.2f}s)")
ax_speed.axvline(ua_time, color='#38bdf8', linestyle='-.', linewidth=1.5, label=f"A* Finish ({ua_time:.2f}s)")
ax_speed.axvline(ub_time, color='#f59e0b', linestyle=':', linewidth=1.2, label=f"Before Finish ({ub_time:.2f}s)")
ax_speed.axvline(b_time, color='#ef4444', linestyle='--', linewidth=1.2, label=f"Baseline Finish ({b_time:.2f}s)")

ax_speed.set_xlabel("Time [s]", color='#cbd5e1', fontsize=10)
ax_speed.set_ylabel("Actual Speed [m/s]", color='#cbd5e1', fontsize=10)
ax_speed.set_ylim(0.0, 1.05)
ax_speed.set_xlim(0.0, b_time + 1.5)
ax_speed.legend(loc="upper right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8.5)

plt.tight_layout()
out_fig = "ablation_comparison.png"
plt.savefig(out_fig, facecolor=fig.get_facecolor(), edgecolor='none', dpi=160)
print(f"Saved ablation comparison figure to {out_fig}")
