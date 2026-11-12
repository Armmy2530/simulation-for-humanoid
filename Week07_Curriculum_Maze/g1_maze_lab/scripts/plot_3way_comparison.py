"""Generate 3-way comparison plot: Baseline vs Before vs New A* Kinodynamic Sprint."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load data
b = np.load("baseline.npz")
u_before = np.load("upgraded.npz")
u_astar = np.load("astar_fast.npz")

b_t, b_pos, b_vel = b["t"], b["pos"], b["lin_vel"]
ub_t, ub_pos, ub_vel = u_before["t"], u_before["pos"], u_before["lin_vel"]
ua_t, ua_pos, ua_vel = u_astar["t"], u_astar["pos"], u_astar["lin_vel"]

b_speed = np.linalg.norm(b_vel, axis=1)
ub_speed = np.linalg.norm(ub_vel, axis=1)
ua_speed = np.linalg.norm(ua_vel, axis=1)

b_len = np.sum(np.linalg.norm(np.diff(b_pos, axis=0), axis=1))
ub_len = np.sum(np.linalg.norm(np.diff(ub_pos, axis=0), axis=1))
ua_len = np.sum(np.linalg.norm(np.diff(ua_pos, axis=0), axis=1))

b_time = float(b["finish_time"])
ub_time = float(u_before["finish_time"])
ua_time = float(u_astar["finish_time"])

print(f"1. Baseline: Time = {b_time:.2f}s, Path = {b_len:.2f}m, Mean Speed = {np.mean(b_speed):.2f}m/s")
print(f"2. Before:   Time = {ub_time:.2f}s, Path = {ub_len:.2f}m, Mean Speed = {np.mean(ub_speed):.2f}m/s")
print(f"3. New A*:   Time = {ua_time:.2f}s, Path = {ua_len:.2f}m, Mean Speed = {np.mean(ua_speed):.2f}m/s")

# Plot 3-way comparison figure
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax_path, ax_speed) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=150)
fig.patch.set_facecolor('#0f172a')

for ax in (ax_path, ax_speed):
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#e2e8f0', labelsize=10)
    ax.grid(color='#334155', linestyle='--', linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color('#475569')

# --- 2D Maze Trajectory ---
ax_path.set_title("3-Stage Trajectory Evolution in Maze", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)

# True walls
import sys; sys.path.insert(0, 'tests')
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
ax_path.text(0.0, 0.0, "START", color='#fdba74', ha='center', va='center', fontweight='bold', fontsize=9)
ax_path.text(8.0, 8.0, "GOAL", color='#86efac', ha='center', va='center', fontweight='bold', fontsize=9)

# 3 Trajectories
ax_path.plot(b_pos[:, 0], b_pos[:, 1], color='#ef4444', linewidth=2.0, linestyle='--', label=f"1. Baseline ({b_time:.1f}s, {b_len:.1f}m)")
ax_path.plot(ub_pos[:, 0], ub_pos[:, 1], color='#f59e0b', linewidth=2.2, linestyle='-.', label=f"2. Before Upgrade ({ub_time:.1f}s, {ub_len:.1f}m)")
ax_path.plot(ua_pos[:, 0], ua_pos[:, 1], color='#38bdf8', linewidth=3.0, linestyle='-', label=f"3. New A* Sprint ({ua_time:.1f}s, {ua_len:.1f}m)")

ax_path.set_xlim(-1.8, 9.8)
ax_path.set_ylim(-1.8, 9.8)
ax_path.set_xlabel("X Position [m]", color='#cbd5e1', fontsize=10)
ax_path.set_ylabel("Y Position [m]", color='#cbd5e1', fontsize=10)
ax_path.legend(loc="lower right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)

# --- Speed Profile Comparison ---
ax_speed.set_title("Forward Velocity Comparison Across 3 Versions", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
ax_speed.plot(b_t, b_speed, color='#ef4444', linewidth=1.6, linestyle='--', label="1. Baseline (Stop & Turn at Corners)")
ax_speed.plot(ub_t, ub_speed, color='#f59e0b', linewidth=1.8, linestyle='-.', label="2. Before Upgrade (Corner Cut Point)")
ax_speed.plot(ua_t, ua_speed, color='#38bdf8', linewidth=2.4, linestyle='-', label="3. New A* Sprint (0.88m/s Straights & 45° Cuts)")

ax_speed.axvline(ua_time, color='#38bdf8', linestyle=':', label=f"A* Finish ({ua_time:.1f}s)")
ax_speed.axvline(ub_time, color='#f59e0b', linestyle=':', label=f"Before Finish ({ub_time:.1f}s)")
ax_speed.axvline(b_time, color='#ef4444', linestyle=':', label=f"Baseline Finish ({b_time:.1f}s)")

ax_speed.set_xlabel("Time [s]", color='#cbd5e1', fontsize=10)
ax_speed.set_ylabel("Actual Speed [m/s]", color='#cbd5e1', fontsize=10)
ax_speed.set_ylim(0.0, 1.05)
ax_speed.set_xlim(0.0, b_time + 2.0)
ax_speed.legend(loc="upper right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8.5)

plt.tight_layout()
out_plot = "snapshots/three_way_comparison.png"
plt.savefig(out_plot, facecolor=fig.get_facecolor(), dpi=150)
print(f"Saved {out_plot}")

artifact_dir = "/home/armmy2530/.gemini/antigravity-cli/brain/325989c9-57cc-438f-93bb-ad536045bcae"
import shutil
shutil.copy(out_plot, os.path.join(artifact_dir, "three_way_comparison.png"))
shutil.copy(out_plot, "/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/three_way_comparison.png")
shutil.copy(out_plot, "/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/g1_maze_lab/artifacts/three_way_comparison.png")
print("Synced three-way comparison plot to all artifact directories.")
