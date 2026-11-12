"""Generate high-quality comparison plots between Baseline and Upgraded Autopilot."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load data
b_data = np.load("baseline.npz")
u_data = np.load("upgraded.npz")

b_t = b_data["t"]
b_pos = b_data["pos"]
b_lin_vel = b_data["lin_vel"]
b_cmd = b_data["cmd"]
b_finish = float(b_data["finish_time"])

u_t = u_data["t"]
u_pos = u_data["pos"]
u_lin_vel = u_data["lin_vel"]
u_cmd = u_data["cmd"]
u_finish = float(u_data["finish_time"])

# Compute actual horizontal speed (norm of vx, vy)
b_speed = np.linalg.norm(b_lin_vel, axis=1)
u_speed = np.linalg.norm(u_lin_vel, axis=1)

# Compute path lengths
b_dists = np.linalg.norm(np.diff(b_pos, axis=0), axis=1)
b_path_len = np.sum(b_dists)
u_dists = np.linalg.norm(np.diff(u_pos, axis=0), axis=1)
u_path_len = np.sum(u_dists)

print(f"=== BENCHMARK METRICS SUMMARY ===")
print(f"Baseline: Time = {b_finish:.2f} s, Path Length = {b_path_len:.2f} m, Mean Speed = {np.mean(b_speed):.2f} m/s, Min Turn Speed = {np.min(b_cmd[:, 0]):.2f} m/s")
print(f"Upgraded: Time = {u_finish:.2f} s, Path Length = {u_path_len:.2f} m, Mean Speed = {np.mean(u_speed):.2f} m/s, Min Turn Speed = {np.min(u_cmd[:, 0]):.2f} m/s")
print(f"Time Reduction: {((b_finish - u_finish) / b_finish) * 100:.1f}% faster!")

# -------------------------------------------------------------
# PLOT 1: Comprehensive Comparison Figure (2x2 grid)
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=150)
fig.patch.set_facecolor('#0f172a') # dark background

for ax in axs.flat:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#e2e8f0', labelsize=10)
    ax.grid(color='#334155', linestyle='--', linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color('#475569')

# (0, 0): 2D Path in Maze
ax_path = axs[0, 0]
ax_path.set_title("2D Navigation Trajectory (G1-Maze-Challenge)", color='#f8fafc', fontsize=13, fontweight='bold', pad=10)

# Draw Maze grid & cells (c = 2.0m, 5x5) from true CHALLENGE layout
import sys
sys.path.insert(0, 'tests')
from test_maze_logic import CHALLENGE
from g1_maze_lab.terrains.maze import parse_layout
true_walls = parse_layout(CHALLENGE)

c = 2.0
n = 5
half = c / 2.0

# Outer boundaries
ax_path.plot([-half, n*c - half], [-half, -half], color='#94a3b8', linewidth=4.0, solid_capstyle='round')
ax_path.plot([-half, n*c - half], [n*c - half, n*c - half], color='#94a3b8', linewidth=4.0, solid_capstyle='round')
ax_path.plot([-half, -half], [-half, n*c - half], color='#94a3b8', linewidth=4.0, solid_capstyle='round')
ax_path.plot([n*c - half, n*c - half], [-half, n*c - half], color='#94a3b8', linewidth=4.0, solid_capstyle='round')

# Exact inner walls from true layout
for i in range(n):
    for j in range(n):
        cx, cy = i * c, j * c
        if true_walls[i, j, 0] and i < n - 1: # East wall
            ax_path.plot([cx + half, cx + half], [cy - half, cy + half], color='#94a3b8', linewidth=4.0, solid_capstyle='round')
        if true_walls[i, j, 1] and j < n - 1: # North wall
            ax_path.plot([cx - half, cx + half], [cy + half, cy + half], color='#94a3b8', linewidth=4.0, solid_capstyle='round')

# Draw Start and Goal cells
start_patch = patches.Rectangle((-1.0, -1.0), 2.0, 2.0, facecolor='#f97316', alpha=0.25, edgecolor='#ea580c', lw=1.5)
goal_patch = patches.Rectangle((7.0, 7.0), 2.0, 2.0, facecolor='#22c55e', alpha=0.25, edgecolor='#16a34a', lw=1.5)
ax_path.add_patch(start_patch)
ax_path.add_patch(goal_patch)
ax_path.text(0.0, 0.0, "START\n(0,0)", color='#fdba74', ha='center', va='center', fontweight='bold', fontsize=9)
ax_path.text(8.0, 8.0, "GOAL\n(4,4)", color='#86efac', ha='center', va='center', fontweight='bold', fontsize=9)

# Plot Trajectories
ax_path.plot(b_pos[:, 0], b_pos[:, 1], color='#ef4444', linewidth=2.4, linestyle='--', label=f"Baseline ({b_finish:.1f}s, {b_path_len:.1f}m)", alpha=0.9)
ax_path.plot(u_pos[:, 0], u_pos[:, 1], color='#38bdf8', linewidth=2.8, linestyle='-', label=f"Upgraded Arc Turning ({u_finish:.1f}s, {u_path_len:.1f}m)", alpha=0.95)

# Annotate Direct Corner Cutting
ax_path.annotate("Direct Corner Cut\n(Arc Turning)", xy=(u_pos[np.argmin(np.abs(u_t - 4.5)), 0], u_pos[np.argmin(np.abs(u_t - 4.5)), 1]),
                 xytext=(3.5, 0.5),
                 arrowprops=dict(facecolor='#38bdf8', edgecolor='#38bdf8', shrink=0.08, width=1.5, headwidth=6),
                 color='#38bdf8', fontsize=8.5, fontweight='semibold')

ax_path.annotate("Square L-Turn\n(Sluggish Stop)", xy=(b_pos[np.argmin(np.abs(b_t - 5.0)), 0], b_pos[np.argmin(np.abs(b_t - 5.0)), 1]),
                 xytext=(0.2, 2.5),
                 arrowprops=dict(facecolor='#ef4444', edgecolor='#ef4444', shrink=0.08, width=1.5, headwidth=6),
                 color='#ef4444', fontsize=8.5, fontweight='semibold')

ax_path.set_xlim(-1.8, 9.8)
ax_path.set_ylim(-1.8, 9.8)
ax_path.set_xlabel("X Position [m]", color='#cbd5e1', fontsize=10)
ax_path.set_ylabel("Y Position [m]", color='#cbd5e1', fontsize=10)
ax_path.legend(loc="lower right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)

# (0, 1): Speed Profile vs Time
ax_speed = axs[0, 1]
ax_speed.set_title("Forward Velocity Profile Comparison", color='#f8fafc', fontsize=13, fontweight='bold', pad=10)
ax_speed.plot(b_t, b_speed, color='#ef4444', linewidth=1.8, linestyle='--', label="Baseline Actual Speed", alpha=0.7)
ax_speed.plot(b_t, b_cmd[:, 0], color='#f87171', linewidth=1.2, label="Baseline Command $v_x$", alpha=0.5)
ax_speed.plot(u_t, u_speed, color='#38bdf8', linewidth=2.2, linestyle='-', label="Upgraded Actual Speed", alpha=0.95)
ax_speed.plot(u_t, u_cmd[:, 0], color='#0284c7', linewidth=1.4, linestyle=':', label="Upgraded Command $v_x$", alpha=0.85)

ax_speed.axvline(u_finish, color='#38bdf8', linestyle='-.', alpha=0.7, label=f"Upgraded Finish ({u_finish:.1f}s)")
ax_speed.axvline(b_finish, color='#ef4444', linestyle='-.', alpha=0.7, label=f"Baseline Finish ({b_finish:.1f}s)")

ax_speed.set_xlabel("Time [s]", color='#cbd5e1', fontsize=10)
ax_speed.set_ylabel("Speed [m/s]", color='#cbd5e1', fontsize=10)
ax_speed.set_ylim(0.0, 0.95)
ax_speed.set_xlim(0.0, max(b_finish, u_finish) + 2.0)
ax_speed.legend(loc="upper right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8.5)

# (1, 0): Angular Velocity Command (Yaw Rate) vs Time
ax_yaw = axs[1, 0]
ax_yaw.set_title(r"Yaw Rate Command $\omega_z$ (Cornering Agility)", color='#f8fafc', fontsize=13, fontweight='bold', pad=10)
ax_yaw.plot(b_t, b_cmd[:, 2], color='#ef4444', linewidth=1.8, linestyle='--', label=r"Baseline $\omega_z$ (max 0.8 rad/s)", alpha=0.85)
ax_yaw.plot(u_t, u_cmd[:, 2], color='#38bdf8', linewidth=2.0, linestyle='-', label=r"Upgraded $\omega_z$ (max 1.25 rad/s)", alpha=0.95)
ax_yaw.axhline(0.0, color='#64748b', linestyle='-', linewidth=0.8)
ax_yaw.set_xlabel("Time [s]", color='#cbd5e1', fontsize=10)
ax_yaw.set_ylabel(r"Yaw Rate $\omega_z$ [rad/s]", color='#cbd5e1', fontsize=10)
ax_yaw.set_xlim(0.0, max(b_finish, u_finish) + 2.0)
ax_yaw.legend(loc="upper right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)

# (1, 1): Metric Bar Chart / Scorecard
ax_bar = axs[1, 1]
ax_bar.set_title("Quantitative Performance Benchmark", color='#f8fafc', fontsize=13, fontweight='bold', pad=10)
metrics = ['Goal Time [s]\n(Lower is better)', 'Path Length [m]\n(Lower is better)', 'Mean Speed [m/s]\n(Higher is better)', 'Min Turn Speed [m/s]\n(Higher is better)']
baseline_vals = [b_finish, b_path_len, float(np.mean(b_speed)), float(np.min(b_cmd[:, 0]))]
upgraded_vals = [u_finish, u_path_len, float(np.mean(u_speed)), float(np.min(u_cmd[:, 0]))]

x = np.arange(len(metrics))
width = 0.35

rects1 = ax_bar.bar(x - width/2, baseline_vals, width, label='Baseline Autopilot', color='#ef4444', alpha=0.85, edgecolor='#b91c1c')
rects2 = ax_bar.bar(x + width/2, upgraded_vals, width, label='Upgraded Autopilot', color='#38bdf8', alpha=0.9, edgecolor='#0284c7')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metrics, color='#cbd5e1', fontsize=9)
ax_bar.legend(loc="upper right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)

# Label bars with values
for rect in rects1:
    h = rect.get_height()
    ax_bar.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color='#fca5a5', fontsize=8.5, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax_bar.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color='#7dd3fc', fontsize=8.5, fontweight='bold')

plt.tight_layout()
out_dir = "snapshots"
os.makedirs(out_dir, exist_ok=True)
plot_path = os.path.join(out_dir, "autopilot_comparison.png")
plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
print(f"[BENCHMARK] Saved plot to {plot_path}")

# Also copy to artifact dir if provided
artifact_dir = "/home/armmy2530/.gemini/antigravity-cli/brain/325989c9-57cc-438f-93bb-ad536045bcae"
if os.path.isdir(artifact_dir):
    import shutil
    shutil.copy(plot_path, os.path.join(artifact_dir, "autopilot_comparison.png"))
    print(f"[BENCHMARK] Copied plot to artifact dir: {artifact_dir}")
