"""Generate visualization matching Fig. 7, Fig. 8, and Fig. 9/10 from the paper."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, 'tests')
from test_maze_logic import CHALLENGE
from g1_maze_lab.terrains.maze import parse_layout
from g1_maze_lab.autopilot import MazeAutopilot
import torch

# 1. Setup maze & plan path
true_walls = parse_layout(CHALLENGE)
pilot = MazeAutopilot(1, 5, 2.0, "cpu", strategy="flood")
pilot.walls[0] = torch.tensor(true_walls)
dist = pilot.flood_fill(pilot.walls)[0].numpy()

# Trace planned shortest path cells (Fig. 7)
cell_path = pilot.plan_full_path((0, 0), env_id=0)
raw_pts = np.array([[c[0] * 2.0, c[1] * 2.0] for c in cell_path])

# Generate dense interpolated waypoints (Fig. 8)
wpts = pilot.interpolate_path(cell_path, pts_per_arc=14).cpu().numpy()

# Load actual executed trajectory from upgraded run
data = np.load("upgraded_smooth.npz")
executed_pos = data["pos"]

# -------------------------------------------------------------
# FIGURE 1: Fig. 7 & Fig. 8 Style Comparison (White Grid Style)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
fig.patch.set_facecolor('#ffffff')

for ax in (ax1, ax2):
    ax.set_facecolor('#ffffff')
    ax.grid(True, linestyle='-', color='#d1d5db', linewidth=0.8)
    ax.tick_params(colors='#374151', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#9ca3af')
    ax.set_xlim(-1.5, 9.5)
    ax.set_ylim(-1.5, 9.5)
    ax.set_aspect('equal')

# --- Fig. 7: Raw Shortest Path Waypoints ---
ax1.plot(raw_pts[:, 0], raw_pts[:, 1], color='#1d4ed8', linestyle=':', marker='o', markersize=5, linewidth=1.5, label="Raw Cell Waypoints")
for idx, (x, y) in enumerate(raw_pts):
    ax1.plot(x, y, marker='o', markersize=6, color='#1e40af')
    if idx % 2 == 0 or idx == len(raw_pts) - 1:
        ax1.text(x + 0.15, y + 0.15, f"W{idx}", fontsize=8, color='#1e3a8a', fontweight='semibold')
ax1.set_title("Fig. 7. Waypoints of the shortest path", fontsize=12, pad=12, fontweight='bold', color='#111827')
ax1.set_xlabel("X [m]", fontsize=10, color='#374151')
ax1.set_ylabel("Y [m]", fontsize=10, color='#374151')

# --- Fig. 8: Interpolated Waypoints with Arc Fillets ---
# Draw raw grid path in red
ax2.plot(raw_pts[:, 0], raw_pts[:, 1], color='#dc2626', linestyle='-', linewidth=2.8, label="Raw Path (Manhattan L-turns)")
# Draw blue dense interpolated waypoints (matching Fig. 8 blue dots!)
ax2.plot(wpts[:, 0], wpts[:, 1], color='#2563eb', marker='.', markersize=7, linestyle='', label="Interpolated Arc Waypoints")
# Overlay actual executed robot trajectory in cyan
ax2.plot(executed_pos[:, 0], executed_pos[:, 1], color='#0284c7', linewidth=1.8, linestyle='--', label=f"Robot Trajectory ({data['finish_time']:.1f}s)", alpha=0.9)

ax2.set_title("Fig. 8. Interpolated waypoints", fontsize=12, pad=12, fontweight='bold', color='#111827')
ax2.set_xlabel("X [m]", fontsize=10, color='#374151')
ax2.set_ylabel("Y [m]", fontsize=10, color='#374151')
ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)

plt.tight_layout()
os.makedirs("snapshots", exist_ok=True)
fig_path1 = "snapshots/fig7_fig8_interpolated_waypoints.png"
plt.savefig(fig_path1, dpi=150)
print(f"Saved {fig_path1}")

# -------------------------------------------------------------
# FIGURE 2: Full Maze 1 & 2 Style (Fig. 9 & Image 2)
# -------------------------------------------------------------
fig2, ax_maze = plt.subplots(figsize=(10, 10), dpi=150)
fig2.patch.set_facecolor('#ffffff')
ax_maze.set_facecolor('#ffffff')
ax_maze.grid(True, linestyle='-', color='#e5e7eb', linewidth=0.8)

c = 2.0
n = 5
half = c / 2.0

# Draw Maze Walls (Dark red/brown outline like Fig. 9 & Image 2)
wall_color = '#991b1b'
ax_maze.plot([-half, n*c - half], [-half, -half], color=wall_color, linewidth=5.0, solid_capstyle='round')
ax_maze.plot([-half, n*c - half], [n*c - half, n*c - half], color=wall_color, linewidth=5.0, solid_capstyle='round')
ax_maze.plot([-half, -half], [-half, n*c - half], color=wall_color, linewidth=5.0, solid_capstyle='round')
ax_maze.plot([n*c - half, n*c - half], [-half, n*c - half], color=wall_color, linewidth=5.0, solid_capstyle='round')

for i in range(n):
    for j in range(n):
        cx, cy = i * c, j * c
        if true_walls[i, j, 0] and i < n - 1:
            ax_maze.plot([cx + half, cx + half], [cy - half, cy + half], color=wall_color, linewidth=5.0, solid_capstyle='round')
        if true_walls[i, j, 1] and j < n - 1:
            ax_maze.plot([cx - half, cx + half], [cy + half, cy + half], color=wall_color, linewidth=5.0, solid_capstyle='round')

# Start and Goal markers
start_p = patches.Rectangle((-half, -half), c, c, facecolor='#fed7aa', alpha=0.35, edgecolor='#ea580c', lw=1.5)
goal_p = patches.Rectangle((4*c - half, 4*c - half), c, c, facecolor='#bbf7d0', alpha=0.35, edgecolor='#16a34a', lw=1.5)
ax_maze.add_patch(start_p)
ax_maze.add_patch(goal_p)
ax_maze.text(0.0, 0.0, "START\n(0,0)", color='#c2410c', ha='center', va='center', fontweight='bold', fontsize=10)
ax_maze.text(8.0, 8.0, "STOP / GOAL\n(4,4)", color='#15803d', ha='center', va='center', fontweight='bold', fontsize=10)

# Draw raw grid path in red dashed
ax_maze.plot(raw_pts[:, 0], raw_pts[:, 1], color='#ef4444', linestyle='--', linewidth=2.0, label="Raw Manhattan Shortest Path", alpha=0.7)
# Draw dense blue dots (Fig. 8 style)
ax_maze.plot(wpts[:, 0], wpts[:, 1], color='#2563eb', marker='.', markersize=8, linestyle='', label="Interpolated Waypoints (Arc Fillets)")
# Draw continuous actual executed curve (Image 2 style)
ax_maze.plot(executed_pos[:, 0], executed_pos[:, 1], color='#0ea5e9', linewidth=3.0, linestyle='-', label=f"Continuous Arc Execution ({data['finish_time']:.1f}s)", alpha=0.95)

# Annotations showing smooth cornering & diagonal transitions
ax_maze.annotate("SS90SR (Smooth 90° Arc)", xy=(2.0, 0.0), xytext=(2.6, -0.6),
                arrowprops=dict(facecolor='#0ea5e9', edgecolor='#0ea5e9', shrink=0.08, width=1.5, headwidth=6),
                color='#0369a1', fontsize=9, fontweight='bold')

ax_maze.annotate("SD45 -> DIA -> DS45\n(Smooth S-Curve Transition)", xy=(6.5, 6.0), xytext=(4.0, 4.2),
                arrowprops=dict(facecolor='#0ea5e9', edgecolor='#0ea5e9', shrink=0.08, width=1.5, headwidth=6),
                color='#0369a1', fontsize=9, fontweight='bold')

ax_maze.annotate("Open Corridor Doorway\n(Between cell (3,3) & (4,3))", xy=(7.0, 6.0), xytext=(4.8, 7.2),
                arrowprops=dict(facecolor='#16a34a', edgecolor='#16a34a', shrink=0.08, width=1.5, headwidth=6),
                color='#15803d', fontsize=9, fontweight='bold')

ax_maze.set_title("G1 Maze Autopilot: Continuous Arc & Diagonal Interpolated Waypoints", fontsize=13, fontweight='bold', pad=14, color='#111827')
ax_maze.set_xlim(-1.8, 9.8)
ax_maze.set_ylim(-1.8, 9.8)
ax_maze.set_aspect('equal')
ax_maze.legend(loc="lower left", fontsize=9.5, framealpha=0.9)

plt.tight_layout()
fig_path2 = "snapshots/fig9_maze_interpolated_path.png"
plt.savefig(fig_path2, dpi=150)
print(f"Saved {fig_path2}")

# Copy to brain artifact directory and project artifacts
artifact_dir = "/home/armmy2530/.gemini/antigravity-cli/brain/325989c9-57cc-438f-93bb-ad536045bcae"
import shutil
shutil.copy(fig_path1, os.path.join(artifact_dir, "fig7_fig8_interpolated_waypoints.png"))
shutil.copy(fig_path2, os.path.join(artifact_dir, "fig9_maze_interpolated_path.png"))
shutil.copy(fig_path1, "/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/fig7_fig8_interpolated_waypoints.png")
shutil.copy(fig_path2, "/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/fig9_maze_interpolated_path.png")
print("Copied all figures to artifact directories.")
