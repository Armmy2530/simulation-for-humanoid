"""Plot 3 random maze benchmark runs: 2D Trajectories and Velocity Profiles."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from g1_maze_lab.terrains.maze import parse_layout

SEEDS = [42, 100, 777]
COLORS = ['#38bdf8', '#a855f7', '#34d399']

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(20, 10.5), dpi=160)
fig.patch.set_facecolor('#0f172a')

# Grid: 3 top/middle subplots for mazes, bottom for velocity profiles
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.75], hspace=0.3, wspace=0.25)

c, n, half = 2.0, 5, 1.0

for idx, seed in enumerate(SEEDS):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#e2e8f0', labelsize=9)
    ax.grid(color='#334155', linestyle='--', linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color('#475569')

    # Load data
    data = np.load(f"random_maze_seed{seed}.npz")
    t, pos, vel = data['t'], data['pos'], data['lin_vel']
    fin_t = float(data['finish_time'])
    layout_str = str(data['layout_ascii'])
    
    true_walls = parse_layout(layout_str)
    
    mask = t <= fin_t
    pos_m = pos[mask]
    vel_m = vel[mask]
    speed_m = np.linalg.norm(vel_m, axis=1)
    path_len = np.sum(np.linalg.norm(np.diff(pos_m, axis=0), axis=1))

    # Draw outer maze boundary
    ax.plot([-half, n*c - half], [-half, -half], color='#94a3b8', linewidth=3.5)
    ax.plot([-half, n*c - half], [n*c - half, n*c - half], color='#94a3b8', linewidth=3.5)
    ax.plot([-half, -half], [-half, n*c - half], color='#94a3b8', linewidth=3.5)
    ax.plot([n*c - half, n*c - half], [-half, n*c - half], color='#94a3b8', linewidth=3.5)

    # Draw inner walls
    for i in range(n):
        for j in range(n):
            cx, cy = i * c, j * c
            if true_walls[i, j, 0] and i < n - 1:
                ax.plot([cx + half, cx + half], [cy - half, cy + half], color='#94a3b8', linewidth=3.5)
            if true_walls[i, j, 1] and j < n - 1:
                ax.plot([cx - half, cx + half], [cy + half, cy + half], color='#94a3b8', linewidth=3.5)

    # Start / Goal patches
    start_p = patches.Rectangle((-half, -half), c, c, facecolor='#f97316', alpha=0.25, edgecolor='#ea580c', lw=1.2)
    goal_p = patches.Rectangle((4*c - half, 4*c - half), c, c, facecolor='#22c55e', alpha=0.25, edgecolor='#16a34a', lw=1.2)
    ax.add_patch(start_p)
    ax.add_patch(goal_p)
    ax.text(0.0, 0.0, "START", color='#fdba74', ha='center', va='center', fontweight='bold', fontsize=8.5)
    ax.text(8.0, 8.0, "GOAL", color='#86efac', ha='center', va='center', fontweight='bold', fontsize=8.5)

    # Trajectory
    ax.plot(pos_m[:, 0], pos_m[:, 1], color=COLORS[idx], linewidth=2.8, label=f"Path ({fin_t:.2f}s, {path_len:.1f}m)")
    ax.scatter([pos_m[0, 0]], [pos_m[0, 1]], color='#f97316', s=40, zorder=5)
    ax.scatter([pos_m[-1, 0]], [pos_m[-1, 1]], color='#22c55e', s=50, marker='*', zorder=5)

    ax.set_title(f"Random Maze: Seed {seed}\nGoal: {fin_t:.2f}s | Dist: {path_len:.1f}m | Avg: {np.mean(speed_m):.2f}m/s", 
                 color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(-1.8, 9.8)
    ax.set_ylim(-1.8, 9.8)
    ax.set_xlabel("X Position [m]", color='#cbd5e1', fontsize=9)
    ax.set_ylabel("Y Position [m]", color='#cbd5e1', fontsize=9)
    ax.legend(loc="lower right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8.5)

# Velocity comparison subplot across the entire bottom width
ax_speed = fig.add_subplot(gs[1, :])
ax_speed.set_facecolor('#1e293b')
ax_speed.tick_params(colors='#e2e8f0', labelsize=10)
ax_speed.grid(color='#334155', linestyle='--', linewidth=0.6, alpha=0.7)
for spine in ax_speed.spines.values():
    spine.set_color('#475569')

ax_speed.set_title("Forward Velocity vs Time Across All 3 Random Mazes", color='#f8fafc', fontsize=12, fontweight='bold', pad=10)

for idx, seed in enumerate(SEEDS):
    data = np.load(f"random_maze_seed{seed}.npz")
    t, vel = data['t'], data['lin_vel']
    fin_t = float(data['finish_time'])
    mask = t <= fin_t
    speed = np.linalg.norm(vel[mask], axis=1)
    ax_speed.plot(t[mask], speed, color=COLORS[idx], linewidth=2.0, label=f"Seed {seed} ({fin_t:.2f}s, Peak: {np.max(speed):.2f}m/s)")
    ax_speed.axvline(fin_t, color=COLORS[idx], linestyle=':', linewidth=1.5, alpha=0.8)

ax_speed.set_xlabel("Time [s]", color='#cbd5e1', fontsize=10)
ax_speed.set_ylabel("Forward Speed [m/s]", color='#cbd5e1', fontsize=10)
ax_speed.set_ylim(0.0, 1.05)
ax_speed.set_xlim(0.0, 42.5)
ax_speed.legend(loc="upper right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)

out_fig = "random_mazes_3way_benchmark.png"
plt.savefig(out_fig, facecolor=fig.get_facecolor(), edgecolor='none', dpi=160)
print(f"Saved figure to {out_fig}")
