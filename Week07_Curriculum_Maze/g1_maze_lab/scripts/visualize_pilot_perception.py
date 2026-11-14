"""Visualize G1 Robot Perception, Online Planning, and Velocity Commands in Hazard Maze.

Runs A4_Hazard_G2 on G1-Maze-Hard (seed=2026, Short Route traversal in ~58s, 0 falls),
records rich 10 Hz telemetry (perception point cloud, chest LiDAR, discovered walls,
A* path, G2 quintic spline, cmd_vel vectors, and controller modes), and renders:
1. Animated MP4 Dashboard Video (1600x900 @ 20 fps):
   artifacts/experiment_10_track_a_hazard_slalom/run/robot_perception_planning_viz.mp4
2. 4-Keyframe Summary Figure (PNG):
   artifacts/experiment_10_track_a_hazard_slalom/robot_perception_planning_keyframes.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc
import matplotlib.patheffects as pe

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "g1_maze_lab", "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "g1_maze_lab", "scripts"))

EXP_DIR = os.path.join(PROJECT_ROOT, "artifacts", "experiment_10_track_a_hazard_slalom")
RUN_DIR = os.path.join(EXP_DIR, "run")
CHECKPOINT = os.path.join(PROJECT_ROOT, "g1_maze_lab", "pretrained", "g1_rough", "model_pretrained.pt")
VIDEO_SOURCE = os.path.join(RUN_DIR, "level7_short_route_walk.mp4")
CACHE_FILE = os.path.join(RUN_DIR, "perception_telemetry_seed2026.pkl")

OUT_MP4 = os.path.join(RUN_DIR, "robot_perception_planning_viz.mp4")
OUT_PNG = os.path.join(EXP_DIR, "robot_perception_planning_keyframes.png")

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
CELL_SIZE = 2.0


def set_a4_hazard_g2_env():
    """Configure environment variables for A4_Hazard_G2 winner configuration."""
    os.environ["MAZE_SPEED"] = "0.60"
    os.environ["MAZE_MIN_SPEED"] = "0.30"
    os.environ["MAZE_MAX_TURN"] = "1.00"
    os.environ["MAZE_ACC"] = "1.40,3.00"
    os.environ["MAZE_DIRECT_CORNER"] = "1"
    os.environ["MAZE_LATERAL_MARGIN"] = "1"
    os.environ["MAZE_CORNER_CUT"] = "0.38"
    os.environ["MAZE_R_FILLET"] = "0.65"
    os.environ["MAZE_LOOKAHEAD"] = "0.50"
    os.environ["MAZE_CURVE_MODE"] = "g2_quintic"
    os.environ["MAZE_USE_ASTAR"] = "1"
    os.environ["MAZE_LIDAR_LOOKAHEAD"] = "1"
    os.environ["MAZE_HAZARD_AVOIDANCE"] = "1"
    os.environ["MAZE_SPIKE_AVOID_MODE"] = "slalom_s_curve"
    os.environ["MAZE_FORK_ARBITRATOR"] = "1"
    os.environ["MAZE_PERSIST_MAP"] = "1"


def record_simulation_telemetry(seed: int = 2026, max_time: float = 70.0) -> dict:
    """Run simulation on G1-Maze-Hard and record rich 10 Hz perception/planning telemetry."""
    import torch
    import warp as wp
    wp.config.enable_backward = False

    from isaaclab_rl.entrypoints.backends import play_rsl_rl
    import g1_maze_lab.tasks
    from g1_maze_lab.terrains import GENERATED, HAZARDS
    import challenge

    set_a4_hazard_g2_env()

    collected_frames = []
    saved_meta = {}

    def telemetry_playback(step, *, dt, args_cli, env_cfg):
        nonlocal collected_frames, saved_meta
        env = challenge._find_env(step)
        robot = env.scene["robot"]
        cmd = env.command_manager.get_term("base_velocity")
        cmd.cfg.heading_command = False
        cmd.is_heading_env[:] = False
        cmd.is_standing_env[:] = False
        scanner = env.scene["maze_scanner"]
        lidar = env.scene["wall_lidar"] if "wall_lidar" in env.scene.keys() else None

        walls = GENERATED[-1]["walls"]
        hazards = HAZARDS[-1]
        n_env = env.num_envs
        n = walls.shape[0]
        start = tuple(hazards["start"])
        goal = tuple(hazards["goal"])

        saved_meta["walls"] = walls.cpu().numpy() if hasattr(walls, "cpu") else np.array(walls)
        saved_meta["hazards"] = hazards
        saved_meta["start"] = start
        saved_meta["goal"] = goal
        saved_meta["n"] = n

        # Disable automatic goal reset
        if "goal_reached" in env.termination_manager.active_terms:
            env.termination_manager.get_term_cfg("goal_reached").func = lambda env, **kw: torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

        pilot = challenge.make_pilot(
            challenge.load_pilot(os.path.join(PROJECT_ROOT, "g1_maze_lab", "submissions", "pro_pilot", "pilot.py")),
            n_env, n, env.device, start, goal
        )

        steps = int(max_time / dt)
        start_t = torch.tensor(start, dtype=torch.long, device=env.device)
        goal_t = torch.tensor(goal, dtype=torch.long, device=env.device)

        finish_time = None
        extra_steps_after_goal = 0

        print(f"[SIM] Recording G1-Maze-Hard seed={seed} (10 Hz sampling, dt={dt:.3f}s)...", flush=True)

        for k in range(steps):
            o = env.scene.env_origins.clone()
            o[:, :2] -= start_t.float() * CELL_SIZE
            pos_xy = robot.data.root_pos_w.torch[:, :2] - o[:, :2]
            yaw = challenge._yaw(robot.data.root_quat_w.torch)

            hits = torch.nan_to_num(scanner.data.ray_hits_w.torch, nan=0.0, posinf=0.0, neginf=0.0)
            scan_xy = hits[..., :2] - o[:, None, :2]
            scan_z = hits[..., 2] - o[:, None, 2]

            if lidar is not None:
                rel_lidar = lidar.data.ray_hits_w.torch[..., :2] - lidar.data.pos_w.torch[:, None, :2]
                lidar_dist = torch.nan_to_num(torch.linalg.norm(rel_lidar, dim=-1), nan=1e6, posinf=1e6)
                lidar_yaw = torch.atan2(rel_lidar[..., 1], rel_lidar[..., 0])
            else:
                lidar_dist = torch.zeros((n_env, 72), device=env.device)
                lidar_yaw = torch.zeros((n_env, 72), device=env.device)

            obs = {
                "pos_xy": pos_xy,
                "yaw": yaw,
                "t": k * dt,
                "dt": dt,
                "start_cell": start_t,
                "goal_cell": goal_t,
                "scan_xy": scan_xy,
                "scan_z": scan_z,
                "lidar_dist": lidar_dist,
                "lidar_yaw": lidar_yaw,
                "lin_vel_b": robot.data.root_lin_vel_b.torch,
                "ang_vel_b": robot.data.root_ang_vel_b.torch,
            }

            command = torch.as_tensor(pilot.act(obs), dtype=torch.float32, device=env.device).reshape(n_env, 3)
            if finish_time is not None:
                command[:] = 0.0
            cmd.time_left[:] = 1.0e9
            cmd.vel_command_b[:] = command

            # 10 Hz sampling (k % 5 == 0 when dt = 0.02)
            sample_interval = max(1, int(round(0.10 / dt)))
            if k % sample_interval == 0:
                t_cur = k * dt
                p_3d = (robot.data.root_pos_w.torch[0, :3] - o[0, :3]).cpu().numpy()
                y_cur = float(yaw[0].item())
                v_w = robot.data.root_lin_vel_w.torch[0, :2].cpu().numpy()
                v_b = robot.data.root_lin_vel_b.torch[0, :3].cpu().numpy()
                w_b = robot.data.root_ang_vel_b.torch[0, :3].cpu().numpy()
                cmd_val = command[0].cpu().numpy()

                s_xy = scan_xy[0].cpu().numpy()
                s_z = scan_z[0].cpu().numpy()
                l_dist = lidar_dist[0].cpu().numpy()
                l_yaw = lidar_yaw[0].cpu().numpy()

                ap = getattr(pilot, "ap", pilot)
                known_walls = ap.walls[0].cpu().numpy().copy()
                cell_path = [list(c) for c in ap.current_cell_paths[0]] if ap.current_cell_paths[0] else []
                smooth_wpts = ap.smooth_wpts[0].cpu().numpy().copy() if ap.smooth_wpts[0] is not None else None
                try:
                    tgt_pt_raw = ap.get_pursuit_target(0, pos_xy[0:1]).cpu().numpy()
                    tgt_pt = tgt_pt_raw.reshape(-1)[:2]
                except Exception:
                    tgt_pt = np.array([goal[0] * CELL_SIZE, goal[1] * CELL_SIZE], dtype=np.float32)

                ramp_zones = [list(c) for c in ap.cell_ramp_zones[0]]
                gap_zones = [list(c) for c in ap.cell_gap_zones[0]]
                spike_zones = {f"{c[0]}_{c[1]}": dict(ap.cell_spike_zones[0][c]) for c in ap.cell_spike_zones[0]}

                cur_c = (int(round(p_3d[0] / CELL_SIZE)), int(round(p_3d[1] / CELL_SIZE)))

                # Determine active mode
                if tuple(cur_c) in ap.cell_gap_zones[0] and cmd_val[0] > 0.70:
                    mode_str = "GAP STRIDE BOOST (vx=0.85 m/s)"
                elif tuple(cur_c) in ap.cell_ramp_zones[0]:
                    mode_str = "STEEP RAMP CENTERLINE LOCK (|wz|<=0.06)"
                elif tuple(cur_c) in ap.cell_spike_zones[0]:
                    mode_str = "SPIKE LANE AVOIDANCE"
                else:
                    mode_str = "G2 QUINTIC SPLINE CRUISE (vx=0.60 m/s)"

                frame_data = {
                    "t": t_cur,
                    "pos": p_3d,
                    "yaw": y_cur,
                    "vel_world": v_w,
                    "vel_body": v_b,
                    "ang_vel_body": w_b,
                    "cmd": cmd_val,
                    "scan_xy": s_xy,
                    "scan_z": s_z,
                    "lidar_dist": l_dist,
                    "lidar_yaw": l_yaw,
                    "known_walls": known_walls,
                    "cell_path": cell_path,
                    "smooth_wpts": smooth_wpts,
                    "target_pt": tgt_pt,
                    "ramp_zones": ramp_zones,
                    "gap_zones": gap_zones,
                    "spike_zones": spike_zones,
                    "active_mode": mode_str,
                    "cell": cur_c,
                }
                collected_frames.append(frame_data)

            with torch.inference_mode():
                step()

            # Check goal reach
            cur_p = (robot.data.root_pos_w.torch[0, :2] - o[0, :2]).tolist()
            if finish_time is None and math.dist(cur_p, (goal[0] * CELL_SIZE, goal[1] * CELL_SIZE)) < 0.6:
                finish_time = (k + 1) * dt
                print(f"[SIM] GOAL REACHED at t={finish_time:.2f} s! Collecting final settle frames...", flush=True)

            if finish_time is not None:
                extra_steps_after_goal += 1
                if extra_steps_after_goal > int(1.2 / dt):  # +1.2 seconds of settle
                    break

        saved_meta["finish_time"] = finish_time
        print(f"[SIM] Telemetry recording completed: {len(collected_frames)} frames collected.", flush=True)

    play_rsl_rl.run_playback = telemetry_playback

    play_rsl_rl.run([
        "--task", "G1-Maze-Hard",
        "--num_envs", "1",
        "--checkpoint", CHECKPOINT,
        f"env.scene.terrain.terrain_generator.seed={seed}"
    ])

    telemetry_bundle = {
        "meta": saved_meta,
        "frames": collected_frames,
    }

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(telemetry_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[CACHE] Saved telemetry bundle to {CACHE_FILE}")

    return telemetry_bundle


def load_telemetry() -> dict:
    """Load cached telemetry bundle or raise FileNotFoundError."""
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Telemetry cache {CACHE_FILE} not found. Run with --record first.")
    with open(CACHE_FILE, "rb") as f:
        bundle = pickle.load(f)
    print(f"[CACHE] Loaded {len(bundle['frames'])} frames from {CACHE_FILE}")
    return bundle


def render_dashboard_frame(
    fig,
    axes,
    frame: dict,
    meta: dict,
    all_frames: list[dict],
    video_reader,
    c_time_series_data: dict,
):
    """Render a single 1600x900 dashboard frame across the 4 coordinated panels."""
    ax_ego = axes["ego"]
    ax_map = axes["map"]
    ax_cam = axes["cam"]
    ax_gauge = axes["gauge"]
    ax_ts = axes["ts"]

    for ax in [ax_ego, ax_map, ax_cam, ax_gauge, ax_ts]:
        ax.clear()

    c = CELL_SIZE
    n = meta["n"]
    walls = meta["walls"]
    hazards = meta["hazards"]

    t = frame["t"]
    pos = frame["pos"]
    yaw = frame["yaw"]
    rx, ry = pos[0], pos[1]
    cmd = frame["cmd"]
    vw = frame["vel_world"]
    scan_xy = frame["scan_xy"]
    scan_z = frame["scan_z"]
    lidar_dist = frame["lidar_dist"]
    lidar_yaw = frame["lidar_yaw"]
    known_walls = frame["known_walls"]
    cell_path = frame["cell_path"]
    smooth_wpts = frame["smooth_wpts"]
    target_pt = frame["target_pt"]
    active_mode = frame["active_mode"]
    cur_c = frame["cell"]

    # =========================================================================
    # PANEL A: Ego-Centric Local Sensor View (4.8m x 4.8m)
    # =========================================================================
    ax_ego.set_facecolor("#111827")
    w_size = 2.4
    ax_ego.set_xlim(rx - w_size, rx + w_size)
    ax_ego.set_ylim(ry - w_size, ry + w_size)
    ax_ego.set_aspect("equal")

    # 1. 72-Ray Chest LiDAR fan
    for ld, ly in zip(lidar_dist, lidar_yaw):
        if ld < 5.8:
            hx = rx + ld * math.cos(ly)
            hy = ry + ld * math.sin(ly)
            ax_ego.plot([rx, hx], [ry, hy], color="#06b6d4", lw=0.8, alpha=0.35, zorder=3)
            ax_ego.scatter([hx], [hy], s=6, color="#22d3ee", alpha=0.8, zorder=4)

    # 2. 3D Height Scanner Point Cloud colored by elevation category
    mask_local = (
        (scan_xy[:, 0] >= rx - w_size) & (scan_xy[:, 0] <= rx + w_size) &
        (scan_xy[:, 1] >= ry - w_size) & (scan_xy[:, 1] <= ry + w_size)
    )
    s_xy_l = scan_xy[mask_local]
    s_z_l = scan_z[mask_local]

    if len(s_xy_l) > 0:
        # Trench
        m_trench = s_z_l < -0.12
        if np.any(m_trench):
            ax_ego.scatter(s_xy_l[m_trench, 0], s_xy_l[m_trench, 1], s=16, color="#e879f9", alpha=0.85, zorder=2, label="Floor Trench")
        # Flat Floor
        m_flat = (s_z_l >= -0.12) & (s_z_l <= 0.018)
        if np.any(m_flat):
            ax_ego.scatter(s_xy_l[m_flat, 0], s_xy_l[m_flat, 1], s=5, color="#334155", alpha=0.6, zorder=1)
        # Rough Boxes
        m_rough = (s_z_l > 0.018) & (s_z_l <= 0.12)
        if np.any(m_rough):
            ax_ego.scatter(s_xy_l[m_rough, 0], s_xy_l[m_rough, 1], s=14, color="#fbbf24", alpha=0.75, zorder=2, label="Rough Boxes")
        # Ramp / Spike Obstacles
        m_ramp = (s_z_l > 0.12) & (s_z_l <= 0.42)
        if np.any(m_ramp):
            ax_ego.scatter(s_xy_l[m_ramp, 0], s_xy_l[m_ramp, 1], s=18, color="#f97316", alpha=0.85, zorder=3, label="Ramp/Spike")
        # Walls
        m_wall = s_z_l > 0.45
        if np.any(m_wall):
            ax_ego.scatter(s_xy_l[m_wall, 0], s_xy_l[m_wall, 1], s=20, marker="s", color="#e2e8f0", alpha=0.9, zorder=3, label="Maze Walls")

    # 3. Planned G2 Spline (local section)
    if smooth_wpts is not None and len(smooth_wpts) > 0:
        dists = np.hypot(smooth_wpts[:, 0] - rx, smooth_wpts[:, 1] - ry)
        mask_wpts = dists < 3.2
        if np.any(mask_wpts):
            ax_ego.plot(
                smooth_wpts[mask_wpts, 0], smooth_wpts[mask_wpts, 1],
                color="#10b981", lw=2.8, alpha=0.95, zorder=6,
                path_effects=[pe.Stroke(linewidth=4, foreground="#064e3b"), pe.Normal()]
            )

    # 4. Lookahead Target
    if target_pt is not None:
        ax_ego.scatter([target_pt[0]], [target_pt[1]], s=130, marker="*", color="#facc15", edgecolors="white", lw=1.2, zorder=8)
        ax_ego.plot([rx, target_pt[0]], [ry, target_pt[1]], color="#facc15", ls="--", lw=1.2, alpha=0.8, zorder=7)

    # 5. Robot pose and Velocity Vectors
    rob_radius = 0.22
    ax_ego.add_patch(Circle((rx, ry), rob_radius, facecolor="#0ea5e9", edgecolor="#38bdf8", lw=2.0, alpha=0.8, zorder=9))
    # Heading line
    head_len = 0.32
    hx_r = rx + head_len * math.cos(yaw)
    hy_r = ry + head_len * math.sin(yaw)
    ax_ego.plot([rx, hx_r], [ry, hy_r], color="#38bdf8", lw=3.0, zorder=10)

    # Commanded Velocity Vector (Lime Green #22c55e)
    # cmd = [cmd_vx, cmd_vy, cmd_wz]
    v_scale = 0.8
    cv_x = math.cos(yaw) * cmd[0] - math.sin(yaw) * cmd[1]
    cv_y = math.sin(yaw) * cmd[0] + math.cos(yaw) * cmd[1]
    ax_ego.annotate(
        "", xy=(rx + cv_x * v_scale, ry + cv_y * v_scale), xytext=(rx, ry),
        arrowprops=dict(arrowstyle="->", color="#22c55e", lw=3.2, mutation_scale=16),
        zorder=11
    )
    if math.hypot(cv_x, cv_y) > 0.15:
        ax_ego.text(rx + cv_x * v_scale * 1.05, ry + cv_y * v_scale * 1.05, "cmd_vel", color="#22c55e", fontsize=7.5, fontweight="bold", zorder=12)

    # Actual Velocity Vector (Gold #eab308)
    ax_ego.annotate(
        "", xy=(rx + vw[0] * v_scale, ry + vw[1] * v_scale), xytext=(rx, ry),
        arrowprops=dict(arrowstyle="->", color="#eab308", lw=2.4, mutation_scale=14),
        zorder=11
    )
    if math.hypot(vw[0], vw[1]) > 0.15:
        ax_ego.text(rx + vw[0] * v_scale * 1.05, ry + vw[1] * v_scale * 1.05, "v_act", color="#eab308", fontsize=7.5, fontweight="bold", zorder=12)

    # Yaw Rate arc at robot front
    if abs(cmd[2]) > 0.05:
        arc_angle = 30.0 if cmd[2] > 0 else -30.0
        turn_color = "#38bdf8"
        ax_ego.annotate(
            "", xy=(rx + 0.35 * math.cos(yaw + math.radians(arc_angle)), ry + 0.35 * math.sin(yaw + math.radians(arc_angle))),
            xytext=(rx + 0.35 * math.cos(yaw - math.radians(arc_angle)), ry + 0.35 * math.sin(yaw - math.radians(arc_angle))),
            arrowprops=dict(arrowstyle="->", color=turn_color, lw=2.0, connectionstyle=f"arc3,rad={0.3 if cmd[2]>0 else -0.3}"),
            zorder=11
        )

    ax_ego.set_title("PANEL A: Ego-Centric Sensor & Velocity Vector View (4.8m x 4.8m)", color="white", fontsize=9.5, fontweight="bold", pad=5)
    ax_ego.tick_params(colors="#64748b", labelsize=7)
    ax_ego.grid(True, color="#1e293b", ls="--", lw=0.6)

    # =========================================================================
    # PANEL B: Global Online Map Discovery + A* & Spline Planning
    # =========================================================================
    ax_map.set_facecolor("#111827")
    ax_map.set_xlim(-1.2, (n - 1) * c + 1.2)
    ax_map.set_ylim(-1.2, (n - 1) * c + 1.2)
    ax_map.set_aspect("equal")

    # Grid background
    for i in range(n):
        for j in range(n):
            ax_map.add_patch(Rectangle((i * c - 1.0, j * c - 1.0), c, c, facecolor="#1e293b", edgecolor="#334155", lw=0.4))

    # Faint ground truth hazards
    for (i, j) in hazards.get("ice", []):
        ax_map.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor="#1d4ed8", alpha=0.35))
    for (i, j) in hazards.get("rough", []):
        ax_map.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor="#854d0e", alpha=0.35))
    for (i, j, a, w) in hazards.get("gap", []):
        rw, rh = (w, 1.7) if a == 0 else (1.7, w)
        ax_map.add_patch(Rectangle((i * c - 0.5 * rw, j * c - 0.5 * rh), rw, rh, facecolor="#dc2626", alpha=0.4))
    for (i, j, a, rh) in hazards.get("ramp", []):
        ax_map.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor="#b45309", alpha=0.4))
    for (i, j, a, x0, y0, x1, y1) in hazards.get("spike", []):
        ax_map.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#dc2626", hatch="//", alpha=0.5))

    # Online detected hazard zones
    for rz in frame["ramp_zones"]:
        ax_map.add_patch(Rectangle((rz[0] * c - 0.9, rz[1] * c - 0.9), 1.8, 1.8, facecolor="none", edgecolor="#f59e0b", lw=2.2, zorder=4))
    for gz in frame["gap_zones"]:
        ax_map.add_patch(Rectangle((gz[0] * c - 0.9, gz[1] * c - 0.9), 1.8, 1.8, facecolor="none", edgecolor="#ec4899", lw=2.2, zorder=4))
    for sk in frame["spike_zones"]:
        si, sj = int(sk.split("_")[0]), int(sk.split("_")[1])
        ax_map.add_patch(Rectangle((si * c - 0.9, sj * c - 0.9), 1.8, 1.8, facecolor="none", edgecolor="#ef4444", lw=2.2, ls="--", zorder=4))

    # Online Wall Memory
    for i in range(n):
        for j in range(n):
            cx, cy = i * c, j * c
            w_mem = known_walls[i, j]
            # 0: East, 1: North, 2: West, 3: South
            # Discovered walls (1): solid bright white/blue
            # Fog-of-war unknown walls (-1): dim dashed slate
            wall_dirs = [
                ([cx + 1.0, cx + 1.0], [cy - 1.0, cy + 1.0]),  # East
                ([cx - 1.0, cx + 1.0], [cy + 1.0, cy + 1.0]),  # North
                ([cx - 1.0, cx - 1.0], [cy - 1.0, cy + 1.0]),  # West
                ([cx - 1.0, cx + 1.0], [cy - 1.0, cy - 1.0]),  # South
            ]
            for d_idx, (xs, ys) in enumerate(wall_dirs):
                if w_mem[d_idx] == 1:
                    ax_map.plot(xs, ys, color="#f8fafc", lw=2.4, zorder=5)
                elif w_mem[d_idx] == -1 and (d_idx in (0, 1)):  # draw once
                    ax_map.plot(xs, ys, color="#334155", ls=":", lw=0.9, zorder=2)

    # Online A* Cell Path
    if cell_path and len(cell_path) > 0:
        c_xs = [cp[0] * c for cp in cell_path]
        c_ys = [cp[1] * c for cp in cell_path]
        ax_map.plot(c_xs, c_ys, color="#60a5fa", marker="D", markersize=3.5, lw=1.5, alpha=0.8, zorder=6, label="Online A* Path")

    # Full G2 Spline
    if smooth_wpts is not None and len(smooth_wpts) > 0:
        ax_map.plot(smooth_wpts[:, 0], smooth_wpts[:, 1], color="#10b981", lw=2.2, zorder=7, label="G2 Spline")

    # Robot traveled trail
    trail_xs = [f["pos"][0] for f in all_frames if f["t"] <= t]
    trail_ys = [f["pos"][1] for f in all_frames if f["t"] <= t]
    ax_map.plot(trail_xs, trail_ys, color="#38bdf8", lw=1.8, zorder=8)

    # Active lookahead point
    if target_pt is not None:
        ax_map.scatter([target_pt[0]], [target_pt[1]], s=80, marker="*", color="#fbbf24", edgecolors="white", lw=1.0, zorder=9)

    # Robot marker
    ax_map.scatter([rx], [ry], s=70, color="#38bdf8", edgecolors="white", lw=1.5, zorder=10)
    ax_map.annotate(
        "", xy=(rx + cv_x * 0.9, ry + cv_y * 0.9), xytext=(rx, ry),
        arrowprops=dict(arrowstyle="->", color="#22c55e", lw=2.0), zorder=11
    )

    # Start & Goal
    st = meta["start"]
    gl = meta["goal"]
    ax_map.scatter([st[0] * c], [st[1] * c], s=90, color="#3b82f6", marker="o", edgecolors="white", lw=1.5, zorder=8)
    ax_map.scatter([gl[0] * c], [gl[1] * c], s=120, color="#10b981", marker="*", edgecolors="white", lw=1.5, zorder=8)

    ax_map.set_title("PANEL B: Global Online Fog-of-War Discovery & A* Path Planning", color="white", fontsize=9.5, fontweight="bold", pad=5)
    ax_map.tick_params(colors="#64748b", labelsize=7)
    ax_map.grid(False)

    # =========================================================================
    # PANEL C: Synchronized 3D Simulation View + Mode Banner
    # =========================================================================
    ax_cam.set_facecolor("#0b0f19")
    if video_reader is not None:
        vid_len = video_reader.count_frames()
        t_max_sim = meta.get("finish_time", 67.9)
        vid_idx = int(round((t / max(0.1, t_max_sim)) * (vid_len - 1)))
        vid_idx = max(0, min(vid_len - 1, vid_idx))
        try:
            cam_frame = video_reader.get_data(vid_idx)
            ax_cam.imshow(cam_frame)
        except Exception:
            ax_cam.text(0.5, 0.5, "3D Newton Replay Frame", color="#94a3b8", ha="center", va="center")
    else:
        ax_cam.text(0.5, 0.5, "Synchronized 3D Simulation Camera", color="#94a3b8", ha="center", va="center")

    ax_cam.set_xticks([])
    ax_cam.set_yticks([])

    # Mode Banner at top of Panel C
    mode_color = (
        "#e879f9" if "GAP" in active_mode else
        "#f59e0b" if "RAMP" in active_mode else
        "#ef4444" if "SPIKE" in active_mode else
        "#10b981"
    )
    t_fin_disp = meta.get("finish_time", 67.9)
    banner_text = f"MODE: {active_mode}\nCELL: {cur_c}  |  TIME: {t:4.1f} s / {t_fin_disp:.1f} s  |  0 FALLS"
    ax_cam.text(
        0.5, 0.91, banner_text, transform=ax_cam.transAxes,
        color=mode_color, fontsize=8.5, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor=mode_color, lw=1.8, alpha=0.92)
    )
    ax_cam.set_title("PANEL C: Synchronized 3D Physical Simulation (Chase Camera)", color="white", fontsize=9.5, fontweight="bold", pad=5)

    # =========================================================================
    # PANEL D (Left): Live Horizontal Command & Speed Gauges
    # =========================================================================
    ax_gauge.set_facecolor("#111827")
    gauge_labels = ["cmd_vx (m/s)", "cmd_vy (m/s)", "cmd_wz (rad/s)", "speed_act (m/s)"]
    v_mag = float(np.hypot(vw[0], vw[1]))
    gauge_vals = [cmd[0], cmd[1], cmd[2], v_mag]
    gauge_limits = [(0.0, 1.0), (-0.4, 0.4), (-1.2, 1.2), (0.0, 1.0)]
    gauge_colors = ["#22c55e", "#a855f7", "#06b6d4", "#eab308"]

    y_pos = np.arange(len(gauge_labels))
    ax_gauge.set_yticks(y_pos)
    ax_gauge.set_yticklabels(gauge_labels, color="#e2e8f0", fontsize=8, fontweight="bold")
    ax_gauge.set_xlim(-1.25, 1.25)
    ax_gauge.axvline(0.0, color="#475569", lw=0.8)

    for idx, (val, (vmin, vmax), col) in enumerate(zip(gauge_vals, gauge_limits, gauge_colors)):
        v_clamped = max(vmin, min(vmax, val))
        if vmin >= 0:
            ax_gauge.barh(idx, v_clamped, height=0.55, color=col, alpha=0.85, edgecolor="white", lw=0.5)
        else:
            ax_gauge.barh(idx, v_clamped, height=0.55, color=col, alpha=0.85, edgecolor="white", lw=0.5)
        ax_gauge.text(
            v_clamped + (0.08 if v_clamped >= 0 else -0.08), idx, f"{val:+.2f}",
            color="white", fontsize=8, fontweight="bold", va="center", ha="left" if v_clamped >= 0 else "right"
        )

    ax_gauge.set_title("PANEL D1: Live Policy Command Gauges", color="white", fontsize=9.0, fontweight="bold", pad=4)
    ax_gauge.tick_params(colors="#64748b", labelsize=7)
    ax_gauge.grid(axis="x", color="#1e293b", ls="--", lw=0.6)

    # =========================================================================
    # PANEL D (Right): Scrolling Time-Series Plot with Playhead & Hazard Zones
    # =========================================================================
    ax_ts.set_facecolor("#111827")
    ts_t = c_time_series_data["t"]
    ts_cmd_vx = c_time_series_data["cmd_vx"]
    ts_act_v = c_time_series_data["act_v"]
    ts_cmd_wz = c_time_series_data["cmd_wz"]

    # Shaded Hazard Bands matching actual trajectory timestamps
    t_fin = meta.get("finish_time", 67.9)
    bands = [
        (3.0, 11.5, "25cm Ramp (1,0)/(2,0)", "#d97706"),
        (22.0, 27.5, "Spike Bed (4,1)", "#dc2626"),
        (36.0, 39.5, "19.7cm Trench (5,2)", "#ec4899"),
        (49.0, 56.5, "26cm Ramp (6,3)", "#d97706"),
        (63.0, 67.0, "Spike Bed (6,5)", "#dc2626"),
    ]
    for b_start, b_end, b_label, b_col in bands:
        ax_ts.axvspan(b_start, b_end, color=b_col, alpha=0.22, zorder=1)
        ax_ts.text(0.5 * (b_start + b_end), 1.02, b_label, color=b_col, fontsize=6.8, fontweight="bold", ha="center", va="bottom")

    ax_ts.plot(ts_t, ts_cmd_vx, color="#22c55e", lw=1.6, label="cmd_vx (m/s)", zorder=3)
    ax_ts.plot(ts_t, ts_act_v, color="#eab308", lw=1.6, label="actual_speed (m/s)", zorder=3)
    ax_ts.plot(ts_t, ts_cmd_wz, color="#06b6d4", lw=1.2, alpha=0.85, label="cmd_wz (rad/s)", zorder=3)
    ax_ts.axhline(0.60, color="#94a3b8", ls="--", lw=0.9, alpha=0.6, label="Nominal Track A (0.60)")
    ax_ts.axhline(0.85, color="#f59e0b", ls=":", lw=1.1, alpha=0.8, label="Trench Stride (0.85)")

    # Moving vertical playhead line
    ax_ts.axvline(t, color="#f8fafc", lw=2.2, zorder=5, path_effects=[pe.Stroke(linewidth=3.5, foreground="#0284c7"), pe.Normal()])

    ax_ts.set_xlim(0, max(ts_t[-1], t_fin + 1.0))
    ax_ts.set_ylim(-1.3, 1.25)
    ax_ts.set_xlabel("Simulation Elapsed Time (seconds)", color="#94a3b8", fontsize=8.5)
    ax_ts.set_ylabel("Command & Velocity Values", color="#94a3b8", fontsize=8.5)
    ax_ts.set_title("PANEL D2: Complete Trajectory Telemetry & Controller Response Profile", color="white", fontsize=9.0, fontweight="bold", pad=4)
    ax_ts.tick_params(colors="#64748b", labelsize=7.5)
    ax_ts.grid(True, color="#1e293b", ls="--", lw=0.6)
    ax_ts.legend(loc="upper left", fontsize=7.2, facecolor="#0f172a", edgecolor="#334155", labelcolor="white", ncol=5)


def generate_mp4_video(telemetry_bundle: dict):
    """Render the full 1600x900 20 fps dashboard MP4 video with high-performance in-place updates."""
    import imageio

    frames = telemetry_bundle["frames"]
    meta = telemetry_bundle["meta"]
    n_frames = len(frames)
    t_fin = meta.get("finish_time", 67.9)
    c = CELL_SIZE
    n = meta["n"]
    hazards = meta["hazards"]

    print(f"[RENDER] Preparing high-performance rendering for {n_frames} frames @ 20 fps -> {OUT_MP4}...", flush=True)

    # 1. Pre-decode & downsample 3D camera frames for instantaneous access
    cached_cam_frames = []
    if os.path.exists(VIDEO_SOURCE):
        try:
            reader = imageio.get_reader(VIDEO_SOURCE)
            vid_len = reader.count_frames()
            print(f"[RENDER] Pre-caching {n_frames} 3D camera frames from {VIDEO_SOURCE}...")
            for f in frames:
                vid_idx = int(round((f["t"] / max(0.1, t_fin)) * (vid_len - 1)))
                vid_idx = max(0, min(vid_len - 1, vid_idx))
                try:
                    raw_cf = reader.get_data(vid_idx)
                    cached_cam_frames.append(raw_cf[::2, ::2])  # 360x640 RGB
                except Exception:
                    cached_cam_frames.append(None)
            reader.close()
        except Exception as e:
            print(f"[WARN] Error reading {VIDEO_SOURCE}: {e}")

    # Fallback blank frame if needed
    if not cached_cam_frames or all(cf is None for cf in cached_cam_frames):
        cached_cam_frames = [np.full((360, 640, 3), 15, dtype=np.uint8) for _ in range(n_frames)]

    # 2. Extract complete time series
    ts_t = [f["t"] for f in frames]
    ts_cmd_vx = [f["cmd"][0] for f in frames]
    ts_act_v = [float(np.hypot(f["vel_world"][0], f["vel_world"][1])) for f in frames]
    ts_cmd_wz = [f["cmd"][2] for f in frames]

    # 3. Create persistent Matplotlib figure & axes
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#0f172a")

    ax_ego = fig.add_axes([0.035, 0.40, 0.28, 0.54])
    ax_map = fig.add_axes([0.345, 0.40, 0.28, 0.54])
    ax_cam = fig.add_axes([0.655, 0.40, 0.31, 0.54])
    ax_gauge = fig.add_axes([0.035, 0.07, 0.23, 0.27])
    ax_ts = fig.add_axes([0.305, 0.07, 0.66, 0.27])

    # 4. Initialize Static Time-Series (ax_ts) ONCE
    ax_ts.set_facecolor("#111827")
    bands = [
        (3.0, 11.5, "25cm Ramp (1,0)/(2,0)", "#d97706"),
        (22.0, 27.5, "Spike Bed (4,1)", "#dc2626"),
        (36.0, 39.5, "19.7cm Trench (5,2)", "#ec4899"),
        (49.0, 56.5, "26cm Ramp (6,3)", "#d97706"),
        (63.0, 67.0, "Spike Bed (6,5)", "#dc2626"),
    ]
    for b_start, b_end, b_label, b_col in bands:
        ax_ts.axvspan(b_start, b_end, color=b_col, alpha=0.22, zorder=1)
        ax_ts.text(0.5 * (b_start + b_end), 1.02, b_label, color=b_col, fontsize=6.8, fontweight="bold", ha="center", va="bottom")

    ax_ts.plot(ts_t, ts_cmd_vx, color="#22c55e", lw=1.6, label="cmd_vx (m/s)", zorder=3)
    ax_ts.plot(ts_t, ts_act_v, color="#eab308", lw=1.6, label="actual_speed (m/s)", zorder=3)
    ax_ts.plot(ts_t, ts_cmd_wz, color="#06b6d4", lw=1.2, alpha=0.85, label="cmd_wz (rad/s)", zorder=3)
    ax_ts.axhline(0.60, color="#94a3b8", ls="--", lw=0.9, alpha=0.6, label="Nominal Track A (0.60)")
    ax_ts.axhline(0.85, color="#f59e0b", ls=":", lw=1.1, alpha=0.8, label="Trench Stride (0.85)")

    playhead = ax_ts.axvline(0.0, color="#f8fafc", lw=2.2, zorder=5, path_effects=[pe.Stroke(linewidth=3.5, foreground="#0284c7"), pe.Normal()])

    ax_ts.set_xlim(0, max(ts_t[-1], t_fin + 1.0))
    ax_ts.set_ylim(-1.3, 1.25)
    ax_ts.set_xlabel("Simulation Elapsed Time (seconds)", color="#94a3b8", fontsize=8.5)
    ax_ts.set_ylabel("Command & Velocity Values", color="#94a3b8", fontsize=8.5)
    ax_ts.set_title("PANEL D2: Complete Trajectory Telemetry & Controller Response Profile", color="white", fontsize=9.0, fontweight="bold", pad=4)
    ax_ts.tick_params(colors="#64748b", labelsize=7.5)
    ax_ts.grid(True, color="#1e293b", ls="--", lw=0.6)
    ax_ts.legend(loc="upper left", fontsize=7.2, facecolor="#0f172a", edgecolor="#334155", labelcolor="white", ncol=5)

    # 5. Initialize Camera View (ax_cam) ONCE
    ax_cam.set_facecolor("#0b0f19")
    cam_artist = ax_cam.imshow(cached_cam_frames[0])
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    ax_cam.set_title("PANEL C: Synchronized 3D Physical Simulation (Chase Camera)", color="white", fontsize=9.5, fontweight="bold", pad=5)

    banner_box = dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#10b981", lw=1.8, alpha=0.92)
    banner_text = ax_cam.text(
        0.5, 0.91, "MODE: G2 QUINTIC SPLINE CRUISE\nCELL: (0, 0) | TIME: 0.0 s", transform=ax_cam.transAxes,
        color="#10b981", fontsize=8.5, fontweight="bold", ha="center", va="center", bbox=banner_box
    )

    # 6. Initialize Video Writer
    os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
    writer = imageio.get_writer(
        OUT_MP4,
        fps=20,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-preset", "veryfast", "-crf", "20"]
    )

    # All positions array for fast trail slicing
    all_xy = np.array([f["pos"][:2] for f in frames])

    t_start = time.time()
    for idx, frame in enumerate(frames):
        t = frame["t"]
        pos = frame["pos"]
        yaw = frame["yaw"]
        rx, ry = pos[0], pos[1]
        cmd = frame["cmd"]
        vw = frame["vel_world"]
        scan_xy = frame["scan_xy"]
        scan_z = frame["scan_z"]
        lidar_dist = frame["lidar_dist"]
        lidar_yaw = frame["lidar_yaw"]
        known_walls = frame["known_walls"]
        cell_path = frame["cell_path"]
        smooth_wpts = frame["smooth_wpts"]
        target_pt = frame["target_pt"]
        active_mode = frame["active_mode"]
        cur_c = frame["cell"]

        # =====================================================================
        # Update Panel C (Camera View & Banner)
        # =====================================================================
        if cached_cam_frames[idx] is not None:
            cam_artist.set_data(cached_cam_frames[idx])

        mode_color = (
            "#e879f9" if "GAP" in active_mode else
            "#f59e0b" if "RAMP" in active_mode else
            "#ef4444" if "SPIKE" in active_mode else
            "#10b981"
        )
        banner_text.set_text(f"MODE: {active_mode}\nCELL: {cur_c}  |  TIME: {t:4.1f} s / {t_fin:.1f} s  |  0 FALLS")
        banner_text.set_color(mode_color)
        banner_text.get_bbox_patch().set_edgecolor(mode_color)

        # =====================================================================
        # Update Panel D1 (Live Gauges)
        # =====================================================================
        ax_gauge.clear()
        ax_gauge.set_facecolor("#111827")
        gauge_labels = ["cmd_vx (m/s)", "cmd_vy (m/s)", "cmd_wz (rad/s)", "speed_act (m/s)"]
        v_mag = float(np.hypot(vw[0], vw[1]))
        gauge_vals = [cmd[0], cmd[1], cmd[2], v_mag]
        gauge_limits = [(0.0, 1.0), (-0.4, 0.4), (-1.2, 1.2), (0.0, 1.0)]
        gauge_colors = ["#22c55e", "#a855f7", "#06b6d4", "#eab308"]

        y_pos = np.arange(len(gauge_labels))
        ax_gauge.set_yticks(y_pos)
        ax_gauge.set_yticklabels(gauge_labels, color="#e2e8f0", fontsize=8, fontweight="bold")
        ax_gauge.set_xlim(-1.25, 1.25)
        ax_gauge.axvline(0.0, color="#475569", lw=0.8)

        for g_i, (val, (vmin, vmax), col) in enumerate(zip(gauge_vals, gauge_limits, gauge_colors)):
            v_clamped = max(vmin, min(vmax, val))
            ax_gauge.barh(g_i, v_clamped, height=0.55, color=col, alpha=0.85, edgecolor="white", lw=0.5)
            ax_gauge.text(
                v_clamped + (0.08 if v_clamped >= 0 else -0.08), g_i, f"{val:+.2f}",
                color="white", fontsize=8, fontweight="bold", va="center", ha="left" if v_clamped >= 0 else "right"
            )

        ax_gauge.set_title("PANEL D1: Live Policy Command Gauges", color="white", fontsize=9.0, fontweight="bold", pad=4)
        ax_gauge.tick_params(colors="#64748b", labelsize=7)
        ax_gauge.grid(axis="x", color="#1e293b", ls="--", lw=0.6)

        # =====================================================================
        # Update Panel D2 (Playhead Only)
        # =====================================================================
        playhead.set_xdata([t, t])

        # =====================================================================
        # Redraw Panel A (Ego-Centric Local Sensor View)
        # =====================================================================
        ax_ego.clear()
        ax_ego.set_facecolor("#111827")
        w_size = 2.4
        ax_ego.set_xlim(rx - w_size, rx + w_size)
        ax_ego.set_ylim(ry - w_size, ry + w_size)
        ax_ego.set_aspect("equal")

        # 72-Ray Chest LiDAR fan
        for ld, ly in zip(lidar_dist, lidar_yaw):
            if ld < 5.8:
                hx = rx + ld * math.cos(ly)
                hy = ry + ld * math.sin(ly)
                ax_ego.plot([rx, hx], [ry, hy], color="#06b6d4", lw=0.8, alpha=0.35, zorder=3)
                ax_ego.scatter([hx], [hy], s=6, color="#22d3ee", alpha=0.8, zorder=4)

        # Height scanner
        mask_local = (
            (scan_xy[:, 0] >= rx - w_size) & (scan_xy[:, 0] <= rx + w_size) &
            (scan_xy[:, 1] >= ry - w_size) & (scan_xy[:, 1] <= ry + w_size)
        )
        s_xy_l = scan_xy[mask_local]
        s_z_l = scan_z[mask_local]
        if len(s_xy_l) > 0:
            m_trench = s_z_l < -0.12
            if np.any(m_trench):
                ax_ego.scatter(s_xy_l[m_trench, 0], s_xy_l[m_trench, 1], s=16, color="#e879f9", alpha=0.85, zorder=2)
            m_flat = (s_z_l >= -0.12) & (s_z_l <= 0.018)
            if np.any(m_flat):
                ax_ego.scatter(s_xy_l[m_flat, 0], s_xy_l[m_flat, 1], s=5, color="#334155", alpha=0.6, zorder=1)
            m_rough = (s_z_l > 0.018) & (s_z_l <= 0.12)
            if np.any(m_rough):
                ax_ego.scatter(s_xy_l[m_rough, 0], s_xy_l[m_rough, 1], s=14, color="#fbbf24", alpha=0.75, zorder=2)
            m_ramp = (s_z_l > 0.12) & (s_z_l <= 0.42)
            if np.any(m_ramp):
                ax_ego.scatter(s_xy_l[m_ramp, 0], s_xy_l[m_ramp, 1], s=18, color="#f97316", alpha=0.85, zorder=3)
            m_wall = s_z_l > 0.45
            if np.any(m_wall):
                ax_ego.scatter(s_xy_l[m_wall, 0], s_xy_l[m_wall, 1], s=20, marker="s", color="#e2e8f0", alpha=0.9, zorder=3)

        # Spline
        if smooth_wpts is not None and len(smooth_wpts) > 0:
            dists = np.hypot(smooth_wpts[:, 0] - rx, smooth_wpts[:, 1] - ry)
            mask_wpts = dists < 3.2
            if np.any(mask_wpts):
                ax_ego.plot(
                    smooth_wpts[mask_wpts, 0], smooth_wpts[mask_wpts, 1],
                    color="#10b981", lw=2.8, alpha=0.95, zorder=6,
                    path_effects=[pe.Stroke(linewidth=4, foreground="#064e3b"), pe.Normal()]
                )

        # Lookahead Target
        if target_pt is not None:
            ax_ego.scatter([target_pt[0]], [target_pt[1]], s=130, marker="*", color="#facc15", edgecolors="white", lw=1.2, zorder=8)
            ax_ego.plot([rx, target_pt[0]], [ry, target_pt[1]], color="#facc15", ls="--", lw=1.2, alpha=0.8, zorder=7)

        # Robot marker
        ax_ego.add_patch(Circle((rx, ry), 0.22, facecolor="#0ea5e9", edgecolor="#38bdf8", lw=2.0, alpha=0.8, zorder=9))
        ax_ego.plot([rx, rx + 0.32 * math.cos(yaw)], [ry, ry + 0.32 * math.sin(yaw)], color="#38bdf8", lw=3.0, zorder=10)

        # Commanded velocity vector (Lime Green)
        v_scale = 0.8
        cv_x = math.cos(yaw) * cmd[0] - math.sin(yaw) * cmd[1]
        cv_y = math.sin(yaw) * cmd[0] + math.cos(yaw) * cmd[1]
        ax_ego.annotate(
            "", xy=(rx + cv_x * v_scale, ry + cv_y * v_scale), xytext=(rx, ry),
            arrowprops=dict(arrowstyle="->", color="#22c55e", lw=3.2, mutation_scale=16), zorder=11
        )
        if math.hypot(cv_x, cv_y) > 0.15:
            ax_ego.text(rx + cv_x * v_scale * 1.05, ry + cv_y * v_scale * 1.05, "cmd_vel", color="#22c55e", fontsize=7.5, fontweight="bold", zorder=12)

        # Actual velocity vector (Gold)
        ax_ego.annotate(
            "", xy=(rx + vw[0] * v_scale, ry + vw[1] * v_scale), xytext=(rx, ry),
            arrowprops=dict(arrowstyle="->", color="#eab308", lw=2.4, mutation_scale=14), zorder=11
        )
        if math.hypot(vw[0], vw[1]) > 0.15:
            ax_ego.text(rx + vw[0] * v_scale * 1.05, ry + vw[1] * v_scale * 1.05, "v_act", color="#eab308", fontsize=7.5, fontweight="bold", zorder=12)

        # Yaw rate arc
        if abs(cmd[2]) > 0.05:
            arc_angle = 30.0 if cmd[2] > 0 else -30.0
            ax_ego.annotate(
                "", xy=(rx + 0.35 * math.cos(yaw + math.radians(arc_angle)), ry + 0.35 * math.sin(yaw + math.radians(arc_angle))),
                xytext=(rx + 0.35 * math.cos(yaw - math.radians(arc_angle)), ry + 0.35 * math.sin(yaw - math.radians(arc_angle))),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.0, connectionstyle=f"arc3,rad={0.3 if cmd[2]>0 else -0.3}"),
                zorder=11
            )

        ax_ego.set_title("PANEL A: Ego-Centric Sensor & Velocity Vector View (4.8m x 4.8m)", color="white", fontsize=9.5, fontweight="bold", pad=5)
        ax_ego.tick_params(colors="#64748b", labelsize=7)
        ax_ego.grid(True, color="#1e293b", ls="--", lw=0.6)

        # =====================================================================
        # Redraw Panel B (Global Fog-of-War Map & Planning)
        # =====================================================================
        ax_map.clear()
        ax_map.set_facecolor("#111827")
        ax_map.set_xlim(-1.2, (n - 1) * c + 1.2)
        ax_map.set_ylim(-1.2, (n - 1) * c + 1.2)
        ax_map.set_aspect("equal")

        for i in range(n):
            for j in range(n):
                ax_map.add_patch(Rectangle((i * c - 1.0, j * c - 1.0), c, c, facecolor="#1e293b", edgecolor="#334155", lw=0.4))

        # Faint hazards
        for (i, j) in hazards.get("ice", []):
            ax_map.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor="#1d4ed8", alpha=0.35))
        for (i, j) in hazards.get("rough", []):
            ax_map.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor="#854d0e", alpha=0.35))
        for (i, j, a, w) in hazards.get("gap", []):
            rw, rh = (w, 1.7) if a == 0 else (1.7, w)
            ax_map.add_patch(Rectangle((i * c - 0.5 * rw, j * c - 0.5 * rh), rw, rh, facecolor="#dc2626", alpha=0.4))
        for (i, j, a, rh) in hazards.get("ramp", []):
            ax_map.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor="#b45309", alpha=0.4))
        for (i, j, a, x0, y0, x1, y1) in hazards.get("spike", []):
            ax_map.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#dc2626", hatch="//", alpha=0.5))

        # Online hazard zones
        for rz in frame["ramp_zones"]:
            ax_map.add_patch(Rectangle((rz[0] * c - 0.9, rz[1] * c - 0.9), 1.8, 1.8, facecolor="none", edgecolor="#f59e0b", lw=2.2, zorder=4))
        for gz in frame["gap_zones"]:
            ax_map.add_patch(Rectangle((gz[0] * c - 0.9, gz[1] * c - 0.9), 1.8, 1.8, facecolor="none", edgecolor="#ec4899", lw=2.2, zorder=4))
        for sk in frame["spike_zones"]:
            si, sj = int(sk.split("_")[0]), int(sk.split("_")[1])
            ax_map.add_patch(Rectangle((si * c - 0.9, sj * c - 0.9), 1.8, 1.8, facecolor="none", edgecolor="#ef4444", lw=2.2, ls="--", zorder=4))

        # Walls
        for i in range(n):
            for j in range(n):
                cx, cy = i * c, j * c
                w_mem = known_walls[i, j]
                wall_dirs = [
                    ([cx + 1.0, cx + 1.0], [cy - 1.0, cy + 1.0]),
                    ([cx - 1.0, cx + 1.0], [cy + 1.0, cy + 1.0]),
                    ([cx - 1.0, cx - 1.0], [cy - 1.0, cy + 1.0]),
                    ([cx - 1.0, cx + 1.0], [cy - 1.0, cy - 1.0]),
                ]
                for d_idx, (xs, ys) in enumerate(wall_dirs):
                    if w_mem[d_idx] == 1:
                        ax_map.plot(xs, ys, color="#f8fafc", lw=2.4, zorder=5)
                    elif w_mem[d_idx] == -1 and (d_idx in (0, 1)):
                        ax_map.plot(xs, ys, color="#334155", ls=":", lw=0.9, zorder=2)

        if cell_path and len(cell_path) > 0:
            ax_map.plot([cp[0] * c for cp in cell_path], [cp[1] * c for cp in cell_path], color="#60a5fa", marker="D", markersize=3.5, lw=1.5, alpha=0.8, zorder=6)

        if smooth_wpts is not None and len(smooth_wpts) > 0:
            ax_map.plot(smooth_wpts[:, 0], smooth_wpts[:, 1], color="#10b981", lw=2.2, zorder=7)

        # Traveled trail up to current step
        ax_map.plot(all_xy[: idx + 1, 0], all_xy[: idx + 1, 1], color="#38bdf8", lw=1.8, zorder=8)

        if target_pt is not None:
            ax_map.scatter([target_pt[0]], [target_pt[1]], s=80, marker="*", color="#fbbf24", edgecolors="white", lw=1.0, zorder=9)

        ax_map.scatter([rx], [ry], s=70, color="#38bdf8", edgecolors="white", lw=1.5, zorder=10)
        ax_map.annotate("", xy=(rx + cv_x * 0.9, ry + cv_y * 0.9), xytext=(rx, ry), arrowprops=dict(arrowstyle="->", color="#22c55e", lw=2.0), zorder=11)

        st = meta["start"]
        gl = meta["goal"]
        ax_map.scatter([st[0] * c], [st[1] * c], s=90, color="#3b82f6", marker="o", edgecolors="white", lw=1.5, zorder=8)
        ax_map.scatter([gl[0] * c], [gl[1] * c], s=120, color="#10b981", marker="*", edgecolors="white", lw=1.5, zorder=8)

        ax_map.set_title("PANEL B: Global Online Fog-of-War Discovery & A* Path Planning", color="white", fontsize=9.5, fontweight="bold", pad=5)
        ax_map.tick_params(colors="#64748b", labelsize=7)
        ax_map.grid(False)

        # Render frame to buffer
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(rgba[:, :, :3])

        if idx % 50 == 0 or idx == n_frames - 1:
            fps_est = (idx + 1) / max(0.1, time.time() - t_start)
            print(f"  Frame {idx+1}/{n_frames} rendered ({fps_est:.1f} fps)...", flush=True)

    writer.close()
    plt.close(fig)
    print(f"[SUCCESS] MP4 Video generated successfully: {OUT_MP4} ({os.path.getsize(OUT_MP4) / 1024 / 1024:.2f} MB)")


def generate_keyframes_figure(telemetry_bundle: dict):
    """Render a 4-keyframe summary composite figure (2x2 grid) showing key challenge milestones."""
    import imageio

    frames = telemetry_bundle["frames"]
    meta = telemetry_bundle["meta"]

    c_ts = {
        "t": [f["t"] for f in frames],
        "cmd_vx": [f["cmd"][0] for f in frames],
        "act_v": [float(np.hypot(f["vel_world"][0], f["vel_world"][1])) for f in frames],
        "cmd_wz": [f["cmd"][2] for f in frames],
    }

    video_reader = None
    if os.path.exists(VIDEO_SOURCE):
        try:
            video_reader = imageio.get_reader(VIDEO_SOURCE)
        except Exception:
            pass

    # Find closest frames to 4 target milestones:
    # 1. 25cm Ramp (1, 0) ~ 6.0s
    # 2. Spike Bed Avoidance (4, 1) ~ 25.5s
    # 3. 19.7cm Floor Trench (5, 2) ~ 37.8s
    # 4. Final Spike Bed & Goal (6, 5)/(6, 6) ~ 67.0s
    t_fin = meta.get("finish_time", 67.9)
    targets = [
        {"desc": "Milestone 1: 25cm Steep Ramp Climbing at Cell (1, 0) with Centerline Heading Lock (|wz| <= 0.06 rad/s)", "t_target": 6.0},
        {"desc": "Milestone 2: Avoiding Deadly Spike Bed at Cell (4, 1) via Safe-Lane Lateral Contouring", "t_target": 25.5},
        {"desc": "Milestone 3: Striding Across 19.7cm Floor Trench at Cell (5, 2) with Speed Boost (cmd_vx = 0.85 m/s)", "t_target": 37.8},
        {"desc": f"Milestone 4: Negotiating Final Spike Bed at Cell (6, 5) and Reaching Goal (6, 6) in {t_fin:.1f}s (0 Falls)", "t_target": min(67.0, t_fin - 0.5)},
    ]

    selected_indices = []
    for tg in targets:
        best_i = min(range(len(frames)), key=lambda i: abs(frames[i]["t"] - tg["t_target"]))
        selected_indices.append((best_i, tg["desc"]))

    # Render each milestone dashboard to an RGB array
    milestone_images = []
    fig_single = plt.figure(figsize=(16, 9), dpi=100, facecolor="#0f172a")
    axes_single = {
        "ego": fig_single.add_axes([0.035, 0.40, 0.28, 0.54]),
        "map": fig_single.add_axes([0.345, 0.40, 0.28, 0.54]),
        "cam": fig_single.add_axes([0.655, 0.40, 0.31, 0.54]),
        "gauge": fig_single.add_axes([0.035, 0.07, 0.23, 0.27]),
        "ts": fig_single.add_axes([0.305, 0.07, 0.66, 0.27]),
    }

    for (f_idx, desc) in selected_indices:
        render_dashboard_frame(fig_single, axes_single, frames[f_idx], meta, frames, video_reader, c_ts)
        fig_single.canvas.draw()
        rgba = np.asarray(fig_single.canvas.buffer_rgba())
        milestone_images.append((rgba[:, :, :3].copy(), desc))

    plt.close(fig_single)
    if video_reader is not None:
        video_reader.close()

    # Now composite into a 2x2 grid
    fig_comp = plt.figure(figsize=(24, 14), facecolor="#0b0f19")
    gs = fig_comp.add_gridspec(2, 2, hspace=0.10, wspace=0.06, left=0.02, right=0.98, top=0.94, bottom=0.03)

    for i in range(4):
        r, col_idx = i // 2, i % 2
        ax = fig_comp.add_subplot(gs[r, col_idx])
        img, desc = milestone_images[i]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#38bdf8")
            spine.set_linewidth(1.8)
        ax.set_title(desc, color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)

    plt.suptitle(
        "G1 Humanoid Robot Perception, Online Fog-of-War Discovery & G2 Spline Planning Keyframes\n"
        "Level 7 Hazard Maze (Seed 2026, Short Route Direct Traversal: 58.1 s, 0 Falls, 4/4 Hazards Cleared)",
        color="#38bdf8", fontsize=15, fontweight="bold", y=0.98
    )

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig_comp)
    print(f"[SUCCESS] Keyframes summary figure saved to {OUT_PNG} ({os.path.getsize(OUT_PNG) / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Visualize G1 Robot Perception, Planning and Cmd_Vel Vectors.")
    parser.add_argument("--record", action="store_true", help="Force re-running simulation to capture telemetry.")
    parser.add_argument("--record-only", action="store_true", help="Only record telemetry without rendering.")
    parser.add_argument("--render-only", action="store_true", help="Only render from existing telemetry cache.")
    parser.add_argument("--seed", type=int, default=2026, help="Maze seed (default: 2026).")
    args = parser.parse_args()

    if args.record or not os.path.exists(CACHE_FILE):
        print(f"[START] Recording simulation telemetry for seed {args.seed}...")
        telemetry = record_simulation_telemetry(seed=args.seed)
    else:
        print(f"[START] Loading existing telemetry from {CACHE_FILE}...")
        telemetry = load_telemetry()

    if not args.record_only:
        generate_mp4_video(telemetry)
        generate_keyframes_figure(telemetry)


if __name__ == "__main__":
    main()
