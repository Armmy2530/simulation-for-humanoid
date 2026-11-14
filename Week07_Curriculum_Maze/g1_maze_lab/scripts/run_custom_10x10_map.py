"""Run, record, and visualize G1 navigation on the custom 10x10 hazard maze.

Generates:
1. 3D Chase-Cam Simulation Video:
   artifacts/experiment_10_track_a_hazard_slalom/run/custom_10x10_walk.mp4
2. Side-by-Side Verification Plot (Original PNG vs Reconstructed 10x10 + Trajectory vs 50 Hz Telemetry):
   artifacts/experiment_10_track_a_hazard_slalom/custom_10x10_map_verification.png
3. 4-Panel Perception + Planning + cmd_vel Vector Dashboard Video:
   artifacts/experiment_10_track_a_hazard_slalom/run/custom_10x10_perception_viz.mp4

Usage (from g1_maze_lab/):
    uv run python scripts/run_custom_10x10_map.py --record --render
    uv run python scripts/run_custom_10x10_map.py --render
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import pickle
import sys
import time

# Load imgui_bundle before Isaac Lab (required by Newton viewer HUD)
with contextlib.suppress(ImportError):
    import imgui_bundle  # noqa: F401

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAB_DIR = os.path.join(PROJECT_ROOT, "g1_maze_lab")
EXP_DIR = os.path.join(PROJECT_ROOT, "artifacts", "experiment_10_track_a_hazard_slalom")
RUN_DIR = os.path.join(EXP_DIR, "run")

REF_IMAGE = "/home/armmy2530/.t3/userdata/attachments/mcp-100274f0-81c3-444a-a332-42b99ce5b352-b05d3bbf-e76a-4a67-a167-c028300efa44.png"
PILOT_PATH = os.path.join(PROJECT_ROOT, "pilot.py")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "policy.pt")

OUT_WALK_MP4 = os.path.join(RUN_DIR, "custom_10x10_walk.mp4")
OUT_TRAJ_NPZ = os.path.join(RUN_DIR, "custom_10x10_traj_50hz.npz")
OUT_TELEM_PKL = os.path.join(RUN_DIR, "custom_10x10_perception_telemetry.pkl")
OUT_VERIFY_PNG = os.path.join(EXP_DIR, "custom_10x10_map_verification.png")
OUT_PERCEPT_MP4 = os.path.join(RUN_DIR, "custom_10x10_perception_viz.mp4")

CELL_SIZE = 2.0


def record_run(max_time: float = 480.0, headless: bool = False):
    """Run G1 on the custom 10x10 hazard maze with Newton 3D recording + 50 Hz/10 Hz telemetry."""
    os.environ["__GL_SYNC_TO_VBLANK"] = "0"
    os.environ["MAZE_CUSTOM_10X10"] = "1"
    os.environ["MAZE_CAM"] = "chase"
    if not headless:
        os.environ["MAZE_REC_MP4"] = OUT_WALK_MP4
    os.environ["MAZE_REC_EVERY"] = "10"

    sys.path.insert(0, os.path.join(LAB_DIR, "src"))
    sys.path.insert(0, os.path.join(LAB_DIR, "scripts"))

    import warp as wp
    wp.config.enable_backward = False
    if not headless:
        import newton.viewer
        _orig_vgl_init = newton.viewer.ViewerGL.__init__

        def _fast_vgl_init(self, *a, **kw):
            if len(a) == 0:
                kw["width"] = 1280
                kw["height"] = 720
            kw["vsync"] = False
            kw["headless"] = True
            return _orig_vgl_init(self, *a, **kw)

        newton.viewer.ViewerGL.__init__ = _fast_vgl_init

    import torch
    from isaaclab_rl.entrypoints.backends import play_rsl_rl
    import g1_maze_lab.tasks  # noqa: F401
    from g1_maze_lab.terrains import GENERATED, HAZARDS
    import challenge

    os.makedirs(RUN_DIR, exist_ok=True)
    challenge.TIME_LIMIT = max_time
    challenge.REC_MP4 = "" if headless else OUT_WALK_MP4
    challenge.REC = bool(challenge.REC_MP4)
    challenge.REC_EVERY = 10
    challenge.CAMERA = "chase"
    challenge.RUN.update(team="pro_pilot", seed=2026, level="10x10", pilot=PILOT_PATH, policy=CHECKPOINT_PATH, score=False)

    traj_50hz = {
        "t": [], "x": [], "y": [], "z": [], "yaw": [],
        "vx_act": [], "vy_act": [], "speed_act": [],
        "cmd_vx": [], "cmd_vy": [], "cmd_wz": [],
        "cell_i": [], "cell_j": [],
    }
    telem_10hz = []
    saved_meta = {}

    def custom_playback(step, *, dt, args_cli, env_cfg):
        env = challenge._find_env(step)
        robot = env.scene["robot"]
        cmd = env.command_manager.get_term("base_velocity")
        cmd.cfg.heading_command = False
        cmd.is_heading_env[:] = False
        cmd.is_standing_env[:] = False
        scanner = env.scene["maze_scanner"]
        lidar = env.scene["wall_lidar"]

        walls = GENERATED[-1]["walls"]
        hazards = HAZARDS[-1]
        n_env, n = env.num_envs, walls.shape[0]
        start, goal = tuple(hazards["start"]), tuple(hazards["goal"])
        challenge.MAZE.update(n=n, start=start, goal=goal)

        saved_meta["walls"] = walls.cpu().numpy() if hasattr(walls, "cpu") else np.array(walls)
        saved_meta["hazards"] = hazards
        saved_meta["start"] = start
        saved_meta["goal"] = goal
        saved_meta["n"] = n

        if "goal_reached" in env.termination_manager.active_terms:
            env.termination_manager.get_term_cfg("goal_reached").func = lambda env, **kw: torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

        to_goal = challenge.cells_to_goal(walls, n, start, goal)
        beds = [tuple(b[3:]) for b in hazards["spike"]]
        spike_top = hazards["spike_top"] + 0.02

        pilot = challenge.make_pilot(challenge.load_pilot(PILOT_PATH), n_env, n, env.device, start, goal)
        viewer = challenge._find_viewer()
        if viewer is not None:
            _markers = challenge._markers(env)  # noqa: F401
            viewer.register_ui_callback(challenge._hud, position="free")

        challenge.HUD.update(team="pro_pilot", seed=2026, level="10x10", limit=max_time)
        recorder = challenge._Recorder() if (viewer is not None and not headless) else None

        stuck_ref = (0.0, 0.0)
        falls, best_left, finish, cell = 0, to_goal[start], None, start
        route_m = to_goal[start] * CELL_SIZE
        start_t = torch.tensor(start, dtype=torch.long, device=env.device)
        goal_t = torch.tensor(goal, dtype=torch.long, device=env.device)
        why = "FELL / CRASH"
        dist_m = best_m = route_m
        last_fall_t = -10.0
        wall_t0 = time.time()
        steps = int(max_time / dt)
        sample_interval = max(1, int(round(0.10 / dt)))

        print(
            f"[CUSTOM 10x10] start={start} goal={goal} shortest_route={to_goal[start]} cells "
            f"(headless={headless})",
            flush=True,
        )

        for k in range(steps):
            o = env.scene.env_origins.clone()
            o[:, :2] -= start_t.float() * CELL_SIZE
            pos_xy = robot.data.root_pos_w.torch[:, :2] - o[:, :2]
            yaw = challenge._yaw(robot.data.root_quat_w.torch)

            hits = torch.nan_to_num(scanner.data.ray_hits_w.torch, nan=0.0, posinf=0.0, neginf=0.0)
            scan_xy = hits[..., :2] - o[:, None, :2]
            scan_z = hits[..., 2] - o[:, None, 2]

            rel = lidar.data.ray_hits_w.torch[..., :2] - lidar.data.pos_w.torch[:, None, :2]
            lidar_dist = torch.nan_to_num(torch.linalg.norm(rel, dim=-1), nan=1e6, posinf=1e6)
            lidar_yaw = torch.atan2(rel[..., 1], rel[..., 0])

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
            if finish is not None:
                command[:] = 0.0
            cmd.time_left[:] = 1.0e9
            cmd.vel_command_b[:] = command

            # Log 10 Hz perception telemetry before step
            if k % sample_interval == 0:
                p_3d = (robot.data.root_pos_w.torch[0, :3] - o[0, :3]).cpu().numpy()
                y_cur = float(yaw[0].item())
                v_w = robot.data.root_lin_vel_w.torch[0, :2].cpu().numpy()
                v_b = robot.data.root_lin_vel_b.torch[0, :3].cpu().numpy()
                w_b = robot.data.root_ang_vel_b.torch[0, :3].cpu().numpy()
                cmd_val = command[0].cpu().numpy()
                ap = getattr(pilot, "ap", pilot)
                known_walls = ap.walls[0].cpu().numpy().copy()
                cell_path = [list(c) for c in ap.current_cell_paths[0]] if ap.current_cell_paths[0] else []
                smooth_wpts = ap.smooth_wpts[0].cpu().numpy().copy() if ap.smooth_wpts[0] is not None else None
                try:
                    tgt_pt = ap.get_pursuit_target(0, pos_xy[0]).cpu().numpy().reshape(-1)[:2]
                except Exception:
                    tgt_pt = np.array([goal[0] * CELL_SIZE, goal[1] * CELL_SIZE], dtype=np.float32)

                cur_c = (
                    min(max(int(round(float(p_3d[0]) / CELL_SIZE)), 0), n - 1),
                    min(max(int(round(float(p_3d[1]) / CELL_SIZE)), 0), n - 1),
                )
                if cur_c in ap.cell_gap_zones[0]:
                    mode_str = "GAP CROSSING (vx=0.85 m/s)"
                elif cur_c in ap.cell_ramp_zones[0]:
                    mode_str = "RAMP CENTERLINE LOCK"
                elif cur_c in ap.cell_spike_zones[0]:
                    spk_kind = ap.cell_spike_zones[0][cur_c].get("kind", "lane").upper()
                    mode_str = f"SPIKE {spk_kind} AVOIDANCE"
                else:
                    mode_str = "G2 QUINTIC SPLINE CRUISE"

                telem_10hz.append({
                    "t": k * dt,
                    "pos": p_3d,
                    "yaw": y_cur,
                    "vel_world": v_w,
                    "vel_body": v_b,
                    "ang_vel_body": w_b,
                    "cmd": cmd_val,
                    "scan_xy": scan_xy[0].cpu().numpy(),
                    "scan_z": scan_z[0].cpu().numpy(),
                    "lidar_dist": lidar_dist[0].cpu().numpy(),
                    "lidar_yaw": lidar_yaw[0].cpu().numpy(),
                    "known_walls": known_walls,
                    "cell_path": cell_path,
                    "smooth_wpts": smooth_wpts,
                    "target_pt": tgt_pt,
                    "ramp_zones": [list(c) for c in ap.cell_ramp_zones[0]],
                    "gap_zones": [list(c) for c in ap.cell_gap_zones[0]],
                    "spike_zones": {f"{c[0]}_{c[1]}": dict(ap.cell_spike_zones[0][c]) for c in ap.cell_spike_zones[0]},
                    "active_mode": mode_str,
                    "cell": cur_c,
                })

            if viewer is not None:
                if not viewer.is_running():
                    break
                challenge._set_camera(viewer, robot, o[0])

            with torch.inference_mode():
                step()

            t = (k + 1) * dt
            prev_cell = cell
            p3 = (robot.data.root_pos_w.torch[0, :3] - o[0, :3]).tolist()
            p = p3[:2]
            vw = robot.data.root_lin_vel_w.torch[0, :2].tolist()
            cur_yaw = float(challenge._yaw(robot.data.root_quat_w.torch)[0].item())
            cell = (min(max(round(p[0] / CELL_SIZE), 0), n - 1), min(max(round(p[1] / CELL_SIZE), 0), n - 1))

            traj_50hz["t"].append(t)
            traj_50hz["x"].append(p3[0])
            traj_50hz["y"].append(p3[1])
            traj_50hz["z"].append(p3[2])
            traj_50hz["yaw"].append(cur_yaw)
            traj_50hz["vx_act"].append(vw[0])
            traj_50hz["vy_act"].append(vw[1])
            traj_50hz["speed_act"].append(math.hypot(vw[0], vw[1]))
            traj_50hz["cmd_vx"].append(float(command[0, 0].item()))
            traj_50hz["cmd_vy"].append(float(command[0, 1].item()))
            traj_50hz["cmd_wz"].append(float(command[0, 2].item()))
            traj_50hz["cell_i"].append(cell[0])
            traj_50hz["cell_j"].append(cell[1])

            left = to_goal.get(cell, best_left)
            best_left = min(best_left, left)
            rem = challenge.route_left_m(p, cell, to_goal, walls)
            if rem is not None and finish is None:
                dist_m = rem
                best_m = min(best_m, dist_m)
            if finish is None and math.dist(p, (goal[0] * CELL_SIZE, goal[1] * CELL_SIZE)) < 0.6:
                finish = t
                dist_m = best_m = 0.0
                print(f"[CUSTOM 10x10] GOAL at {t:.1f} s, falls {falls}", flush=True)

            if k % int(challenge.STUCK_TIME / dt) == 0:
                if finish is None and k > 0 and math.dist(p, stuck_ref) < 0.25:
                    with torch.inference_mode():
                        env._reset_idx(torch.arange(n_env, device=env.device))
                        env.reset_buf[:] = True
                    why = "STUCK"
                    p = [start[0] * CELL_SIZE, start[1] * CELL_SIZE]
                stuck_ref = p

            if finish is None and beds and not bool(env.reset_buf[0]):
                bodies = robot.data.body_pos_w.torch[0].clone()
                bodies[:, :2] -= o[0, :2]
                bodies[:, 2] -= o[0, 2]
                if challenge.spike_hit(bodies, beds, spike_top):
                    with torch.inference_mode():
                        env._reset_idx(torch.arange(n_env, device=env.device))
                        env.reset_buf[:] = True
                    why = "SPIKE"
                    stuck_ref = [start[0] * CELL_SIZE, start[1] * CELL_SIZE]

            if finish is None and bool(env.reset_buf[0]) and k + 1 < steps:
                falls += 1
                last_fall_t = t
                pilot.reset(torch.arange(n_env, device=env.device))
                print(f"[CUSTOM 10x10] t={t:.1f}s fall/crash #{falls} in cell {prev_cell}", flush=True)
                challenge.HUD["why"], why = why, "FELL / CRASH"

            challenge.HUD.update(
                t=finish if finish is not None else t,
                left=0 if finish is not None else left,
                falls=falls,
                goal=None if finish is None else f"GOAL  {finish:.1f} s",
                flash=t - last_fall_t < 1.5,
                dist=dist_m,
                best=best_m,
                route=route_m,
            )
            if recorder is not None and k % challenge.REC_EVERY == 0:
                recorder.add(viewer, k // challenge.REC_EVERY)

            if finish is not None and t >= finish + 2.5:
                break

            if k % int(10.0 / dt) == 0:
                print(
                    f"[CUSTOM 10x10] t={t:5.1f}s cell={cell} left={left} dist={dist_m:.1f}m falls={falls} wall={time.time() - wall_t0:.0f}s",
                    flush=True,
                )

        if recorder is not None:
            recorder.close()

        saved_meta["finish_time"] = finish if finish is not None else max_time
        saved_meta["falls"] = falls
        print(
            f"[CUSTOM 10x10] DONE: finish={finish} s, falls={falls}, wall={time.time() - wall_t0:.0f} s",
            flush=True,
        )

        np.savez(OUT_TRAJ_NPZ, **{k: np.array(v) for k, v in traj_50hz.items()})
        with open(OUT_TELEM_PKL, "wb") as f:
            pickle.dump({"meta": saved_meta, "frames": telem_10hz}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[CUSTOM 10x10] Saved {OUT_TRAJ_NPZ} & {OUT_TELEM_PKL}", flush=True)

    play_rsl_rl.run_playback = custom_playback
    args = [
        "--task", "G1-Maze-Hard",
        "--num_envs", "1",
        "--checkpoint", CHECKPOINT_PATH,
        "env.scene.terrain.terrain_generator.seed=2026",
    ]
    if not headless:
        args += ["--visualizer", "newton", "env.sim.render_interval=40"]
    play_rsl_rl.run(args)


def _draw_reconstructed_map(ax, walls: np.ndarray, hazards: dict, start: tuple, goal: tuple, traj_xy: np.ndarray | None = None, traj_t: np.ndarray | None = None):
    """Draw the 10x10 hazard maze with exact walls, hazard patches, labels, and optional trajectory."""
    from matplotlib.patches import Rectangle
    from matplotlib.collections import LineCollection

    n, c = walls.shape[0], CELL_SIZE
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(-1.15, (n - 1) * c + 1.15)
    ax.set_ylim(-1.15, (n - 1) * c + 1.15)
    ax.set_aspect("equal")

    # Cell grid
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((i * c - 1.0, j * c - 1.0), c, c, facecolor="#ffffff", edgecolor="#e2e8f0", lw=0.6, zorder=0))

    # Ice tiles (cyan)
    for (i, j) in hazards.get("ice", []):
        ax.add_patch(Rectangle((i * c - 0.9, j * c - 0.9), 1.8, 1.8, facecolor="#7dd3fc", alpha=0.55, edgecolor="#0284c7", lw=0.8, zorder=1))
        ax.text(i * c, j * c, "ICE", color="#0369a1", fontsize=5.5, fontweight="bold", ha="center", va="center", zorder=2)

    # Rough cells (brown/amber dots)
    for (i, j) in hazards.get("rough", []):
        ax.add_patch(Rectangle((i * c - 0.9, j * c - 0.9), 1.8, 1.8, facecolor="#fde68a", alpha=0.50, edgecolor="#b45309", lw=0.7, zorder=1))
        ax.text(i * c, j * c, "RGH", color="#92400e", fontsize=5.2, fontweight="bold", ha="center", va="center", zorder=2)

    # Gaps (dark trench)
    for (i, j, a, w) in hazards.get("gap", []):
        rw, rh = (w, 1.8) if a == 0 else (1.8, w)
        ax.add_patch(Rectangle((i * c - 0.5 * rw, j * c - 0.5 * rh), rw, rh, facecolor="#1e293b", alpha=0.90, zorder=2))
        ax.text(i * c, j * c + (0.65 if a == 0 else 0.0), "GAP", color="#be123c", fontsize=5.2, fontweight="bold", ha="center", va="center", zorder=3)

    # Ramps (orange with height)
    for (i, j, a, rh) in hazards.get("ramp", []):
        col = "#ea580c" if rh >= 0.215 else "#fdba74"
        tcol = "#ffffff" if rh >= 0.215 else "#7c2d12"
        ax.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor=col, alpha=0.72, edgecolor="#9a3412", lw=1.0, zorder=2))
        ax.text(i * c, j * c, f"R {rh:.2f}", color=tcol, fontsize=5.3, fontweight="bold", ha="center", va="center", zorder=3)

    # Spikes & Slaloms (red hatched)
    slaloms = set(map(tuple, hazards.get("slalom", [])))
    for (i, j, a, x0, y0, x1, y1) in hazards.get("spike", []):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#ef4444", hatch="///", edgecolor="#991b1b", alpha=0.78, lw=0.8, zorder=3))
    for (i, j) in slaloms:
        ax.text(i * c, j * c, "SLM", color="#7f1d1d", fontsize=5.0, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#dc2626", alpha=0.85, lw=0.6), zorder=4)

    # Maze walls
    for i in range(n):
        for j in range(n):
            cx, cy = i * c, j * c
            if walls[i, j, 0]:
                ax.plot([cx + 1.0, cx + 1.0], [cy - 1.0, cy + 1.0], color="#0f172a", lw=2.4, solid_capstyle="round", zorder=5)
            if walls[i, j, 1]:
                ax.plot([cx - 1.0, cx + 1.0], [cy + 1.0, cy + 1.0], color="#0f172a", lw=2.4, solid_capstyle="round", zorder=5)
            if i == 0 and walls[i, j, 2]:
                ax.plot([cx - 1.0, cx - 1.0], [cy - 1.0, cy + 1.0], color="#0f172a", lw=2.4, solid_capstyle="round", zorder=5)
            if j == 0 and walls[i, j, 3]:
                ax.plot([cx - 1.0, cx + 1.0], [cy - 1.0, cy - 1.0], color="#0f172a", lw=2.4, solid_capstyle="round", zorder=5)

    # Start & Goal markers
    ax.add_patch(Rectangle((start[0] * c - 0.65, start[1] * c - 0.65), 1.3, 1.3, facecolor="#3b82f6", alpha=0.28, edgecolor="#2563eb", lw=1.5, zorder=2))
    ax.text(start[0] * c, start[1] * c, "S", color="#1d4ed8", fontsize=11, fontweight="bold", ha="center", va="center", zorder=6)
    ax.add_patch(Rectangle((goal[0] * c - 0.65, goal[1] * c - 0.65), 1.3, 1.3, facecolor="#22c55e", alpha=0.32, edgecolor="#16a34a", lw=1.5, zorder=2))
    ax.text(goal[0] * c, goal[1] * c, "G", color="#15803d", fontsize=11, fontweight="bold", ha="center", va="center", zorder=6)

    # Trajectory overlay colored by elapsed time
    if traj_xy is not None and len(traj_xy) > 2:
        pts = traj_xy.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t_vals = traj_t[:-1] if traj_t is not None else np.linspace(0, 1, len(segs))
        lc = LineCollection(segs, cmap="plasma", norm=None, linewidths=2.2, alpha=0.92, zorder=7)
        lc.set_array(t_vals)
        ax.add_collection(lc)
        ax.scatter([traj_xy[-1, 0]], [traj_xy[-1, 1]], s=65, color="#16a34a", edgecolors="white", lw=1.4, zorder=8)

    ax.set_xticks(np.arange(0, n * c, c))
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=7.5)
    ax.set_yticks(np.arange(0, n * c, c))
    ax.set_yticklabels([str(j) for j in range(n)], fontsize=7.5)
    ax.set_xlabel("Cell i (x = 2i m)", fontsize=8.5)
    ax.set_ylabel("Cell j (y = 2j m)", fontsize=8.5)


def render_verification_png():
    """Render side-by-side PNG: Original Reference Image vs Reconstructed 10x10 + Trajectory vs 50 Hz Telemetry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    with open(OUT_TELEM_PKL, "rb") as f:
        bundle = pickle.load(f)
    meta = bundle["meta"]
    traj = np.load(OUT_TRAJ_NPZ)

    t = traj["t"]
    xy = np.stack([traj["x"], traj["y"]], axis=1)
    v_act = traj["speed_act"]
    cmd_vx = traj["cmd_vx"]
    cmd_wz = traj["cmd_wz"]
    yaw_deg = np.degrees(traj["yaw"])

    t_fin = meta.get("finish_time", float(t[-1]))
    falls = meta.get("falls", 0)

    fig = plt.figure(figsize=(19.5, 6.8), dpi=160, facecolor="#ffffff")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.12, 1.18], wspace=0.22, hspace=0.30, left=0.03, right=0.98, top=0.87, bottom=0.10)

    ax_orig = fig.add_subplot(gs[:, 0])
    if os.path.exists(REF_IMAGE):
        ax_orig.imshow(np.asarray(Image.open(REF_IMAGE).convert("RGB")))
    else:
        ax_orig.text(0.5, 0.5, "Reference Image Missing", ha="center", va="center")
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])
    ax_orig.set_title("1. Original 10x10 Target Map", fontsize=11, fontweight="bold", pad=8)

    ax_rec = fig.add_subplot(gs[:, 1])
    _draw_reconstructed_map(ax_rec, meta["walls"], meta["hazards"], meta["start"], meta["goal"], traj_xy=xy, traj_t=t)
    ax_rec.set_title(f"2. Reconstructed Map & G1 Trajectory ({t_fin:.1f} s, {falls} Falls)", fontsize=11, fontweight="bold", pad=8)

    ax_v = fig.add_subplot(gs[0, 2])
    ax_v.plot(t, cmd_vx, color="#16a34a", lw=1.2, alpha=0.85, label="cmd_vx [m/s]")
    ax_v.plot(t, v_act, color="#0284c7", lw=1.0, alpha=0.85, label="actual speed [m/s]")
    ax_v.set_xlim(0, t[-1])
    ax_v.set_ylim(-0.05, 1.05)
    ax_v.set_ylabel("Linear Speed [m/s]", fontsize=9)
    ax_v.set_title("3. 50 Hz Velocity & Heading Telemetry", fontsize=11, fontweight="bold", pad=8)
    ax_v.grid(True, ls="--", alpha=0.35)
    ax_v.legend(loc="upper right", fontsize=8, frameon=True)

    ax_h = fig.add_subplot(gs[1, 2], sharex=ax_v)
    ax_h.plot(t, yaw_deg, color="#7c3aed", lw=0.9, label="yaw [deg]")
    ax_h_r = ax_h.twinx()
    ax_h_r.plot(t, cmd_wz, color="#ea580c", lw=0.8, alpha=0.65, label="cmd_wz [rad/s]")
    ax_h.set_xlim(0, t[-1])
    ax_h.set_ylim(-195, 195)
    ax_h_r.set_ylim(-2.0, 2.0)
    ax_h.set_xlabel("Simulation Time [s]", fontsize=9)
    ax_h.set_ylabel("Heading Yaw [deg]", fontsize=9, color="#7c3aed")
    ax_h_r.set_ylabel("cmd_wz [rad/s]", fontsize=9, color="#ea580c")
    ax_h.grid(True, ls="--", alpha=0.35)

    # Mark key slalom/ramp crossings
    ci, cj = traj["cell_i"], traj["cell_j"]
    for idx_cell, label, col in [((0, 3), "(0,3) Slalom", "#dc2626"), ((1, 7), "(1,7) Spike", "#b91c1c"), ((4, 8), "(4,8) Spike", "#b91c1c")]:
        hits_c = np.where((ci == idx_cell[0]) & (cj == idx_cell[1]))[0]
        if len(hits_c) > 0:
            t0, t1 = float(t[hits_c[0]]), float(t[hits_c[-1]])
            ax_v.axvspan(t0, t1, color=col, alpha=0.14)
            ax_h.axvspan(t0, t1, color=col, alpha=0.14)
            ax_v.text(0.5 * (t0 + t1), 0.96, label, color=col, fontsize=7, fontweight="bold", ha="center", va="top")

    fig.suptitle(
        f"Custom 10x10 Hazard Maze Verification — G1 Reached Goal (9,9) in {t_fin:.1f} s with {falls} Falls",
        fontsize=13, fontweight="bold", y=0.96
    )
    os.makedirs(EXP_DIR, exist_ok=True)
    fig.savefig(OUT_VERIFY_PNG, dpi=160)
    plt.close(fig)
    print(f"[VERIFY] Saved {OUT_VERIFY_PNG}", flush=True)


def render_perception_dashboard_mp4():
    """Render 4-panel Perception + Planning + cmd_vel Vector Dashboard MP4 @ 20 fps."""
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Circle, Rectangle

    with open(OUT_TELEM_PKL, "rb") as f:
        bundle = pickle.load(f)
    frames = bundle["frames"]
    meta = bundle["meta"]
    n_frames = len(frames)
    t_fin = meta.get("finish_time", frames[-1]["t"])
    falls = meta.get("falls", 0)
    c = CELL_SIZE
    n = meta["n"]
    hazards = meta["hazards"]

    print(f"[DASHBOARD] Rendering {n_frames} frames @ 20 fps -> {OUT_PERCEPT_MP4}...", flush=True)

    # Pre-load synchronized 3D camera frames
    cached_cam = [None] * n_frames
    if os.path.exists(OUT_WALK_MP4):
        try:
            reader = imageio.get_reader(OUT_WALK_MP4)
            vid_len = reader.count_frames()
            for idx, f in enumerate(frames):
                vid_idx = int(round((f["t"] / max(0.1, t_fin)) * (vid_len - 1)))
                vid_idx = max(0, min(vid_len - 1, vid_idx))
                raw_cf = reader.get_data(vid_idx)
                cached_cam[idx] = raw_cf[::2, ::2]
            reader.close()
        except Exception as e:
            print(f"[WARN] Could not decode {OUT_WALK_MP4}: {e}", flush=True)
    for idx in range(n_frames):
        if cached_cam[idx] is None:
            cached_cam[idx] = np.full((360, 640, 3), 15, dtype=np.uint8)

    # Pre-render static Panel B background once (eliminates 130+ Rectangle patches per frame!)
    x0_m, x1_m = -1.2, (n - 1) * c + 1.2
    fig_bg = plt.figure(figsize=(5.0, 5.0), dpi=120, facecolor="#111827")
    ax_bg = fig_bg.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_bg.set_facecolor("#111827")
    ax_bg.set_xlim(x0_m, x1_m)
    ax_bg.set_ylim(x0_m, x1_m)
    ax_bg.set_aspect("equal")
    for i in range(n):
        for j in range(n):
            ax_bg.add_patch(Rectangle((i * c - 1.0, j * c - 1.0), c, c, facecolor="#1e293b", edgecolor="#334155", lw=0.45))
    for (i, j) in hazards.get("ice", []):
        ax_bg.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor="#1d4ed8", alpha=0.35))
    for (i, j) in hazards.get("rough", []):
        ax_bg.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor="#854d0e", alpha=0.35))
    for (i, j, a, w) in hazards.get("gap", []):
        rw, rh = (w, 1.7) if a == 0 else (1.7, w)
        ax_bg.add_patch(Rectangle((i * c - 0.5 * rw, j * c - 0.5 * rh), rw, rh, facecolor="#dc2626", alpha=0.45))
    for (i, j, a, rh) in hazards.get("ramp", []):
        ax_bg.add_patch(Rectangle((i * c - 0.85, j * c - 0.85), 1.7, 1.7, facecolor="#b45309", alpha=0.45))
    for (i, j, a, x0, y0, x1, y1) in hazards.get("spike", []):
        ax_bg.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#dc2626", hatch="//", alpha=0.55))
    ax_bg.axis("off")
    fig_bg.canvas.draw()
    bg_map_rgb = np.asarray(fig_bg.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig_bg)

    ts_t = np.array([f["t"] for f in frames])
    ts_cmd_vx = np.array([f["cmd"][0] for f in frames])
    ts_act_v = np.array([float(np.hypot(f["vel_world"][0], f["vel_world"][1])) for f in frames])
    ts_cmd_wz = np.array([f["cmd"][2] for f in frames])
    all_xy = np.array([f["pos"][:2] for f in frames])

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#0f172a")
    ax_ego = fig.add_axes([0.035, 0.40, 0.28, 0.54])
    ax_map = fig.add_axes([0.345, 0.40, 0.28, 0.54])
    ax_cam = fig.add_axes([0.655, 0.40, 0.31, 0.54])
    ax_gauge = fig.add_axes([0.035, 0.07, 0.23, 0.27])
    ax_ts = fig.add_axes([0.305, 0.07, 0.66, 0.27])

    # Static Panel D2
    ax_ts.set_facecolor("#111827")
    for idx_cell, label, col in [((0, 3), "(0,3) Slalom", "#ef4444"), ((1, 7), "(1,7) Spike", "#f87171"), ((4, 8), "(4,8) Spike", "#f87171")]:
        hits_i = [f["t"] for f in frames if tuple(f["cell"]) == idx_cell]
        if hits_i:
            t0, t1 = min(hits_i), max(hits_i)
            ax_ts.axvspan(t0, t1, color=col, alpha=0.22, zorder=1)
            ax_ts.text(0.5 * (t0 + t1), 1.02, label, color=col, fontsize=7.0, fontweight="bold", ha="center", va="bottom")

    ax_ts.plot(ts_t, ts_cmd_vx, color="#22c55e", lw=1.4, label="cmd_vx (m/s)", zorder=3)
    ax_ts.plot(ts_t, ts_act_v, color="#eab308", lw=1.2, alpha=0.85, label="actual_speed (m/s)", zorder=3)
    ax_ts.plot(ts_t, ts_cmd_wz, color="#06b6d4", lw=1.0, alpha=0.75, label="cmd_wz (rad/s)", zorder=3)
    ax_ts.axhline(0.60, color="#94a3b8", ls="--", lw=0.8, alpha=0.6, label="Nominal (0.60)")
    playhead = ax_ts.axvline(0.0, color="#f8fafc", lw=2.2, zorder=5, path_effects=[pe.Stroke(linewidth=3.5, foreground="#0284c7"), pe.Normal()])
    ax_ts.set_xlim(0, max(float(ts_t[-1]), t_fin + 1.0))
    ax_ts.set_ylim(-1.3, 1.25)
    ax_ts.set_xlabel("Simulation Elapsed Time (seconds)", color="#94a3b8", fontsize=8.5)
    ax_ts.set_ylabel("Command & Velocity Values", color="#94a3b8", fontsize=8.5)
    ax_ts.set_title("PANEL D2: Complete 10x10 Trajectory Telemetry & Controller Response", color="white", fontsize=9.0, fontweight="bold", pad=4)
    ax_ts.tick_params(colors="#64748b", labelsize=7.5)
    ax_ts.grid(True, color="#1e293b", ls="--", lw=0.6)
    ax_ts.legend(loc="upper left", fontsize=7.2, facecolor="#0f172a", edgecolor="#334155", labelcolor="white", ncol=4)

    # Static Panel C setup
    ax_cam.set_facecolor("#0b0f19")
    cam_artist = ax_cam.imshow(cached_cam[0])
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    ax_cam.set_title("PANEL C: Synchronized 3D Physical Simulation (Chase Camera)", color="white", fontsize=9.5, fontweight="bold", pad=5)
    banner_box = dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#10b981", lw=1.8, alpha=0.92)
    banner_text = ax_cam.text(
        0.5, 0.91, "MODE: INIT", transform=ax_cam.transAxes,
        color="#10b981", fontsize=8.5, fontweight="bold", ha="center", va="center", bbox=banner_box
    )

    # Static Panel D1 Gauges setup (persistent bars & labels)
    ax_gauge.set_facecolor("#111827")
    gauge_labels = ["cmd_vx (m/s)", "cmd_vy (m/s)", "cmd_wz (rad/s)", "speed_act (m/s)"]
    gauge_limits = [(0.0, 1.0), (-0.4, 0.4), (-1.2, 1.2), (0.0, 1.0)]
    gauge_colors = ["#22c55e", "#a855f7", "#06b6d4", "#eab308"]
    ax_gauge.set_yticks(np.arange(4))
    ax_gauge.set_yticklabels(gauge_labels, color="#e2e8f0", fontsize=8, fontweight="bold")
    ax_gauge.set_xlim(-1.25, 1.25)
    ax_gauge.set_ylim(-0.6, 3.6)
    ax_gauge.axvline(0.0, color="#475569", lw=0.8)
    ax_gauge.set_title("PANEL D1: Live Policy Command Gauges", color="white", fontsize=9.0, fontweight="bold", pad=4)
    ax_gauge.tick_params(colors="#64748b", labelsize=7)
    ax_gauge.grid(axis="x", color="#1e293b", ls="--", lw=0.6)
    gauge_bars = []
    gauge_txts = []
    for g_i, col in enumerate(gauge_colors):
        rect = Rectangle((0.0, g_i - 0.275), 0.0, 0.55, facecolor=col, alpha=0.85, edgecolor="white", lw=0.5)
        ax_gauge.add_patch(rect)
        gauge_bars.append(rect)
        txt = ax_gauge.text(0.08, g_i, "+0.00", color="white", fontsize=8, fontweight="bold", va="center", ha="left")
        gauge_txts.append(txt)

    # Static Panel B Map setup (pre-rendered background + persistent artists)
    st = meta["start"]
    gl = meta["goal"]
    ax_map.set_facecolor("#111827")
    ax_map.set_xlim(x0_m, x1_m)
    ax_map.set_ylim(x0_m, x1_m)
    ax_map.set_aspect("equal")
    ax_map.imshow(bg_map_rgb, extent=[x0_m, x1_m, x0_m, x1_m], origin="upper", zorder=0)
    ax_map.scatter([st[0] * c], [st[1] * c], s=75, color="#3b82f6", marker="o", edgecolors="white", lw=1.2, zorder=8)
    ax_map.scatter([gl[0] * c], [gl[1] * c], s=100, color="#10b981", marker="*", edgecolors="white", lw=1.2, zorder=8)
    ax_map.set_title("PANEL B: Global 10x10 Online Discovery & A* Planning", color="white", fontsize=9.5, fontweight="bold", pad=5)
    ax_map.tick_params(colors="#64748b", labelsize=7)

    (walls_line,) = ax_map.plot([], [], color="#f8fafc", lw=2.0, zorder=5)
    (cell_path_line,) = ax_map.plot([], [], color="#60a5fa", marker="D", markersize=3.0, lw=1.3, alpha=0.8, zorder=6)
    (smooth_line,) = ax_map.plot([], [], color="#10b981", lw=2.0, zorder=7)
    (traj_line,) = ax_map.plot([], [], color="#38bdf8", lw=1.8, zorder=8)
    target_scat = ax_map.scatter([], [], s=70, marker="*", color="#fbbf24", edgecolors="white", lw=0.8, zorder=9)
    robot_scat = ax_map.scatter([], [], s=60, color="#38bdf8", edgecolors="white", lw=1.2, zorder=10)
    zone_patches = []

    writer = imageio.get_writer(
        OUT_PERCEPT_MP4,
        fps=20,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-preset", "veryfast", "-crf", "22"],
    )

    t0_wall = time.time()

    for idx, frame in enumerate(frames):
        t = frame["t"]
        pos = frame["pos"]
        yaw = frame["yaw"]
        rx, ry = float(pos[0]), float(pos[1])
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

        # Panel C
        cam_artist.set_data(cached_cam[idx])
        mode_color = (
            "#e879f9" if "GAP" in active_mode else
            "#f59e0b" if "RAMP" in active_mode else
            "#ef4444" if "SPIKE" in active_mode else
            "#10b981"
        )
        banner_text.set_text(f"MODE: {active_mode}\nCELL: {cur_c}  |  TIME: {t:4.1f} s / {t_fin:.1f} s  |  {falls} FALLS")
        banner_text.set_color(mode_color)
        banner_text.get_bbox_patch().set_edgecolor(mode_color)

        # Panel D1 Gauges (fast in-place update)
        v_mag = float(np.hypot(vw[0], vw[1]))
        gauge_vals = [cmd[0], cmd[1], cmd[2], v_mag]
        for rect, txt, val, (vmin, vmax) in zip(gauge_bars, gauge_txts, gauge_vals, gauge_limits):
            vc = max(vmin, min(vmax, val))
            rect.set_x(min(0.0, vc))
            rect.set_width(abs(vc))
            txt.set_x(vc + (0.08 if vc >= 0 else -0.08))
            txt.set_text(f"{val:+.2f}")
            txt.set_ha("left" if vc >= 0 else "right")

        # Panel D2 Playhead
        playhead.set_xdata([t, t])

        # Panel A: Ego Sensor & Vector View
        ax_ego.clear()
        ax_ego.set_facecolor("#111827")
        w_size = 2.4
        ax_ego.set_xlim(rx - w_size, rx + w_size)
        ax_ego.set_ylim(ry - w_size, ry + w_size)
        ax_ego.set_aspect("equal")

        valid_l = lidar_dist < 5.8
        if np.any(valid_l):
            hx = rx + lidar_dist[valid_l] * np.cos(lidar_yaw[valid_l])
            hy = ry + lidar_dist[valid_l] * np.sin(lidar_yaw[valid_l])
            l_xs, l_ys = [], []
            for lx_i, ly_i in zip(hx, hy):
                l_xs.extend([rx, lx_i, np.nan])
                l_ys.extend([ry, ly_i, np.nan])
            ax_ego.plot(l_xs, l_ys, color="#06b6d4", lw=0.7, alpha=0.30, zorder=3)
            ax_ego.scatter(hx, hy, s=6, color="#22d3ee", alpha=0.8, zorder=4)

        mask_local = (
            (scan_xy[:, 0] >= rx - w_size) & (scan_xy[:, 0] <= rx + w_size) &
            (scan_xy[:, 1] >= ry - w_size) & (scan_xy[:, 1] <= ry + w_size)
        )
        s_xy_l = scan_xy[mask_local]
        s_z_l = scan_z[mask_local]
        if len(s_xy_l) > 0:
            for m_mask, col, sz, mk, zo in [
                (s_z_l < -0.12, "#e879f9", 16, "o", 2),
                ((s_z_l >= -0.12) & (s_z_l <= 0.018), "#334155", 5, "o", 1),
                ((s_z_l > 0.018) & (s_z_l <= 0.12), "#fbbf24", 14, "o", 2),
                ((s_z_l > 0.12) & (s_z_l <= 0.42), "#f97316", 18, "o", 3),
                (s_z_l > 0.45, "#e2e8f0", 20, "s", 3),
            ]:
                if np.any(m_mask):
                    ax_ego.scatter(s_xy_l[m_mask, 0], s_xy_l[m_mask, 1], s=sz, marker=mk, color=col, alpha=0.85, zorder=zo)

        if smooth_wpts is not None and len(smooth_wpts) > 0:
            mw = np.hypot(smooth_wpts[:, 0] - rx, smooth_wpts[:, 1] - ry) < 3.2
            if np.any(mw):
                ax_ego.plot(smooth_wpts[mw, 0], smooth_wpts[mw, 1], color="#10b981", lw=2.6, zorder=6)

        if target_pt is not None:
            ax_ego.scatter([target_pt[0]], [target_pt[1]], s=120, marker="*", color="#facc15", edgecolors="white", lw=1.0, zorder=8)
            ax_ego.plot([rx, target_pt[0]], [ry, target_pt[1]], color="#facc15", ls="--", lw=1.2, alpha=0.8, zorder=7)

        ax_ego.add_patch(Circle((rx, ry), 0.22, facecolor="#0ea5e9", edgecolor="#38bdf8", lw=2.0, alpha=0.85, zorder=9))
        ax_ego.plot([rx, rx + 0.32 * math.cos(yaw)], [ry, ry + 0.32 * math.sin(yaw)], color="#38bdf8", lw=3.0, zorder=10)

        v_scale = 0.8
        cv_x = math.cos(yaw) * cmd[0] - math.sin(yaw) * cmd[1]
        cv_y = math.sin(yaw) * cmd[0] + math.cos(yaw) * cmd[1]
        ax_ego.annotate("", xy=(rx + cv_x * v_scale, ry + cv_y * v_scale), xytext=(rx, ry),
                        arrowprops=dict(arrowstyle="->", color="#22c55e", lw=3.0, mutation_scale=15), zorder=11)
        ax_ego.annotate("", xy=(rx + vw[0] * v_scale, ry + vw[1] * v_scale), xytext=(rx, ry),
                        arrowprops=dict(arrowstyle="->", color="#eab308", lw=2.2, mutation_scale=13), zorder=11)

        ax_ego.set_title("PANEL A: Ego-Centric Sensor & cmd_vel Vector (4.8m x 4.8m)", color="white", fontsize=9.5, fontweight="bold", pad=5)
        ax_ego.tick_params(colors="#64748b", labelsize=7)
        ax_ego.grid(True, color="#1e293b", ls="--", lw=0.6)

        # Panel B: Global 10x10 Discovery & Planning (in-place update)
        for zp in zone_patches:
            zp.remove()
        zone_patches.clear()
        for rz in frame["ramp_zones"]:
            p_r = Rectangle((rz[0] * c - 0.9, rz[1] * c - 0.9), 1.8, 1.8, facecolor="none", edgecolor="#f59e0b", lw=1.8, zorder=4)
            ax_map.add_patch(p_r)
            zone_patches.append(p_r)
        for gz in frame["gap_zones"]:
            p_g = Rectangle((gz[0] * c - 0.9, gz[1] * c - 0.9), 1.8, 1.8, facecolor="none", edgecolor="#ec4899", lw=1.8, zorder=4)
            ax_map.add_patch(p_g)
            zone_patches.append(p_g)
        for sk in frame["spike_zones"]:
            si, sj = int(sk.split("_")[0]), int(sk.split("_")[1])
            p_s = Rectangle((si * c - 0.9, sj * c - 0.9), 1.8, 1.8, facecolor="none", edgecolor="#ef4444", lw=1.8, ls="--", zorder=4)
            ax_map.add_patch(p_s)
            zone_patches.append(p_s)

        w_xs, w_ys = [], []
        for i in range(n):
            for j in range(n):
                cx, cy = i * c, j * c
                wm = known_walls[i, j]
                if wm[0] == 1:
                    w_xs.extend([cx + 1.0, cx + 1.0, np.nan]); w_ys.extend([cy - 1.0, cy + 1.0, np.nan])
                if wm[1] == 1:
                    w_xs.extend([cx - 1.0, cx + 1.0, np.nan]); w_ys.extend([cy + 1.0, cy + 1.0, np.nan])
                if i == 0 and wm[2] == 1:
                    w_xs.extend([cx - 1.0, cx - 1.0, np.nan]); w_ys.extend([cy - 1.0, cy + 1.0, np.nan])
                if j == 0 and wm[3] == 1:
                    w_xs.extend([cx - 1.0, cx + 1.0, np.nan]); w_ys.extend([cy - 1.0, cy - 1.0, np.nan])
        walls_line.set_data(w_xs, w_ys)

        if cell_path:
            cell_path_line.set_data([cp[0] * c for cp in cell_path], [cp[1] * c for cp in cell_path])
        else:
            cell_path_line.set_data([], [])
        if smooth_wpts is not None and len(smooth_wpts) > 0:
            smooth_line.set_data(smooth_wpts[:, 0], smooth_wpts[:, 1])
        else:
            smooth_line.set_data([], [])
        traj_line.set_data(all_xy[:idx + 1, 0], all_xy[:idx + 1, 1])
        if target_pt is not None:
            target_scat.set_offsets([[target_pt[0], target_pt[1]]])
        robot_scat.set_offsets([[rx, ry]])

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(buf)

        if idx % 200 == 0 or idx == n_frames - 1:
            print(f"[DASHBOARD] frame {idx + 1}/{n_frames} ({time.time() - t0_wall:.1f} s elapsed)", flush=True)

    writer.close()
    plt.close(fig)
    print(f"[DASHBOARD] Saved {OUT_PERCEPT_MP4}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Run simulation with 3D video + telemetry recording")
    parser.add_argument("--headless", action="store_true", help="Record telemetry headlessly (skip Newton 3D MP4)")
    parser.add_argument("--render", action="store_true", help="Render verification PNG and 4-panel dashboard MP4")
    parser.add_argument("--max-time", type=float, default=480.0)
    args = parser.parse_args()

    if not args.record and not args.render:
        args.record = True
        args.render = True

    if args.record:
        record_run(max_time=args.max_time, headless=args.headless)
    if args.render:
        render_verification_png()
        render_perception_dashboard_mp4()


if __name__ == "__main__":
    main()
