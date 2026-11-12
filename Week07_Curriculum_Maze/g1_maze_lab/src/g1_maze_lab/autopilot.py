# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Maze autopilot: height scan -> walls -> next cell -> velocity command (for every env at once).

Three layers, each one a function you can change on its own:

1. ``walls_from_height_scan``  (perception)  which sides of the current cell are blocked?
2. ``choose_direction``        (planning)    which neighbour cell do we go to next?
3. ``velocity_command``        (control)     v_x, v_y, w_z that drive the robot to that cell

The pretrained G1 walking policy then turns the velocity command into joint targets, exactly like the
keyboard in Week 3. Directions: 0 = +x (east), 1 = +y (north), 2 = -x (west), 3 = -y (south).
"""

from __future__ import annotations

import math
import os

import torch

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def generate_subcell_waypoints(prev_cell, curr_cell, next_cell, sz, c=2.0, mode="slalom_s_curve"):
    """Generate sub-cell waypoints for cell curr_cell given spike zone info sz and travel direction."""
    ci, cj = curr_cell
    cx, cy = ci * c, cj * c

    if prev_cell is not None:
        di, dj = curr_cell[0] - prev_cell[0], curr_cell[1] - prev_cell[1]
    elif next_cell is not None:
        di, dj = next_cell[0] - curr_cell[0], next_cell[1] - curr_cell[1]
    else:
        di, dj = 1, 0

    kind = sz.get("kind", "lane")
    axis = sz.get("axis", 0 if dj == 0 else 1)

    if mode == "centroid":
        v_offset = sz.get("v_centroid", 0.0)
        if axis == 0:
            return [(cx - 0.55 * di, cy + v_offset), (cx, cy + v_offset), (cx + 0.55 * di, cy + v_offset)]
        else:
            return [(cx + v_offset, cy - 0.55 * dj), (cx + v_offset, cy), (cx + v_offset, cy + 0.55 * dj)]

    if mode == "old_slalom":
        if kind == "lane":
            v_safe = sz.get("v_safe", 0.46)
            if axis == 0:
                return [(cx - 0.55 * di, cy + v_safe), (cx, cy + v_safe), (cx + 0.55 * di, cy + v_safe)]
            else:
                return [(cx + v_safe, cy - 0.55 * dj), (cx + v_safe, cy), (cx + v_safe, cy + 0.55 * dj)]
        elif kind == "slalom":
            v_minus = sz.get("v_minus", 0.46)
            v_plus = sz.get("v_plus", -0.46)
            if axis == 0:
                if di >= 0:
                    p_enter = (cx - 0.55, cy + v_minus)
                    p_mid = (cx, cy)
                    p_exit = (cx + 0.55, cy + v_plus)
                else:
                    p_enter = (cx + 0.55, cy + v_plus)
                    p_mid = (cx, cy)
                    p_exit = (cx - 0.55, cy + v_minus)
            else:
                if dj >= 0:
                    p_enter = (cx + v_minus, cy - 0.55)
                    p_mid = (cx, cy)
                    p_exit = (cx + v_plus, cy + 0.55)
                else:
                    p_enter = (cx + v_plus, cy + 0.55)
                    p_mid = (cx, cy)
                    p_exit = (cx + v_minus, cy - 0.55)
            return [p_enter, p_mid, p_exit]

    # Method 1: Extended Lead-In / Lead-Out waypoints at v_safe = ±0.44 m
    s_dir = 1 if (di > 0 or dj > 0) else -1
    if kind == "lane":
        v_raw = sz.get("v_safe", 0.44)
        v_safe = 0.44 if v_raw >= 0 else -0.44
        u_offsets = [-0.95, -0.60, 0.0, 0.60, 0.95]
        if axis == 0:
            return [(cx + u * s_dir, cy + v_safe) for u in u_offsets]
        else:
            return [(cx + v_safe, cy + u * s_dir) for u in u_offsets]

    elif kind == "slalom":
        v_lat = 0.50
        v_m = sz.get("v_minus", v_lat)
        v_p = sz.get("v_plus", -v_lat)
        v_minus = v_lat if v_m >= 0 else -v_lat
        v_plus = v_lat if v_p >= 0 else -v_lat
        if v_minus == v_plus:
            v_plus = -v_minus

        v_entry = v_minus if s_dir > 0 else v_plus
        v_exit = v_plus if s_dir > 0 else v_minus

        pts_u_v = [
            (-0.95 * s_dir, v_entry),
            (-0.60 * s_dir, v_entry),
            (-0.30 * s_dir, v_entry),
            (0.0, 0.0),
            (0.15 * s_dir, v_exit),
            (0.60 * s_dir, v_exit),
            (0.95 * s_dir, v_exit),
        ]
        if axis == 0:
            return [(cx + u, cy + v) for u, v in pts_u_v]
        else:
            return [(cx + v, cy + u) for u, v in pts_u_v]

    return [(cx, cy)]


class MazeAutopilot:
    """Cell-by-cell maze solver for ``num_envs`` robots.

    Args:
        num_envs: number of robots.
        num_cells: maze cells per side.
        cell_size: cell width [m].
        strategy: ``"left"`` / ``"right"`` (wall following) or ``"flood"`` (flood fill on the walls seen so far).
        speed: forward speed command [m/s].
        turn_early: pick the next cell once the robot is this close to the centre of the cell it is walking to [m].
            0.35 = walk to the centre, then turn there. 0.5 = start the turn half a metre early and walk it as an arc.
        min_speed: forward speed during a sharp turn [m/s] (0.15 = nearly on the spot).
        max_turn: largest yaw rate command [rad/s].
        start, goal: start / goal cell (default ``(0, 0)`` / ``(n-1, n-1)``). Positions are always relative to the
            centre of cell ``(0, 0)``.
    """

    def __init__(
        self,
        num_envs: int,
        num_cells: int,
        cell_size: float,
        device: str,
        strategy: str = "left",
        speed: float = 0.6,
        turn_early: float = 0.35,
        min_speed: float = 0.15,
        max_turn: float = 0.8,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
    ):
        self.n, self.c, self.device = num_cells, cell_size, device
        self.start = tuple(int(v) for v in start)
        self.goal = (num_cells - 1, num_cells - 1) if goal is None else tuple(int(v) for v in goal)
        self.strategy, self.speed = strategy, speed
        self.turn_early = float(os.environ.get("MAZE_TURN_EARLY", turn_early))
        self.min_speed = float(os.environ.get("MAZE_MIN_SPEED", min_speed))
        self.num_envs = num_envs
        self._dirs = torch.tensor(DIRS, device=device)
        # wall map: -1 unknown, 0 open, 1 wall (outer border is known from the start)
        self.walls = torch.full((num_envs, num_cells, num_cells, 4), -1, dtype=torch.int8, device=device)
        self.walls[:, -1, :, 0] = 1
        self.walls[:, :, -1, 1] = 1
        self.walls[:, 0, :, 2] = 1
        self.walls[:, :, 0, 3] = 1
        start_t = torch.tensor(self.start, dtype=torch.long, device=device)
        self.target = start_t.repeat(num_envs, 1)  # cell we are walking to
        self.origin = start_t.repeat(num_envs, 1)  # cell the last decision was made in
        self.heading = torch.zeros(num_envs, dtype=torch.long, device=device)  # last direction travelled
        self.next_heading = torch.full((num_envs,), -1, dtype=torch.long, device=device)  # lookahead direction
        self.need_decision = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.cells_walked = torch.zeros(num_envs, dtype=torch.long, device=device)
        # command shaping (see velocity_command)
        self.cmd = torch.zeros(num_envs, 3, device=device)
        self.max_turn, self.max_acc, self.max_ang_acc = max_turn, 1.6, 3.6
        self.direct_corner = os.environ.get("MAZE_DIRECT_CORNER", "1") == "1"
        self.lateral_margin = os.environ.get("MAZE_LATERAL_MARGIN", "1") == "1"
        self.use_astar = os.environ.get("MAZE_USE_ASTAR", "0") == "1"
        self.curve_mode = os.environ.get("MAZE_CURVE_MODE", "bezier")
        self.corner_cut = float(os.environ.get("MAZE_CORNER_CUT", 0.35))  # lateral cut offset for direct corner paths [m]
        self.wall_margin = float(os.environ.get("MAZE_WALL_MARGIN", 0.45))  # safe clearance to walls [m]
        default_r = 0.65 if self.curve_mode == "g2_quintic" else 0.60
        self.r_fillet = float(os.environ.get("MAZE_R_FILLET", default_r))  # corner arc fillet radius [m] (Fig. 8)
        default_lookahead = 0.50 if self.curve_mode == "g2_quintic" else 0.65
        self.lookahead_dist = float(os.environ.get("MAZE_LOOKAHEAD", default_lookahead))  # pure-pursuit lookahead [m]
        self.smooth_wpts = [None] * num_envs  # list of (K, 2) waypoint tensors per env
        self.smooth_curvatures = [None] * num_envs  # list of (K,) curvature tensors per env
        self.smooth_tangents = [None] * num_envs  # list of (K, 2) tangent tensors per env
        self.smooth_is_spike = [None] * num_envs  # list of (K,) bool tensors per env
        self.use_lidar_lookahead = os.environ.get("MAZE_LIDAR_LOOKAHEAD", "0") == "1"
        self.wall_hit_counts = torch.zeros((num_envs, num_cells, num_cells, 4), dtype=torch.int32, device=device)
        self.current_cell_paths = [None] * num_envs
        # Hazard avoidance and arbitration attributes
        self.use_hazard_avoidance = os.environ.get("MAZE_HAZARD_AVOIDANCE", "0") == "1"
        self.spike_avoid_mode = os.environ.get("MAZE_SPIKE_AVOID_MODE", "none")  # "none", "centroid", "slalom_s_curve"
        self.use_fork_arbitrator = os.environ.get("MAZE_FORK_ARBITRATOR", "0") == "1"
        self.persist_map_on_reset = os.environ.get("MAZE_PERSIST_MAP", "0") == "1"
        self.cell_hazard_cost = torch.zeros((num_envs, num_cells, num_cells), dtype=torch.float32, device=device)
        self.cell_spike_zones = [{} for _ in range(num_envs)]
        self.cell_ramp_zones = [set() for _ in range(num_envs)]
        self.cell_gap_zones = [set() for _ in range(num_envs)]
        self.replan_needed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.debug = False

    def update_hazards_from_height_scan(self, pos_xy: torch.Tensor, scan_xy: torch.Tensor, scan_z: torch.Tensor):
        """Analyze 3.2m x 3.2m height scan to detect floor trenches, steep ramps, and spike beds."""
        N = pos_xy.shape[0]
        c, n = self.c, self.n
        for e in range(N):
            s_xy = scan_xy[e]  # (R, 2)
            s_z = scan_z[e]    # (R,)

            ci = torch.round(s_xy[:, 0] / c).long()
            cj = torch.round(s_xy[:, 1] / c).long()
            valid_mask = (ci >= 0) & (ci < n) & (cj >= 0) & (cj < n) & (s_z <= 0.48)
            if not valid_mask.any():
                continue

            valid_ci = ci[valid_mask]
            valid_cj = cj[valid_mask]
            unique_cells = torch.unique(torch.stack([valid_ci, valid_cj], dim=1), dim=0)

            for u_cell in unique_cells:
                i, j = int(u_cell[0].item()), int(u_cell[1].item())
                cell_mask = valid_mask & (ci == i) & (cj == j)
                if cell_mask.sum() < 4:
                    continue

                rx = s_xy[cell_mask, 0] - i * c
                ry = s_xy[cell_mask, 1] - j * c
                rz = s_z[cell_mask]

                # Exclude perimeter wall margins
                inner = (rx.abs() <= 0.78) & (ry.abs() <= 0.78)
                if inner.sum() < 4:
                    continue
                rx = rx[inner]
                ry = ry[inner]
                rz = rz[inner]

                # 1. Floor Trench / Gap Check (z < -0.15 m)
                gap_pts = rz < -0.15
                if gap_pts.sum() >= 2:
                    if (i, j) not in self.cell_gap_zones[e]:
                        self.cell_gap_zones[e].add((i, j))
                        cur_ci = int(round(pos_xy[e, 0].item() / c))
                        cur_cj = int(round(pos_xy[e, 1].item() / c))
                        if (i, j) != (cur_ci, cur_cj):
                            self.replan_needed[e] = True
                    if (i, j) in self.cell_spike_zones[e]:
                        del self.cell_spike_zones[e][(i, j)]
                        self.replan_needed[e] = True
                    if self.cell_hazard_cost[e, i, j] > 0.0 and self.cell_hazard_cost[e, i, j] < 30.0:
                        self.cell_hazard_cost[e, i, j] = 0.0
                        self.replan_needed[e] = True

                # 2. Elevated Obstacle Check (0.12 m < z < 0.42 m)
                elev_pts = (rz > 0.12) & (rz < 0.42)
                if elev_pts.sum() >= 4:
                    # Corridor orientation: axis 0 along x, axis 1 along y
                    axis = None
                    c_path = self.current_cell_paths[e]
                    if c_path and (i, j) in c_path:
                        idx_c = c_path.index((i, j))
                        neighbors = []
                        if idx_c > 0:
                            neighbors.append(c_path[idx_c - 1])
                        if idx_c + 1 < len(c_path):
                            neighbors.append(c_path[idx_c + 1])
                        for nb in neighbors:
                            if nb[0] != i and nb[1] == j:
                                axis = 0
                                break
                            elif nb[1] != j and nb[0] == i:
                                axis = 1
                                break

                    if axis is None:
                        is_axis_0 = (self.walls[e, i, j, 1] == 1) or (self.walls[e, i, j, 3] == 1) or (self.wall_hit_counts[e, i, j, 1] > 0) or (self.wall_hit_counts[e, i, j, 3] > 0)
                        is_axis_1 = (self.walls[e, i, j, 0] == 1) or (self.walls[e, i, j, 2] == 1) or (self.wall_hit_counts[e, i, j, 0] > 0) or (self.wall_hit_counts[e, i, j, 2] > 0)
                        if is_axis_0 and not is_axis_1:
                            axis = 0
                        elif is_axis_1 and not is_axis_0:
                            axis = 1
                        else:
                            mean_rx = rx[elev_pts].mean().item()
                            mean_ry = ry[elev_pts].mean().item()
                            if abs(mean_rx) > abs(mean_ry) + 0.10:
                                axis = 1
                            elif abs(mean_ry) > abs(mean_rx) + 0.10:
                                axis = 0
                            else:
                                span_x = (rx.max() - rx.min()).item()
                                span_y = (ry.max() - ry.min()).item()
                                axis = 0 if span_x >= span_y else 1

                    u = rx if axis == 0 else ry
                    v = ry if axis == 0 else rx

                    # Check Ramp vs Spike Bed:
                    # Ramps span across the centerline (|v| <= 0.18 has elevated points rz > 0.12,
                    # or (vm < -0.20).any() and (vm > 0.20).any(), whereas a real spike bed has elevated cones
                    # on one side AND flat floor rz < 0.05 on the opposite side).
                    v_elev = v[elev_pts]
                    has_neg_elev = bool((v_elev < -0.20).any().item())
                    has_pos_elev = bool((v_elev > 0.20).any().item())
                    flat_neg = bool(((rz < 0.06) & (v < -0.20)).any().item())
                    flat_pos = bool(((rz < 0.06) & (v > 0.20)).any().item())
                    centerline_elev = int((elev_pts & (v.abs() <= 0.18)).sum().item())

                    z_minus = elev_pts & (u >= -0.92) & (u <= -0.25)
                    z_mid = elev_pts & (u >= -0.25) & (u <= 0.25)
                    z_plus = elev_pts & (u >= 0.25) & (u <= 0.92)

                    is_ramp = False
                    if has_neg_elev and has_pos_elev:
                        is_ramp = True
                    elif centerline_elev >= 4 and not (flat_neg or flat_pos):
                        is_ramp = True

                    if is_ramp:
                        if (i, j) not in self.cell_ramp_zones[e]:
                            self.cell_ramp_zones[e].add((i, j))
                            cur_ci = int(round(pos_xy[e, 0].item() / c))
                            cur_cj = int(round(pos_xy[e, 1].item() / c))
                            if (i, j) != (cur_ci, cur_cj):
                                self.replan_needed[e] = True
                        if (i, j) in self.cell_spike_zones[e]:
                            del self.cell_spike_zones[e][(i, j)]
                            self.replan_needed[e] = True
                        if self.cell_hazard_cost[e, i, j] > 0.0 and self.cell_hazard_cost[e, i, j] < 30.0:
                            self.cell_hazard_cost[e, i, j] = 0.0
                            self.replan_needed[e] = True
                    elif (i, j) not in self.cell_ramp_zones[e] and (i, j) not in self.cell_gap_zones[e]:
                        # Spike Bed (lane or slalom)
                        # Require that for a spike bed, the opposite side of the corridor actually has flat floor points (rz < 0.06)
                        flat_pts = rz < 0.06
                        use_m1 = self.spike_avoid_mode in ("slalom_s_curve", "method1_lead_in_out", "method1", "holonomic_vy")
                        v_lat = 0.44

                        v_minus_safe = None
                        v_plus_safe = None
                        if z_minus.sum() >= 3:
                            mean_vm = v[z_minus].mean().item()
                            if mean_vm < -0.15 and (flat_pts & (v > 0.15)).sum() >= 1:
                                v_minus_safe = v_lat
                            elif mean_vm > 0.15 and (flat_pts & (v < -0.15)).sum() >= 1:
                                v_minus_safe = -v_lat

                        if z_plus.sum() >= 3:
                            mean_vp = v[z_plus].mean().item()
                            if mean_vp < -0.15 and (flat_pts & (v > 0.15)).sum() >= 1:
                                v_plus_safe = v_lat
                            elif mean_vp > 0.15 and (flat_pts & (v < -0.15)).sum() >= 1:
                                v_plus_safe = -v_lat

                        v_whole_centroid = v[elev_pts].mean().item()
                        centroid_safe = None
                        if v_whole_centroid < -0.15 and (flat_pts & (v > 0.15)).sum() >= 1:
                            centroid_safe = v_lat
                        elif v_whole_centroid > 0.15 and (flat_pts & (v < -0.15)).sum() >= 1:
                            centroid_safe = -v_lat

                        prev_zone = self.cell_spike_zones[e].get((i, j))

                        if use_m1:
                            # Method 1: Persist/accumulate v_minus and v_plus memory across frames
                            prev_vm = prev_zone.get("v_minus") if prev_zone else None
                            prev_vp = prev_zone.get("v_plus") if prev_zone else None
                            m_vm = v_minus_safe if v_minus_safe is not None else prev_vm
                            m_vp = v_plus_safe if v_plus_safe is not None else prev_vp

                            if m_vm is not None and m_vp is not None and (m_vm != m_vp):
                                # 2-bed SLALOM
                                new_zone = {
                                    "kind": "slalom",
                                    "axis": axis,
                                    "v_minus": m_vm,
                                    "v_plus": m_vp,
                                    "v_centroid": centroid_safe if centroid_safe is not None else 0.0,
                                }
                            elif m_vm is not None or m_vp is not None or centroid_safe is not None:
                                v_safe = m_vm if m_vm is not None else (m_vp if m_vp is not None else centroid_safe)
                                new_zone = {
                                    "kind": "lane",
                                    "axis": axis,
                                    "v_safe": v_safe,
                                    "v_minus": m_vm,
                                    "v_plus": m_vp,
                                    "v_centroid": centroid_safe if centroid_safe is not None else v_safe,
                                }
                            else:
                                new_zone = None

                            if new_zone is not None and prev_zone != new_zone:
                                self.cell_spike_zones[e][(i, j)] = new_zone
                                self.replan_needed[e] = True
                                if self.use_fork_arbitrator and new_zone["kind"] == "slalom" and self.cell_hazard_cost[e, i, j] < 25.0:
                                    self.cell_hazard_cost[e, i, j] = 25.0
                        else:
                            # Baseline / Exp 10 old logic: overwrite without persistence & no spike replan trigger
                            c_safe = centroid_safe if centroid_safe is not None else 0.0
                            if v_minus_safe is not None and v_plus_safe is not None and (v_minus_safe != v_plus_safe):
                                self.cell_spike_zones[e][(i, j)] = {
                                    "kind": "slalom",
                                    "axis": axis,
                                    "v_minus": v_minus_safe,
                                    "v_plus": v_plus_safe,
                                    "v_centroid": c_safe,
                                }
                                if self.use_fork_arbitrator and self.cell_hazard_cost[e, i, j] < 25.0:
                                    self.cell_hazard_cost[e, i, j] = 25.0
                                    self.replan_needed[e] = True
                            elif v_minus_safe is not None or v_plus_safe is not None or centroid_safe is not None:
                                v_safe = v_minus_safe if v_minus_safe is not None else (v_plus_safe if v_plus_safe is not None else c_safe)
                                self.cell_spike_zones[e][(i, j)] = {
                                    "kind": "lane",
                                    "axis": axis,
                                    "v_safe": v_safe,
                                    "v_centroid": c_safe,
                                }

    # ------------------------------------------------------------------ helpers
    def cell_of(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Maze cell index of a position given relative to the centre of cell (0, 0)."""
        return torch.round(pos_xy / self.c).long().clamp(0, self.n - 1)

    def centre_of(self, cell: torch.Tensor) -> torch.Tensor:
        return cell.float() * self.c

    # ------------------------------------------------------------------ 1. perception
    def walls_from_height_scan(
        self, hits_xy: torch.Tensor, hits_z: torch.Tensor, cell: torch.Tensor, ground_z: float = 0.0
    ) -> torch.Tensor:
        """Return ``(num_envs, 4)`` bools: is there a wall on side d of ``cell``?

        ``hits_xy`` are the ray hit points (relative to the centre of cell (0, 0)) and ``hits_z`` their heights.
        A side is blocked when at least ``min_hits`` tall points (> 0.5 m) fall inside a thin box
        that sits on that side of the cell (0.3 m deep, 1.2 m long).
        """
        centre = self.centre_of(cell).unsqueeze(1)  # (N, 1, 2)
        rel = hits_xy - centre  # (N, R, 2)
        tall = hits_z > ground_z + 0.5
        out = torch.zeros(hits_xy.shape[0], 4, dtype=torch.bool, device=self.device)
        half = 0.5 * self.c
        for d, (dx, dy) in enumerate(DIRS):
            along = rel[..., 0] * dx + rel[..., 1] * dy  # distance towards side d
            across = rel[..., 0] * dy - rel[..., 1] * dx
            in_box = (torch.abs(along - half) < 0.15) & (torch.abs(across) < 0.6)
            out[:, d] = (in_box & tall).sum(dim=1) >= 3
        return out

    def walls_from_lidar(self, distances: torch.Tensor, angles_w: torch.Tensor, offset_xy: torch.Tensor):
        """Same output as :meth:`walls_from_height_scan`, from a horizontal 2D lidar (Exercise 7.4).

        ``distances``: (N, R) range of every ray, ``angles_w``: (N, R) world yaw of every ray,
        ``offset_xy``: (N, 2) robot position minus the current cell centre.
        """
        out = torch.zeros(distances.shape[0], 4, dtype=torch.bool, device=self.device)
        for d in range(4):
            yaw = d * math.pi / 2
            diff = torch.atan2(torch.sin(angles_w - yaw), torch.cos(angles_w - yaw))
            in_cone = torch.abs(diff) < math.radians(10)
            dist = torch.where(in_cone, distances, torch.full_like(distances, 1e6)).min(dim=1).values
            # distance from the robot to side d of the cell, corrected for the robot's offset from the centre
            to_side = 0.5 * self.c - (offset_xy[:, 0] * DIRS[d][0] + offset_xy[:, 1] * DIRS[d][1])
            out[:, d] = dist < to_side + 0.3
        return out

    def update_walls_from_lidar_raycast(
        self, pos_xy: torch.Tensor, lidar_dist: torch.Tensor, lidar_yaw: torch.Tensor
    ):
        """Update occupancy grid self.walls using 72-ray 6m LiDAR."""
        N = pos_xy.shape[0]
        n, c = self.n, self.c
        px = pos_xy[:, 0]
        py = pos_xy[:, 1]

        valid_dist = (lidar_dist < 5.8) & (lidar_dist > 0.1)
        hx = pos_xy[:, None, 0] + lidar_dist * torch.cos(lidar_yaw)
        hy = pos_xy[:, None, 1] + lidar_dist * torch.sin(lidar_yaw)

        for e in range(N):
            p_x = float(px[e].item())
            p_y = float(py[e].item())
            valid_e = valid_dist[e]
            if not valid_e.any():
                continue

            h_x = hx[e, valid_e].detach().cpu().numpy()
            h_y = hy[e, valid_e].detach().cpu().numpy()
            d_e = lidar_dist[e, valid_e].detach().cpu().numpy()
            y_e = lidar_yaw[e, valid_e].detach().cpu().numpy()

            # 1. Wall hit classification
            for rx, ry, yaw in zip(h_x, h_y, y_e):
                cos_ray = math.cos(float(yaw))
                sin_ray = math.sin(float(yaw))

                # Vertical walls at x = (k + 0.5) * c (normal along x -> cos_incidence = |cos_ray|)
                k_cand = int(round((rx / c) - 0.5))
                x_wall = (k_cand + 0.5) * c
                if abs(rx - x_wall) <= 0.20:
                    j = int(round(ry / c))
                    if 0 <= j < n and -1 <= k_cand <= n - 1:
                        offset = ry - j * c
                        if abs(offset) < 0.55 and abs(cos_ray) > 0.35:
                            if 0 <= k_cand < n:
                                self.wall_hit_counts[e, k_cand, j, 0] += 1
                                if self.wall_hit_counts[e, k_cand, j, 0] >= 2:
                                    self.walls[e, k_cand, j, 0] = 1
                            if 0 <= k_cand + 1 < n:
                                self.wall_hit_counts[e, k_cand + 1, j, 2] += 1
                                if self.wall_hit_counts[e, k_cand + 1, j, 2] >= 2:
                                    self.walls[e, k_cand + 1, j, 2] = 1

                # Horizontal walls at y = (m + 0.5) * c (normal along y -> cos_incidence = |sin_ray|)
                m_cand = int(round((ry / c) - 0.5))
                y_wall = (m_cand + 0.5) * c
                if abs(ry - y_wall) <= 0.20:
                    i = int(round(rx / c))
                    if 0 <= i < n and -1 <= m_cand <= n - 1:
                        offset = rx - i * c
                        if abs(offset) < 0.55 and abs(sin_ray) > 0.35:
                            if 0 <= m_cand < n:
                                self.wall_hit_counts[e, i, m_cand, 1] += 1
                                if self.wall_hit_counts[e, i, m_cand, 1] >= 2:
                                    self.walls[e, i, m_cand, 1] = 1
                            if 0 <= m_cand + 1 < n:
                                self.wall_hit_counts[e, i, m_cand + 1, 3] += 1
                                if self.wall_hit_counts[e, i, m_cand + 1, 3] >= 2:
                                    self.walls[e, i, m_cand + 1, 3] = 1

            # 2. Free space ray marching
            for dist, yaw in zip(d_e, y_e):
                march_limit = max(0.0, float(dist) - 0.25)
                if march_limit < 0.5:
                    continue
                dx = math.cos(float(yaw))
                dy = math.sin(float(yaw))

                # Vertical grid lines (aperture center |offset| < 0.35 and cos_incidence |dx| > 0.50)
                if abs(dx) > 0.50:
                    for k in range(-1, n):
                        xw = (k + 0.5) * c
                        s = (xw - p_x) / dx
                        if 0.15 < s < march_limit:
                            y_cross = p_y + s * dy
                            j = int(round(y_cross / c))
                            if 0 <= j < n:
                                offset = y_cross - j * c
                                if abs(offset) < 0.35:
                                    if 0 <= k < n and self.walls[e, k, j, 0] == -1:
                                        self.walls[e, k, j, 0] = 0
                                    if 0 <= k + 1 < n and self.walls[e, k + 1, j, 2] == -1:
                                        self.walls[e, k + 1, j, 2] = 0

                # Horizontal grid lines (aperture center |offset| < 0.35 and cos_incidence |dy| > 0.50)
                if abs(dy) > 0.50:
                    for m in range(-1, n):
                        yw = (m + 0.5) * c
                        s = (yw - p_y) / dy
                        if 0.15 < s < march_limit:
                            x_cross = p_x + s * dx
                            i = int(round(x_cross / c))
                            if 0 <= i < n:
                                offset = x_cross - i * c
                                if abs(offset) < 0.35:
                                    if 0 <= m < n and self.walls[e, i, m, 1] == -1:
                                        self.walls[e, i, m, 1] = 0
                                    if 0 <= m + 1 < n and self.walls[e, i, m + 1, 3] == -1:
                                        self.walls[e, i, m + 1, 3] = 0

    # ------------------------------------------------------------------ 2. planning
    def flood_fill(self, walls: torch.Tensor) -> torch.Tensor:
        """Steps from every cell to the goal, treating unknown walls as open. (N, n, n, 4) -> (N, n, n)."""
        n = self.n
        big = n * n + 1
        dist = torch.full(walls.shape[:3], big, dtype=torch.long, device=self.device)
        dist[:, self.goal[0], self.goal[1]] = 0
        blocked = walls == 1
        for _ in range(n * n):
            new = dist.clone()
            # neighbour in direction d is reachable if side d is not blocked
            new[:, :-1, :] = torch.minimum(new[:, :-1, :], torch.where(blocked[:, :-1, :, 0], big, dist[:, 1:, :] + 1))
            new[:, 1:, :] = torch.minimum(new[:, 1:, :], torch.where(blocked[:, 1:, :, 2], big, dist[:, :-1, :] + 1))
            new[:, :, :-1] = torch.minimum(new[:, :, :-1], torch.where(blocked[:, :, :-1, 1], big, dist[:, :, 1:] + 1))
            new[:, :, 1:] = torch.minimum(new[:, :, 1:], torch.where(blocked[:, :, 1:, 3], big, dist[:, :, :-1] + 1))
            if torch.equal(new, dist):
                break
            dist = new
        return dist

    def choose_direction(
        self, cell: torch.Tensor, open_sides: torch.Tensor, heading: torch.Tensor, walls: torch.Tensor
    ) -> torch.Tensor:
        """Pick the next direction for each robot.

        Args:
            cell: (B, 2) current cell, open_sides: (B, 4) which sides are open,
            heading: (B,) direction the robot came in with, walls: (B, n, n, 4) wall map seen so far.
        """
        h = heading
        if self.strategy in ("left", "right"):
            # wall following: try left, straight, right, back (mirror for the right hand)
            turns = (1, 0, 3, 2) if self.strategy == "left" else (3, 0, 1, 2)
            choice = (h + 2) % 4
            for t in reversed(turns):  # the first open side in the preference list wins
                d = (h + t) % 4
                choice = torch.where(open_sides.gather(1, d.unsqueeze(1)).squeeze(1), d, choice)
            return choice
        if self.strategy in ("flood", "astar"):
            dist = self.flood_fill(walls)
            idx = torch.arange(len(h), device=self.device)
            best = torch.full_like(h, -1)
            best_cost = torch.full(h.shape, 10**6, dtype=torch.long, device=self.device)
            # straight first, so ties keep the robot walking straight (fewer turns)
            for t in (0, 1, 3, 2):
                d = (h + t) % 4
                nb = (cell + self._dirs[d]).clamp(0, self.n - 1)
                cost = dist[idx, nb[:, 0], nb[:, 1]]
                ok = open_sides[idx, d] & (cost < best_cost)
                best = torch.where(ok, d, best)
                best_cost = torch.where(ok, cost, best_cost)
            return torch.where(best >= 0, best, (h + 2) % 4)
        raise ValueError(f"unknown strategy {self.strategy!r}")

    def plan_full_path(self, start_cell: tuple[int, int] | list[int], env_id: int = 0, turn_penalty: float = 0.05) -> list[tuple[int, int]]:
        """A* search with turn penalties for minimum-time path execution."""
        import heapq
        import numpy as np

        n = self.n
        walls = self.walls[env_id].cpu().numpy()
        start = tuple(int(v) for v in start_cell)
        goal = self.goal
        has_moved = int(self.cells_walked[env_id].item()) > 0

        # Priority queue: (f_score, g_score, (i, j), heading)
        pq = [(0.0, 0.0, start, int(self.heading[env_id].item()))]
        g_costs = {(start, int(self.heading[env_id].item())): 0.0}
        came_from = {}
        goal_state = None

        while pq:
            f, g, curr, h = heapq.heappop(pq)
            if curr == goal:
                goal_state = (curr, h)
                break
            if g > g_costs.get((curr, h), 1e9):
                continue
            for t in (0, 1, 3, 2):
                d = (h + t) % 4
                di, dj = DIRS[d]
                if walls[curr[0], curr[1], d] == 1:
                    continue
                ni, nj = curr[0] + di, curr[1] + dj
                if 0 <= ni < n and 0 <= nj < n:
                    turn_cost = turn_penalty if (d != h) else 0.0
                    unk_cost = 0.08 if (self.use_lidar_lookahead and walls[curr[0], curr[1], d] == -1) else 0.0
                    rev_cost = 0.08 if (curr == start and has_moved and d == (h + 2) % 4) else 0.0
                    hazard_cost = float(self.cell_hazard_cost[env_id, ni, nj].item())
                    new_g = g + 1.0 + turn_cost + unk_cost + rev_cost + hazard_cost
                    st = ((ni, nj), d)
                    if new_g < g_costs.get(st, 1e9) - 1e-6:
                        g_costs[st] = new_g
                        h_cost = float(np.sqrt((ni - goal[0]) ** 2 + (nj - goal[1]) ** 2))
                        heapq.heappush(pq, (new_g + h_cost, new_g, (ni, nj), d))
                        came_from[st] = (curr, h)

        if goal_state is None:
            return [start]
        path = []
        st = goal_state
        while st in came_from:
            path.append(st[0])
            st = came_from[st]
        path.append(start)
        path.reverse()
        return path

    def interpolate_path(
        self, cell_path: list[tuple[int, int]], pts_per_arc: int = 10, return_derivatives: bool = False, env_id: int = 0
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate dense waypoints with smooth arc fillets at every corner (Fig. 8 & Fig. 9)."""
        import numpy as np

        if len(cell_path) < 2:
            wpts = torch.tensor(cell_path, dtype=torch.float, device=self.device) * self.c
            if return_derivatives:
                curvs = torch.zeros(len(wpts), device=self.device)
                tangs = torch.zeros((len(wpts), 2), device=self.device)
                return wpts, curvs, tangs
            return wpts

        raw_pts = []
        is_spike_cell = []
        spike_zones = self.cell_spike_zones[env_id] if env_id < len(self.cell_spike_zones) else {}
        ramp_zones = self.cell_ramp_zones[env_id] if env_id < len(self.cell_ramp_zones) else set()
        gap_zones = self.cell_gap_zones[env_id] if env_id < len(self.cell_gap_zones) else set()
        for k, c_pt in enumerate(cell_path):
            cell_tuple = tuple(c_pt)
            if self.spike_avoid_mode != "none" and cell_tuple in spike_zones:
                prev_c = cell_path[k - 1] if k > 0 else None
                next_c = cell_path[k + 1] if k + 1 < len(cell_path) else None
                sub_pts = generate_subcell_waypoints(prev_c, cell_tuple, next_c, spike_zones[cell_tuple], c=self.c, mode=self.spike_avoid_mode)
                for sp in sub_pts:
                    raw_pts.append(np.array(sp, dtype=float))
                    is_spike_cell.append(True)
            elif cell_tuple in ramp_zones or cell_tuple in gap_zones:
                prev_c = cell_path[k - 1] if k > 0 else None
                next_c = cell_path[k + 1] if k + 1 < len(cell_path) else None
                if prev_c is not None:
                    di, dj = cell_tuple[0] - prev_c[0], cell_tuple[1] - prev_c[1]
                elif next_c is not None:
                    di, dj = next_c[0] - cell_tuple[0], next_c[1] - cell_tuple[1]
                else:
                    di, dj = 1, 0

                cx = cell_tuple[0] * self.c
                cy = cell_tuple[1] * self.c
                if dj == 0:  # corridor axis along x
                    s_dir = 1 if di >= 0 else -1
                    for u in (-0.65, 0.0, 0.65):
                        raw_pts.append(np.array([cx + u * s_dir, cy], dtype=float))
                        is_spike_cell.append(False)
                else:  # corridor axis along y
                    s_dir = 1 if dj >= 0 else -1
                    for u in (-0.65, 0.0, 0.65):
                        raw_pts.append(np.array([cx, cy + u * s_dir], dtype=float))
                        is_spike_cell.append(False)
            else:
                pt = [c_pt[0] * self.c, c_pt[1] * self.c]
                # Pre-align approach corner waypoint if turning into a spike corridor
                if self.spike_avoid_mode in ("slalom_s_curve", "method1_lead_in_out", "method1", "holonomic_vy") and k + 1 < len(cell_path):
                    next_c = tuple(cell_path[k + 1])
                    if next_c in spike_zones:
                        sz = spike_zones[next_c]
                        axis_next = sz.get("axis", 0)
                        di_next = next_c[0] - cell_tuple[0]
                        dj_next = next_c[1] - cell_tuple[1]
                        s_dir_next = di_next if axis_next == 0 else dj_next
                        if s_dir_next != 0:
                            kind_next = sz.get("kind", "lane")
                            v_lat = 0.50 if kind_next == "slalom" else 0.44
                            if kind_next == "slalom":
                                v_m = sz.get("v_minus", v_lat)
                                v_p = sz.get("v_plus", -v_lat)
                                v_minus = v_lat if v_m >= 0 else -v_lat
                                v_plus = v_lat if v_p >= 0 else -v_lat
                                if v_minus == v_plus:
                                    v_plus = -v_minus
                                v_entry = v_minus if s_dir_next > 0 else v_plus
                            else:
                                v_raw = sz.get("v_safe", v_lat)
                                v_entry = v_lat if v_raw >= 0 else -v_lat

                            if k > 0:
                                prev_c = cell_path[k - 1]
                                di_in = cell_tuple[0] - prev_c[0]
                                dj_in = cell_tuple[1] - prev_c[1]
                                if (di_in != 0 and di_next == 0) or (dj_in != 0 and dj_next == 0):
                                    # Corner turn into spike corridor: shift turn point to line up with entry lane
                                    if axis_next == 0:
                                        pt[1] = next_c[1] * self.c + v_entry
                                    else:
                                        pt[0] = next_c[0] * self.c + v_entry

                raw_pts.append(np.array(pt, dtype=float))
                is_spike_cell.append(False)

        wpts = [raw_pts[0]]
        wpts_is_spike = [is_spike_cell[0]]
        curvatures = [0.0]
        init_v = raw_pts[1] - raw_pts[0]
        init_t = init_v / max(float(np.linalg.norm(init_v)), 1e-6)
        tangents = [init_t]

        for k in range(1, len(raw_pts) - 1):
            p_prev = raw_pts[k - 1]
            p_curr = raw_pts[k]
            p_next = raw_pts[k + 1]

            v_in = p_curr - p_prev
            d_in = float(np.linalg.norm(v_in))
            u_in = v_in / max(d_in, 1e-6)

            v_out = p_next - p_curr
            d_out = float(np.linalg.norm(v_out))
            u_out = v_out / max(d_out, 1e-6)

            is_turn = float(np.dot(u_in, u_out)) < 0.98
            if is_turn:
                near_spike = is_spike_cell[k] or (k > 0 and is_spike_cell[k - 1]) or (k + 1 < len(is_spike_cell) and is_spike_cell[k + 1])
                use_m1 = self.spike_avoid_mode in ("slalom_s_curve", "method1_lead_in_out", "method1")
                fillet_limit = (0.12 if use_m1 else 0.22) if near_spike else self.r_fillet
                if self.curve_mode == "g2_quintic":
                    r = min(fillet_limit, 0.48 * d_in, 0.48 * d_out)
                    p0 = p_curr - r * u_in
                    p1 = p_curr + r * u_out
                    eta = 1.15 * r
                    m0 = eta * u_in
                    m1 = eta * u_out

                    s_vals = np.linspace(0.0, 1.0, pts_per_arc + 2)
                    for s in s_vals:
                        h00 = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5
                        h10 = s - 6.0 * s**3 + 8.0 * s**4 - 3.0 * s**5
                        h01 = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
                        h11 = -4.0 * s**3 + 7.0 * s**4 - 3.0 * s**5

                        h00_p = -30.0 * s**2 + 60.0 * s**3 - 30.0 * s**4
                        h10_p = 1.0 - 18.0 * s**2 + 32.0 * s**3 - 15.0 * s**4
                        h01_p = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
                        h11_p = -12.0 * s**2 + 28.0 * s**3 - 15.0 * s**4

                        h00_pp = -60.0 * s + 180.0 * s**2 - 120.0 * s**3
                        h10_pp = -36.0 * s + 96.0 * s**2 - 60.0 * s**3
                        h01_pp = 60.0 * s - 180.0 * s**2 + 120.0 * s**3
                        h11_pp = -24.0 * s + 84.0 * s**2 - 60.0 * s**3

                        P = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                        Pp = h00_p * p0 + h10_p * m0 + h01_p * p1 + h11_p * m1
                        Ppp = h00_pp * p0 + h10_pp * m0 + h01_pp * p1 + h11_pp * m1

                        spd_sq = Pp[0] ** 2 + Pp[1] ** 2
                        kappa = (Pp[0] * Ppp[1] - Pp[1] * Ppp[0]) / max(spd_sq**1.5, 1e-6)
                        tang = Pp / max(math.sqrt(spd_sq), 1e-6)

                        wpts.append(P)
                        wpts_is_spike.append(near_spike)
                        curvatures.append(float(kappa))
                        tangents.append(tang)
                else:
                    r = min(self.r_fillet, 0.45 * d_in, 0.45 * d_out)
                    p_enter = p_curr - r * u_in
                    p_exit = p_curr + r * u_out
                    wpts.append(p_enter)
                    wpts_is_spike.append(near_spike)
                    curvatures.append(0.0)
                    tangents.append(u_in)
                    for s in np.linspace(0.12, 0.88, pts_per_arc):
                        arc_pt = ((1.0 - s) ** 2) * p_enter + 2.0 * (1.0 - s) * s * p_curr + (s ** 2) * p_exit
                        dp = 2.0 * (1.0 - s) * (p_curr - p_enter) + 2.0 * s * (p_exit - p_curr)
                        tang = dp / max(float(np.linalg.norm(dp)), 1e-6)
                        wpts.append(arc_pt)
                        wpts_is_spike.append(near_spike)
                        curvatures.append(0.0)
                        tangents.append(tang)
                    wpts.append(p_exit)
                    wpts_is_spike.append(near_spike)
                    curvatures.append(0.0)
                    tangents.append(u_out)
            else:
                wpts.append(p_curr)
                wpts_is_spike.append(is_spike_cell[k])
                curvatures.append(0.0)
                tangents.append(u_in)

        wpts.append(raw_pts[-1])
        wpts_is_spike.append(is_spike_cell[-1])
        curvatures.append(0.0)
        tangents.append(tangents[-1] if len(tangents) > 0 else np.zeros(2))

        wpts_t = torch.tensor(np.array(wpts), dtype=torch.float, device=self.device)
        curvs_t = torch.tensor(np.array(curvatures), dtype=torch.float, device=self.device)
        tangs_t = torch.tensor(np.array(tangents), dtype=torch.float, device=self.device)
        if hasattr(self, "smooth_is_spike") and env_id < len(self.smooth_is_spike):
            self.smooth_is_spike[env_id] = torch.tensor(np.array(wpts_is_spike), dtype=torch.bool, device=self.device)

        if return_derivatives:
            return wpts_t, curvs_t, tangs_t
        return wpts_t

    def get_pursuit_target_info(self, env_id: int, pos_xy: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
        """Get pure-pursuit waypoint, curvature, and tangent along interpolated trajectory."""
        wpts = self.smooth_wpts[env_id]
        if wpts is None or len(wpts) < 2:
            return self.centre_of(self.target[env_id:env_id+1]).squeeze(0), 0.0, torch.zeros(2, device=self.device)

        dists = torch.linalg.norm(wpts - pos_xy, dim=1)
        closest_idx = torch.argmin(dists).item()

        # Near spike check: tighten pure-pursuit lookahead to prevent cutting into spike beds
        is_spk = self.smooth_is_spike[env_id] if hasattr(self, "smooth_is_spike") and env_id < len(self.smooth_is_spike) else None
        near_spike = is_spk is not None and closest_idx < len(is_spk) and bool(is_spk[closest_idx].item())
        near_spike &= self.spike_avoid_mode in ("slalom_s_curve", "method1_lead_in_out", "method1")
        la_dist = 0.42 if near_spike else self.lookahead_dist

        accum = 0.0
        target_idx = closest_idx
        for i in range(closest_idx, len(wpts) - 1):
            seg_len = torch.linalg.norm(wpts[i + 1] - wpts[i]).item()
            accum += seg_len
            target_idx = i + 1
            if accum >= la_dist:
                break

        target_pt = wpts[target_idx]
        curv = 0.0
        tangent = torch.zeros(2, device=self.device)
        if self.smooth_curvatures[env_id] is not None:
            curv = float(self.smooth_curvatures[env_id][target_idx].item())
        if self.smooth_tangents[env_id] is not None:
            tangent = self.smooth_tangents[env_id][target_idx]
        return target_pt, curv, tangent

    def get_pursuit_target(self, env_id: int, pos_xy: torch.Tensor) -> torch.Tensor:
        """Get pure-pursuit lookahead waypoint along the interpolated smooth trajectory (Fig. 8)."""
        target_pt, _, _ = self.get_pursuit_target_info(env_id, pos_xy)
        return target_pt

    def peek_next_direction(self, target_cell: torch.Tensor, heading: torch.Tensor, walls: torch.Tensor) -> torch.Tensor:
        """Look ahead one step from target_cell to predict the next direction."""
        if self.strategy != "flood":
            return heading
        dist = self.flood_fill(walls)
        idx = torch.arange(len(heading), device=self.device)
        best = torch.full_like(heading, -1)
        best_cost = torch.full(heading.shape, 10**6, dtype=torch.long, device=self.device)
        h = heading
        for t in (0, 1, 3, 2):
            d = (h + t) % 4
            nb = (target_cell + self._dirs[d]).clamp(0, self.n - 1)
            cost = dist[idx, nb[:, 0], nb[:, 1]]
            not_blocked = walls[idx, target_cell[:, 0], target_cell[:, 1], d] != 1
            ok = not_blocked & (cost < best_cost)
            best = torch.where(ok, d, best)
            best_cost = torch.where(ok, cost, best_cost)
        return torch.where(best >= 0, best, heading)

    def get_target_waypoint(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Compute the pursuit waypoint in metres, with direct corner-cutting if turning."""
        target_centre = self.centre_of(self.target)
        if not self.direct_corner:
            return target_centre

        is_turn = (self.next_heading >= 0) & (self.next_heading != self.heading)
        is_goal = (self.target[:, 0] == self.goal[0]) & (self.target[:, 1] == self.goal[1])
        apply_cut = is_turn & ~is_goal

        waypoint = target_centre.clone()
        if apply_cut.any():
            k = torch.nonzero(apply_cut).squeeze(1)
            dir_in = self._dirs[self.heading[k]].float()
            dir_out = self._dirs[self.next_heading[k]].float()
            cut_offset = -self.corner_cut * dir_in + self.corner_cut * dir_out
            waypoint[k] = target_centre[k] + cut_offset
        return waypoint

    # ------------------------------------------------------------------ 3. control
    def velocity_command(self, pos_xy: torch.Tensor, yaw: torch.Tensor, dt: float) -> torch.Tensor:
        """(v_x, v_y, w_z) in the robot frame driving along smooth interpolated arc waypoints or direct corner cuts."""
        if self.use_astar:
            targets = []
            kappas = []
            for i in range(self.num_envs):
                if self.direct_corner and self.smooth_wpts[i] is not None:
                    target_pt, curv, _ = self.get_pursuit_target_info(i, pos_xy[i])
                    targets.append(target_pt)
                    kappas.append(curv)
                else:
                    targets.append(self.centre_of(self.target[i]))
                    kappas.append(0.0)
            to_target = torch.stack(targets) - pos_xy
            kappa_ff = torch.tensor(kappas, device=self.device, dtype=torch.float)
        else:
            waypoint = self.get_target_waypoint(pos_xy)
            to_target = waypoint - pos_xy
            kappa_ff = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        yaw_des = torch.atan2(to_target[:, 1], to_target[:, 0])
        yaw_err = yaw_des - yaw
        yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))

        # Heading alignment and cornering velocity: maintain smooth momentum through turns
        # Normal cornering (|yaw_err| <= 75 deg): full speed profile (100% Exp 07/08 momentum)
        # In-place pivot (|yaw_err| >= 95 deg): in_place_scale = 0 -> zero forward/lateral drift
        cos_align = torch.clamp(torch.cos(yaw_err), min=0.0)
        v_des = self.min_speed + (self.speed - self.min_speed) * (cos_align ** 1.2)
        in_place_scale = torch.clamp((math.radians(95.0) - torch.abs(yaw_err)) / math.radians(20.0), 0.0, 1.0)
        v_des = v_des * in_place_scale

        if self.use_astar and self.curve_mode == "g2_quintic":
            w_des = torch.clamp(v_des * kappa_ff + 2.0 * yaw_err, -self.max_turn, self.max_turn)
        else:
            w_des = torch.clamp(2.4 * yaw_err, -self.max_turn, self.max_turn)

        # Non-holonomic G1 locomotion: G1Rough policy trained with lin_vel_y = 0.0. Zero lateral strafe.
        v_y_des = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # Holonomic Slalom Gap Speed Moderation & Crossing Assistance (Method 1 + Method 2)
        fast_decel_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        use_holonomic_spike = self.use_hazard_avoidance and (
            self.spike_avoid_mode in ("slalom_s_curve", "method1_lead_in_out", "method1", "holonomic_vy")
        )

        if use_holonomic_spike:
            curr_cells = self.cell_of(pos_xy)
            for e in range(self.num_envs):
                c_cur = (int(curr_cells[e, 0].item()), int(curr_cells[e, 1].item()))
                c_tgt = (int(self.target[e, 0].item()), int(self.target[e, 1].item()))

                if c_cur in self.cell_ramp_zones[e]:
                    h_idx = int(self.heading[e].item())
                    c_path = self.current_cell_paths[e]
                    if c_path and c_cur in c_path:
                        idx_c = c_path.index(c_cur)
                        if idx_c + 1 < len(c_path):
                            di_h = c_path[idx_c + 1][0] - c_cur[0]
                            dj_h = c_path[idx_c + 1][1] - c_cur[1]
                        elif idx_c > 0:
                            di_h = c_cur[0] - c_path[idx_c - 1][0]
                            dj_h = c_cur[1] - c_path[idx_c - 1][1]
                        else:
                            di_h, dj_h = DIRS[h_idx]
                    else:
                        di_h, dj_h = DIRS[h_idx]

                    axis = 0 if dj_h == 0 else 1
                    s_dir = di_h if axis == 0 else dj_h
                    ci, cj = c_cur
                    u_r = (pos_xy[e, 0].item() - ci * self.c) if axis == 0 else (pos_xy[e, 1].item() - cj * self.c)
                    u_prog = u_r * s_dir

                    # Active ramp climbing window: from ramp entrance (-0.85 m) through ramp descent (+0.80 m)
                    if -0.85 <= u_prog <= 0.80:
                        if di_h == 1:
                            corridor_yaw = 0.0
                        elif di_h == -1:
                            corridor_yaw = math.pi
                        elif dj_h == 1:
                            corridor_yaw = math.pi / 2.0
                        else:
                            corridor_yaw = -math.pi / 2.0

                        psi_cur = float(yaw[e].item())
                        yaw_err = math.atan2(math.sin(corridor_yaw - psi_cur), math.cos(corridor_yaw - psi_cur))
                        # Gentle heading lock: cap at 0.06 rad/s so humanoid climbs incline without leg twisting/stalling
                        w_des[e] = max(-0.06, min(0.06, 0.8 * yaw_err))
                        v_y_des[e] = 0.0
                        v_des[e] = float(self.speed)
                        continue

                if c_cur in self.cell_gap_zones[e]:
                    h_idx = int(self.heading[e].item())
                    c_path = self.current_cell_paths[e]
                    if c_path and c_cur in c_path:
                        idx_c = c_path.index(c_cur)
                        if idx_c + 1 < len(c_path):
                            di_h = c_path[idx_c + 1][0] - c_cur[0]
                            dj_h = c_path[idx_c + 1][1] - c_cur[1]
                        elif idx_c > 0:
                            di_h = c_cur[0] - c_path[idx_c - 1][0]
                            dj_h = c_cur[1] - c_path[idx_c - 1][1]
                        else:
                            di_h, dj_h = DIRS[h_idx]
                    else:
                        di_h, dj_h = DIRS[h_idx]

                    axis = 0 if dj_h == 0 else 1
                    s_dir = di_h if axis == 0 else dj_h
                    ci, cj = c_cur
                    u_r = (pos_xy[e, 0].item() - ci * self.c) if axis == 0 else (pos_xy[e, 1].item() - cj * self.c)
                    u_prog = u_r * s_dir

                    # Active gap crossing window: from cell entrance (-1.05 m) through trench clearance (+0.18 m)
                    if -1.05 <= u_prog <= 0.18:
                        if di_h == 1:
                            corridor_yaw = 0.0
                        elif di_h == -1:
                            corridor_yaw = math.pi
                        elif dj_h == 1:
                            corridor_yaw = math.pi / 2.0
                        else:
                            corridor_yaw = -math.pi / 2.0

                        psi_cur = float(yaw[e].item())
                        yaw_err = math.atan2(math.sin(corridor_yaw - psi_cur), math.cos(corridor_yaw - psi_cur))
                        # Gentle heading lock: cap at 0.06 rad/s so humanoid steps straight over trench without diagonal swerve
                        w_des[e] = max(-0.06, min(0.06, 0.8 * yaw_err))
                        v_y_des[e] = 0.0
                        v_des[e] = 0.85
                        continue

                # Active spike cell determination:
                # Check current cell, target cell, or up to 2 cells ahead in planned path
                spk_cell = None
                if c_cur in self.cell_spike_zones[e]:
                    spk_cell = c_cur
                elif c_tgt in self.cell_spike_zones[e]:
                    spk_cell = c_tgt
                elif self.current_cell_paths[e] is not None:
                    for cp in self.current_cell_paths[e][:3]:
                        cp_t = tuple(cp)
                        if cp_t in self.cell_spike_zones[e]:
                            spk_cell = cp_t
                            break

                if spk_cell is not None:
                    spk = self.cell_spike_zones[e][spk_cell]
                    axis = spk.get("axis", 0)
                    di_h, dj_h = DIRS[int(self.heading[e].item())]
                    s_dir = di_h if axis == 0 else dj_h
                    if s_dir != 0:
                        si, sj = spk_cell
                        u_r = (pos_xy[e, 0].item() - si * self.c) if axis == 0 else (pos_xy[e, 1].item() - sj * self.c)
                        v_r = (pos_xy[e, 1].item() - sj * self.c) if axis == 0 else (pos_xy[e, 0].item() - si * self.c)
                        u_prog = u_r * s_dir

                        # Active corridor window: from approach (-1.45 m) through cell exit (+0.95 m)
                        if -1.45 <= u_prog <= 0.95:
                            # Corridor travel heading
                            if axis == 0:
                                corridor_yaw = 0.0 if s_dir > 0 else math.pi
                            else:
                                corridor_yaw = (math.pi / 2.0) if s_dir > 0 else (-math.pi / 2.0)

                            kind = spk.get("kind", "lane")
                            v_along = float(self.speed)
                            max_lead_deg = 15.0
                            is_crossing = False

                            if kind == "lane":
                                v_lat = 0.44
                                v_raw = spk.get("v_safe", v_lat)
                                v_safe = v_lat if v_raw >= 0 else -v_lat
                                v_des_lat = v_safe

                                err_v = v_des_lat - v_r
                                abs_err_v = abs(err_v)
                                if abs_err_v > 0.25:
                                    v_along = 0.22
                                    fast_decel_mask[e] = True
                                elif abs_err_v > 0.12:
                                    v_along = 0.32
                                else:
                                    v_along = float(self.speed)

                            elif kind == "slalom":
                                v_lat = 0.50
                                v_m = spk.get("v_minus", v_lat)
                                v_p = spk.get("v_plus", -v_lat)
                                v_minus = v_lat if v_m >= 0 else -v_lat
                                v_plus = v_lat if v_p >= 0 else -v_lat
                                if v_minus == v_plus:
                                    v_plus = -v_minus
                                v_entry = v_minus if s_dir > 0 else v_plus
                                v_exit = v_plus if s_dir > 0 else v_minus

                                in_exit_lane = abs(v_r - v_exit) <= 0.15
                                if u_prog < -0.50:
                                    v_des_lat = v_entry
                                    v_along = float(self.speed)
                                    is_crossing = False
                                    in_post_exit = False
                                elif u_prog < -0.25:
                                    v_des_lat = v_entry
                                    v_along = 0.30
                                    fast_decel_mask[e] = True
                                    is_crossing = False
                                    in_post_exit = False
                                elif in_exit_lane:
                                    v_des_lat = v_exit
                                    is_crossing = False
                                    in_post_exit = True
                                else:
                                    v_des_lat = v_exit
                                    is_crossing = True
                                    in_post_exit = False

                                err_v = v_des_lat - v_r
                                abs_err_v = abs(err_v)
                            else:
                                v_des_lat = 0.0
                                err_v = 0.0
                                abs_err_v = 0.0
                                in_post_exit = False

                            psi_cur = float(yaw[e].item())
                            if kind == "slalom" and is_crossing:
                                fast_decel_mask[e] = True
                                s_across = 1.0 if err_v > 0 else -1.0
                                if axis == 0:
                                    psi_cross_raw = (math.pi / 2.0) if s_across > 0 else (-math.pi / 2.0)
                                    delta_psi = 1.6 * (u_prog - 0.0) * s_dir * s_across
                                else:
                                    psi_cross_raw = 0.0 if s_across > 0 else math.pi
                                    delta_psi = -1.6 * (u_prog - 0.0) * s_dir * s_across

                                psi_des = psi_cross_raw + max(-0.25, min(0.25, delta_psi))
                                yaw_err = math.atan2(math.sin(psi_des - psi_cur), math.cos(psi_des - psi_cur))
                                turn_cap = 1.8
                                w_des[e] = max(-turn_cap, min(turn_cap, 3.2 * yaw_err))

                                if abs(yaw_err) > math.radians(12.0):
                                    v_des[e] = 0.00  # Zero-drift in-place pivot
                                else:
                                    v_des[e] = 0.32  # Walk steadily across gap to exit lane

                            elif kind == "slalom" and in_post_exit:
                                # Lateral crossing complete! Re-align with corridor heading and walk straight out!
                                psi_des = corridor_yaw
                                yaw_err = math.atan2(math.sin(psi_des - psi_cur), math.cos(psi_des - psi_cur))
                                turn_cap = 1.8
                                w_des[e] = max(-turn_cap, min(turn_cap, 3.2 * yaw_err))

                                if abs(yaw_err) > math.radians(12.0):
                                    v_des[e] = 0.00  # Pivot back to corridor heading
                                    fast_decel_mask[e] = True
                                else:
                                    v_des[e] = float(self.speed)  # Walk straight past Bed 2 at full speed!

                            else:
                                # Normal lane corridor traversal
                                v_across = max(-0.28, min(0.28, 1.4 * err_v))
                                if axis == 0:
                                    v_wx = s_dir * v_along
                                    v_wy = v_across
                                else:
                                    v_wy = s_dir * v_along
                                    v_wx = v_across

                                psi_motion = math.atan2(v_wy, v_wx)
                                d_lead = math.atan2(math.sin(psi_motion - corridor_yaw), math.cos(psi_motion - corridor_yaw))
                                max_lead = math.radians(max_lead_deg)
                                d_lead_clamped = max(-max_lead, min(max_lead, d_lead))
                                psi_des = corridor_yaw + d_lead_clamped

                                yaw_err = math.atan2(math.sin(psi_des - psi_cur), math.cos(psi_des - psi_cur))
                                w_des[e] = max(-self.max_turn, min(self.max_turn, 2.6 * yaw_err))
                                v_des[e] = max(0.20, min(self.speed, v_along))

                            v_y_des[e] = 0.0

        # Rate limits [m/s^2], [rad/s^2]
        # Fast deceleration (3.6 m/s^2) and fast angular accel (6.0 rad/s^2) when entering slalom gap
        for e in range(self.num_envs):
            dv_x = v_des[e] - self.cmd[e, 0]
            max_dec = 3.6 if (dv_x < 0 and fast_decel_mask[e]) else self.max_acc
            self.cmd[e, 0] += torch.clamp(dv_x, -max_dec * dt, self.max_acc * dt)

            dv_y = v_y_des[e] - self.cmd[e, 1]
            self.cmd[e, 1] += torch.clamp(dv_y, -self.max_acc * dt, self.max_acc * dt)

            dw_z = w_des[e] - self.cmd[e, 2]
            max_a_acc = 6.0 if fast_decel_mask[e] else self.max_ang_acc
            self.cmd[e, 2] += torch.clamp(dw_z, -max_a_acc * dt, max_a_acc * dt)

        return self.cmd.clone()

    # ------------------------------------------------------------------ main step
    def step(self, pos_xy: torch.Tensor, yaw: torch.Tensor, walls_now: torch.Tensor, dt: float) -> torch.Tensor:
        """Update the plan and return the velocity command. ``walls_now`` = sensed walls of the current cell."""
        cell = self.cell_of(pos_xy)
        near_centre = torch.linalg.norm(pos_xy - self.centre_of(cell), dim=1) < self.turn_early
        arrived = near_centre & (cell == self.target).all(dim=1)
        # a turn picked early sees the far wall of the cell only at the edge of the scan: if the side we are
        # heading out of turns out to be a wall while we are still in that cell, decide again
        still_there = (cell == self.origin).all(dim=1) & ~self.need_decision
        surprise = still_there & walls_now.gather(1, self.heading.unsqueeze(1)).squeeze(1)
        surprise &= self.walls[torch.arange(len(cell), device=self.device), cell[:, 0], cell[:, 1], self.heading] == 0
        # Check if currently planned path is blocked by a wall discovered ahead
        plan_blocked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.use_lidar_lookahead and self.use_astar:
            for e in range(self.num_envs):
                c_path = self.current_cell_paths[e]
                if c_path is not None and len(c_path) >= 2:
                    for idx_p in range(len(c_path) - 1):
                        c1, c2 = c_path[idx_p], c_path[idx_p + 1]
                        di = c2[0] - c1[0]
                        dj = c2[1] - c1[1]
                        for d_idx, ddir in enumerate(DIRS):
                            if ddir == (di, dj):
                                if self.walls[e, c1[0], c1[1], d_idx] == 1:
                                    plan_blocked[e] = True
                                break
                        if plan_blocked[e]:
                            break

        decide = self.need_decision | arrived | surprise | plan_blocked | self.replan_needed
        if decide.any():
            idx = torch.nonzero(decide).squeeze(1)
            # remember what we saw (and the same wall from the neighbour's side)
            # Local 3D height scanner (scan_z) is the ultimate ground truth for cell[idx]
            for d, (di, dj) in enumerate(DIRS):
                ci, cj = cell[idx, 0], cell[idx, 1]
                val = walls_now[idx, d].to(torch.int8)
                self.walls[idx, ci, cj, d] = val
                ni, nj = ci + di, cj + dj
                inside = (ni >= 0) & (ni < self.n) & (nj >= 0) & (nj < self.n)
                k = idx[inside]
                self.walls[k, ni[inside], nj[inside], (d + 2) % 4] = val[inside]
            open_sides = self.walls[idx, cell[idx, 0], cell[idx, 1]] == 0
            if self.debug and decide[0]:
                print(
                    f"[PILOT] robot0 cell={cell[0].tolist()} sensed={walls_now[0].int().tolist()} "
                    f"open={open_sides[0].int().tolist()}",
                    flush=True,
                )
            moved = ~self.need_decision[idx] & ~surprise[idx] & ~plan_blocked[idx] & ~self.replan_needed[idx]
            self.cells_walked[idx] += moved.long()

            if self.use_astar and self.strategy == "flood":
                for e in idx.tolist():
                    c_path = self.plan_full_path(cell[e].tolist(), e)
                    self.current_cell_paths[e] = c_path
                    if len(c_path) > 1:
                        next_c = c_path[1]
                        di = next_c[0] - cell[e, 0].item()
                        dj = next_c[1] - cell[e, 1].item()
                        for d_idx, ddir in enumerate(DIRS):
                            if ddir == (di, dj):
                                self.heading[e] = d_idx
                                self.target[e] = torch.tensor(next_c, device=self.device)
                                break
                        if len(c_path) > 2:
                            next_c2 = c_path[2]
                            di2 = next_c2[0] - next_c[0]
                            dj2 = next_c2[1] - next_c[1]
                            for d_idx2, ddir in enumerate(DIRS):
                                if ddir == (di2, dj2):
                                    self.next_heading[e] = d_idx2
                                    break
                        else:
                            self.next_heading[e] = self.heading[e]
                    wpts, curvs, tangs = self.interpolate_path(c_path, return_derivatives=True, env_id=e)
                    self.smooth_wpts[e] = wpts
                    self.smooth_curvatures[e] = curvs
                    self.smooth_tangents[e] = tangs
                    self.replan_needed[e] = False
            else:
                d = self.choose_direction(cell[idx], open_sides, self.heading[idx], self.walls[idx])
                self.heading[idx] = d
                self.target[idx] = (cell[idx] + self._dirs[d]).clamp(0, self.n - 1)
                self.next_heading[idx] = self.peek_next_direction(self.target[idx], d, self.walls[idx])
                for e in idx.tolist():
                    self.replan_needed[e] = False

            self.origin[idx] = cell[idx]
            self.need_decision[idx] = False

        cmd = self.velocity_command(pos_xy, yaw, dt)
        return cmd

    def reset(self, env_ids: torch.Tensor, fall_cell: tuple[int, int] | None = None):
        """Forget or retain map for the given robots (e.g. after a fall)."""
        if not self.persist_map_on_reset:
            self.walls[env_ids] = -1
            self.walls[env_ids, -1, :, 0] = 1
            self.walls[env_ids, :, -1, 1] = 1
            self.walls[env_ids, 0, :, 2] = 1
            self.walls[env_ids, :, 0, 3] = 1
            self.wall_hit_counts[env_ids] = 0
            self.cell_hazard_cost[env_ids] = 0.0
            for e in env_ids.tolist():
                self.cell_spike_zones[e].clear()
                self.cell_ramp_zones[e].clear()
                self.cell_gap_zones[e].clear()
        elif fall_cell is not None:
            self.cell_hazard_cost[env_ids, fall_cell[0], fall_cell[1]] += 30.0

        start = torch.tensor(self.start, dtype=torch.long, device=self.device)
        self.target[env_ids] = start
        self.origin[env_ids] = start
        self.heading[env_ids] = 0
        self.next_heading[env_ids] = -1
        self.need_decision[env_ids] = True
        self.replan_needed[env_ids] = True
        self.cells_walked[env_ids] = 0
        self.cmd[env_ids] = 0.0
        for e in env_ids.tolist():
            self.smooth_wpts[e] = None
            self.smooth_curvatures[e] = None
            self.smooth_tangents[e] = None
            self.current_cell_paths[e] = None

    def map_ascii(self, env_id: int = 0) -> str:
        """Draw the walls robot ``env_id`` has seen: '-'/'|' wall, '?' never looked at."""
        w = self.walls[env_id].cpu()
        n = self.n

        def ch(v, wall):
            return wall if v == 1 else ("?" if v == -1 else " ")

        rows = []
        for j in range(n - 1, -1, -1):
            rows.append("+" + "".join(ch(w[i, j, 1], "-") + "+" for i in range(n)))
            line = ch(w[0, j, 2], "|")
            for i in range(n):
                label = "S" if (i, j) == self.start else "G" if (i, j) == self.goal else " "
                line += label + ch(w[i, j, 0], "|")
            rows.append(line)
        rows.append("+" + "".join(ch(w[i, 0, 3], "-") + "+" for i in range(n)))
        return "\n".join(rows)

    reset_envs = reset


class A4_Hazard_G2(MazeAutopilot):
    """Preset configuration for A4_Hazard_G2 winner autopilot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_acc = 1.40
        self.max_ang_acc = 3.00
        self.direct_corner = True
        self.lateral_margin = True
        self.use_astar = True
        self.curve_mode = "g2_quintic"
        self.r_fillet = 0.65
        self.lookahead_dist = 0.50
        self.use_lidar_lookahead = True
        self.use_hazard_avoidance = True
        self.spike_avoid_mode = "slalom_s_curve"
        self.use_fork_arbitrator = True
        self.persist_map_on_reset = True

