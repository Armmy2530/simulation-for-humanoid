"""High-Performance Hazard Maze Autopilot for Challenge 7.5.

Wraps the Track A winner: A4_Hazard_G2
- G2 Quintic Hermite Spline path smoothing with curvature feedforward
- 6m multi-cell LiDAR corridor raycast lookahead mapping
- 3D Height-Scan Sub-Cell 2-Bed Slalom & Single-Lane S-Curve Contouring
- Canadian Traveler Start-Fork Arbitrator with Hazard-Aware A* replanning
- Persistent map retention with fall-cell penalty
"""

from __future__ import annotations

import os
import torch
from g1_maze_lab.autopilot import MazeAutopilot, A4_Hazard_G2


class Pilot:
    """Challenge 7.5 Hazard Maze Autopilot (A4_Hazard_G2)."""

    def __init__(
        self,
        num_envs: int,
        num_cells: int,
        cell_size: float,
        device: str,
        start_cell: tuple[int, int] = (0, 0),
        goal_cell: tuple[int, int] | None = None,
    ):
        self.num_envs = num_envs
        self.n = num_cells
        self.c = cell_size
        self.device = device
        self.start = tuple(int(v) for v in start_cell)
        self.goal = (num_cells - 1, num_cells - 1) if goal_cell is None else tuple(int(v) for v in goal_cell)

        speed = float(os.environ.get("MAZE_SPEED", "0.60"))
        min_speed = float(os.environ.get("MAZE_MIN_SPEED", "0.30"))
        turn_early = float(os.environ.get("MAZE_TURN_EARLY", "0.50"))
        max_turn = float(os.environ.get("MAZE_MAX_TURN", "1.00"))

        self.ap = A4_Hazard_G2(
            num_envs=num_envs,
            num_cells=num_cells,
            cell_size=cell_size,
            device=device,
            strategy="flood",
            speed=speed,
            turn_early=turn_early,
            min_speed=min_speed,
            max_turn=max_turn,
            start=self.start,
            goal=self.goal,
        )
        self.ap.max_acc = 1.40
        self.ap.max_ang_acc = 3.00
        self.ap.direct_corner = True
        self.ap.lateral_margin = True
        self.ap.use_astar = True
        self.ap.curve_mode = "g2_quintic"
        self.ap.r_fillet = 0.65
        self.ap.lookahead_dist = 0.50
        self.ap.use_lidar_lookahead = True
        self.ap.use_hazard_avoidance = True
        self.ap.spike_avoid_mode = "slalom_s_curve"
        self.ap.use_fork_arbitrator = True
        self.ap.persist_map_on_reset = True

    def act(self, obs: dict) -> torch.Tensor:
        pos_xy = obs["pos_xy"]
        yaw = obs["yaw"]
        dt = obs["dt"]
        scan_xy = obs["scan_xy"]
        scan_z = obs["scan_z"]
        cell = self.ap.cell_of(pos_xy)

        # 1. Chest LiDAR lookahead
        if "lidar_dist" in obs and "lidar_yaw" in obs:
            self.ap.update_walls_from_lidar_raycast(pos_xy, obs["lidar_dist"], obs["lidar_yaw"])

        # 2. Hazard perception: trenches, steep ramps, and spike beds
        self.ap.update_hazards_from_height_scan(pos_xy, scan_xy, scan_z)

        # 3. Local cell wall perception
        walls_now = self.ap.walls_from_height_scan(scan_xy, scan_z, cell)

        # 4. Controller step
        cmd = self.ap.step(pos_xy, yaw, walls_now, dt)
        return cmd

    def reset(self, env_ids: torch.Tensor):
        self.ap.reset(env_ids)
