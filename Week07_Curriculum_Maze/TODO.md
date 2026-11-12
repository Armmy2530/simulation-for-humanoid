# Humanoid Maze Challenge 7.5 — R&D Master Roadmap & Todo List

> **Experimental Governance**:
> - **Track A (Path Planning & Topological Exploration Benchmark)**: Strictly lock `policy.pt = pretrained/g1_rough/model_pretrained.pt` and velocity/acceleration bounds (`MAZE_SPEED=0.60`, `MAZE_MIN_SPEED=0.30`, `MAZE_MAX_TURN=1.00`, `MAZE_ACC="1.40,3.00"`). Compare candidate planning algorithms purely on path geometry, topological discovery, obstacle clearance, and momentum preservation.
> - **Track B (Locomotion, Speed & RL Policy Optimization)**: Lock the winning Track A planner and systematically optimize dynamic speed profiles, ZMP cornering limiters, slip recovery, and RL policy fine-tuning.
> - **Artifact Standard**: Every experiment follows [`artifact_file_structor.md`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/artifact_file_structor.md) (`README.md` + `*.png` at experiment root; `*.npz` + `*.mp4` inside `run/`).

---

## Phase 0: Project Standards, Literature Review & Simulation Acceleration
- [x] **0.1 Artifact Directory Standard**: Establish [`artifact_file_structor.md`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/artifact_file_structor.md) and master catalog [`artifacts/README.md`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/README.md).
- [x] **0.2 Git Hygiene**: Configure [`.gitignore`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/.gitignore) to track only source code and ignore `artifacts/`, `*.pt`, `*.onnx`, `*.npz`, and `*.mp4`.
- [x] **0.3 Simulator & Draft Audit**: Audit [`hazard_maze.py`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/g1_maze_lab/src/g1_maze_lab/hazard_maze.py), [`challenge.py`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/g1_maze_lab/scripts/challenge.py), [`challenge_server.py`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/g1_maze_lab/scripts/challenge_server.py), and [`autopilot.py`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/g1_maze_lab/src/g1_maze_lab/autopilot.py); identify the 5 critical competition gaps (G1–G5) and establish the Two-Track Experimental Protocol.
- [x] **0.4 Fast Vectorized Benchmark Harness (`--num_envs K` + Zero-Sync GPU Logging)**: Upgrade [`benchmark_autopilot.py`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/g1_maze_lab/scripts/benchmark_autopilot.py) to evaluate all $K$ candidate pilots simultaneously in parallel environments (`--num_envs 4`) with pre-allocated GPU logging buffers.
  - **Result**: Evaluated all 4 candidates in a single simulation pass per map, sharing the maze mesh and RayCaster BVH, reducing 16-run wall-clock time from ~1,840 s to ~520 s (~3.5× acceleration) with zero per-step CPU-GPU synchronization.

---

## Track A: Path Planning & Topological Exploration Benchmark (Locked Speed @ `0.60 m/s`)

- [x] **Experiment 07 — Geometric Cornering & Curvature Continuity (`A0` vs `A1` vs `A2_G1` vs `A2_G2`)**
  - **Folder**: [`artifacts/experiment_07_track_a_g2_spline/`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_07_track_a_g2_spline/)
  - **Sub-Agent**: `[Exp 07 Track A: Fixed-Speed Path Planning Benchmark (A0 vs A1 vs A2 G...](t3-thread://v2/c519f0e9-dc13-4a60-acbc-71daae372a9a/thread:delegated-task:command%3Amcp%3A31d2b30c-c2fe-4c30-a726-4429ad8384fd%3Adelegate-task%3A380c9d6c-839c-4bc8-b45e-26504d2c90f3)`
  - **Result**: `A2_G2` ($G^2$ Quintic Hermite Spline + Curvature Feedforward $\omega_{\text{des}} = v_{\text{des}}\kappa_{\text{path}} + 2.0\psi_{\text{err}}$) won across every metric on the $5\times 5$ Challenge Maze: **`32.56 s`** (vs `33.46 s` for `A1`, `32.94 s` for `A2_G1`, `38.60 s` for `A0`), shortest path **`17.55 m`**, lowest yaw jerk **`5.32 rad/s²`**, and highest wall clearance **`0.878 m`**.

- [x] **Experiment 08 — Multi-Cell 6m Chest LiDAR Ray-Cast Mapping on Unseen Random Mazes (`A1` vs `A2_G2` vs `A3_Lidar_G2`)**
  - **Folder**: [`artifacts/experiment_08_track_a_lidar_exploration/`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_08_track_a_lidar_exploration/)
  - **Sub-Agent**: `[Exp 08 Track A: 6m LiDAR Ray-Cast Mapping + G2 Spline on Random Mazes...](t3-thread://v2/c519f0e9-dc13-4a60-acbc-71daae372a9a/thread:delegated-task:command%3Amcp%3A31d2b30c-c2fe-4c30-a726-4429ad8384fd%3Adelegate-task%3Ac267a9e7-f2fb-47d2-b8d2-d4f89fb8a59e)`
  - **Result**: `A3_Lidar_G2` mapped corridor walls up to 3 cells ahead and pruned dead-end branches before entry, winning all 3 unseen procedural seeds (Seed 42: **`32.48 s`**, Seed 100: **`24.82 s`** vs `56.98 s` for `A1` — a **`-56.4%`** time reduction, Seed 777: **`30.78 s`**). Macro-average: **`29.36 s` / `15.73 m`** vs `41.57 s / 21.99 m` for `A1`.

- [x] **Experiment 09 — Arbitrary Non-Corner Start & Goal Generalization (`--random-ends` across 4 Quadrants)**
  - **Folder**: [`artifacts/experiment_09_track_a_random_ends/`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_09_track_a_random_ends/)
  - **Sub-Agents**:
    - Initial Run: `[Exp 09 Track A: Randomized Start & Goal Benchmark Across 4 Mazes](t3-thread://v2/c519f0e9-dc13-4a60-acbc-71daae372a9a/thread:delegated-task:command%3Amcp%3A31d2b30c-c2fe-4c30-a726-4429ad8384fd%3Adelegate-task%3Aee49b6f0-9d3d-4c43-9461-b9ff5c5d9d57)`
    - Refinement & Audit Fixes: `[Exp 09 Refinement: Fix A* Turn Penalty & LiDAR Grazing Bugs on Random...](t3-thread://v2/c519f0e9-dc13-4a60-acbc-71daae372a9a/thread:delegated-task:command%3Amcp%3A31d2b30c-c2fe-4c30-a726-4429ad8384fd%3Adelegate-task%3Acb642f7d-44ff-48e9-8e59-c284033acf49)`
  - **Result**: With `turn_penalty = 0.05` tie-breaking, LiDAR grazing incidence filters, and `in_place_scale` pivot gating (`>= 95°` zero-drift turn in place), `A3_Lidar_G2` achieved strict monotonic superiority on Map 1 (**`22.00 s` / `12.00 m`**), Map 2 (**`83.20 s` / `44.59 m`**), and Map 4 (**`32.56 s` / `17.83 m`**), and won the overall 4-map macro-average (**`44.11 s` / `23.77 m`** vs `45.59 s / 24.11 m` for `A1` and `49.90 s` for `A2_G2`).

- [x] **Experiment 10 — Hazard Maze & Slalom Contouring + Canadian Traveler Fork Arbitrator (`A4_Hazard_G2`)**
  - **Folder**: [`artifacts/experiment_10_track_a_hazard_slalom/`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_10_track_a_hazard_slalom/)
  - **Sub-Agents**:
    - Phase 0.4 + Exp 10 Main Benchmark: `[Phase 0.4 Fast Multi-Env Harness + Exp 10 Track A Hazard & Slalom Ben...](t3-thread://v2/c519f0e9-dc13-4a60-acbc-71daae372a9a/thread:delegated-task:command%3Amcp%3Ad72bda79-e155-47c5-99ef-786d5c441422%3Adelegate-task%3Aexp10-phase04-hazard-slalom-01)`
    - Exp 10b (Isolated Spike+Slalom Map + Method 1): `[Exp 10b: Simple Spike+Slalom Isolation Map & Method 1 Benchmark](t3-thread://v2/c519f0e9-dc13-4a60-acbc-71daae372a9a/thread:delegated-task:command%3Amcp%3Ae63002a8-d479-4566-8d8b-4fd900198dff%3Adelegate-task%3Aexp10b-spike-slalom-isolation-test-01)`
    - Exp 10c (Perpendicular Gap Crossing + Newton 3D Video): `[Exp 10c: Holonomic vy Slalom Crossing + Export 3D Video on Spike Course](t3-thread://v2/c519f0e9-dc13-4a60-acbc-71daae372a9a/thread:delegated-task:command%3Amcp%3A02e01d8c-5d8d-46b8-9ab5-fcdf3fadbfc3%3Adelegate-task%3Aexp10c-holonomic-spike-video-01)`
    - Exp 10d (Full Level 7 / Level 9 Re-Benchmark): `[Exp 10d: Re-Benchmark Upgraded A4_Hazard_G2 on Full Level 7 & Level 9...](t3-thread://v2/c519f0e9-dc13-4a60-acbc-71daae372a9a/thread:delegated-task:command%3Amcp%3A6972c8eb-a39b-4063-b10f-4d11960aec70%3Adelegate-task%3Aexp10d-full-l7-l9-hazard-retest-01)`
    - Exp 10e (Level 7 Direct Short-Route Traversal + Video): `[Exp 10e: Fix Short-Route Ramp/Trench/Rough Alignment + Run A4_Hazard_...](t3-thread://v2/c519f0e9-dc13-4a60-acbc-71daae372a9a/thread:delegated-task:command%3Amcp%3A443beac1-a8e1-40d4-bc62-a0a9cfc509bb%3Adelegate-task%3Aexp10e-level7-short-route-video-01)`
  - **Result**:
    - **Direct Short-Route Traversal on Level 7 (`Exp 10e`)**: With `cell_ramp_zones` and `cell_gap_zones` centerline heading lock (`|wz| <= 0.06 rad/s`), trench momentum stride (`0.85 m/s`), and `vy = 0` non-holonomic control, `A4_Hazard_G2` walks **straight right `(0, 0) -> (1, 0) -> (2, 0)...` along the Short Route** without reversing, reaching GOAL in **`58.1 s` (`0 falls`)** on `seed=2026` (down from `106.12 s`, a **`-45.2%`** time reduction!) and **`56.3 s` (`0 falls`)** on `seed=42` (down from `140.02 s`, **`-59.8%`**!).
    - **Full Level 7 / Level 9 Hazard Mazes (Re-Verified Exp 10d)**: `A4_Hazard_G2` achieved **100% mission completion (4/4 goals reached) with a perfect 0 falls / 0 resets across all 4 Level 7 and Level 9 mazes** (Map 1 `L7_seed2026`: **`106.12 s` / 0 falls**, Map 2 `L7_seed42`: **`140.02 s` / 0 falls**, Map 3 `L9_seed2026`: **`94.16 s` / 0 falls**, Map 4 `L9_seed42_rand`: **`119.42 s` / 0 falls**). `A4_Slalom_Only` also achieved 4/4 goals (175.76 s, 175.06 s, 94.86 s, 137.78 s) with 0 falls on both Level 9 mazes. Baselines `A1` and `A3` suffered DNF on all maps with 7 to 16 falls each.
    - **Isolated Spike & 2-Bed Opposing Slalom Courses (`Course A` & `Course B`)**: Upgrading Method 1 (immediate `replan_needed` on spike discovery + persistent lane memory + extended $[-1.05, +1.05]\text{ m}$ waypoints) with **3-Phase Perpendicular In-Place Slalom Gap Crossing & Closed-Loop Longitudinal Centering** achieved **100% hazard clearance (4/4 beds, 0 falls/resets)** on both `Course A` (**`69.26 s`**) and `Course B` (**`66.50 s`**), whereas all baselines had 20–22 resets and 0/4 cleared.
    - **Newton 3D Videos**: Recorded [`artifacts/experiment_10_track_a_hazard_slalom/run/level7_short_route_walk.mp4`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_10_track_a_hazard_slalom/run/level7_short_route_walk.mp4) and [`artifacts/experiment_10_track_a_hazard_slalom/run/spike_course_walk.mp4`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_10_track_a_hazard_slalom/run/spike_course_walk.mp4).
  - **Crown Winner of Track A**: `A4_Hazard_G2` is locked as the baseline planner for Track B locomotion optimization.

---

## Track B: Speed, Locomotion & RL Policy Optimization (Locked Winning Track A Planner)

- [ ] **Experiment 11 — LIPM Centripetal ZMP Velocity Profiler & Corridor Sprint (`B1_Dynamic_Speed`)**
  - **Folder**: `artifacts/experiment_11_track_b_zmp_speed_profile/`
  - **Scope**: Lock the winning Track A planner (`A4_Hazard_G2`) and replace fixed speed (`0.60 m/s`) with an analytical curvature-bounded velocity profile:
    $$v_{\text{des}}(s) = \min\left(v_{\max},\; \sqrt{\frac{a_{y,\max}}{\max(|\kappa(s)|, \kappa_{\min})}}\right)$$
    Test increasing straight-line sprint speeds ($0.85 \to 1.15\text{ m/s}$) while constraining centripetal acceleration $a_{y,\max}$ on $G^2$ spline turns so the G1 never tips over.

- [ ] **Experiment 12 — Proprioceptive Ice Slip Detector & Trench/Ramp Traversal Controller (`B2_Hazard_Reflex`)**
  - **Folder**: `artifacts/experiment_12_track_b_hazard_reflex/`
  - **Scope**:
    1. **Ice Slip Reflex**: Since ice is only $1\text{ cm}$ tall (`0.01 m` in `scan_z`), detect low-friction patches via velocity-tracking residual $e_v = \|\mathbf{v}_{xy,\text{body}} - \mathbf{v}_{\text{cmd}}\|$ + yaw-rate residual and clamp lateral acceleration / yaw rate over ice.
    2. **Trench ($15\text{ cm}$ gap) & Ramp ($\pm 15^\circ$) Alignment**: Square up heading orthogonal to the hazard axis and apply a controlled momentum burst across the $15\text{ cm}$ gap.

- [ ] **Experiment 13 — Surgical RL Locomotion Fine-Tuning (`B3_RL_Policy`)**
  - **Folder**: `artifacts/experiment_13_track_b_rl_finetune/`
  - **Scope**: Warm-start from `pretrained/g1_rough/model_pretrained.pt` in `g1_hazard_starter` and fine-tune with exact Challenge 7.5 terrain primitives ($15^\circ$ trapezoid ramps, $15\text{ cm}$ trenches, randomized friction $[0.05, 1.0]$) and higher velocity command ranges ($v_x \in [-0.6, 1.4]\text{ m/s}$). Export final `policy.pt`.

- [ ] **Experiment 14 — Final Competition Dress Rehearsal (`10x10` Final Level, `--random-ends`, Hidden Seeds)**
  - **Folder**: `artifacts/experiment_14_final_competition_benchmark/`
  - **Scope**: Full end-to-end evaluation of [`pilot.py`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/pilot.py) + `policy.pt` via [`challenge.py`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/g1_maze_lab/scripts/challenge.py) on Level 7, Level 9, and Final Level ($10\times 10$, all hazards, `--random-ends`) across multiple unseen seeds.
