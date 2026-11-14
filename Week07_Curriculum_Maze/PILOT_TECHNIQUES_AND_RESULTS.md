# G1 Maze Challenge 7.5 — Latest Pilot (`A4_Hazard_G2`): Techniques, Results, Run & Submission Guide

This document summarizes everything built in **Week 07 (`Week07_Curriculum_Maze`)**, the techniques inside our latest standalone [`pilot.py`](pilot.py) and [`policy.pt`](policy.pt), all benchmark results, and step-by-step instructions to run and submit for **Challenge 7.5**.

> **Ready-to-Submit Files at Root (`Week07_Curriculum_Maze/`)**:
> - [`pilot.py`](pilot.py) (`76 KB`) — 100% self-contained autopilot (`class Pilot` wrapping `A4_Hazard_G2`), verified with `challenge_server.check_python`.
> - `policy.pt` (`7.8 MB`) — Pretrained `rsl_rl` G1 Rough walking checkpoint (`g1_maze_lab/pretrained/g1_rough/model_pretrained.pt`), verified with `challenge_server.check_policy`. *(Ignored by `.gitignore`; copy from `g1_maze_lab/pretrained/g1_rough/model_pretrained.pt` after a fresh `git clone`.)*
> - Pre-installed copy: [`g1_maze_lab/submissions/pro_pilot/`](g1_maze_lab/submissions/pro_pilot/)

---

## 1. Summary of What We Have Done

1. **System Audit & Vectorized Multi-Env Benchmark Harness (`Phase 0`)**:
   - Upgraded [`g1_maze_lab/scripts/benchmark_autopilot.py`](g1_maze_lab/scripts/benchmark_autopilot.py) with `--num_envs K`, `--batch-json`, shared spatial BVH, and zero-sync GPU telemetry logging (~3.5× faster benchmarking).
   - Organized all notes, `.png` plots, `.npz` telemetry, and `.mp4` videos under [`artifacts/`](artifacts/) (see [`artifacts/README.md`](artifacts/README.md)) and updated [`.gitignore`](.gitignore) so the repo cleanly tracks source code, markdown docs, plots, and videos.
2. **Track A Path Planning Evolution (`Exp 01 – Exp 10e` @ Locked `0.60 m/s` Nominal Speed)**:
   - **Exp 01–06**: Evaluated baseline stop-and-pivot (`41.42 s`) vs. corner-cutting (`27.64 s`), quadratic Bézier arcs (`24.46 s`), and early hazard prototypes.
   - **Exp 07 (`A2_G2`)**: Replaced $G^1$ corners with $G^2$ Quintic Hermite splines + curvature feedforward, reducing yaw jerk and achieving `32.56 s` at `0.60 m/s`.
   - **Exp 08 (`A3_Lidar_G2`)**: Added 6m 72-ray chest LiDAR ray-casting to map corridors up to 3 cells ahead and prune dead ends before entering (`-56.4%` time on Seed 100).
   - **Exp 09 (`--random-ends`)**: Generalized A\* planning to arbitrary non-corner start/goal cells with `0.05` turn tie-breaking, incidence-angle LiDAR filtering, and zero-drift in-place pivots (`>= 95°`).
   - **Exp 10 / 10b / 10c / 10d / 10e (`A4_Hazard_G2`)**: Added online 3D height-scan hazard classification, persistent spike memory with pre-entry lead-in/lead-out waypoints, **3-phase perpendicular in-place slalom gap crossing**, and **ramp/trench centerline alignment** — conquering both full Level 7/9 hazard mazes (`4/4 GOAL, 0 falls`) and the direct Level 7 Short Route (`56.3–58.1 s, 0 falls`).
3. **Standalone Submission Packaging**:
   - Exported `A4_Hazard_G2` + `class Pilot` into a single dependency-free [`pilot.py`](pilot.py) alongside `policy.pt` at the root of `Week07_Curriculum_Maze/` and inside [`g1_maze_lab/submissions/pro_pilot/`](g1_maze_lab/submissions/pro_pilot/).

---

## 2. Techniques Used in the Latest Pilot (`pilot.py` + `policy.pt`)

### 2.1 Walking Locomotion Policy (`policy.pt`)
- **Model**: Pretrained `rsl_rl` **G1 Rough** policy (`g1_maze_lab/pretrained/g1_rough/model_pretrained.pt`).
- **Command Interface**: Receives `(v_x, v_y, w_z)` in the robot base frame at 50 Hz.
- **Key Physical Constraint**: Because `G1 Rough` was trained with $v_y = 0$, commanding lateral strafe ($v_y \neq 0$) causes leg crossing and instability. Our latest pilot strictly locks **`v_y_des = 0.0` everywhere** and executes all turns and lateral lane shifts via heading ($\psi$) and yaw rate ($\omega_z$).

### 2.2 Navigation & Planning Core (`A4_Hazard_G2` in `pilot.py`)
1. **A\* Planner with Turn Tie-Breaking & Persistent Map Memory**:
   - Plans dynamically over the online-discovered wall grid (`self.walls`) from any `start_cell` to `goal_cell` with a `0.05` turn penalty to avoid unnecessary zigzags.
   - `persist_map_on_reset = True`: Retains discovered walls, spike zones, ramp zones, and trench zones across resets (if a fall occurs, adds `+30.0` hazard cost to that cell).
2. **$G^2$ Quintic Hermite Splines + Curvature Feedforward (`Exp 07`)**:
   - Smooths corners (`r_fillet = 0.65 m`, tightened to `0.12 m` near spike cells) using $G^2$-continuous quintic Hermite polynomials ($\kappa(0)=\kappa(1)=0$).
   - Computes analytical path curvature $\kappa(s)$ and applies feedforward yaw control:
     $$\omega_{\text{des}} = \text{clamp}\left(v_{\text{des}}\,\kappa_{\text{ff}} + 2.0\,\psi_{\text{err}},\; -\omega_{\max},\; +\omega_{\max}\right)$$
   - Uses pure pursuit (`lookahead_dist = 0.50 m`, tightened to `0.42 m` near spikes) and zero-drift in-place pivot gating (`in_place_scale -> 0` when $|\psi_{\text{err}}| \ge 95^\circ$).
3. **6m 72-Ray Chest LiDAR Far-Vision Ray-Casting & Bug Fixes (`Exp 08` + Far-Vision Overhaul)**:
   - **Bug A Fix (Max-Range 6.0m Free-Space Marching)**: Previously, `valid_dist = (lidar_dist < 5.8)` discarded rays reaching max range (`6.0 m`), blinding the robot to open corridors 3 cells ahead. Fixed by marching all non-hit rays up to `5.85 m` (and hit rays to `dist - 0.20 m`), clearing open doorways and corridors up to 3 cells ahead without delay.
   - **Bug B Fix (Angled Ray Doorway Clearance)**: Widened doorway offset tolerance to `abs(offset) < 0.52 m` (doorway opening is 1.85 m wide) and reduced minimum ray incidence to `abs(dx) > 0.30`, enabling angled rays looking around corners to reliably clear cross-corridors 2–3 cells ahead.
   - **Bug C Fix (Far Wall Hit Detection)**: Expanded wall edge detection offset to `abs(offset) < 0.78 m` and incidence threshold to `abs(cos_ray) > 0.22`, allowing the 72-ray LiDAR to reliably register side walls of T-junctions and cul-de-sacs up to 3 full cells ahead.
4. **Multi-Cell 3D Height-Scanner Mapping (`update_walls_from_height_scan_multicell`)**:
   - Exploits the full $3.2\text{ m} \times 3.2\text{ m}$ (1089 points) scanner footprint spanning $\pm 1.6\text{ m}$ around the robot center to map wall boundaries in adjacent and diagonal cells *before* entering them.
   - Confirms wall edges (`1`) if $\ge 3$ points exceed $z > 0.5\text{ m}$; confirms open doorways (`0`) if footprint coverage $\ge 8$ points and $0$ points exceed $z > 0.5\text{ m}$.
5. **Recursive Dead-End Pruning (`prune_dead_ends`) & Instant Dynamic Replanning**:
   - Detects when any non-goal, non-current cell $(i, j)$ has 3 sides confirmed as `walls == 1` and iteratively seals cul-de-sacs, propagating backward down single corridors.
   - Applies a heavy cost penalty (`+50.0`) in A* against entering dead-end cells, preventing the robot from ever turning into a blind corridor.
   - Automatically triggers `replan_needed = True` immediately when:
     - An active path edge becomes blocked (`1`).
     - A cell along the active path becomes a pruned dead end.
     - A newly discovered open doorway (`0`) provides a shorter route to the goal.
6. **Adaptive Straight-Corridor Sprint Boost (`0.76 m/s`)**:
   - When heading error is small ($|\psi_{\text{err}}| < 0.15\text{ rad}$) and analytical path curvature is low ($|\kappa_{\text{ff}}| < 0.12\text{ m}^{-1}$), cruise speed smoothly ramps from nominal $0.60\text{ m/s}$ up to **$0.76\text{ m/s}$**.
   - As the spline approaches any turn, curvature rises and speed seamlessly drops back to $0.60\text{ m/s}$ and down to $0.30\text{ m/s}$ for safe, zero-drift cornering with **0 falls**.
7. **Online 3D Height-Scan Hazard Discrimination (`Exp 10d / 10e`)**:
   - Uses `scan_xy` and `scan_z` ($33 \times 33$ height scanner) to distinguish four terrain primitives without false positives:
     - **Single-Lane Spike Beds (`"lane"`)**: High cones ($0.12\text{ m} < z < 0.42\text{ m}$) confined to one side of the corridor with flat floor ($z < 0.06\text{ m}$) on the opposite side; routes along safe lane $v_{\text{safe}} = \pm 0.44\text{ m}$ with pre-entry lead-in (`-1.05 m`) and post-exit lead-out (`+1.05 m`) waypoints.
     - **2-Bed Opposing Spike Slaloms (`"slalom"`)**: Persistent dual-bed memory (`v_minus` / `v_plus`) detects opposing beds separated by a $0.60\text{ m}$ central gap ($u \in [-0.30, +0.30]\text{ m}$) and immediately sets `replan_needed = True` when a zone is discovered or upgraded from `"lane"` to `"slalom"`.
     - **Steep Ramps (`cell_ramp_zones`)**: Full-width elevated slopes ($25\text{–}26\text{ cm}$) are explicitly tracked; any overlapping false-positive spike classifications in that cell are purged immediately.
     - **Floor Trenches (`cell_gap_zones`)**: Deep floor drops ($z < -0.15\text{ m}$, $16\text{–}20\text{ cm}$ gap) are explicitly tracked and purged from spike classification.
8. **3-Phase Perpendicular In-Place Slalom Gap Crossing (`Exp 10c`)**:
   - *Phase 1 (Approach & Stop)*: Decelerates to `0.30 m/s` at $u_{\text{prog}} \in [-0.50, -0.25)\text{ m}$, then brakes (`3.6 m/s²`) to `0.00 m/s` and pivots $90^\circ$ in place ($\omega_z \le 1.8\text{ rad/s}$) to face straight across the $0.60\text{ m}$ gap.
   - *Phase 2 (Closed-Loop Gap Walk)*: Walks straight across at `0.32 m/s` once $|\psi_{\text{err}}| \le 12^\circ$, applying closed-loop longitudinal centering $\Delta\psi = \pm 1.6 \cdot u_{\text{prog}}$ to stay dead-center between Bed 1 and Bed 2.
   - *Phase 3 (Realign & Exit)*: Upon reaching the exit lane ($|v_r - v_{\text{exit}}| \le 0.15\text{ m}$), pivots back in place to `corridor_yaw` and accelerates to `0.60 m/s` past Bed 2.
9. **Ramp & Trench Centerline Alignment (`Exp 10e`)**:
   - **Ramp Climbing** ($u_{\text{prog}} \in [-0.85, +0.80]\text{ m}$): Locks heading to the straight corridor centerline (`|wz| <= 0.06 rad/s`, `vy = 0.0`, `vx = 0.60 m/s`) so the robot climbs $25\text{–}26\text{ cm}$ ramps squarely without stepping off an edge.
   - **Trench Clearance** ($u_{\text{prog}} \in [-1.05, +0.18]\text{ m}$): Locks heading (`|wz| <= 0.06 rad/s`, `vy = 0.0`) with a controlled `0.85 m/s` forward stride so the G1 clears `16–20 cm` floor trenches cleanly in stride instead of braking at the lip.
   - **Short Route Commitment**: Removes the pre-emptive `+25.0` fork diversion penalty on ramps and trenches so the robot takes the direct Short Route rather than reversing to the longer detour.

---

## 3. Results of the Latest Pilot (`A4_Hazard_G2`)

### 3.1 Level 7 Direct Short-Route Traversal (`Exp 10e`, Latest `pilot.py`)
Walking straight through the right-hand Short Route (`(0,0) -> (1,0) -> (2,0)...`) over steep `25–26 cm` ramps, `16–20 cm` trenches, rough boxes, and spike beds:

| Level 7 Map | Previous Long Route (`Exp 10d`) | Latest Short Route (`Exp 10e`) | Time Saved | Falls / Resets |
| :--- | :---: | :---: | :---: | :---: |
| **Level 7 (`seed=2026`)** | `106.12 s` (0 falls) | **`58.1 s` (GOAL)** 🏆 | **`-48.0 s` (`-45.2%`)** ⚡ | **`0` / `0`** |
| **Level 7 (`seed=42`)** | `140.02 s` (0 falls) | **`56.3 s` (GOAL)** 🏆 | **`-83.7 s` (`-59.8%`)** ⚡ | **`0` / `0`** |

- **3D Newton Replay Video**: [`artifacts/experiment_10_track_a_hazard_slalom/run/level7_short_route_walk.mp4`](artifacts/experiment_10_track_a_hazard_slalom/run/level7_short_route_walk.mp4)
- **Verification Telemetry Plot**: [`artifacts/experiment_10_track_a_hazard_slalom/level7_short_route_verification.png`](artifacts/experiment_10_track_a_hazard_slalom/level7_short_route_verification.png)

### 3.2 Full Level 7 & Level 9 Multi-Hazard Benchmark (`Exp 10d`)
Evaluated across 4 hazard mazes (`--num_envs 4`, nominal `0.60 m/s`):

| Map | `A4_Hazard_G2` (Ours) | `A4_Slalom_Only` | `A3_Centroid_Spike` | `A1_Baseline_Cut` |
| :--- | :---: | :---: | :---: | :---: |
| **Map 1 (`L7_seed2026`)** | **GOAL `106.12 s` (0 falls)** 🏆 | GOAL `175.76 s` (3 falls) | DNF (5 falls, `2.92 m` left) | DNF (8 falls, `13.53 m` left) |
| **Map 2 (`L7_seed42`)** | **GOAL `140.02 s` (0 falls)** 🏆 | GOAL `175.06 s` (5 falls) | DNF (3 falls, `2.92 m` left) | DNF (7 falls, `10.78 m` left) |
| **Map 3 (`L9_seed2026`)** | **GOAL `94.16 s` (0 falls)** 🏆 | GOAL `94.86 s` (0 falls) | DNF (2 falls, `7.59 m` left) | DNF (2 falls, `7.28 m` left) |
| **Map 4 (`L9_seed42_rand`)** | **GOAL `119.42 s` (0 falls)** 🏆 | GOAL `137.78 s` (0 falls) | DNF (6 falls, `3.26 m` left) | DNF (16 falls, `14.34 m` left) |
| **Completion / Total Falls** | **4/4 (100%) / 0 Falls** | 4/4 (100%) / 8 Falls | 0/4 (0%) / 16 Falls | 0/4 (0%) / 33 Falls |

- **Benchmark Plot**: [`artifacts/experiment_10_track_a_hazard_slalom/track_a_hazard_slalom_comparison.png`](artifacts/experiment_10_track_a_hazard_slalom/track_a_hazard_slalom_comparison.png)

### 3.3 Isolated Spike & 2-Bed Opposing Slalom Courses (`Exp 10b / 10c`)

| Course | `A4_Hazard_G2` (Method 1 + 3-Phase Crossing) | Baselines (`C0`, `C1`, `C2`) |
| :--- | :---: | :---: |
| **Course A (2 Lane + 2 Slalom)** | **GOAL `68.06 s` — `4/4 (100%)` cleared, `0` falls** 🏆 | DNF — `0/4` cleared, `20–21` resets |
| **Course B (Mirror Slalom + Lane)** | **GOAL `66.50 s` — `4/4 (100%)` cleared, `0` falls** 🏆 | DNF — `0/4` cleared, `21–22` resets |

- **Plot**: [`artifacts/experiment_10_track_a_hazard_slalom/spike_isolation_method1_comparison.png`](artifacts/experiment_10_track_a_hazard_slalom/spike_isolation_method1_comparison.png)
- **3D Video**: [`artifacts/experiment_10_track_a_hazard_slalom/run/spike_course_walk.mp4`](artifacts/experiment_10_track_a_hazard_slalom/run/spike_course_walk.mp4)

### 3.4 Earlier Track A Navigation Benchmarks (`Exp 07 – Exp 09` @ `0.60 m/s`)
- **Exp 07 ($G^2$ Quintic Spline, $5\times 5$)**: `32.56 s` / `17.55 m` (vs `38.60 s` / `20.14 m` for `A0` Manhattan, `-15.6%`).
- **Exp 08 + Far-Vision Overhaul (Chest LiDAR + Multi-Cell Height Scan + Dead-End Pruning + Sprint Boost)**:
  - **Seed 42**: **`27.76 s`** (was `32.48 s`, **`-14.5%`**, `0` falls) ⚡
  - **Seed 100**: **`22.80 s`** (was `24.82 s`, **`-8.1%`**, `0` falls) ⚡
  - **Seed 777**: **`27.92 s`** (was `30.78 s`, **`-9.3%`**, `0` falls) ⚡
  - **Macro-Average**: **`26.16 s`** (was `29.36 s`, **`-10.9%`**, **`0` falls** across all seeds) 🏆
- **Exp 09 (`--random-ends`, 4 Quadrants)**: Won macro-average **`44.11 s` / `23.77 m`** vs `45.59 s` for `A1` and `49.90 s` for `A2_G2`.

---

## 4. How to Run the Latest Pilot

Run all simulation commands from inside `Week07_Curriculum_Maze/g1_maze_lab/`:

```bash
cd g1_maze_lab

# If you just cloned the repo and root policy.pt is missing, copy it first:
cp -n pretrained/g1_rough/model_pretrained.pt ../policy.pt
```

### A. Run with the 3D Newton Viewer (`--viz newton`)
```bash
# Level 7 (7x7 Hazard Maze, 300 s limit)
uv run python scripts/challenge.py --policy ../policy.pt --pilot ../pilot.py --level 7 --seed 2026 --viz newton

# Level 9 (9x9 Hazard Maze + 2-bed opposing spike slaloms, 480 s limit)
uv run python scripts/challenge.py --policy ../policy.pt --pilot ../pilot.py --level 9 --seed 2026 --viz newton

# Practice random start & goal cells on Level 7 or 9
uv run python scripts/challenge.py --policy ../policy.pt --pilot ../pilot.py --level 9 --seed 42 --random-ends --viz newton

# Final Competition Level (10x10 Hazard Maze + random start/goal, 600 s limit; requires challenge_final task)
uv run python scripts/challenge.py --policy ../policy.pt --pilot ../pilot.py --level final --seed 2026 --viz newton

# Isolated Spike & 2-Bed Opposing Slalom Course (A or B)
MAZE_SPIKE_ISOLATION_TEST=A uv run python scripts/challenge.py --policy ../policy.pt --pilot ../pilot.py --level 7 --viz newton
```

### B. Fast Headless Scoring (No Viewer)
```bash
uv run python scripts/challenge.py --policy ../policy.pt --pilot ../pilot.py --level 7 --seed 2026
```

### C. Record an `.mp4` Replay Video
```bash
MAZE_CAM=chase \
MAZE_REC_MP4=../my_run.mp4 \
uv run python scripts/challenge.py --policy ../policy.pt --pilot ../pilot.py --level 7 --seed 2026 --viz newton
```

### D. Generate the 4-Panel Perception, Planned Path & `cmd_vel` Vector Visualization `.mp4`
Run [`g1_maze_lab/scripts/visualize_pilot_perception.py`](g1_maze_lab/scripts/visualize_pilot_perception.py) to generate the synchronized 4-panel perception/planning/vector dashboard `.mp4` video and 4-keyframe summary `.png`:
```bash
cd g1_maze_lab

# Run simulation, record 10 Hz sensor/planner telemetry, and render MP4 + PNG:
uv run python scripts/visualize_pilot_perception.py --record --seed 2026

# Fast re-render from cached telemetry (.pkl) without re-running simulation:
uv run python scripts/visualize_pilot_perception.py --render-only
```
- **Output Video**: [`artifacts/experiment_10_track_a_hazard_slalom/run/robot_perception_planning_viz.mp4`](artifacts/experiment_10_track_a_hazard_slalom/run/robot_perception_planning_viz.mp4)
- **Output Keyframe Plot**: [`artifacts/experiment_10_track_a_hazard_slalom/robot_perception_planning_keyframes.png`](artifacts/experiment_10_track_a_hazard_slalom/robot_perception_planning_keyframes.png)

---

## 5. How to Submit `pilot.py` and `policy.pt`

### Method 1: Classroom Web Upload (`challenge_server.py`)
1. Open the instructor's upload URL in a browser: `http://<instructor-ip>:8000`
2. Fill in the form fields:
   - **Team Name (`team`)**: 2–32 characters using `a-z`, `A-Z`, `0-9`, `_`, or `-` (re-uploading the same team name replaces your previous submission).
   - **`policy.pt` (`policy`, required)**: select `Week07_Curriculum_Maze/policy.pt`.
   - **`pilot.py` (`pilot`, optional)**: select `Week07_Curriculum_Maze/pilot.py`.
   - **Helper `.py` files (`helpers`)**: **leave empty** — `pilot.py` is 100% self-contained.
3. Click **Submit**. The server automatically runs `check_policy` (verifies `rsl_rl` G1 Rough tensor shapes) and `check_python` (compiles UTF-8 Python, checks `class Pilot`, and blocks forbidden map imports like `GENERATED`, `HAZARDS`, `HARD_CELLS`, `g1_maze_lab.terrains`), then queues your team on `http://<instructor-ip>:8000/board`.

### Method 2: Direct Folder Copy (Local / Instructor Machine)
```bash
mkdir -p g1_maze_lab/submissions/<team_name>
cp policy.pt pilot.py g1_maze_lab/submissions/<team_name>/

cd g1_maze_lab
uv run python scripts/challenge.py --team <team_name> --level 7 --seed 2026 --viz newton
```

### Pre-Submission Sanity Check (Optional)
Run this from `Week07_Curriculum_Maze/` before uploading to verify both files pass the server checks:
```bash
uv run python -c "
import sys; sys.path.insert(0, 'g1_maze_lab/scripts')
from challenge_server import check_policy, check_python
print('check_policy:', check_policy(open('policy.pt', 'rb').read()))
print('check_python:', check_python('pilot.py', open('pilot.py', 'rb').read()))
"
# Expected output:
# check_policy: None
# check_python: None
```
