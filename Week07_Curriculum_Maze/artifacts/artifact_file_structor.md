# Artifact File Structure & Experiment Directory Standard

This document establishes the official file structure, naming conventions, and runtime organization template for all experiment runs, benchmarks, and deliverables in the **G1 Humanoid Maze Autopilot** project.

---

## 📁 1. Directory Hierarchy Overview

All experiment outputs and benchmark deliverables must be stored under [`artifacts/`](file:///home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/).

Every experiment is allocated a dedicated, self-contained directory named with a zero-padded index and a descriptive snake_case milestone name:

```text
artifacts/
├── README.md                                 # Master experiment catalog and benchmark scoreboard
├── artifact_file_structor.md                 # This standard template specification
├── experiment_01_baseline_vs_upgraded/       # Milestone 1: Stop-and-pivot vs Corner-cut
│   ├── README.md                             # Experiment report, hypothesis, methodology, analysis
│   ├── *.png                                 # High-resolution figures, trajectory plots, visual comparisons
│   └── run/                                  # Dynamic execution artifacts
│       ├── *.mp4                             # High-definition video recordings (dual-view/single-view)
│       └── *.npz                             # 50 Hz trajectory data (t, pos, lin_vel, yaw, cmd, cell, walls)
├── experiment_02_waypoint_interpolation/     # Milestone 2: Fig 7, 8, 9 quadratic Bézier fillets
├── experiment_03_three_way_evolution/        # Milestone 3: 3-way progression & trajectory analysis
├── experiment_04_velocity_ablation/          # Milestone 4: Direct corner cut vs dense A* waypoints
├── experiment_05_random_mazes_evaluation/    # Milestone 5: Procedural maze generalization (Seeds 42, 100, 777)
├── experiment_06_challenge_hazard_run/       # Milestone 6: Challenge 7.5 spike & hazard avoidance
└── archive/                                  # Intermediate frames, JSON specs, and raw render caches
```

---

## 📐 2. Structural Rules and Separation of Concerns

### A. Experiment Folder Naming
Format: `experiment_{NN}_{descriptive_snake_case}/`
- `{NN}`: 2-digit zero-padded sequence number (e.g., `01`, `02`, `07`, `10`).
- `{descriptive_snake_case}`: Short, human-readable slug indicating the core research focus (e.g., `chest_lidar_lookahead`, `dynamic_obstacle_avoidance`, `elevation_stepping`).

### B. Root Level Files (`artifacts/experiment_{NN}_{name}/`)
Only human-facing analytical documentation and static figures belong at the root:
1. `README.md` (Mandatory): Full experiment report formatted according to the [Section 4 Template](#4-boilerplate-readmemd-template-for-new-experiments).
2. `*.png` (Recommended): Trajectory path plots, velocity time-series curves, state-machine transitions, and side-by-side snapshot comparisons.

### C. Run Output Subdirectory (`artifacts/experiment_{NN}_{name}/run/`)
All large, binary, and dynamic simulation outputs must be saved in `run/`:
1. `*.mp4`: Rendered camera video replays (e.g., 60 FPS H.264 MP4 with top-down or split-screen HUD).
2. `*.npz`: Compressed NumPy archives containing 50 Hz telemetric telemetry:
   - `t`: timestamps array `(N,)`
   - `pos`: 3D Cartesian coordinates `(N, 3)`
   - `lin_vel`: linear velocity `(N, 3)`
   - `yaw`: heading angle `(N,)`
   - `cmd`: commanded linear/angular velocities `(N, 3)`
   - `cell`: current grid coordinates `(N, 2)`
   - `walls`: sensed wall configuration `(N, 4)`
   - `metadata`: dictionary with `env_seed`, `controller`, `runtime_s`, `path_length_m`, etc.

### D. Temporary / Intermediate Files (`artifacts/archive/`)
Never pollute project root (`Week07_Curriculum_Maze/`) or package roots (`g1_maze_lab/`) with intermediate files:
- Individual render frames (`frames_*/`)
- Video stitch specifications (`spec_*.json`)
- Deprecated debug scripts or temporary caches
Move them to `artifacts/archive/` or delete them upon successful video compilation.

---

## 🛠️ 3. Standard Workflow for Running Experiments

When developing, running, and documenting an experiment:

```bash
# Step 1: Create the experiment directory structure
EXP_DIR="artifacts/experiment_07_your_experiment_name"
mkdir -p "${EXP_DIR}/run"

# Step 2: Execute the simulation script and direct outputs to run/
python3 g1_maze_lab/experiments/run_experiment.py \
    --output-npz "${EXP_DIR}/run/telemetry.npz" \
    --output-video "${EXP_DIR}/run/recording.mp4"

# Step 3: Generate comparative trajectory/performance plots in the root of EXP_DIR
python3 g1_maze_lab/scripts/plot_trajectory.py \
    --input "${EXP_DIR}/run/telemetry.npz" \
    --output "${EXP_DIR}/trajectory_analysis.png"

# Step 4: Write the experiment report
cp artifacts/artifact_file_structor.md "${EXP_DIR}/README.md"  # then edit with findings

# Step 5: Update the master index artifacts/README.md
```

---

## 📝 4. Boilerplate `README.md` Template for New Experiments

Copy and paste the template below when creating a new `README.md` for any future experiment:

````markdown
# Experiment [NN]: [Descriptive Experiment Title]

**Date**: YYYY-MM-DD  
**Status**: Completed / In-Progress  
**Lead Contributor**: [Author / Pilot Team]  
**Environment**: Mujoco Warp / Isaac Lab 3 (G1 Humanoid)  
**Map / Challenge**: [e.g., 5x5 Standard / Random Seed 100 / 7x7 Hazard Challenge]

---

## 🎯 1. Objective & Hypothesis
- **Problem Statement**: What bottleneck, limitation, or challenge does this experiment address?
- **Hypothesis**: What measurable improvement is expected (e.g., latency reduction, path smoothness, collision-free avoidance)?

---

## 🔬 2. Methodology & Experimental Setup
- **Policy / Controller Architecture**: (e.g., `pro_pilot/pilot.py` with modified lookahead radius)
- **Velocity Configuration**: ($v_{\max} = \dots \text{ m/s}, v_{\min} = \dots \text{ m/s}, \omega_{\max} = \dots \text{ rad/s}$)
- **Waypoint / Planning Strategy**: (e.g., Direct Bézier filleting vs A* grid search)
- **Sensors Used**: (e.g., 2D Corridor Raycast, 3D Chest LiDAR Pointcloud, Odometry)

---

## 📊 3. Performance Scorecard & Results

| Metric | Baseline / Target | This Run | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Arrival Time ($t_{\text{goal}}$)** | 00.00 s | **00.00 s** | -0.00 s (-00.0%) |
| **Total Path Length ($L$)** | 00.00 m | **00.00 m** | -0.00 m (-00.0%) |
| **Average Linear Speed ($\bar{v}$)** | 0.00 m/s | **0.00 m/s** | +0.00 m/s |
| **Collision Count** | 0 | **0** | Pass |
| **Success / Goal Reached** | Yes / No | **Yes** | 100% |

---

## 📈 4. Visual Analysis & Figures

![Trajectory Analysis](trajectory_analysis.png)

### Key Observations:
1. **Cornering Behavior**: [Description of turning radius, clearance to corners]
2. **Speed Profile**: [Acceleration and braking characteristics near waypoints]
3. **Obstacle Response**: [Lateral displacement or deviation around obstacles/hazards]

---

## 🎬 5. Run Artifacts & Replay Files

All telemetry data and video recordings are stored in the [`run/`](run/) directory:
- 🎥 **Video Recording**: [`run/experiment_video.mp4`](run/experiment_video.mp4)
- 💾 **50 Hz Telemetry Log**: [`run/telemetry.npz`](run/telemetry.npz)

### Data Inspection Snippet:
```python
import numpy as np

data = np.load("artifacts/experiment_NN_title/run/telemetry.npz")
print("Recorded keys:", list(data.keys()))
print("Total time:", data["t"][-1], "seconds")
print("Total distance:", np.sum(np.linalg.norm(np.diff(data["pos"][:, :2], axis=0), axis=1)), "m")
```

---

## 💡 6. Key Takeaways & Next Steps
- **Conclusion**: [Summary of whether the hypothesis was supported]
- **Implications**: [How this affects the overall humanoid navigation stack]
- **Next Iteration**: [Specific next experiment or parameter tune suggested]
````

---

## 🔒 5. Verification Checklist

Before marking an experiment as complete:
- [ ] Experiment folder is placed under `artifacts/experiment_{NN}_{name}/`.
- [ ] `README.md` is populated at the root of the experiment directory.
- [ ] High-resolution plots (`*.png`) are saved in the root of the experiment directory.
- [ ] Video files (`*.mp4`) are in `run/`.
- [ ] Telemetry archives (`*.npz`) are in `run/`.
- [ ] Master [`artifacts/README.md`](file:///home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/README.md) catalog has been updated with the new entry and scoreboard metrics.
- [ ] No loose or duplicated files remain in `g1_maze_lab/` or repository root.
