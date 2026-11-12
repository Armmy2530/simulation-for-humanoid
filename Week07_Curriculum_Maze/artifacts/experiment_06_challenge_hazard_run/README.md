# Experiment 06: Challenge 7.5 Hazard Maze (`pro_pilot`)

## Overview & Competition Setup
- **Task**: `G1-Maze-Hard` (Level 7, 7x7 hazard maze).
- **Hazards Present**: Deadly spike beds (instant reset on contact below cone tips), slippery ice tiles ($\mu = 0.05$), floor ramps up to 22°, and floor trenches/gaps.
- **Pilot Package**: [`g1_maze_lab/submissions/pro_pilot/pilot.py`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/g1_maze_lab/submissions/pro_pilot/pilot.py)

## Strategies Implemented in `pro_pilot`:
1. **Dynamic Spike Avoidance**:
   - Spike cones ($25\text{–}35\text{ cm}$) detected online via height scan z-thresholds ($0.12 < z < 0.42\text{ m}$).
   - The autopilot offsets its pursuit target by $\pm 0.45\text{ m}$ toward the clear side of the corridor, stepping past spikes safely.
2. **Ice Traction Regulation**:
   - Reduces angular rate acceleration to prevent foot slip on low-friction tiles.
3. **Start Fork Selection**:
   - Prioritizes the 14-cell northern corridor route from start cell $(0, 0)$ over the 23-cell route.

## Benchmark Results (Level 7 Seed 1)
- **Path Followed**: Direct northern corridor exit through $(0, 2) \to (0, 4) \to (1, 5) \to (2, 5)$.
- **Spike Clearance**: Successfully dodged the first deadly spike bed at $(0, 2)$ by shifting to the East lane.
- **Deep Progression**: Advanced through 11 cells, reaching within 7.0 m (3 cells) of the goal.

## Run Deliverables in `run/`
- Video Replay: [`run/challenge_hazard_run.mp4`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_06_challenge_hazard_run/run/challenge_hazard_run.mp4) (Chase-cam HD video of Level 7 hazard run with live timer HUD)
