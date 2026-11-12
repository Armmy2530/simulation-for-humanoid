# Experiment 03: 3-Way Evolution Comparison

## Overview
- **Progression Evaluated**:
  1. **Run 1: Baseline** (41.42 s) — Stop and turn at cell centers ($v = 0.60\text{ m/s}, v_{\min} = 0.15\text{ m/s}$).
  2. **Run 2: Upgraded Direct Corner Cut** (27.64 s) — Momentum corner cut ($v = 0.75\text{ m/s}, v_{\min} = 0.38\text{ m/s}$).
  3. **Run 3: New A\* Kinodynamic Sprint** (24.46 s) — Dense waypoints tracked with pure pursuit ($v = 0.88\text{ m/s}, v_{\min} = 0.46\text{ m/s}$).

## Visual Trajectory & Velocity Comparison
![3-Way Evolution](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_03_three_way_evolution/three_way_evolution_comparison.png)

## Comparative Scorecard

| Metric | 1. Baseline | 2. Upgraded (Run 2) | 3. New A\* Sprint | Delta (A\* vs Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Arrival Time (Headless)** | 41.42 s | 27.64 s | **24.46 s** | **-16.96 s (-41.0%)** |
| **Arrival Time (With HUD / Video)** | 40.8 s | 29.1 s | **24.3 s** | **-16.50 s (-40.4%)** |
| **Total Path Distance** | 20.47 m | 17.92 m | **18.43 m** | **-2.04 m shorter** |
| **Average Forward Speed** | 0.497 m/s | 0.652 m/s | **0.759 m/s** | **+52.7% boost** |
| **Peak Speed** | 0.681 m/s | 0.798 m/s | **0.925 m/s** | Full sprint |
| **Falls / Resets** | 0 | 0 | **0** | 100% collision-free |

## Run Deliverables in `run/`
- Full Demonstration Video: [`run/three_way_evolution_video.mp4`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_03_three_way_evolution/run/three_way_evolution_video.mp4) (116s complete HD comparison with Thai title cards)
- A\* Sprint Raw Data: [`run/astar_fast.npz`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_03_three_way_evolution/run/astar_fast.npz)
