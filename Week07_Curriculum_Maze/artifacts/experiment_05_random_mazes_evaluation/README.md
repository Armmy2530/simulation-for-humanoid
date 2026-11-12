# Experiment 05: Random Mazes Generalization Evaluation

## Overview
- **Goal**: Verify robustness, zero-shot generalization, and obstacle avoidance across 3 unseen random 5x5 mazes synthesized via depth-first procedural generation (`open_ratio = 0.15`):
  - **Seed 42**: 10-cell L-corridor with right-angle turns.
  - **Seed 100**: 16-cell long winding perimeter maze.
  - **Seed 777**: 10-cell alternating zigzag S-curve maze.

## Visual Trajectories & Speed Profiles
![Random Mazes Benchmark](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_05_random_mazes_evaluation/random_mazes_3way_benchmark.png)

## Multi-Seed Quantitative Results

| Random Maze | Corridor Layout | Cells Walked | Total Path | Arrival Time | Avg Speed | Peak Speed | Falls / Resets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 10-Cell L-Corridor | 10 | 18.23 m | **24.42 s** | 0.751 m/s | 0.933 m/s | **0** |
| **Seed 100** | 16-Cell Winding | 16 | 30.82 m | **40.58 s** | 0.764 m/s | 0.904 m/s | **0** |
| **Seed 777** | 10-Cell Zigzag | 10 | 18.36 m | **25.32 s** | 0.732 m/s | 0.903 m/s | **0** |

## Discussion: Why Seed 100 Detoured & How to Prevent It
- **True Shortest Route**: 9 cells (`(0,0) -> (0,1) -> (0,2) -> (1,2) -> (1,3) -> ...`).
- **Autopilot Path Taken**: 16 cells (`(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (3,2) -> ...`).
- **Root Cause**: At cell $(1, 2)$, optimistic flood fill evaluated both East $(2, 2)$ and North $(1, 3)$ with an identical cost of 4. Inertia tie-breaking (`t=0`) chose East into an unmapped dead-end corridor.
- **Solution**: Utilizing the 6-meter chest lidar ray returns down corridors detects the dead-end wall at $(3, 2)$ while the robot is still at $(1, 2)$, forcing North $(1, 3)$ and cutting the route to 9 cells.

## Run Deliverables in `run/`
- Seed 42 Trajectory Data: [`run/random_maze_seed42.npz`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_05_random_mazes_evaluation/run/random_maze_seed42.npz)
- Seed 100 Trajectory Data: [`run/random_maze_seed100.npz`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_05_random_mazes_evaluation/run/random_maze_seed100.npz)
- Seed 777 Trajectory Data: [`run/random_maze_seed777.npz`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_05_random_mazes_evaluation/run/random_maze_seed777.npz)
