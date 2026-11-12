# Experiment 01: Baseline vs. Upgraded Direct Corner Cut

## Overview & Hypothesis
- **Hypothesis**: Replacing discrete cell-center stop-and-turn pivots with lookahead corner-cutting offsets ($W = C_{\text{target}} - \delta_{\text{cut}}\hat{\mathbf{d}}_{\text{in}} + \delta_{\text{cut}}\hat{\mathbf{d}}_{\text{out}}$) and momentum-preserving velocity shaping will significantly reduce travel time without inducing collisions or balance loss.
- **Environment**: `G1-Maze-Challenge` (5x5 maze, 10 cells traversal).
- **Robot/Policy**: Unitree G1 Humanoid with pretrained `Isaac-Velocity-Rough-G1` walking policy.

## Visual Comparison
![Baseline vs Upgraded Comparison](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_01_baseline_vs_upgraded/baseline_vs_upgraded_comparison.png)

## Performance Metrics

| Metric | Baseline (Run 1) | Upgraded (Run 2) | Improvement |
| :--- | :---: | :---: | :---: |
| **Arrival Time (Headless)** | 41.42 s | **27.64 s** | **-13.78 s (-33.3%)** |
| **Arrival Time (With HUD / Video)** | 40.8 s | **29.1 s** | **-11.70 s (-28.7%)** |
| **Total Path Distance** | 20.47 m | **17.92 m** | **-2.55 m shorter** |
| **Average Forward Speed** | 0.497 m/s | **0.652 m/s** | **+31.2% speedup** |
| **Peak Speed** | 0.681 m/s | 0.798 m/s | Sprint maintained |
| **Cornering Minimum Speed** | 0.112 m/s (stop-turn) | **0.217 m/s** | Momentum preserved |
| **Falls / Wall Collisions** | 0 | 0 | 100% collision-free |

## Run Deliverables in `run/`
- Video Replay: [`run/baseline_vs_upgraded_video.mp4`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_01_baseline_vs_upgraded/run/baseline_vs_upgraded_video.mp4) (85s side-by-side HD video with real-time HUD telemetry)
- Baseline Raw Data: [`run/baseline.npz`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_01_baseline_vs_upgraded/run/baseline.npz)
- Upgraded Raw Data: [`run/upgraded.npz`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_01_baseline_vs_upgraded/run/upgraded.npz)
