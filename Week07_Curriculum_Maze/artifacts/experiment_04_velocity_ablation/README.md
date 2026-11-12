# Experiment 04: Velocity Profile Ablation Study

## Hypothesis & Research Question
- **Question**: Did the speedup from 27.64 s down to ~24.4 s come primarily from global A\* dense waypoint tracking, or did it come from the aggressive kinodynamic velocity profile ($v_{\max} = 0.88\text{ m/s}, v_{\min} = 0.46\text{ m/s}, a_{\text{ang}} = 4.0\text{ rad/s}^2$)?
- **Ablation Setup**: Run the "Before Upgrade" autopilot (Direct Corner Cut without A\*) using the exact same aggressive velocity profile.

## Visual Comparison
![Ablation Comparison](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_04_velocity_ablation/ablation_fast_velocity_vs_astar.png)

## Findings & Scorecard

| Configuration | Trajectory Control | Speed Profile ($v_{\max} / v_{\min}$) | Total Path | Goal Time | Result |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Baseline** | Stop & Turn (Cell Centers) | 0.60 / 0.15 m/s | 20.47 m | 41.42 s | Slowest |
| **2. Before Upgrade (Original Vel)** | Direct Corner Cut | 0.75 / 0.38 m/s | 17.92 m | 27.64 s | Momentum baseline |
| **3. Upgraded with A\* Spline** | Dense Bézier Arc Waypoints | 0.88 / 0.46 m/s | 18.43 m | 24.46 s | +0.28m outer swing |
| **4. Before Upgrade + Fast Vel (No A\*)** | **Direct Corner Cut (Without A\*)** | **0.88 / 0.46 m/s** | **18.15 m** | **24.26 s** 🏆 | **Fastest Overall** |

## Why Direct Corner Cut Wins:
1. **Inner Apex Cutting**: Direct corner cutting sets a single shifted pursuit point across the corner apex, cutting the path to **18.15 m**.
2. **Lookahead Widening in A\***: Tracking dense waypoints via pure pursuit ($l_{\text{ahead}} = 0.65\text{ m}$) bulges outward on 90° corners, adding **+0.28 m** of walking distance (**18.43 m**).
3. **Conclusion**: Global A\* dense waypoint tracking adds zero value in this maze topology; the velocity profile and tight apex cornering account for 100% of the speedup.

## Run Deliverables in `run/`
- Winning Run Trajectory Data: [`run/before_fast_velocity.npz`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_04_velocity_ablation/run/before_fast_velocity.npz) (24.26 s goal time)
