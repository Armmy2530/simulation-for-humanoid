# Experiment 02: Waypoint Interpolation & Continuous Fillets

## Overview & Mathematical Formulation
- **Objective**: Implement reference specification (Figs. 7, 8, 9) converting discrete Manhattan cell centers into dense quadratic Bézier arc fillets for continuous turning.
- **Reference Geometry**:
  - Manhattan cell centers connected at 90° corners: $\mathbf{p}_{\text{prev}}, \mathbf{p}_{\text{curr}}, \mathbf{p}_{\text{next}}$.
  - Fillet entry point: $\mathbf{p}_{\text{enter}} = \mathbf{p}_{\text{curr}} - r \hat{\mathbf{u}}_{\text{in}}$
  - Fillet exit point: $\mathbf{p}_{\text{exit}} = \mathbf{p}_{\text{curr}} + r \hat{\mathbf{u}}_{\text{out}}$
  - Continuous curve parameterized by $s \in [0, 1]$:
    $$\mathbf{B}(s) = (1-s)^2 \mathbf{p}_{\text{enter}} + 2(1-s)s \mathbf{p}_{\text{curr}} + s^2 \mathbf{p}_{\text{exit}}$$

## Visual Figures
- **Figure 7 vs. Figure 8 Implementation**:
  ![Discrete vs Fillet Waypoints](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_02_waypoint_interpolation/fig7_fig8_interpolated_waypoints.png)
- **Figure 9 Full Continuous Maze Path**:
  ![Full Continuous Maze Arc](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_02_waypoint_interpolation/fig9_maze_interpolated_path.png)

## Key Technical Details
- **Smooth 90° Turns (`SS90SR`)**: Clean circular fillet rounding isolated corners with radius $r = 0.60\text{ m}$.
- **Diagonal S-Curve Transitions (`SD45 -> DIA -> DS45`)**: Consecutive alternating corners (e.g. $(3,4) \to (3,3) \to (4,3)$) blend into continuous diagonal sweeps.
- **Doorway Corridor Passage**: Clean entry through the open doorway between cells $(3,3)$ and $(4,3)$ directly into goal cell $(4,4)$.

## Run Deliverables in `run/`
- Trajectory Data: [`run/upgraded_smooth.npz`](/home/armmy2530/Project/cu-robo-class-ws/simulation-for-humanoid/Week07_Curriculum_Maze/artifacts/experiment_02_waypoint_interpolation/run/upgraded_smooth.npz) (27.90 s traversal)
