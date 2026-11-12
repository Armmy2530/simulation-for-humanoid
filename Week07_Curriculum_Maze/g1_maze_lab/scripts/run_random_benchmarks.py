"""Run benchmark on 3 random mazes."""

import subprocess
import sys
import numpy as np
from g1_maze_lab.terrains.maze import generate_maze, maze_to_ascii

SEEDS = [42, 100, 777]

def run_seed(seed):
    rng = np.random.default_rng(seed)
    w = generate_maze(5, rng, open_ratio=0.15)
    layout_str = maze_to_ascii(w)
    out_file = f"random_maze_seed{seed}.npz"
    print(f"\n==========================================")
    print(f"RUNNING BENCHMARK ON RANDOM MAZE SEED {seed}")
    print(f"==========================================")
    print(layout_str)
    
    cmd = [
        "uv", "run", "python", "scripts/benchmark_autopilot.py",
        "--task", "G1-Maze-Challenge",
        "--num_envs", "1",
        "--checkpoint", "pretrained/g1_rough/model_pretrained.pt"
    ]
    env = dict(
        BENCHMARK_UPGRADED="1",
        BENCHMARK_OUTPUT=out_file,
        MAZE_SPEED="0.88",
        MAZE_MIN_SPEED="0.46",
        MAZE_TURN_EARLY="0.60",
        MAZE_MAX_TURN="1.35",
        MAZE_ACC="2.0,4.0",
        MAZE_DIRECT_CORNER="1",
        MAZE_USE_ASTAR="0",
        MAZE_LAYOUT_OVERRIDE=layout_str
    )
    import os
    full_env = os.environ.copy()
    full_env.update(env)
    
    res = subprocess.run(cmd, env=full_env)
    if res.returncode != 0:
        print(f"Error running seed {seed}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_seed(int(sys.argv[1]))
    else:
        for s in SEEDS:
            run_seed(s)
