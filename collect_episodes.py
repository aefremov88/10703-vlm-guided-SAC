# collect_episodes.py
"""
Collect environment rollouts and save original states (images + distances).

This script is the single source of truth for raw episodes:
each step stores:
    - "img": HxWx3 uint8 RGB array from env.render()
    - "d":   float, env distance to goal
    - "t":   int, step index within the episode

Usage (examples):
    python collect_episodes.py
    python collect_episodes.py --num_episodes 50 --max_steps 60
"""

import argparse
import os
import pickle
from typing import List, Dict, Any

from env_utils import make_env, distance_to_goal


def collect_episodes(
    num_episodes: int = 20,
    max_steps: int = 50,
) -> List[List[Dict[str, Any]]]:
    """
    Collect full episodes with a random policy.

    Args:
        num_episodes: how many episodes to sample.
        max_steps:    safety cap on steps per episode (truncated if exceeded).

    Returns:
        episodes: list of episodes
            each episode: list of dicts with keys:
                - "img": HxWx3 uint8 RGB array
                - "d":   float distance to goal
                - "t":   int step index in episode
    """
    env = make_env()
    episodes: List[List[Dict[str, Any]]] = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_idx = 0
        traj: List[Dict[str, Any]] = []

        while not done and step_idx < max_steps:
            img = env.render()
            d = float(distance_to_goal(obs))

            traj.append(
                {
                    "img": img,
                    "d": d,
                    "t": step_idx,
                }
            )

            # Random policy for now; later you can plug in SAC / PPO here.
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_idx += 1
            done = terminated or truncated

        episodes.append(traj)
        print(
            f"Collected episode {ep + 1}/{num_episodes}, "
            f"length={len(traj)}"
        )

    env.close()
    return episodes


def main():
    parser = argparse.ArgumentParser(
        description="Collect FetchReach episodes (images + distances)."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20,
        help="Number of episodes to collect (default: 20)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Max steps per episode (default: 50)",
    )
    args = parser.parse_args()

    # Hardwired output path
    output_path = "data/episodes_random.pkl"

    # Ensure data/ exists
    os.makedirs("data", exist_ok=True)

    episodes = collect_episodes(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )

    with open(output_path, "wb") as f:
        pickle.dump(episodes, f)

    print(f"Saved {len(episodes)} episodes to {output_path}")


if __name__ == "__main__":
    main()
