# build_pairs_by_k.py
"""
Build k-step state pairs from saved episodes.

Each pair comes from the SAME episode (same target) and has:
    - img_a, img_b
    - d_a, d_b, delta_d
    - k           (steps apart)
    - gt          ("A" or "B", which is closer to the goal)
    - ep_idx, t_a, t_b

Output is saved to:
    data/pairs_by_k.pkl
"""

import argparse
import os
import pickle
from typing import Any, Dict, List

import numpy as np

EPISODES_PATH = "data/episodes_random.pkl"
PAIRS_PATH = "data/pairs_by_k.pkl"


def load_episodes(path: str = EPISODES_PATH) -> List[List[Dict[str, Any]]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_pairs_by_k(
    episodes: List[List[Dict[str, Any]]],
    k_min: int,
    k_max: int,
    pairs_per_k: int,
    min_delta_d: float,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    pairs_by_k: Dict[int, List[Dict[str, Any]]] = {
        k: [] for k in range(k_min, k_max + 1)
    }

    # Collect ALL candidate pairs per k
    for ep_idx, traj in enumerate(episodes):
        T = len(traj)
        if T < k_min + 1:
            continue

        ds = np.array([float(s["d"]) for s in traj], dtype=float)

        for t_a in range(T):
            for k in range(k_min, k_max + 1):
                t_b = t_a + k
                if t_b >= T:
                    continue

                d_a = ds[t_a]
                d_b = ds[t_b]
                delta_d = float(abs(d_b - d_a))

                if delta_d < min_delta_d:
                    continue

                gt = "A" if d_a < d_b else "B"

                pair = {
                    "img_a": traj[t_a]["img"],
                    "img_b": traj[t_b]["img"],
                    "d_a": float(d_a),
                    "d_b": float(d_b),
                    "delta_d": delta_d,
                    "k": int(k),
                    "gt": gt,
                    "ep_idx": int(ep_idx),
                    "t_a": int(t_a),
                    "t_b": int(t_b),
                }
                pairs_by_k[k].append(pair)

    # For each k, subsample up to pairs_per_k
    final_pairs: List[Dict[str, Any]] = []
    for k in range(k_min, k_max + 1):
        candidates = pairs_by_k[k]
        n = len(candidates)
        if n == 0:
            print(f"[build_pairs_by_k] k={k}: 0 candidates")
            continue

        if n > pairs_per_k:
            idx = rng.choice(n, size=pairs_per_k, replace=False)
            selected = [candidates[i] for i in idx]
        else:
            selected = candidates

        print(
            f"[build_pairs_by_k] k={k}: using {len(selected)} pairs "
            f"(out of {n} candidates)"
        )
        final_pairs.extend(selected)

    print(f"[build_pairs_by_k] Total pairs collected: {len(final_pairs)}")
    return final_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Build k-step pairs from saved episodes."
    )
    parser.add_argument(
        "--k_min",
        type=int,
        default=1,
        help="Minimum k (steps apart) (default: 1)",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=30,
        help="Maximum k (steps apart) (default: 30)",
    )
    parser.add_argument(
        "--pairs_per_k",
        type=int,
        default=10,  # ↓ smaller default so VLM eval is cheaper
        help="Target number of pairs per k (default: 10)",
    )
    parser.add_argument(
        "--min_delta_d",
        type=float,
        default=0.0,
        help=(
            "Minimum |d_b - d_a| required to keep a pair "
            "(default: 0.0, no filter)"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(EPISODES_PATH):
        raise FileNotFoundError(
            f"Episodes file not found at {EPISODES_PATH}. "
            "Run collect_episodes.py first."
        )

    episodes = load_episodes(EPISODES_PATH)
    print(f"Loaded {len(episodes)} episodes from {EPISODES_PATH}")

    pairs = build_pairs_by_k(
        episodes=episodes,
        k_min=args.k_min,
        k_max=args.k_max,
        pairs_per_k=args.pairs_per_k,
        min_delta_d=args.min_delta_d,
    )

    os.makedirs("data", exist_ok=True)
    with open(PAIRS_PATH, "wb") as f:
        pickle.dump(pairs, f)

    print(f"Saved {len(pairs)} k-step pairs to {PAIRS_PATH}")


if __name__ == "__main__":
    main()
