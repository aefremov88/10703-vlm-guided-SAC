# build_pairs_by_dist.py
"""
Build state pairs bucketed by distance difference Δd = |d_b - d_a|.

Each pair comes from the SAME episode (same target) and has:
    - img_a, img_b
    - d_a, d_b, delta_d
    - gt          ("A" or "B", which is closer to the goal)
    - ep_idx, t_a, t_b

Pairs are distributed across specified Δd buckets, up to a maximum
number per bucket. Within each bucket, we:
    - prefer pairs with the LARGEST Δd,
    - enforce that each state (ep_idx, t) is used at most once
      to increase diversity.

Output is saved to:
    data/pairs_by_dist.pkl
"""

import argparse
import os
import pickle
from typing import Any, Dict, List

import numpy as np

EPISODES_PATH = "data/episodes_random.pkl"
PAIRS_PATH = "data/pairs_by_dist.pkl"


def load_episodes(path: str = EPISODES_PATH) -> List[List[Dict[str, Any]]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_pairs_by_dist(
    episodes: List[List[Dict[str, Any]]],
    bucket_edges: List[float],
    pairs_per_bucket: int,
    min_delta_d: float,
) -> List[Dict[str, Any]]:
    """
    Build within-episode pairs and bucket them by Δd.

    Args:
        episodes:         list of episodes from collect_episodes.py
        bucket_edges:     sorted list of bucket boundaries for Δd
        pairs_per_bucket: target number of pairs per Δd bucket
        min_delta_d:      global minimum Δd to keep any pair

    Returns:
        List of pair dicts.
    """
    bucket_edges = np.asarray(bucket_edges, dtype=float)
    assert np.all(np.diff(bucket_edges) > 0), "bucket_edges must be strictly increasing"
    num_buckets = len(bucket_edges) - 1

    # Collect ALL candidate pairs per bucket.
    bucket_candidates: List[List[Dict[str, Any]]] = [[] for _ in range(num_buckets)]

    for ep_idx, traj in enumerate(episodes):
        T = len(traj)
        if T < 2:
            continue

        ds = np.array([float(s["d"]) for s in traj], dtype=float)

        # Loop over all (t_a, t_b) with t_a < t_b
        for t_a in range(T):
            for t_b in range(t_a + 1, T):
                d_a = ds[t_a]
                d_b = ds[t_b]
                delta_d = float(abs(d_b - d_a))

                if delta_d < min_delta_d:
                    continue

                # Find bucket index
                b = int(np.searchsorted(bucket_edges, delta_d, side="right") - 1)
                if b < 0 or b >= num_buckets:
                    continue  # outside all buckets

                gt = "A" if d_a < d_b else "B"

                pair = {
                    "img_a": traj[t_a]["img"],
                    "img_b": traj[t_b]["img"],
                    "d_a": float(d_a),
                    "d_b": float(d_b),
                    "delta_d": delta_d,
                    "gt": gt,
                    "ep_idx": int(ep_idx),
                    "t_a": int(t_a),
                    "t_b": int(t_b),
                }
                bucket_candidates[b].append(pair)

    # Now select up to pairs_per_bucket from each bucket,
    # preferring the largest Δd and enforcing state diversity.
    all_pairs: List[Dict[str, Any]] = []
    for b_idx, candidates in enumerate(bucket_candidates):
        lo = bucket_edges[b_idx]
        hi = bucket_edges[b_idx + 1]
        n_cand = len(candidates)

        if n_cand == 0:
            print(
                f"[build_pairs_by_dist] bucket [{lo:.4f}, {hi:.4f}): "
                f"used 0 pairs (out of 0 candidates)"
            )
            continue

        # sort by delta_d descending
        candidates_sorted = sorted(
            candidates, key=lambda p: p["delta_d"], reverse=True
        )

        selected: List[Dict[str, Any]] = []
        used_states = set()  # (ep_idx, t)

        for pair in candidates_sorted:
            ep = pair["ep_idx"]
            t_a = pair["t_a"]
            t_b = pair["t_b"]
            key_a = (ep, t_a)
            key_b = (ep, t_b)

            # enforce diversity: don't reuse any state in this bucket
            if key_a in used_states or key_b in used_states:
                continue

            selected.append(pair)
            used_states.add(key_a)
            used_states.add(key_b)

            if len(selected) >= pairs_per_bucket:
                break

        used = len(selected)
        if used > 0:
            deltas = [p["delta_d"] for p in selected]
            min_delta = float(min(deltas))
            max_delta = float(max(deltas))
            avg_delta = float(sum(deltas) / used)
            delta_info = (
                f", |Δd| in selected ≈ "
                f"[{min_delta:.4f}, {max_delta:.4f}], avg={avg_delta:.4f}"
            )
        else:
            delta_info = ""

        print(
            f"[build_pairs_by_dist] bucket [{lo:.4f}, {hi:.4f}): "
            f"used {used} pairs (out of {n_cand} candidates){delta_info}"
        )

        all_pairs.extend(selected)

    print(f"[build_pairs_by_dist] Total pairs collected: {len(all_pairs)}")
    return all_pairs


def parse_bucket_edges(s: str) -> List[float]:
    """
    Parse a comma-separated list of floats into sorted bucket edges.

    Example:
        "0.0,0.02,0.05,0.10,0.20,0.40"
    """
    vals = [float(x) for x in s.split(",")]
    vals_sorted = sorted(vals)
    if vals != vals_sorted:
        print("[build_pairs_by_dist] Warning: bucket edges were not sorted; sorting them.")
    return vals_sorted


def main():
    parser = argparse.ArgumentParser(
        description="Build Δd-bucketed pairs from saved episodes."
    )
    parser.add_argument(
        "--bucket_edges",
        type=str,
        default="0.0,0.02,0.05,0.10,0.20,0.40",
        help=(
            "Comma-separated list of bucket edges for Δd. "
            "Example: '0.0,0.02,0.05,0.10,0.20,0.40'"
        ),
    )
    parser.add_argument(
        "--pairs_per_bucket",
        type=int,
        default=10,
        help="Target number of pairs per Δd bucket (default: 10)",
    )
    parser.add_argument(
        "--min_delta_d",
        type=float,
        default=0.0,
        help="Global minimum Δd to keep any pair (default: 0.0, no filter)",
    )
    args = parser.parse_args()

    if not os.path.exists(EPISODES_PATH):
        raise FileNotFoundError(
            f"Episodes file not found at {EPISODES_PATH}. "
            "Run collect_episodes.py first."
        )

    episodes = load_episodes(EPISODES_PATH)
    print(f"Loaded {len(episodes)} episodes from {EPISODES_PATH}")

    bucket_edges = parse_bucket_edges(args.bucket_edges)
    pairs = build_pairs_by_dist(
        episodes=episodes,
        bucket_edges=bucket_edges,
        pairs_per_bucket=args.pairs_per_bucket,
        min_delta_d=args.min_delta_d,
    )

    os.makedirs("data", exist_ok=True)
    with open(PAIRS_PATH, "wb") as f:
        pickle.dump(pairs, f)

    print(f"Saved {len(pairs)} Δd-bucketed pairs to {PAIRS_PATH}")


if __name__ == "__main__":
    main()
