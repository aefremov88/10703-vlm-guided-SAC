# eval_vlm_by_dist.py
"""
Evaluate VLM on distance-bucketed pairs.

- Input:  data/pairs_by_dist.pkl (from build_pairs_by_dist.py)
- Output:
    - Console + logs/eval_vlm_by_dist.log
    - plots/<plot_name> (accuracy vs |Δd| bucket)
    - pairs/by_dist/... PNGs with GT/VLM labels
"""

import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from requests.exceptions import HTTPError, RequestException

from vlm_client import vlm_compare

PAIRS_PATH = "data/pairs_by_dist.pkl"
PAIRS_VIS_DIR = "pairs/by_dist"
PLOTS_DIR = "plots"
LOGS_DIR = "logs"

os.makedirs(PAIRS_VIS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TASK_TEXT = (
    "Move the robot end-effector so that it reaches the red target point "
    "on top of the table as accurately as possible."
)


def load_pairs(path: str = PAIRS_PATH) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pair_image_with_vlm(res: Dict[str, Any], rank: int) -> None:
    """
    Save a side-by-side image for a single pair, including GT and VLM labels.
    Images are used as-is (no cropping/rotation; handled by camera).
    """
    img_a = res["img_a"]
    img_b = res["img_b"]
    d_a = res["d_a"]
    d_b = res["d_b"]
    delta_d = res["delta_d"]
    gt = res["gt"]           # "A" or "B"
    y: Optional[int] = res["y"]  # 0, 1 or None

    gt_label = gt
    if y is None:
        vlm_label = "UNK"
    else:
        vlm_label = "A" if y == 0 else "B"

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].imshow(img_a)
    axs[0].axis("off")
    axs[0].set_title("State A")

    axs[1].imshow(img_b)
    axs[1].axis("off")
    axs[1].set_title("State B")

    fig.suptitle(
        f"Rank {rank} | |Δd|={delta_d:.4f} | "
        f"d_A={d_a:.3f}, d_B={d_b:.3f} | GT={gt_label}, VLM={vlm_label}",
        fontsize=8,
    )

    plt.tight_layout()
    fname = (
        f"pair_{rank:03d}_dd_{delta_d:.3f}_"
        f"gt_{gt_label}_vlm_{vlm_label}.png"
    )
    out_path = os.path.join(PAIRS_VIS_DIR, fname)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_bucket_edges(s: str, deltas: np.ndarray) -> np.ndarray:
    """
    If s is non-empty, parse as comma-separated edges.
    If s is empty, create 10 equal-width bins over [min(d), max(d)].
    """
    if s.strip():
        vals = [float(x) for x in s.split(",")]
        vals_sorted = sorted(vals)
        if vals != vals_sorted:
            print("[eval_vlm_by_dist] Warning: bucket edges were not sorted; sorting them.")
        edges = np.asarray(vals_sorted, dtype=float)
    else:
        d_min = float(deltas.min())
        d_max = float(deltas.max())
        edges = np.linspace(d_min, d_max, 11)
    return edges


def eval_vlm_by_dist(
    pairs: List[Dict[str, Any]],
    bucket_edges_str: str,
    plot_name: str,
) -> None:
    log_lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    results = []

    log(f"Evaluating VLM on {len(pairs)} distance-bucketed pairs...")

    for i, p in enumerate(pairs):
        img_a = p["img_a"]
        img_b = p["img_b"]
        d_a = float(p["d_a"])
        d_b = float(p["d_b"])
        delta_d = float(p["delta_d"])
        gt_label = p["gt"]  # "A" or "B"

        gt = 0 if gt_label == "A" else 1

        try:
            y = vlm_compare(img_a, img_b, TASK_TEXT)
        except (HTTPError, RequestException) as e:
            log(f"[VLM error] pair {i}: {e}")
            y = None

        results.append(
            {
                "img_a": img_a,
                "img_b": img_b,
                "d_a": d_a,
                "d_b": d_b,
                "delta_d": delta_d,
                "gt": gt_label,
                "gt_idx": gt,
                "y": y,
            }
        )

    # ---------- global stats ----------
    n_total = len(results)
    decisions = [r for r in results if r["y"] is not None]
    n_dec = len(decisions)
    n_correct = sum(1 for r in decisions if r["y"] == r["gt_idx"])
    n_incorrect = n_dec - n_correct
    n_unknown = n_total - n_dec

    log("\n=== Global VLM stats (Δd-bucketed pairs) ===")
    log(f"Total pairs: {n_total}")
    log(f"Decisions (A/B): {n_dec}")
    log(f"  correct:   {n_correct:3d} ({n_correct / max(1, n_dec):.3f})")
    log(f"  incorrect: {n_incorrect:3d} ({n_incorrect / max(1, n_dec):.3f})")
    log(
        f"Unknown:     {n_unknown:3d} "
        f"({n_unknown / max(1, n_total):.3f} of all pairs)"
    )

    # ---------- buckets by |Δd| ----------
    deltas = np.array([r["delta_d"] for r in results], dtype=float)
    edges = parse_bucket_edges(bucket_edges_str, deltas)
    num_buckets = len(edges) - 1

    log("\n=== Accuracy by |Δd| bucket (only A/B decisions) ===")
    bucket_stats = []

    for i in range(num_buckets):
        lo, hi = edges[i], edges[i + 1]
        in_bucket = [r for r in results if lo <= r["delta_d"] < hi]
        dec_bucket = [r for r in in_bucket if r["y"] is not None]
        unk_bucket = [r for r in in_bucket if r["y"] is None]

        nb = len(dec_bucket)
        if nb > 0:
            correct_b = sum(1 for r in dec_bucket if r["y"] == r["gt_idx"])
            incorrect_b = nb - correct_b
            pct_correct = 100.0 * correct_b / nb
            pct_incorrect = 100.0 * incorrect_b / nb
        else:
            correct_b = incorrect_b = 0
            pct_correct = pct_incorrect = 0.0

        if len(in_bucket) > 0:
            pct_unknown = 100.0 * len(unk_bucket) / len(in_bucket)
        else:
            pct_unknown = 0.0

        log(
            f"|Δd| in [{lo:.3f}, {hi:.3f}): "
            f"n_dec={nb:3d}, %correct={pct_correct:5.1f}, "
            f"%incorrect={pct_incorrect:5.1f}, %unknown={pct_unknown:5.1f}"
        )

        bucket_stats.append(
            {
                "lo": lo,
                "hi": hi,
                "n_dec": nb,
                "correct": correct_b,
                "incorrect": incorrect_b,
            }
        )

    # ---------- bar chart: |Δd| buckets ----------
    labels_d = [f"[{b['lo']:.2f},{b['hi']:.2f})" for b in bucket_stats]
    frac_correct_d = [
        (b["correct"] / b["n_dec"]) if b["n_dec"] > 0 else 0.0
        for b in bucket_stats
    ]
    frac_incorrect_d = [
        (b["incorrect"] / b["n_dec"]) if b["n_dec"] > 0 else 0.0
        for b in bucket_stats
    ]

    x = np.arange(len(labels_d))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, frac_correct_d, width, label="correct")
    plt.bar(x + width / 2, frac_incorrect_d, width, label="incorrect")
    plt.xticks(x, labels_d, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Fraction of VLM decisions")
    plt.xlabel("|Δdistance| bucket")
    plt.title("VLM decision quality vs. |Δdistance|")
    plt.legend()

    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"\nSaved |Δd| bar chart to {plot_path}")

    # ---------- save visual pair images sorted by |Δd| ----------
    log(
        f"\nSaving images with GT/VLM labels to ./{PAIRS_VIS_DIR} "
        "sorted by |Δd| ..."
    )
    sorted_results = sorted(results, key=lambda r: r["delta_d"], reverse=True)
    for rank, r in enumerate(sorted_results):
        save_pair_image_with_vlm(r, rank)
    log("Done saving visual distance-bucketed pairs.")

    # ---------- write log file ----------
    log_path = os.path.join(LOGS_DIR, "eval_vlm_by_dist.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nWrote log to {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate VLM on distance-bucketed pairs."
    )
    parser.add_argument(
        "--bucket_edges",
        type=str,
        default="0.0,0.02,0.05,0.10,0.20,0.40",
        help=(
            "Comma-separated bucket edges for |Δd|. "
            "Default matches build_pairs_by_dist.py: "
            "'0.0,0.02,0.05,0.10,0.20,0.40'. "
            "If empty, use 10 equal-width bins over the data range."
        ),
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        default="vlm_acc_by_dist.png",
        help="Filename for the plot saved under plots/ (default: vlm_acc_by_dist.png)",
    )
    args = parser.parse_args()

    pairs = load_pairs(PAIRS_PATH)
    print(f"Loaded {len(pairs)} Δd-bucketed pairs from {PAIRS_PATH}")
    eval_vlm_by_dist(
        pairs,
        bucket_edges_str=args.bucket_edges,
        plot_name=args.plot_name,
    )
