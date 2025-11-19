# VLM Critic Workbench

This repo is a **small workbench** to:

- pick and compare **vision-language models (VLMs)** as critics
- tune the **pair-generation scripts** (what images we show, how we bucket difficulty)
- and choose a good **k / “nudge” design** for a VLM-enhanced SAC / PPO workflow.

Everything here is **offline**: we sample rollouts once, then build and analyze pair datasets to understand when a VLM can reliably say “state A is better than state B”.

------

## High-level workflow

1. **Sample rollouts** from the environment
    → store raw states as images + distances to the target in `data/episodes_*.pkl`.
2. **Build pair datasets**:
   - by **distance difference** `Δd = |d_B − d_A|` (same episode / same target),
   - by **number of steps k apart** along a trajectory.
3. **Evaluate a VLM** on those pairs:
   - by Δd buckets → “how accuracy changes with how different the states are”
   - by k buckets → “how accuracy changes with temporal separation”
4. Use these results to:
   - choose a VLM + prompt format,
   - choose k and weighting for the VLM “nudge” in SAC.

------

## Directory & File Structure

### Environment and VLM utilities

- **`env_utils.py`**
  - Creates `FetchReachDense-v4` with a **top/zoomed camera**.
  - Provides helpers to compute **environment distance** `d` between gripper and goal.
  - Used by all other scripts that need an environment.
- **`vlm_client.py`**
  - Encodes images (rotation, resize, base64) and calls the **VLM API**
  - Given `(img_a, img_b)` returns a **preference** (`A` / `B` / `unknown`) 
  - All model choice, endpoint URLs, and prompt formatting live here
- **`quick_view_camera.py`** 
  - Fast visual check of the **camera configuration**
  - Renders one frame with `make_env()`, applies the same rotation as the VLM client, and shows it.
  - Use this when tweaking azimuth / elevation / distance / lookat

------

### Data collection

- **`collect_episodes.py`**
   **Purpose:** sample environment rollouts once and save original states

  - **Key parameters to tune**

    - `--num_episodes`
    - `--max_steps_per_episode`
    - `--output` (e.g. `data/episodes_random.pkl`)

  - **Output**

    - `data/episodes_*.pkl`: list of episodes, each an ordered list of dicts:

      ```
      {
        "img": <H×W×3 uint8>,   # original state image (camera view, no crop)
        "d": float,             # env distance to goal
        "t": int,               # step index within episode
      }
      ```

------

### Pair dataset builders

- **`build_pairs_by_dist.py`**
   **Purpose:** create a dataset of pairs `(A,B)` bucketed by **distance difference** `Δd`, with both states from the **same episode** (same target).

  - **Input**

    - `data/episodes_*.pkl` (from `collect_episodes.py`)

  - **Key parameters to tune**

    - `--bucket_edges` for `Δd` (e.g. `[0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]`)
    - `--pairs_per_bucket` - maximum pairs per bucket (less than maximum if not enough data for a bucket)

  - **Output**

    - `data/pairs_by_dist.pkl`: list of dicts:

      ```
      {
        "img_a", "img_b",
        "d_a", "d_b", "delta_d",
        "gt",                # "A" or "B" (which is closer to goal)
        "ep_idx", "t_a", "t_b",
      }
      ```

- **`build_pairs_by_k.py`**
   **Purpose:** create a dataset of pairs `(A,B)` bucketed by **number of steps k apart** along the same episode.

  - **Input**

    - `data/episodes_*.pkl`

  - **Key parameters to tune**

    - `--episodes_path`
    - `--k_min`, `--k_max` (e.g. 1–30)
    - `--pairs_per_k`
    - Optional: `--min_delta_d`

  - **Output**

    - `data/pairs_by_k.pkl`: list of dicts:

      ```
      {
        "img_a", "img_b",
        "d_a", "d_b", "delta_d",
        "k",                 # step difference
        "gt",                # "A" or "B"
        "ep_idx", "t_a", "t_b",
      }
      ```

------

### Evaluation & visualization scripts

- **`eval_vlm_by_dist.py`**
   **Purpose:** evaluate a VLM on distance-bucketed pairs; see where it’s confident and where it struggles.
  - **Input**
    - `data/pairs_by_dist.pkl`
    - `vlm_client.py` (for queries)
  - **Key parameters to tune**
    - `--pairs_path`
    - `--batch_size` or rate-limiting options
    - `--model_name` / `--api_endpoint` (handled inside `vlm_client`)
  - **Output**
    - Console summary (also logged to `logs/` – see below).
    - `plots/vlm_acc_by_dist.png`: bar chart of accuracy per `Δd` bucket.
    - Visual exports into `pairs/by_dist/` of pictures of both states, distance to target in both states, Δd, ground truth label and VLM-generated label sorted by Δd
- **`eval_vlm_by_k.py`**
   **Purpose:** evaluate a VLM on k-bucketed pairs (step distance).
  - **Input**
    - `data/pairs_by_k.pkl`
    - `vlm_client.py`
  - **Key parameters to tune**
    - `--pairs_path`
    - `--k_bins` (e.g. [1–5], [6–10], …)
  - **Output**
    - Console summary (also logged).
    - `plots/vlm_acc_by_k.png`: bar chart of accuracy per k-range
    - Visual exports into `pairs/by_k/` of pictures of both states, distance to target in both states, Δd, ground truth label and VLM-generated label sorted by Δd

------

### Folders

- **`data/`**
  - Serialized datasets:
    - `episodes_random.pkl` (original state images + distances from random policy)
    - `episodes_sac.pkl` (later, from SAC policy)
    - `pairs_by_dist.pkl`
    - `pairs_by_k.pkl`
- **`pairs/`**
  - Exported image pairs with overlaid GT / VLM labels, useful for eyeballing failure modes:
    - `pairs/by_dist/…`
    - `pairs/by_k/…`
- **`plots/`**
  - Summary charts:
    - `vlm_acc_by_dist.png`
    - `vlm_acc_by_k.png`
    - any other diagnostic figures
- **`logs/`**
  - Text log files with console outputs from long runs, e.g.:
    - `logs/eval_vlm_by_dist.log`
    - `logs/eval_vlm_by_k.log`
