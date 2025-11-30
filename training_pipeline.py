"""
SAC on FetchReach with sparse reward + nudge, using action-repeat (K).

Two nudge modes:

  1) Ground-truth nudge (use_vlm = False)
     - We look at the true distance to the goal.
     - If |Δd| <= delta_eps   → nudge = 0  ("don't know")
     - If Δd > 0              → nudge = +gamma  (moved closer)
     - If Δd < 0              → nudge = -gamma  (moved farther)

  2) VLM nudge (use_vlm = True)
     - We call vlm_compare(prev_img, cur_img, task_text):
          returns 1 → B better (cur state closer)  → nudge = +gamma
          returns 0 → A better (prev state closer) → nudge = -gamma
          returns None / "unknown"                → nudge = 0
    - We use true |Δd| ONLY as a *gate* (don't ask the VLM if the move is tiny): 
    if |Δd| <= delta_eps   → nudge = 0  and NO VLM call

Base sparse reward is always:
    r_sparse = success_reward * success - step_penalty

Total reward:
    r_total = r_sparse + nudge

If we set:
  --repeat 1 --nudge-gamma 0
we effectively get a pure sparse reward with no nudge (just step penalty + success bonus).
"""

import os
import time
import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import gymnasium_robotics  
except ImportError:
    gymnasium_robotics = None

from env_utils import make_env as make_camera_env  
from vlm_client import vlm_compare  

# ----------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------
DEFAULT_SUCCESS_RADIUS = 0.05
DEFAULT_STEP_PENALTY = 1.0
DEFAULT_SUCCESS_REWARD = 5.0

DEFAULT_NUDGE_GAMMA = 0.0          # size of +/- nudge
DELTA_EPS = 0.10                   # "don't know" gate on |Δdistance|
TRAINING_STEPS = 150_000
LOG_INTERVAL = 20

DEFAULT_REPEAT = 2                 # K
DEFAULT_MAX_STEPS = 40             # RL steps per episode (outer horizon)

DEFAULT_USE_VLM = False
DEFAULT_TASK_TEXT = (
    "The robot must move its gripper tip to touch "
    "the small red target sphere."
)


# ----------------------------------------------------------
# Config dataclass
# ----------------------------------------------------------
@dataclass
class FetchVLMConfig:
    success_radius: float = DEFAULT_SUCCESS_RADIUS
    step_penalty: float = DEFAULT_STEP_PENALTY
    success_reward: float = DEFAULT_SUCCESS_REWARD
    nudge_gamma: float = DEFAULT_NUDGE_GAMMA
    delta_eps: float = DELTA_EPS
    repeat: int = DEFAULT_REPEAT
    max_steps: int = DEFAULT_MAX_STEPS
    use_vlm: bool = DEFAULT_USE_VLM
    task_text: str = DEFAULT_TASK_TEXT


# ----------------------------------------------------------
# Wrappers
# ----------------------------------------------------------
class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat the same action `repeat` times in the base environment.

    One step of this wrapper = `repeat` steps of the underlying env.
    Rewards are summed; termination if any inner step terminates.
    """

    def __init__(self, env, repeat: int):
        super().__init__(env)
        assert repeat >= 1
        self.repeat = repeat

    def step(self, action):
        total_r = 0.0
        terminated = truncated = False
        info = {}
        obs = None
        for _ in range(self.repeat):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_r += r
            if terminated or truncated:
                break
        return obs, total_r, terminated, truncated, info


class TruncateAfterNSteps(gym.Wrapper):
    """
    Truncate episode after max_steps *outer* steps (RL decisions).
    """

    def __init__(self, env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        if self._step_count >= self.max_steps:
            truncated = True
        return obs, r, terminated, truncated, info


class FetchReachSparseVLMWrapper(gym.Wrapper):
    """
    Sparse reward + (optionally VLM-based) nudge on top of FetchReach.

    - Uses env_utils.make_env() camera, so images match the offline workbench
    - Maintains previous distance + frame to compute / query a nudge each RL step.
    """

    def __init__(self, env, cfg: FetchVLMConfig):
        super().__init__(env)
        self.cfg = cfg

        self.prev_distance: float | None = None
        self.prev_image: np.ndarray | None = None

        # Flatten observation space (Fetch dict -> single Box)
        obs, _ = env.reset()
        assert isinstance(obs, dict), "FetchReach env should return dict observations"
        flat = self._flatten_obs(obs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat.shape,
            dtype=np.float32,
        )

    # ----- helpers -----
    @staticmethod
    def _flatten_obs(obs_dict):
        base = np.asarray(obs_dict["observation"], dtype=np.float32)
        goal = np.asarray(obs_dict["desired_goal"], dtype=np.float32)
        return np.concatenate([base, goal], axis=-1)

    @staticmethod
    def _distance(obs_dict) -> float:
        ag = np.asarray(obs_dict["achieved_goal"], dtype=np.float32)
        dg = np.asarray(obs_dict["desired_goal"], dtype=np.float32)
        return float(np.linalg.norm(ag - dg, ord=2))

    def _grab_frame(self) -> np.ndarray:
        """
        Grab RGB frame from the env (after camera config in env_utils).
        Assumes render_mode='rgb_array'.
        """
        img = self.env.render()
        if img is None:
            raise RuntimeError(
                "env.render() returned None; make sure render_mode='rgb_array'."
            )
        return np.asarray(img)

    # ----- gym API -----
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        flat = self._flatten_obs(obs)
        dist = self._distance(obs)

        self.prev_distance = dist
        # initial frame (state A for first step)
        self.prev_image = self._grab_frame()

        success = float(dist <= self.cfg.success_radius)
        info = dict(info)
        info["distance"] = dist
        info["is_success"] = success
        info["sparse_reward"] = 0.0
        info["vlm_nudge"] = 0.0
        info["shaped_reward"] = 0.0
        return flat, info

    def step(self, action):
        # previous state image is the image from the end of the last step
        prev_img = self.prev_image

        obs, env_r, terminated, truncated, info = self.env.step(action)
        flat = self._flatten_obs(obs)
        dist = self._distance(obs)
        cur_img = self._grab_frame()

        success = float(dist <= self.cfg.success_radius)

        # sparse part
        r_sparse = self.cfg.success_reward * success - self.cfg.step_penalty

        # nudge part
        nudge = 0.0
        if self.cfg.nudge_gamma != 0.0 and self.prev_distance is not None:
            delta = self.prev_distance - dist
            abs_delta = abs(delta)

            # small move → "don't know" (no nudge, no VLM query)
            if abs_delta > self.cfg.delta_eps:
                if self.cfg.use_vlm:
                    # Ask VLM: is B (current) better than A (previous)?
                    choice = vlm_compare(prev_img, cur_img, self.cfg.task_text)
                    if choice == 1:        # B better → moved closer
                        nudge = +self.cfg.nudge_gamma
                    elif choice == 0:      # A better → moved farther
                        nudge = -self.cfg.nudge_gamma
                    else:                  # unknown / parse failure
                        nudge = 0.0
                else:
                    # Ground-truth sign of delta distance
                    nudge = (
                        +self.cfg.nudge_gamma if delta > 0.0 else -self.cfg.nudge_gamma
                    )

        reward = r_sparse + nudge

        # terminate on success
        if success:
            terminated = True

        # update previous state for next step
        self.prev_distance = dist
        self.prev_image = cur_img

        info = dict(info)
        info["distance"] = dist
        info["is_success"] = success
        info["sparse_reward"] = float(r_sparse)
        info["vlm_nudge"] = float(nudge)
        info["shaped_reward"] = float(reward)

        return flat, reward, terminated, truncated, info


# ----------------------------------------------------------
# Env factory
# ----------------------------------------------------------
def make_fetch_env(cfg: FetchVLMConfig):
    if gymnasium_robotics is None:
        raise ImportError(
            "gymnasium-robotics is not installed. "
            "Install with: pip install gymnasium-robotics mujoco"
        )

    # Camera-configured FetchReachDense-v4 (render_mode='rgb_array')
    base_env = make_camera_env()  # from env_utils.py :contentReference[oaicite:3]{index=3}

    # 1) action repeat (K)
    base_env = ActionRepeatWrapper(base_env, repeat=cfg.repeat)

    # 2) outer episode length (in RL steps)
    base_env = TruncateAfterNSteps(base_env, max_steps=cfg.max_steps)

    # 3) sparse + (GT or VLM) nudge
    wrapped = FetchReachSparseVLMWrapper(base_env, cfg)
    return wrapped


# ----------------------------------------------------------
# Main training loop
# ----------------------------------------------------------
def main(cfg: FetchVLMConfig):
    # log dir
    mode_label = "vlm" if cfg.use_vlm else "gt"
    run_label = (
        f"{mode_label}_K{cfg.repeat}"
        f"_g{cfg.nudge_gamma:g}"
        f"_sr{cfg.success_radius:g}"
        f"_sp{cfg.step_penalty:g}"
        f"_srw{cfg.success_reward:g}"
        f"_T{cfg.max_steps}"
    )

    log_root = "runs"
    log_dir = os.path.join(log_root, run_label)
    os.makedirs(log_dir, exist_ok=True)

    print("=== Running SAC on FetchReach (repeat + nudge) ===")
    print(f"  mode            : {'VLM' if cfg.use_vlm else 'ground-truth'}")
    print(f"  repeat (K)      : {cfg.repeat}")
    print(f"  nudge_gamma     : {cfg.nudge_gamma}")
    print(f"  success_radius  : {cfg.success_radius}")
    print(f"  step_penalty    : {cfg.step_penalty}")
    print(f"  success_reward  : {cfg.success_reward}")
    print(f"  max_steps (RL)  : {cfg.max_steps}")
    print(f"  delta_eps       : {cfg.delta_eps}")
    print(f"  log_dir         : {log_dir}")
    print()

    def _env_fn():
        env = make_fetch_env(cfg)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        return env

    vec_env = DummyVecEnv([_env_fn])

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tb_fetchreach"),
    )

    start_time = time.time()
    model.learn(total_timesteps=TRAINING_STEPS, log_interval=LOG_INTERVAL)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"\nTraining finished in {elapsed:.1f} seconds "
          f"({elapsed/60.0:.2f} minutes).")

    model.save(os.path.join(log_dir, "sac_fetchreach_vlm_repeat"))
    vec_env.close()


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Use online VLM nudge instead of ground-truth nudge.",
    )
    parser.add_argument(
        "--nudge-gamma",
        type=float,
        default=DEFAULT_NUDGE_GAMMA,
        help=f"Nudge magnitude gamma (default: {DEFAULT_NUDGE_GAMMA}).",
    )
    parser.add_argument(
        "--success-radius",
        type=float,
        default=DEFAULT_SUCCESS_RADIUS,
        help=f"Success radius (default: {DEFAULT_SUCCESS_RADIUS}).",
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=DEFAULT_STEP_PENALTY,
        help=f"Sparse step penalty (default: {DEFAULT_STEP_PENALTY}).",
    )
    parser.add_argument(
        "--success-reward",
        type=float,
        default=DEFAULT_SUCCESS_REWARD,
        help=f"Sparse success reward (default: {DEFAULT_SUCCESS_REWARD}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max RL steps per episode (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help=f"Action repeat K (default: {DEFAULT_REPEAT}).",
    )
    parser.add_argument(
        "--delta-eps",
        type=float,
        default=DELTA_EPS,
        help=f"|Δdistance| gate before nudge or VLM call (default: {DELTA_EPS}).",
    )

    args = parser.parse_args()

    cfg = FetchVLMConfig(
        success_radius=args.success_radius,
        step_penalty=args.step_penalty,
        success_reward=args.success_reward,
        nudge_gamma=args.nudge_gamma,
        delta_eps=args.delta_eps,
        repeat=args.repeat,
        max_steps=args.max_steps,
        use_vlm=args.use_vlm,
        task_text=DEFAULT_TASK_TEXT,
    )

    main(cfg)
