# env_utils.py
import numpy as np
import gymnasium as gym
import gymnasium_robotics


def make_env():
    """Create FetchReachDense-v4 with a configured top/zoomed camera."""
    # Register robotics envs
    gym.register_envs(gymnasium_robotics)

    env_id = "FetchReachDense-v4"
    env = gym.make(env_id, render_mode="rgb_array")

    # ---- Camera config: zoomed, rotated, slightly tilted ----
    renderer = env.unwrapped.mujoco_renderer
    cfg = renderer.default_cam_config

    # Zoom: smaller distance → bigger objects
    cfg["distance"] = 1.5  # tweak as needed (e.g., 0.8–1.2)

    # Horizontal rotation around the scene
    # cfg["azimuth"] = 180.0

    # Vertical tilt: -90 = top-down, 0 = side view
    # cfg["elevation"] = -30.0

    # Center camera on the workspace
    # cfg["lookat"] = np.array([1.3, 0.75, 0.55])
    # --------------------------------------------------------

    return env


def distance_to_goal(obs: dict) -> float:
    """Return raw environment distance between achieved and desired goal."""
    achieved = obs["achieved_goal"]
    desired = obs["desired_goal"]
    d = np.linalg.norm(achieved - desired)
    return float(d)
