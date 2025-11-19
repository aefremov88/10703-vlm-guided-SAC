# quick_view_camera.py
import matplotlib.pyplot as plt
from env_utils import make_env


def main():
    env = make_env()
    obs, info = env.reset()
    img = env.render()

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title("Current camera view")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
