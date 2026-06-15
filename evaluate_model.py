import os

import numpy as np
import torch

from env import CrawlEnv
from sb3_contrib import MaskablePPO

from constants import MODEL_PATH, REPLAY_OUTPUT_DIR


def select_device() -> str:
    """Pick the best available device. SB3's "auto" only knows cpu/cuda, so we
    detect MPS (Apple Silicon) ourselves and fall back to cpu otherwise."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_path: str = MODEL_PATH,
    env: CrawlEnv | None = None,
    device: str | None = None,
    **kwargs,
) -> MaskablePPO:
    """Load a saved MaskablePPO agent onto the best available device.

    The policy class and its kwargs (``n_residual_blocks`` ...) are restored from
    the saved archive, so callers only pass overrides they need -- e.g. an ``env``
    and ``tensorboard_log`` to resume training. ``device`` defaults to cuda/mps/cpu
    via ``select_device``.
    """
    return MaskablePPO.load(
        model_path,
        env=env,
        device=device or select_device(),
        **kwargs,
    )


def setup_output_dir():
    if not os.path.exists(REPLAY_OUTPUT_DIR):
        os.makedirs(REPLAY_OUTPUT_DIR)

    return REPLAY_OUTPUT_DIR


def run_episode(
    env: CrawlEnv,
    model: MaskablePPO,
    seed: int | None = None,
    deterministic: bool = False,
):
    obs, _ = env.reset(seed=seed)

    done = False
    while not done:
        action, _ = model.predict(
            obs, action_masks=env.action_masks(), deterministic=deterministic
        )
        obs, reward, done, truncated, info = env.step(action)


def run_n_episodes(
    env: CrawlEnv,
    model: MaskablePPO,
    n: int,
    output_dir: str = None,
    seed: int | None = None,
    deterministic: bool = False,
):
    replay_dir = output_dir if output_dir is not None else setup_output_dir()
    os.makedirs(replay_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(n):
        episode_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        run_episode(
            env,
            model,
            seed=episode_seed,
            deterministic=deterministic,
        )
        html_out = env.render(mode="html", width=800, height=800)

        if html_out is not None:
            with open(os.path.join(replay_dir, f"replay_{i}.html"), "w") as f:
                f.write(html_out)
                print(
                    f"Episode {i} finished with seed {episode_seed}. "
                    f"Written to replay_{i}.html"
                )


if __name__ == "__main__":
    EPISODES = 10

    crawl_env = CrawlEnv()

    model = load_model()
    run_n_episodes(crawl_env, model, EPISODES)

    print("Evaluation Finished successfully.")
