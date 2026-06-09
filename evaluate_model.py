import os

import numpy as np

from env import CrawlEnv
from stable_baselines3 import PPO

from constants import MODEL_PATH, REPLAY_OUTPUT_DIR
from model import CNNFeatureExtractor


def setup_output_dir():
    if not os.path.exists(REPLAY_OUTPUT_DIR):
        os.makedirs(REPLAY_OUTPUT_DIR)

    return REPLAY_OUTPUT_DIR


def run_episode(
    env: CrawlEnv, model: PPO, seed: int | None = None, deterministic: bool = False
):
    obs, _ = env.reset(seed=seed)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated


def run_n_episodes(
    env: CrawlEnv,
    model: PPO,
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

    model = PPO.load(
        MODEL_PATH,
        policy_kwargs={"features_extractor_class": CNNFeatureExtractor},
    )
    run_n_episodes(crawl_env, model, EPISODES)

    print(f"Evaluation Finished successfully.")
