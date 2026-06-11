import os

from env import CrawlEnv
from stable_baselines3 import PPO
from env import CrawlEnv
from kaggle_environments import make

from constants import MODEL_PATH, REPLAY_OUTPUT_DIR
from model import CNNFeatureExtractor


def setup_output_dir():
    if not os.path.exists(REPLAY_OUTPUT_DIR):
        os.makedirs(REPLAY_OUTPUT_DIR)

    return REPLAY_OUTPUT_DIR


def run_episode(env: CrawlEnv, model: PPO):
    obs, _ = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)


def run_n_episodes(env: CrawlEnv, model: PPO, n: int, output_dir: str = None):
    replay_dir = output_dir if output_dir is not None else setup_output_dir()
    os.makedirs(replay_dir, exist_ok=True)
    for i in range(n):
        run_episode(env, model)
        html_out = env.render(mode="html", width=800, height=800)

        if html_out is not None:
            with open(os.path.join(replay_dir, f"replay_{i}.html"), "w") as f:
                f.write(html_out)
                print(f"Episode {i} finished. Written to replay_{i}.html")


if __name__ == "__main__":
    EPISODES = 10

    crawl_env = CrawlEnv()

    model = PPO.load(
        MODEL_PATH,
        policy_kwargs={"features_extractor_class": CNNFeatureExtractor},
    )
    run_n_episodes(crawl_env, model, EPISODES)

    print(f"Evaluation Finished successfully.")
