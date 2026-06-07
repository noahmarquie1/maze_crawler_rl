import os

from agent import CrawlEnv
from stable_baselines3 import PPO
from agent import CrawlEnv
from kaggle_environments import make

from constants import MODEL_PATH, REPLAY_OUTPUT_DIR
from model import CNNFeatureExtractor


def setup_output_dir():
    if not os.path.exists(REPLAY_OUTPUT_DIR):
        os.makedirs(REPLAY_OUTPUT_DIR)

    return REPLAY_OUTPUT_DIR


def run_episode(env: CrawlEnv, model: PPO):
    obs, _ = crawl_env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = crawl_env.step(action.item())


if __name__ == "__main__":
    EPISODES = 5

    crawl_env = CrawlEnv()

    model = PPO.load(
        MODEL_PATH, policy_kwargs=dict(feature_extractor_class=CNNFeatureExtractor)
    )
    for i in range(EPISODES):
        run_episode(crawl_env, model)
        html_out = crawl_env.render(mode="html", width=800, height=800)

        replay_dir = setup_output_dir()

        if html_out is not None:
            with open(os.path.join(replay_dir, f"replay_{i}.html"), "w") as f:
                f.write(html_out)

    print(f"Evaluation Finished successfully.")
