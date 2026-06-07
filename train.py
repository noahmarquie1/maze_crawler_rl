from constants import ACTION_MAPPING
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make
from kaggle_environments.utils import Struct
from stable_baselines3 import PPO
import os
from env import CrawlEnv

from model import CNNFeatureExtractor


if __name__ == "__main__":
    out = "ppo_crawl"
    env = CrawlEnv()
    try:
        agent = PPO(
            "MlpPolicy",
            env,
            #policy_kwargs={"features_extractor_class": CNNFeatureExtractor},
            verbose=1,
        )
        agent.learn(total_timesteps=int(1e4), log_interval=1, progress_bar=True)
        if os.path.exists(out + ".zip"):
            os.remove(out + ".zip")

        agent.save(out)

    except Exception as e:
        print(e)