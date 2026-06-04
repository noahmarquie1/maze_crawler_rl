import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make
from kaggle_environments.utils import Struct
from stable_baselines3 import PPO
import os

action_mapping = {
    0: "BUILD_SCOUT",
    1: "BUILD_WORKER",
    2: "BUILD_MINER",
    3: "JUMP_NORTH",
    4: "JUMP_SOUTH",
    5: "JUMP_EAST",
    6: "JUMP_WEST",
}

class CrawlEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Action space is:
        # 0: BUILD_SCOUT, 1: BUILD_WORKER, 2: BUILD_MINER
        # 3: JUMP_NORTH, 4: JUMP_SOUTH, 5: JUMP_EAST, 6: JUMP_WEST
        self.action_space = spaces.Discrete(7)

        # Observation space is a flattened array of factory state and walls
        #  - factory state: 8 elements
        #  - walls: 400 elements
        self.observation_space = spaces.Box(low=-1, high=5000, shape=(408,), dtype=np.float32)

        base_env = make("crawl", configuration={"randomSeed": 42})
        self.trainer: Struct = base_env.train([None, "random"])

    def format_obs(self, base_obs):
        obs = np.zeros(408, dtype=np.float32)
        if "0-0" in base_obs.robots.keys():
            obs[:8] = np.array(base_obs.robots["0-0"], dtype=np.float32)
        obs[8:] = np.array(base_obs.walls, dtype=np.float32)
        return obs

    def step(self, action):
        action = action_mapping[action]
        base_obs, _, done, info = self.trainer.step(action)
        reward = 1 # Temporary constant reward for staying alive
        return self.format_obs(base_obs), reward, done, 0, info 

    def reset(self, seed=None, options=None):
        base_obs = self.trainer.reset()
        return self.format_obs(base_obs), {}

    def render(self, mode="human"):
        self.trainer.render(mode=mode)

    def close(self):
        self.trainer.close()


if __name__ == "__main__":
    out = "ppo_crawl"
    if os.path.exists(out + ".zip"):
        os.remove(out + ".zip")

    env = CrawlEnv()
    try:
        agent = PPO("MlpPolicy", env)
        agent.learn(total_timesteps=int(1e4), log_interval=100, progress_bar=True)
        agent.save(out)

    except Exception as e:
        print(e)
