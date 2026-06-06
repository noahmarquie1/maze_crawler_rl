from constants import ACTION_MAPPING
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make
from kaggle_environments.utils import Struct
from stable_baselines3 import PPO
import os


# Game and RL Agents
def game_agent(obs, fac_action):
    actions = {}
    for uid, data in obs.robots.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        if owner != obs.player:
            continue
        if rtype == 0:  # Factory
            actions[uid] = fac_action
        else:
            actions[uid] = "NORTH"
    return actions


# Environmnet
class CrawlEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Action space is:
        # - DIFFERENT DIRECTIONS JUMPING
        self.action_space = spaces.Discrete(4)
        self.timestep: int = 0

        # Observation space is a flattened array of factory state and walls
        #  - factory state: 8 elements
        #  - walls: 400 elements
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20, 20, 5), dtype=np.float32
        )
        self.game_obs = None

        self.base_env = make("crawl")
        self.trainer = self.base_env.train([None, "random"])

    def format_obs(self, base_obs, timestep):
        obs = np.zeros((20, 20, 5), dtype=np.float32)
        if "0-0" in base_obs.robots.keys():
            robot_obs = base_obs.robots["0-0"]
            obs[int(robot_obs[1]), int(robot_obs[2]), 0] = 1

        # Walls (10, 20) - first 4 items
        n_walls = np.array([wall == 1 for wall in base_obs.walls]).reshape((20, 20))
        e_walls = np.array([wall == 2 for wall in base_obs.walls]).reshape((20, 20))
        s_walls = np.array([wall == 4 for wall in base_obs.walls]).reshape((20, 20))
        w_walls = np.array([wall == 8 for wall in base_obs.walls]).reshape((20, 20))

        obs[:, :, 1] = n_walls
        obs[:, :, 2] = e_walls
        obs[:, :, 3] = s_walls
        obs[:, :, 4] = w_walls

        return obs

    def step(self, action):
        self.timestep += 1
        agent_action = ACTION_MAPPING[action]
        game_action = game_agent(self.game_obs, agent_action)
        self.game_obs, _, done, info = self.trainer.step(game_action)
        if done:
            reward = -100.0
        else:
            reward = 1.0

        truncated = 0
        return (
            self.format_obs(self.game_obs, self.timestep),
            reward,
            done,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.game_obs = self.trainer.reset()
        return self.format_obs(self.game_obs, self.timestep), {}

    def render(self, mode="human", width=800, height=800, **kwargs):
        return self.base_env.render(mode=mode, width=width, height=height, **kwargs)

    def close(self):
        self.trainer.close()


if __name__ == "__main__":
    out = "ppo_crawl"
    env = CrawlEnv()
    try:
        agent = PPO("MlpPolicy", env, verbose=1)
        agent.learn(total_timesteps=int(1e5), log_interval=1, progress_bar=True)
        if os.path.exists(out + ".zip"):
            os.remove(out + ".zip")

        agent.save(out)

    except Exception as e:
        print(e)
