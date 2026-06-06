import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make
from kaggle_environments.utils import Struct
from stable_baselines3 import PPO
import os

ACTION_MAPPING = {
    0: "JUMP_NORTH",
    1: "JUMP_SOUTH",
    2: "JUMP_EAST",
    3: "JUMP_WEST",
}

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
        self.observation_space = spaces.Box(low=-1, high=20, shape=(409,), dtype=np.float32)
        self.game_obs = None

        base_env = make("crawl")
        self.trainer: Struct = base_env.train([None, None])

    def format_obs(self, base_obs, timestep):
        obs = np.zeros(409, dtype=np.float32)
        if "0-0" in base_obs.robots.keys():
            robot_obs = base_obs.robots['0-0']
            robot_obs[3] = robot_obs[3] / 100
            obs[:8] = np.array(robot_obs, dtype=np.float32)
        obs[8:408] = np.array(base_obs.walls, dtype=np.float32)
        obs[408] = timestep / 100
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
        return self.format_obs(self.game_obs, self.timestep), reward, done, 0, info

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.game_obs = self.trainer.reset()
        return self.format_obs(self.game_obs, self.timestep), {}

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
        agent = PPO("MlpPolicy", env, verbose=1)
        agent.learn(total_timesteps=int(3e4), log_interval=1, progress_bar=True)
        agent.save(out)

    except Exception as e:
        print(e)
