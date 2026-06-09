from constants import ACTION_MAPPING
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make


def decision_tree_opponent(obs, config):
    actions = {}
    walls = obs.walls  # flat 400-element list, row-major: walls[row * 20 + col]

    for uid, data in obs.robots.items():
        rtype, col, row, energy, owner = int(data[0]), int(data[1]), int(data[2]), data[3], data[4]

        if owner != obs.player:
            continue
        if rtype != 0:  # only factory acts
            continue
        if row == 19:  # at top edge, do nothing
            continue

        has_north_wall = bool(walls[row * 20 + col] & 1)
        actions[uid] = "JUMP_NORTH" if has_north_wall else "NORTH"

    return actions


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
        self.action_space = spaces.Discrete(8)
        self.timestep: int = 0

        #self.observation_space = spaces.Box(
        #    low=0, high=1, shape=(5, 20, 20), dtype=np.float32
        #)

        self.observation_space = spaces.Dict({
            "spatial": spaces.Box(0, 1, shape=(5, 20, 20)),
            "energy": spaces.Box(0, 5, shape=(1,)),
        })

        self.game_obs = None

        self.base_env = make("crawl")
        self.trainer = self.base_env.train([None, decision_tree_opponent])

    def format_obs(self, base_obs, timestep):
        # Shape: (C=5, H=20, W=20) — channels-first for PyTorch CNN
        obs = {
            "spatial": np.zeros((5, 20, 20), dtype=np.float32),
        }
        if "0-0" in base_obs.robots.keys():
            robot_obs = base_obs.robots["0-0"]
            obs["spatial"][0, int(robot_obs[1]) - 0, int(robot_obs[2]) - 0] = 1
            obs["energy"] = np.array([ base_obs.robots['0-0'][3] / 1000])

        walls = np.array(base_obs.walls, dtype=np.float32).reshape(20, 20)
        obs["spatial"][1] = (walls == 1).astype(np.float32)
        obs["spatial"][2] = (walls == 2).astype(np.float32)
        obs["spatial"][3] = (walls == 4).astype(np.float32)
        obs["spatial"][4] = (walls == 8).astype(np.float32)

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
        pass
