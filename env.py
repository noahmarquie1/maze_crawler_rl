from constants import FACTORY_MAPPING, SCOUT_MAPPING, WORKER_MAPPING, MINER_MAPPING
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


def game_agent(obs, agent_action): # Obs should be formatted already
    actions = {}

    for uid, data in obs.robots.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        if owner != obs.player:
            continue

        action = agent_action[(row - 1) * 20 + (col - 1)]
        mappings = {
            0: FACTORY_MAPPING,
            1: SCOUT_MAPPING,
            2: WORKER_MAPPING,
            3: MINER_MAPPING,
        }

        if action in mappings[rtype].keys():
            actions[uid] = mappings[rtype][action]
        else:
            actions[uid] = "IDLE"

    return actions


class CrawlEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.MultiDiscrete([13] * 400)
        self.timestep: int = 0

        #self.observation_space = spaces.Box(
        #    low=0, high=1, shape=(5, 20, 20), dtype=np.float32
        #)

        self.observation_space = spaces.Dict({
            # 0-3, walls n, e, s, w
            # 4-7, robots factory, scout, worker, miner
            "spatial": spaces.Box(0, 1, shape=(8, 20, 20)),
            "stats": spaces.Box(0, 5, shape=(1,)),
        })

        self.game_obs = None

        self.base_env = make("crawl")
        self.trainer = self.base_env.train([None, decision_tree_opponent])


    def action_masks(self):
        mask = np.zeros((400, 13), dtype=bool)
        mask[:, 0] = True  # "IDLE" always valid 

        type_valid_actions = {
            0: range(12),  # Factory: 0-11
            1: range(5),   # Scout: 0-4
            2: range(13),  # Worker: 0-12
            3: range(6),   # Miner: 0-5
        }

        for uid, data in self.game_obs.robots.items():
            rtype, col, row, energy, owner = int(data[0]), int(data[1]), int(data[2]), data[3], data[4]
            if owner != self.game_obs.player:
                continue
            idx = (row - 1) * 20 + (col - 1)
            mask[idx, :] = False
            mask[idx, list(type_valid_actions[rtype])] = True

        return mask.flatten()


    def format_obs(self, base_obs, timestep):
        # Shape: (C=8, H=20, W=20) — channels-first for PyTorch CNN
        obs = {
            "spatial": np.zeros((8, 20, 20), dtype=np.float32),
            "stats": np.zeros((1,), dtype=np.float32)
        }
        for robot, r_vals in base_obs.robots.items():
            type = r_vals[0]
            row = r_vals[2]
            col = r_vals[1]
            obs["spatial"][4+type, row - 1, col - 1] = 1

        walls = np.array(base_obs.walls, dtype=np.float32).reshape(20, 20)
        obs["spatial"][0] = (walls == 1).astype(np.float32)
        obs["spatial"][1] = (walls == 2).astype(np.float32)
        obs["spatial"][2] = (walls == 4).astype(np.float32)
        obs["spatial"][3] = (walls == 8).astype(np.float32)

        return obs

    def step(self, action):
        game_action = game_agent(self.game_obs, action)
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
