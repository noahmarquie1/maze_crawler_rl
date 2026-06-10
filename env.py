from constants import FACTORY_MAPPING, SCOUT_MAPPING, WORKER_MAPPING, MINER_MAPPING
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make
from opponent import decision_tree_opponent


def game_agent(obs, agent_action):
    actions = {}

    for robot, robot_obs in obs.robots.items():
        rtype = robot_obs[0]
        owner = robot_obs[4]

        row = min(int(robot_obs[2]) - int(obs.southBound), 19)
        col = min(int(robot_obs[1]), 19)

        if owner != obs.player:
            continue

        action = agent_action[row * 20 + col]
        mappings = {
            0: FACTORY_MAPPING,
            1: SCOUT_MAPPING,
            2: WORKER_MAPPING,
            3: MINER_MAPPING,
        }

        if action in mappings[rtype].keys():
            actions[robot] = mappings[rtype][action]
        else:
            actions[robot] = "IDLE"

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
            "spatial": spaces.Box(0, 1, shape=(10, 20, 20)),
            "stats": spaces.Box(0, 5, shape=(4,)),
        })

        self.game_obs = None

        self.base_env = make("crawl")
        self.trainer = self.base_env.train([None, decision_tree_opponent])
        self.southBound = 0


    def action_masks(self):
        mask = np.zeros((400, 13), dtype=bool)
        mask[:, 0] = True  # "IDLE" always valid 

        type_valid_actions = {
            0: range(12),  # Factory: 0-11
            1: range(5),   # Scout: 0-4
            2: range(13),  # Worker: 0-12
            3: range(6),   # Miner: 0-5
        }

        for robot, robot_obs in self.game_obs.robots.items():
            rtype, col, row, energy, owner = int(robot_obs[0]), int(robot_obs[1]), int(robot_obs[2]), robot_obs[3], robot_obs[4]
            if owner != self.game_obs.player:
                continue

            row = min(int(robot_obs[2]) - int(self.game_obs.southBound), 19)
            col = min(int(robot_obs[1]), 19)

            idx = row * 20 + col
            mask[idx, :] = False
            mask[idx, list(type_valid_actions[rtype])] = True

        return mask.flatten()


    def format_obs(self, base_obs):
        # Shape: (C=8, H=20, W=20) — channels-first for PyTorch CNN
        obs = {
            # Spatial is:
            # 1. (0-3) walls
            # 2. (4-7) robot types
            # 3. (8) crystals
            # 4. (9) mines
            "spatial": np.zeros((10, 20, 20), dtype=np.float32),
            # Stats are:
            # 1. factory energy
            # 2. game timestep
            # 3. factory move cd
            # 4. factory jump cd
            "stats": np.zeros((4,), dtype=np.float32)
        }
        self.southBound = base_obs.southBound
        for robot, robot_obs in base_obs.robots.items():
            type = robot_obs[0]
            row = min(int(robot_obs[2]) - int(self.southBound), 19)
            col = min(int(robot_obs[1]), 19)
            obs["spatial"][4+type, row, col] = 1

            if robot == "0-0":
                obs['stats'][0] = robot_obs[3] / 1000
                obs['stats'][2] = robot_obs[5] / 10
                obs['stats'][3] = robot_obs[6] / 10

        for coord, energy in base_obs.crystals.items():
            row = min(int(coord.split(",")[1]) - int(self.southBound), 19)
            col = min(int(coord.split(",")[0]), 19)
            obs["spatial"][8,row,col] = 1

        for coord, info in base_obs.mines.items():
            row = min(int(coord.split(",")[1]) - int(self.southBound), 19)
            col = min(int(coord.split(",")[0]), 19)
            obs["spatial"][9,row,col] = 1


        # Wall information
        walls = np.array(base_obs.walls, dtype=np.float32).reshape(20, 20)
        obs["spatial"][0] = (walls == 1).astype(np.float32)
        obs["spatial"][1] = (walls == 2).astype(np.float32)
        obs["spatial"][2] = (walls == 4).astype(np.float32)
        obs["spatial"][3] = (walls == 8).astype(np.float32)

        # More stats
        obs["stats"][1] = self.timestep

        return obs
    

    def reward(self, obs, done):
        mines = obs.mines
        mine_count = len(mines.keys())

        reward = 0
        if done:
            reward -= 100.0
        else:
            reward += 1.0
            reward += 2 * mine_count

        for uid, data in self.game_obs.robots.items():
            rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
            if owner != self.game_obs.player:
                continue
            is_close_to_bottom = row - self.game_obs.southBound < 3
            if rtype == 0 and is_close_to_bottom:  # Factory
                reward -= 2.0

        return reward


    def step(self, action):
        game_action = game_agent(self.game_obs, action)
        self.game_obs, _, done, info = self.trainer.step(game_action)
        reward = self.reward(self.game_obs, done)
        self.timestep += 1

        truncated = 0
        return (
            self.format_obs(self.game_obs),
            reward,
            done,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.game_obs = self.trainer.reset()
        return self.format_obs(self.game_obs), {}

    def render(self, mode="human", width=800, height=800, **kwargs):
        return self.base_env.render(mode=mode, width=width, height=height, **kwargs)

    def close(self):
        pass
