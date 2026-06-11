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

        # self.observation_space = spaces.Box(
        #    low=0, high=1, shape=(5, 20, 20), dtype=np.float32
        # )

        self.observation_space = spaces.Dict({
            # 0-3, walls n, e, s, w
            # 4-7, robots factory, scout, worker, miner
            "spatial": spaces.Box(0, 1, shape=(10, 20, 20)),
            "stats": spaces.Box(0, 5, shape=(4,)),
        })

        self.game_obs = None

        self.trainer = None
        self.make_trainer_env()
        self.southBound = 0

        self.prev_factory_row = 0
        self.prev_mine_count = 0
        self.prev_robot_count = 0


    def make_trainer_env(self):
        base_env = make("crawl")
        self.trainer = base_env.train([None, decision_tree_opponent])
        self.game_obs = self.trainer.reset()
    
    
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
            if robot_obs[4] != base_obs.player:
                continue

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
        walls = np.array(base_obs.walls, dtype=np.int8).reshape(20, 20)
        obs["spatial"][0] = ((walls & 1) != 0).astype(np.float32)
        obs["spatial"][1] = ((walls & 2) != 0).astype(np.float32)
        obs["spatial"][2] = ((walls & 4) != 0).astype(np.float32)
        obs["spatial"][3] = ((walls & 8) != 0).astype(np.float32)

        # More stats
        obs["stats"][1] = self.timestep

        return obs
    

    def reward(self, obs, done):
        if done:
            return -50.0

        reward = 0.0
        player = obs.player
        southBound = int(obs.southBound)

        # ── Collect current state ──────────────────────────────────────
        my_robots = {
            uid: data for uid, data in obs.robots.items()
            if data[4] == player
        }

        factory = my_robots.get("0-0")
        miners  = {uid: d for uid, d in my_robots.items() if d[0] == 3}
        scouts  = {uid: d for uid, d in my_robots.items() if d[0] == 1}
        workers = {uid: d for uid, d in my_robots.items() if d[0] == 2}

        mines   = obs.mines    # dict: "col,row" -> info
        crystals = obs.crystals  # dict: "col,row" -> energy

        # ── 1. Factory northward progress ─────────────────────────────
        # Reward moving north (increasing relative row), penalize moving south
        if factory is not None:
            factory_row = int(factory[2]) - southBound
            row_delta = factory_row - self.prev_factory_row
            reward += row_delta * 3.0
            self.prev_factory_row = factory_row

            # Survival bonus — just being far north is good
            reward += factory_row * 0.05

            # Penalty for being close to the southern boundary
            if factory_row < 3:
                reward -= 3.0

        # ── 2. Mine rewards ───────────────────────────────────────────
        mine_count = len(mines)

        # Reward creating new mines, penalize losing them
        mine_delta = mine_count - self.prev_mine_count
        reward += mine_delta * 5.0       # big reward for placing a new mine
        reward += mine_count * 1.0       # ongoing reward for keeping mines alive
        self.prev_mine_count = mine_count

        # ── 3. Miner proximity to mines ───────────────────────────────
        # Reward miners for being close to mine locations
        if miners and mines:
            mine_coords = [
                (int(c.split(",")[1]) - southBound, int(c.split(",")[0]))
                for c in mines.keys()
            ]
            for uid, mdata in miners.items():
                mrow = int(mdata[2]) - southBound
                mcol = int(mdata[1])
                # Minimum manhattan distance to any mine
                min_dist = min(
                    abs(mrow - mr) + abs(mcol - mc)
                    for mr, mc in mine_coords
                )
                reward += max(0, (10 - min_dist)) * 0.1  # closer = more reward, max 1.0

        # ── 4. Scout/Worker proximity to crystals ─────────────────────
        if (scouts or workers) and crystals:
            crystal_coords = [
                (int(c.split(",")[1]) - southBound, int(c.split(",")[0]))
                for c in crystals.keys()
            ]
            for uid, rdata in {**scouts, **workers}.items():
                rrow = int(rdata[2]) - southBound
                rcol = int(rdata[1])
                min_dist = min(
                    abs(rrow - cr) + abs(rcol - cc)
                    for cr, cc in crystal_coords
                )
                reward += max(0, (10 - min_dist)) * 0.05  # max 0.5 per robot

        # ── 5. Robot diversity bonus ──────────────────────────────────
        # Encourage building a balanced team rather than spamming one type
        robot_count = len(my_robots)
        robot_delta = robot_count - self.prev_robot_count
        reward += robot_delta * 2.0   # reward for building new robots
        self.prev_robot_count = robot_count

        reward += len(miners) * 0.3   # ongoing bonus per miner
        reward += len(scouts) * 0.1   # smaller bonus per scout
        reward += len(workers) * 0.2  # medium bonus per worker

        # ── 6. Wall navigation bonus ──────────────────────────────────
        # Reward scouts for covering new ground — proxy for maze navigation
        # Uses the northernmost scout row as a signal
        if scouts:
            best_scout_row = max(
                int(d[2]) - southBound for d in scouts.values()
            )
            reward += best_scout_row * 0.1

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

        self.prev_factory_row = 0
        self.prev_mine_count = 0
        self.prev_robot_count = 0

        self.make_trainer_env()
        return self.format_obs(self.game_obs), {}
    

    def close(self):
        pass
