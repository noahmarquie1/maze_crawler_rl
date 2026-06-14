from constants import (
    FACTORY_MAPPING,
    SCOUT_MAPPING,
    WORKER_MAPPING,
    MINER_MAPPING,
    MAX_HEIGHT_REWARD,
    USE_NOAHS_REWARD_FUNC,
)
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make
import os
import contextlib
from opponent import decision_tree_opponent
from kaggle_patches import patch_kaggle_schema_validation

# Applied at import so every SubprocVecEnv worker (which imports this module)
# skips the redundant per-step schema validation.
patch_kaggle_schema_validation()

MAX_GAME_STEPS = 500  # Crawl runs to step 500; used to normalize the timestep stat


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

        self.observation_space = spaces.Dict(
            {
                # 0-3,  walls n, e, s, w
                # 4-7,  robots factory, scout, worker, miner
                # 8,    crystals
                # 9,    mines
                # 10,   factory energy (painted at the factory cell)
                # 11,   factory move cooldown (painted at the factory cell)
                # 12,   factory jump cooldown (painted at the factory cell)
                "spatial": spaces.Box(0, np.inf, shape=(13, 20, 20)),
                # 0, normalized game timestep (global FiLM conditioning)
                "stats": spaces.Box(0, 1, shape=(1,)),
            }
        )

        self.game_obs = None
        self.prev_game_obs = None

        self.base_env = None
        self.trainer = None
        self.make_trainer_env()

        self.prev_factory_row = 0
        self.prev_mine_count = 0
        self.prev_robot_count = 0

        self.cardinal = ["NORTH", "EAST", "SOUTH", "WEST"]
        self.cardinal_bitwise = {
            "NORTH": 1,
            "EAST": 2,
            "SOUTH": 4,
            "WEST": 8,
        }

    def make_trainer_env(self, seed=None):
        configuration = {"randomSeed": int(seed)} if seed is not None else None
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            self.base_env = make("crawl", configuration=configuration)
        self.trainer = self.base_env.train([None, decision_tree_opponent])
        self.game_obs = self.trainer.reset()

    def action_masks(self):
        mask = np.zeros((400, 13), dtype=bool)
        mask[:, 0] = True  # "IDLE" always valid

        type_valid_actions = {
            0: range(12),  # Factory: 0-11
            1: range(5),  # Scout: 0-4
            2: range(13),  # Worker: 0-12
            3: range(6),  # Miner: 0-5
        }

        for robot, robot_obs in self.game_obs.robots.items():
            rtype, col, row, energy, owner = (
                int(robot_obs[0]),
                int(robot_obs[1]),
                int(robot_obs[2]),
                robot_obs[3],
                robot_obs[4],
            )
            if owner != self.game_obs.player:
                continue

            row = min(int(robot_obs[2]) - int(self.game_obs.southBound), 19)
            col = min(int(robot_obs[1]), 19)

            idx = row * 20 + col
            mask[idx, :] = False
            mask[idx, list(type_valid_actions[rtype])] = True

        return mask.flatten()

    def format_obs(self, base_obs):
        # Shape: (C=10, H=20, W=20) — channels-first for PyTorch CNN
        obs = {
            # Spatial is:
            # 1. (0-3) walls
            # 2. (4-7) robot types
            # 3. (8) crystals
            # 4. (9) mines
            # 5. (10) factory energy, painted at the factory cell
            # 6. (11) factory move cooldown, painted at the factory cell
            # 7. (12) factory jump cooldown, painted at the factory cell
            "spatial": np.zeros((13, 20, 20), dtype=np.float32),
            # Stats are:
            # 1. normalized game timestep
            "stats": np.zeros((1,), dtype=np.float32),
        }
        for robot, robot_obs in base_obs.robots.items():
            if robot_obs[4] != base_obs.player:
                continue

            type = robot_obs[0]
            row = min(int(robot_obs[2]) - int(base_obs.southBound), 19)
            col = min(int(robot_obs[1]), 19)
            obs["spatial"][4 + type, row, col] = 1

            if robot == "0-0":
                obs["spatial"][10, row, col] = robot_obs[3] / 1000  # factory energy
                obs["spatial"][11, row, col] = robot_obs[5] / 10  # factory move cd
                obs["spatial"][12, row, col] = robot_obs[6] / 10  # factory jump cd

        for coord, energy in base_obs.crystals.items():
            row = min(int(coord.split(",")[1]) - int(base_obs.southBound), 19)
            col = min(int(coord.split(",")[0]), 19)
            obs["spatial"][8, row, col] = 1

        for coord, info in base_obs.mines.items():
            row = min(int(coord.split(",")[1]) - int(base_obs.southBound), 19)
            col = min(int(coord.split(",")[0]), 19)
            obs["spatial"][9, row, col] = 1

        # Wall information
        walls = np.array(base_obs.walls, dtype=np.int8).reshape(20, 20)
        obs["spatial"][0] = ((walls & 1) != 0).astype(np.float32)
        obs["spatial"][1] = ((walls & 2) != 0).astype(np.float32)
        obs["spatial"][2] = ((walls & 4) != 0).astype(np.float32)
        obs["spatial"][3] = ((walls & 8) != 0).astype(np.float32)

        # Global stats: normalized game timestep
        obs["stats"][0] = self.timestep / MAX_GAME_STEPS

        return obs

    def detect_outcome(self, done, info):
        # Win/loss/draw detection runs regardless of the active reward function so
        # GameMetricsCallback can log win rate and score margin either way.
        if not done:
            return

        our_score = self.base_env.state[0].reward
        opponent_score = self.base_env.state[1].reward

        if our_score > opponent_score:
            outcome = "win"
        elif our_score < opponent_score:
            outcome = "loss"
        else:
            outcome = "draw"

        info["outcome"] = outcome
        info["final_scores"] = [our_score, opponent_score]

    def noahs_reward(self, obs, action, done):
        # Obs are formatted how Kaggle provides them
        # Action follows Kaggle formatting as well (per-robot action strings)
        reward = 0
        if done:
            reward = -100
            return reward

        reward += 1  # Survival

        walls = np.array(obs.walls, dtype=np.int8).reshape(20, 20)
        for robot, robot_obs in obs.robots.items():
            if robot_obs[4] != obs.player:
                continue

            row = min(int(robot_obs[2]) - int(obs.southBound), 19)
            col = min(int(robot_obs[1]), 19)

            if robot == "0-0":  # Factory
                if "JUMP" in action[robot]:
                    reward -= 0.5  # Jumping is costly and should be avoided
                elif action[robot] in self.cardinal:
                    if (
                        walls[row, col] & self.cardinal_bitwise[action[robot]]
                    ) == 0:  # Wall is not in the way
                        reward += 2  # reward for going in a correct direction
                        if action[robot] == "NORTH":
                            reward += 0.25  # extra (small) reward for going north when possible
                    else:
                        reward -= 1  # penalty for bumping into a wall
                elif action[robot] == "IDLE":
                    reward -= 0.25  # Idle is not the worst thing in the world but unideal, penalized

        return reward

    def michaels_reward(self, obs, action, done):
        # Terminal win/loss plus height shaping. `action` is the per-robot action
        # dict; the factory's action is read from it for the jump penalty.
        reward = 0
        if done:
            our_score = self.base_env.state[0].reward
            opponent_score = self.base_env.state[1].reward
            if our_score > opponent_score:
                reward += 50.0
            elif our_score < opponent_score:
                reward -= 50.0
        else:
            reward += 0.5

        # Penalize being close to the bottom and reward height using the current factory row
        curr_factory_obs = obs.robots.get("0-0")
        if curr_factory_obs is not None:
            board_height = 20
            row = curr_factory_obs[2]
            relative_height = (row - obs.southBound) / board_height
            reward += MAX_HEIGHT_REWARD * relative_height

            is_close_to_bottom = row - obs.southBound < 3
            if is_close_to_bottom:
                reward -= 5.0

        # Penalize invalid jumps using the pre-step factory jump cooldown
        factory_action = action.get("0-0")
        prev_factory_obs = (
            self.prev_game_obs.robots.get("0-0")
            if self.prev_game_obs is not None
            else None
        )
        if factory_action is not None and prev_factory_obs is not None:
            prev_jump_cooldown = prev_factory_obs[6]
            if factory_action.startswith("JUMP") and prev_jump_cooldown > 0:
                reward -= 2.0

        return reward

    def reward(self, obs, action, done):
        if USE_NOAHS_REWARD_FUNC:
            return self.noahs_reward(obs, action, done)
        return self.michaels_reward(obs, action, done)

    def step(self, action):
        self.prev_game_obs = self.game_obs
        game_action = game_agent(self.game_obs, action)
        self.game_obs, _, done, info = self.trainer.step(game_action)
        self.timestep += 1

        reward = self.reward(self.game_obs, game_action, done)
        self.detect_outcome(done, info)

        truncated = 0
        return (
            self.format_obs(self.game_obs),
            reward,
            done,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep = 0

        self.prev_factory_row = 0
        self.prev_mine_count = 0
        self.prev_robot_count = 0
        self.prev_game_obs = None

        game_seed = (
            seed
            if seed is not None
            else self.np_random.integers(0, np.iinfo(np.int32).max)
        )
        self.make_trainer_env(game_seed)
        return self.format_obs(self.game_obs), {"seed": int(game_seed)}

    def render(self, mode="html", width=800, height=800, **kwargs):
        return self.base_env.render(mode=mode, width=width, height=height, **kwargs)

    def close(self):
        pass
