from constants import ACTION_MAPPING
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make

from opponent import decision_tree_opponent


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

        # self.observation_space = spaces.Box(
        #    low=0, high=1, shape=(5, 20, 20), dtype=np.float32
        # )

        self.observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(0, 1, shape=(5, 20, 20)),
                # Stats: energy, move cooldown, jump cooldown
                "stats": spaces.Box(
                    low=np.zeros(3, dtype=np.float32),
                    high=np.array([5, 2, 20], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        self.game_obs = None

        self.base_env = None
        self.trainer = None
        self._create_game()

    def _create_game(self, seed=None):
        configuration = {"randomSeed": int(seed)} if seed is not None else None
        self.base_env = make("crawl", configuration=configuration)
        self.trainer = self.base_env.train([None, decision_tree_opponent])

    def format_obs(self, base_obs, timestep):
        # Shape: (C=5, H=20, W=20) — channels-first for PyTorch CNN
        obs = {
            "spatial": np.zeros((5, 20, 20), dtype=np.float32),
            "stats": np.zeros((3,), dtype=np.float32),
        }
        if "0-0" in base_obs.robots.keys():
            robot_obs = base_obs.robots["0-0"]
            obs["spatial"][
                0,
                min(int(robot_obs[2]) - int(base_obs.southBound), 19),  # row
                min(int(robot_obs[1]), 19),  # col
            ] = 1
            obs["stats"] = np.array(
                [robot_obs[3] / 1000, robot_obs[5], robot_obs[6]],
                dtype=np.float32,
            )

        walls = np.array(base_obs.walls, dtype=np.int8).reshape(20, 20)
        obs["spatial"][1] = ((walls & 1) != 0).astype(np.float32)
        obs["spatial"][2] = ((walls & 2) != 0).astype(np.float32)
        obs["spatial"][3] = ((walls & 4) != 0).astype(np.float32)
        obs["spatial"][4] = ((walls & 8) != 0).astype(np.float32)

        return obs

    def step(self, action):
        self.timestep += 1
        agent_action = ACTION_MAPPING[action]
        game_action = game_agent(self.game_obs, agent_action)

        prev_game_obs = self.game_obs
        self.game_obs, _, done, info = self.trainer.step(game_action)

        reward = 0
        if done:
            our_score = self.base_env.state[0].reward
            opponent_score = self.base_env.state[1].reward

            if our_score > opponent_score:
                reward += 100.0
                outcome = "win"
            elif our_score < opponent_score:
                reward -= 100.0
                outcome = "loss"
            else:
                outcome = "draw"

            info["outcome"] = outcome
            info["final_scores"] = [our_score, opponent_score]
        else:
            reward += 1.0

        # Penalize Invalid Moves
        prev_factory_obs = prev_game_obs.robots.get("0-0", None)
        if prev_factory_obs is not None:
            # prev_move_cd = prev_factory_obs[5]
            prev_jump_cooldown = prev_factory_obs[6]
            if agent_action.startswith("JUMP") and prev_jump_cooldown > 0:
                reward -= 2.0

        curr_factory_obs = self.game_obs.robots.get("0-0", None)
        if curr_factory_obs is not None:
            row = curr_factory_obs[2]
            is_close_to_bottom = row - self.game_obs.southBound < 3
            if is_close_to_bottom:
                reward -= 2.0

        truncated = 0
        return (
            self.format_obs(self.game_obs, self.timestep),
            reward,
            done,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep = 0
        game_seed = (
            seed
            if seed is not None
            else self.np_random.integers(0, np.iinfo(np.int32).max)
        )
        self._create_game(game_seed)
        self.game_obs = self.trainer.reset()
        return self.format_obs(self.game_obs, self.timestep), {"seed": int(game_seed)}

    def render(self, mode="human", width=800, height=800, **kwargs):
        return self.base_env.render(mode=mode, width=width, height=height, **kwargs)

    def close(self):
        pass
