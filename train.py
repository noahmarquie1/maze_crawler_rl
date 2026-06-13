from collections import deque

import numpy as np
import os
import glob
import zipfile
import torch
import io
import contextlib
import warnings
from misc.log_stopper import LogStopper

# Constants
n_envs = 16

# Suppress warnings on import
with LogStopper():
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from model import CrawlMaskablePolicy
    from env import CrawlEnv
    from evaluate_model import run_n_episodes


class GameMetricsCallback(BaseCallback):
    def __init__(self, window_size: int = 100):
        super().__init__()
        self.outcomes = deque(maxlen=window_size)
        self.score_margins = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.completed_episodes = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done or "outcome" not in info:
                continue

            outcome = info["outcome"]
            self.outcomes.append(outcome)
            self.completed_episodes += 1

            final_scores = info.get("final_scores")
            if final_scores is not None:
                self.score_margins.append(final_scores[0] - final_scores[1])

            episode = info.get("episode")
            if episode is not None:
                self.episode_lengths.append(episode["l"])

        if self.outcomes:
            self.logger.record(
                "game/win_rate",
                np.mean([outcome == "win" for outcome in self.outcomes]),
            )
            self.logger.record(
                "game/draw_rate",
                np.mean([outcome == "draw" for outcome in self.outcomes]),
            )
            self.logger.record(
                "game/loss_rate",
                np.mean([outcome == "loss" for outcome in self.outcomes]),
            )
            self.logger.record("game/episodes", self.completed_episodes)

        if self.score_margins:
            self.logger.record("game/mean_score_margin", np.mean(self.score_margins))

        if self.episode_lengths:
            self.logger.record(
                "game/mean_episode_length", np.mean(self.episode_lengths)
            )

        return True


# Callbacks
class EvalCallback(BaseCallback):
    def __init__(
        self, eval_freq: int, n_episodes: int = 5, replay_dir: str = "eval_replays"
    ):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.replay_dir = replay_dir
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps // self.eval_freq > self._last_eval:
            self._last_eval = self.num_timesteps // self.eval_freq
            eval_env = CrawlEnv()
            output_dir = os.path.join(self.replay_dir, f"step_{self.num_timesteps}")
            run_n_episodes(eval_env, self.model, self.n_episodes, output_dir=output_dir)
            eval_env.close()
        return True


# Helpers
def get_latest_checkpoint(checkpoint_dir, prefix):
    # Find all matching checkpoint files
    files = glob.glob(os.path.join(checkpoint_dir, f"{prefix}_*.zip"))
    if not files:
        return None
    
    # Sort files by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    
    for file_path in files:
        try:
            with zipfile.ZipFile(file_path, "r") as archive:
                pth_files = [name for name in archive.namelist() if name.endswith('.pth')]
                
                if not pth_files:
                    raise RuntimeError("No PyTorch weight files found in archive.")

                for pth_file in pth_files:
                    weight_data = archive.read(pth_file)
                    buffer = io.BytesIO(weight_data)
                    torch.load(buffer, map_location="cpu", weights_only=True)
            
            return file_path
            
        except (zipfile.BadZipFile, RuntimeError, KeyError, EOFError) as e:
            print(f"Warning: Checkpoint {os.path.basename(file_path)} is corrupted ({type(e).__name__}). Skipping...")
            continue
            
    return None


if __name__ == "__main__":
    USE_CHECKPOINT = True

    checkpoint_file = "ppo_crawl.zip"
    out = "ppo_crawl"
    checkpoint_dir = "checkpoints"
    tensorboard_log_dir = "logs/tensorboard"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    with LogStopper():
        env = SubprocVecEnv([lambda: Monitor(CrawlEnv()) for _ in range(n_envs)])
    print(f"Num envs: {env.num_envs}")

    dir_checkpoint = get_latest_checkpoint(checkpoint_dir, out)
    checkpoint_file = dir_checkpoint if not dir_checkpoint is None else checkpoint_file

    checkpoint_exists = USE_CHECKPOINT and os.path.exists(checkpoint_file)
    if checkpoint_exists:
        agent = MaskablePPO.load(
            checkpoint_file, env=env, tensorboard_log=tensorboard_log_dir
        )
        print(f"Resuming from {checkpoint_file}")

    else:
        agent = MaskablePPO(
            CrawlMaskablePolicy,
            env,
            n_steps=512,
            batch_size=512,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
        )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=50_000 // n_envs,
                save_path=checkpoint_dir,
                name_prefix=out,
            ),
            GameMetricsCallback(window_size=100),
            EvalCallback(eval_freq=100_000),
        ]
    )

    try:
        agent.learn(
            total_timesteps=int(3e6),
            log_interval=1,
            progress_bar=True,
            callback=callbacks,
            reset_num_timesteps=not checkpoint_exists,
            tb_log_name="crawl_ppo",
        )

    except Exception as e:
        raise e
    finally:
        if os.path.exists(out + ".zip"):
            os.remove(out + ".zip")
        agent.save(out)
        env.close()
        print(f"Agent saved to {out}.zip - training complete.")
