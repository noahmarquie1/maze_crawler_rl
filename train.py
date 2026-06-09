from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import argparse
import os
from env import CrawlEnv

from model import CNNFeatureExtractor
from evaluate_model import run_n_episodes


class EvalCallback(BaseCallback):
    def __init__(self, eval_freq: int, n_episodes: int = 5, replay_dir: str = "eval_replays"):
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


if __name__ == "__main__":
    CHECKPOINT_FILE = "ppo_crawl.zip"
    out = "ppo_crawl"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    n_envs = 32
    n_envs = 8
    env = SubprocVecEnv([lambda: Monitor(CrawlEnv()) for _ in range(n_envs)])

    checkpoint_exists = os.path.exists(CHECKPOINT_FILE)
    if checkpoint_exists:
        agent = PPO.load(CHECKPOINT_FILE, env=env)
        print(f"Resuming from {CHECKPOINT_FILE}")

    else:
        agent = PPO(
            "MultiInputPolicy",
            env,
            n_steps=512,
            #policy_kwargs={"features_extractor_class": CNNFeatureExtractor},
            batch_size=128,
            verbose=1,
        )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=50_000 // n_envs,
            save_path=checkpoint_dir,
            name_prefix=out,
        ),
        EvalCallback(eval_freq=100_000),
    ])

    try:
        agent.learn(
            total_timesteps=int(1e5),
            log_interval=1,
            progress_bar=True,
            callback=callbacks,
            reset_num_timesteps=not checkpoint_exists,
        )

    except Exception as e:
        raise e
    finally:
        if os.path.exists(out + ".zip"):
            os.remove(out + ".zip")
        agent.save(out)
        html_out = env.render(mode="html")
        if html_out is not None:
            with open("error_replay.html", "w") as f:
                f.write(html_out)
