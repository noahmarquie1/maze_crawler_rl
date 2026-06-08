from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
import os
from env import CrawlEnv

from model import CNNFeatureExtractor


if __name__ == "__main__":
    CHECKPOINT_FILE = "ppo_crawl.zip"
    out = "ppo_crawl"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    n_envs = 32
    env = SubprocVecEnv([lambda: Monitor(CrawlEnv()) for _ in range(n_envs)])

    checkpoint_exists = os.path.exists(CHECKPOINT_FILE)
    if checkpoint_exists:
        agent = PPO.load(CHECKPOINT_FILE, env=env)
        print(f"Resuming from {CHECKPOINT_FILE}")

    else:
        agent = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            policy_kwargs={"features_extractor_class": CNNFeatureExtractor},
            batch_size=128,
            verbose=1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=checkpoint_dir,
        name_prefix=out,
    )

    try:
        agent.learn(
            total_timesteps=int(4e5),
            log_interval=1,
            progress_bar=True,
            callback=checkpoint_callback,
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
