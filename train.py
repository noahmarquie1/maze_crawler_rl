from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import os
from env import CrawlEnv

from model import CNNFeatureExtractor


if __name__ == "__main__":
    out = "ppo_crawl"
    n_envs = 32
    env = SubprocVecEnv([lambda: Monitor(CrawlEnv()) for _ in range(n_envs)])
    agent = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        #policy_kwargs={"features_extractor_class": CNNFeatureExtractor},
        batch_size=128,
        verbose=1,
    )
    try:
        agent.learn(total_timesteps=int(4e5), log_interval=1, progress_bar=True)

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