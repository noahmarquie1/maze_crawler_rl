from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import os
from env import CrawlEnv
import glob
import zipfile
import torch 
import io
from evaluate_model import run_n_episodes
from model import CNNFeatureExtractor

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
    os.makedirs(checkpoint_dir, exist_ok=True)
    n_envs = 16

    env = SubprocVecEnv([lambda: Monitor(CrawlEnv()) for _ in range(n_envs)])
    print(f"Num envs: {env.num_envs}")

    dir_checkpoint = get_latest_checkpoint(checkpoint_dir, out)
    checkpoint_file = dir_checkpoint if not dir_checkpoint is None else checkpoint_file

    checkpoint_exists = USE_CHECKPOINT and os.path.exists(checkpoint_file)
    if checkpoint_exists:
        agent = MaskablePPO.load(checkpoint_file, env=env, tensorboard_log="./agent_logs/")
        print(f"Resuming from {checkpoint_file}")

    else:
        agent = MaskablePPO(
            "MultiInputPolicy",
            env,
            n_steps=1024,
            batch_size=512,
            policy_kwargs={"features_extractor_class": CNNFeatureExtractor},
            verbose=1,
            tensorboard_log="./agent_logs/"
        )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=100_000 // n_envs,
                save_path=checkpoint_dir,
                name_prefix=out,
            ),
        ]
    )

    try:
        agent.learn(
            total_timesteps=int(1e4),
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
        print(f"Agent saved to {out}.zip - training complete.")
