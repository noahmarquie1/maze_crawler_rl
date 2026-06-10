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
    n_envs = 8

    env = SubprocVecEnv([lambda: Monitor(CrawlEnv()) for _ in range(n_envs)])

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
            n_steps=512,
            batch_size=128,
            verbose=1,
            tensorboard_log="./agent_logs/"
        )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=20_000 // n_envs,
                save_path=checkpoint_dir,
                name_prefix=out,
            ),
        ]
    )

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
