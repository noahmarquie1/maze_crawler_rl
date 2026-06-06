from unittest.mock import MagicMock
import sys

fake_module = MagicMock()

sys.modules["tensorboard"] = fake_module
sys.modules["tensorboard.compat"] = fake_module
sys.modules["tensorboard.compat.tf"] = fake_module
sys.modules["torch.utils.tensorboard"] = fake_module
sys.modules["torch.utils.tensorboard.writer"] = fake_module

from stable_baselines3 import PPO
from kaggle_environments import make
from eval import my_agent

MODEL_PATH = "ppo_crawl"
ACTION_MAPPING = {
    0: "NORTH",
    1: "SOUTH",
    2: "EAST",
    3: "WEST",
}

model = PPO.load(MODEL_PATH)
agent = lambda obs, config: my_agent(obs, config, model)

if __name__ == "__main__":
    kaggle_env = make("crawl", configuration={"randomSeed": 42})
    kaggle_env.run([agent, "random"])

    html_out = kaggle_env.render(mode="html", width=800, height=800)
    with open("replay.html", "w") as f:
        f.write(html_out)

    print(f"Game finished successfully. Written to replay.html")