import os

from constants import *
from opponent import decision_tree_opponent
import numpy as np
import torch
from torch import nn
from typing import Mapping
import contextlib

with contextlib.redirect_stdout(open(os.devnull, 'w')), \
     contextlib.redirect_stderr(open(os.devnull, 'w')):
    
    from kaggle_environments import make
    from env import CrawlEnv, game_agent
    rl_env = CrawlEnv()

# Config
OBS_DIM = 20*20*10 + 4
ACTION_DIM = 13*400
DEBUG = True


class InferenceWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.pi_extractor = policy.pi_features_extractor
        self.action_net   = policy.action_net

    def forward(self, spatial: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        obs = {"spatial": spatial, "stats": stats}
        trunk  = self.pi_extractor(obs)
        logits = self.action_net(trunk)
        return logits


if __name__ == "__main__":
    BASE_DIR = ""
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load("ppo_crawl.zip", env=rl_env)
    p = model.policy
    p.eval()
    wrapper = InferenceWrapper(p)
    wrapper.eval()

    example_spatial = torch.zeros(1, 13, 20, 20)
    example_stats   = torch.zeros(1, 1)

    traced = torch.jit.trace(wrapper, (example_spatial, example_stats))
    traced.save("policy_traced.pt")

else:
    BASE_DIR = "/kaggle_simulations/agent"
    
    
_policy = torch.jit.load(
    os.path.join(BASE_DIR, "policy_traced.pt"), map_location="cpu"
)
_policy.eval()

def agent(obs, config):
    rl_obs  = rl_env.format_obs(obs)
    spatial = torch.tensor(rl_obs["spatial"], dtype=torch.float32).unsqueeze(0)
    stats   = torch.tensor(rl_obs["stats"],   dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = _policy(spatial, stats)
    action = logits.squeeze(0).view(400, 13).argmax(dim=-1).numpy()
    return game_agent(obs, action)


if __name__ == "__main__":
    kaggle_env = make("crawl")
    if not DEBUG:
        kaggle_env.run([agent, decision_tree_opponent])

    # Optional - debugging mode (does not render)
    else:
        trainer = kaggle_env.train([None, decision_tree_opponent])
        obs = trainer.reset()
        done = False
        while not done:
            action = agent(obs, None)
            obs, reward, done, info = trainer.step(action)


    html_out = kaggle_env.render(mode="html", width=800, height=800)
    with open("replay.html", "w") as f:
        f.write(html_out)

    print(f"Game finished successfully. Written to replay.html")

