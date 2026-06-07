import os

from constants import *
from kaggle_environments import make
from env import CrawlEnv, debug_agent
import numpy as np
import torch
from torch import nn

# Config
OBS_DIM = 20*20*5
ACTION_DIM = 8
rl_env = CrawlEnv()

if __name__ == "__main__":
    BASE_DIR = ""
else:
    BASE_DIR = "/kaggle_simulations/agent"


# RL Classes
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, net_arch=None):
        net_arch = net_arch or [64, 64]
        super().__init__()
        
        # Match SB3's mlp_extractor.policy_net
        layers = []
        in_dim = obs_dim
        for hidden in net_arch:
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        self.mlp_extractor = nn.ModuleDict({
            "policy_net": nn.Sequential(*layers)
        })
        
        # Match SB3's action_net
        self.action_net = nn.Linear(in_dim, action_dim)

    def forward(self, x):
        x = self.mlp_extractor["policy_net"](x)
        return self.action_net(x)
    

# Custom RL Setup
policy = PolicyNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
policy.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "policy_weights.pt")), 
    strict=False
)
policy.eval()


def rl_agent(obs) -> int: 
    obs_array = np.array(obs).flatten()
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action_logits = policy(obs_tensor)
    action = action_logits.argmax(dim=-1).item()
    return action


def agent(obs, config): # Main kaggle agent
    actions = {}
    for uid, data in obs.robots.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        if owner != obs.player:
            continue
        if rtype == 0:  
            actions[uid] = ACTION_MAPPING[rl_agent(rl_env.format_obs(obs, rl_env.timestep))]
        else:
            actions[uid] = "NORTH"
    return actions


# Main Loop - for Debugging
if __name__ == "__main__":
    kaggle_env = make("crawl", configuration={"randomSeed": 42})
    kaggle_env.run([agent, debug_agent])

    html_out = kaggle_env.render(mode="html", width=800, height=800)
    with open("replay.html", "w") as f:
        f.write(html_out)

    print(f"Game finished successfully. Written to replay.html")