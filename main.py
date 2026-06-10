import os

from constants import *
from kaggle_environments import make
from env import CrawlEnv, game_agent
from opponent import decision_tree_opponent
import numpy as np
import torch
from torch import nn

# Config
OBS_DIM = 20*20*10 + 4
ACTION_DIM = 13*400

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
    action = action_logits.view(400, 13).argmax(dim=-1).numpy()
    return action


def agent(obs, config): # Main kaggle agent
    rl_obs = rl_env.format_obs(obs)
    flattened_obs = np.append(rl_obs['spatial'].flatten(), rl_obs['stats'], axis=0)
    agent_action = rl_agent(flattened_obs)
    return game_agent(obs, agent_action)


# Main Loop - for Debugging
DEBUG = True



if __name__ == "__main__":
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load("checkpoints/ppo_crawl_179936_steps.zip", env=rl_env)
    torch.save(model.policy.state_dict(), "policy_weights.pt")

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