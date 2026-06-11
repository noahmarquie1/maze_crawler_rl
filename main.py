import os

from constants import *
from kaggle_environments import make
from env import CrawlEnv, game_agent
from opponent import decision_tree_opponent
import numpy as np
import torch
from torch import nn
from typing import Mapping

# Config
OBS_DIM = 20*20*10 + 4
ACTION_DIM = 13*400
rl_env = CrawlEnv()


# Custom Policy Class
class PureCNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        spatial_shape: tuple = (10, 20, 20),
        metadata_dim: int = 4,           
        features_dim: int = 128,
        cnn_head_dim: int = 64,
        metadata_head_dim: int = 4,
    ):
        super(PureCNNFeatureExtractor, self).__init__()

        self.cnn = nn.Sequential(
            nn.LazyConv2d(8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(4, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.zeros((1, *spatial_shape))
            cnn_output_dim = self.cnn(sample_input).shape[1]

        self.cnn_head = nn.Sequential(
            nn.Linear(cnn_output_dim, cnn_head_dim),
            nn.ReLU(),
        )
        self.metadata_head = nn.Sequential(
            nn.Linear(metadata_dim, metadata_head_dim),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(cnn_head_dim + metadata_head_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cnn_features = self.cnn(observations["spatial"])
        cnn_out = self.cnn_head(cnn_features)
        metadata_out = self.metadata_head(observations["stats"])
        return self.linear(torch.concat((cnn_out, metadata_out), dim=1))


# 2. The Complete Actor Network
class KaggleActor(nn.Module):
    def __init__(self, action_dim=13*400):
        super(KaggleActor, self).__init__()
        
        self.features_extractor = PureCNNFeatureExtractor()
        
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        self.action_net = nn.Linear(64, action_dim)

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        features = self.features_extractor(observations)
        hidden = self.policy_net(features)
        action_logits = self.action_net(hidden)
        return action_logits


if __name__ == "__main__":
    BASE_DIR = ""
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load("checkpoints/ppo_crawl_1900000_steps.zip", env=rl_env)
    torch.save(model.policy.state_dict(), "policy_weights.pt")
else:
    BASE_DIR = "/kaggle_simulations/agent"


# Custom RL Setup
policy = KaggleActor()

policy.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "policy_weights.pt")), 
    strict=False
)
policy.eval()


def rl_agent(obs_dict) -> np.ndarray: 
    tensor_obs = {
        key: torch.tensor(value, dtype=torch.float32).unsqueeze(0) 
        for key, value in obs_dict.items()
    }
    with torch.no_grad():
        action_logits = policy(tensor_obs)
        
    action = action_logits.squeeze(0).view(400, 13).argmax(dim=-1).numpy()
    return action


def agent(obs, config): # Main kaggle agent
    rl_obs = rl_env.format_obs(obs)
    flattened_obs = np.append(rl_obs['spatial'].flatten(), rl_obs['stats'], axis=0)
    agent_action = rl_agent(rl_obs)
    return game_agent(obs, agent_action)


# Main Loop - for Debugging
DEBUG = True


if __name__ == "__main__":
    #from sb3_contrib import MaskablePPO
    #model = MaskablePPO.load("checkpoints/ppo_crawl_200000_steps.zip", env=rl_env)
    #torch.save(model.policy.state_dict(), "policy_weights.pt")

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