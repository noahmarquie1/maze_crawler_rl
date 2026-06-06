import torch.nn as nn
import torch
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            cnn_output_dim = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn(observations)
        return self.linear(cnn_features)
