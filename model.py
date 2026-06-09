from typing import Mapping

from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.LazyConv2d(8, kernel_size=3, stride=1, padding=1),  # (8, 20, 20)
            nn.ReLU(),
            nn.LazyConv2d(16, kernel_size=3, stride=1, padding=1),  # (16, 20, 20)
            nn.ReLU(),
            nn.LazyConv2d(16, kernel_size=3, stride=1),  # (16, 18, 18)
            nn.ReLU(),
            nn.Flatten(),
        )

        spatial_observation_space = observation_space["spatial"]
        metadata_observation_space = observation_space["stats"]
        assert isinstance(spatial_observation_space, spaces.Box)
        assert isinstance(metadata_observation_space, spaces.Box)
        spatial_shape = spatial_observation_space.shape
        metadata_shape = metadata_observation_space.shape

        with torch.no_grad():
            sample_input = torch.zeros((1, *spatial_shape))
            cnn_output_dim = self.cnn(sample_input).shape[1]

        self.cnn_head = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim // 2),
            nn.ReLU(),
        )
        self.metadata_head = nn.Sequential(
            nn.Linear(metadata_shape[0], features_dim // 2),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cnn_features = self.cnn(observations["spatial"])
        x = self.cnn_head(cnn_features) + self.metadata_head(observations["stats"])

        return self.linear(x)
