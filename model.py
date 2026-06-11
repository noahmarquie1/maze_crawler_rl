from typing import Mapping

from gymnasium import spaces
import torch
from torch.nn import functional as F
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        cnn_head_dim: int = 64,
        metadata_head_dim: int = 4,
    ):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.LazyConv2d(8, kernel_size=3, stride=1, padding=1),  # (8, 20, 20)
            nn.ReLU(),
            nn.LazyConv2d(16, kernel_size=3, stride=1, padding=1),  # (16, 20, 20)
            nn.ReLU(),
            nn.LazyConv2d(4, kernel_size=1, stride=1),  # (4, 20, 20)
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
            nn.Linear(cnn_output_dim, cnn_head_dim),
            nn.ReLU(),
        )
        self.metadata_head = nn.Sequential(
            nn.Linear(metadata_shape[0], metadata_head_dim),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(cnn_head_dim + metadata_head_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cnn_features = self.cnn(observations["spatial"])
        cnn_out: torch.Tensor = self.cnn_head(cnn_features)
        metadata_out: torch.Tensor = self.metadata_head(
            # F.normalize(observations["stats"], dim=1)
            observations["stats"]
        )

        return self.linear(torch.concat((cnn_out, metadata_out), dim=1))