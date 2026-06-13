from functools import partial
from typing import Mapping

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy


N_ACTION_TYPES = 13  # per-cell action vocabulary, matches MultiDiscrete([13] * 400)


class FiLM(nn.Module):
    """Feature-wise linear modulation conditioned on global stats.

    Projects the conditioning vector to a per-channel scale and shift and
    applies them to a (B, C, H, W) feature map. No normalization is applied, so
    this is a surgical addition on top of the existing conv stack. The scale is
    centered at 1 (``features * (1 + scale)``) so an untrained FiLM starts close
    to identity.
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return features * (1 + scale) + shift


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """Shared convolutional trunk with FiLM conditioning on global stats.

    FiLM is applied after each conv block (before the activation) so the global
    signal is injected early enough for a downstream 3x3 conv to propagate it
    spatially. Unlike a stock extractor this returns the 4D (B, C, H, W) trunk
    map rather than a flat vector: the spatial heads consume it directly, and the
    identity mlp_extractor (net_arch=[]) passes it through untouched. Only valid
    paired with CrawlMaskablePolicy -- a stock net_arch with Linear layers would
    expect a flat input.
    """

    def __init__(self, observation_space: spaces.Dict):
        self.out_channels = 16

        spatial_space = observation_space["spatial"]
        stats_space = observation_space["stats"]
        assert isinstance(spatial_space, spaces.Box)
        assert isinstance(stats_space, spaces.Box)

        in_channels, height, width = spatial_space.shape
        cond_dim = stats_space.shape[0]

        super().__init__(
            observation_space, features_dim=self.out_channels * height * width
        )

        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.film1 = FiLM(cond_dim, 8)
        self.conv2 = nn.Conv2d(8, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.film2 = FiLM(cond_dim, self.out_channels)
        self.conv2 = nn.Conv2d(8, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.film2 = FiLM(cond_dim, self.out_channels)

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        spatial = observations["spatial"]
        stats = observations["stats"]

        x = torch.relu(self.film1(self.conv1(spatial), stats))
        x = torch.relu(self.film2(self.conv2(x), stats))
        return x


class SpatialActionHead(nn.Module):
    """1x1 conv mapping the trunk map to per-cell action logits.

    Input is the (B, C, H, W) trunk map. Output is (B, H*W*A) ordered as
    ``cell * A + action_type`` to match the environment's flattened action mask
    (``mask.flatten()`` over a (H*W, A) array with ``cell = row * W + col``).
    """

    def __init__(self, in_channels: int, n_action_types: int = N_ACTION_TYPES):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_action_types, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.conv(features)  # (B, A, H, W)
        # permute to (B, H, W, A) then flatten -> (row * W + col) * A + action_type
        # basically just put the channels (action types) last and then flatten so its the flat action space
        return logits.permute(0, 2, 3, 1).reshape(features.shape[0], -1)


class PooledValueHead(nn.Module):
    """Mean-pooled critic head producing a scalar value.

    Input is the (B, C, H, W) trunk map; it is mean-pooled over the spatial dims
    to a global (B, C) summary, then mapped to a scalar.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=(2, 3))  # (B, C)
        return self.mlp(pooled)


class CrawlMaskablePolicy(MaskableMultiInputActorCriticPolicy):
    """Fully-convolutional spatial policy for the grid action space.

    Replaces SB3's dense ``action_net``/``value_net`` with a 1x1 conv action head
    and a mean-pooled value head. Every base-class method routes through
    ``action_net(latent_pi)`` and ``value_net(latent_vf)``, so overriding
    ``_build`` is sufficient -- ``forward``, ``evaluate_actions``, masking, and
    sampling are unchanged. ``net_arch`` is forced empty so the ``mlp_extractor``
    is an identity and the (B, C, H, W) trunk map reaches the heads untouched.
    """

    def __init__(self, *args, **kwargs):
        kwargs["net_arch"] = []
        super().__init__(*args, **kwargs)

    def make_features_extractor(self) -> CNNFeatureExtractor:
        # Own the trunk instead of receiving it via features_extractor_class, so
        # the policy is self-contained. This is the single seam SB3's inherited
        # methods route obs through, so building it here keeps every one of them
        # (forward, evaluate_actions, get_distribution, predict_values) working.
        return CNNFeatureExtractor(self.observation_space)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        channels = self.features_extractor.out_channels

        self.action_net = SpatialActionHead(channels)
        self.value_net = PooledValueHead(channels)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
