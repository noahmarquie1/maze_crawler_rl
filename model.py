from functools import partial
from typing import Mapping

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.distributions import (
    MaskableMultiCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy


N_ACTION_TYPES = 13  # per-cell action vocabulary, matches MultiDiscrete([13] * 400)
HUGE_NEG = -1e8  # masked-logit fill; large enough to zero the softmax probability


class ConditionalInstanceNorm(nn.Module):
    """Conditional Instance Normalization conditioned on global stats.

    Normalizes a (B, C, H, W) feature map per-instance and per-channel over the
    spatial dims, then applies a conditioned per-channel scale and shift derived
    from the conditioning vector. InstanceNorm uses ``affine=False`` so the only
    affine parameters are the conditioned ones from ``proj``. The scale is
    centered at 1 (``normed * (1 + scale)``) so an untrained module starts as a
    plain instance norm.
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        normed = self.norm(features)
        scale, shift = self.proj(cond).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return normed * (1 + scale) + shift


class ResidualBlock(nn.Module):
    """Original (post-activation) residual block with Conditional Instance Norm.

    Applies ``conv -> CIN -> relu -> conv -> CIN``, adds the block input, then a
    final relu. Channel count is preserved so the skip connection lines up
    without a projection. Both CIN layers are conditioned on the same global
    stats vector.
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.norm1 = ConditionalInstanceNorm(cond_dim, num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.norm2 = ConditionalInstanceNorm(cond_dim, num_channels)

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.norm1(self.conv1(features), cond))
        x = self.norm2(self.conv2(x), cond)
        return torch.relu(x + features)


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """Shared convolutional trunk with Conditional Instance Norm on global stats.

    Conditional Instance Norm is applied after each conv block (before the
    activation) so the global signal is injected early enough for a downstream
    3x3 conv to propagate it spatially. Unlike a stock extractor this returns the
    4D (B, C, H, W) trunk
    map rather than a flat vector: the spatial heads consume it directly, and the
    identity mlp_extractor (net_arch=[]) passes it through untouched. Only valid
    paired with CrawlMaskablePolicy -- a stock net_arch with Linear layers would
    expect a flat input.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        n_residual_blocks: int = 0,
        out_channels: int = 16,
    ):
        self.out_channels = out_channels

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
        self.norm1 = ConditionalInstanceNorm(cond_dim, 8)
        self.conv2 = nn.Conv2d(8, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = ConditionalInstanceNorm(cond_dim, self.out_channels)
        self.res_blocks = nn.ModuleList(
            ResidualBlock(cond_dim, self.out_channels) for _ in range(n_residual_blocks)
        )

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        spatial = observations["spatial"]
        stats = observations["stats"]

        x = torch.relu(self.norm1(self.conv1(spatial), stats))
        x = torch.relu(self.norm2(self.conv2(x), stats))
        for block in self.res_blocks:
            x = block(x, stats)
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


class VecMaskableMultiCategoricalDistribution(MaskableMultiCategoricalDistribution):
    """
    AI SLOPIFIED Performance improvements

    Vectorized drop-in for sb3-contrib's MaskableMultiCategoricalDistribution.

    The stock implementation builds a Python list of one ``MaskableCategorical``
    per sub-action (400 of them here) and loops over that list for every
    ``log_prob``/``entropy``/``sample`` call -- the dominant cost in both rollout
    collection and training. This keeps all logits as a single
    ``(B, n_cells, n_act)`` tensor and does masking, log-prob, entropy and
    sampling with pure tensor ops. Numerically identical to the stock version
    (``log_prob`` exact; ``entropy`` agrees to float noise). Assumes a uniform
    per-cell action count, which holds for our ``MultiDiscrete([13] * 400)``.
    """

    def __init__(self, action_dims):
        super().__init__(action_dims)
        self.n_cells = len(action_dims)
        self.n_act = int(action_dims[0])
        assert all(int(d) == self.n_act for d in action_dims), (
            "VecMaskableMultiCategoricalDistribution requires a uniform action count"
        )
        self._logits = None
        self._logp = None
        self._masks = None

    def proba_distribution(self, action_logits):
        self._logits = action_logits.view(-1, self.n_cells, self.n_act)
        self._masks = None
        self._recompute()
        return self

    def apply_masking(self, masks):
        if masks is None:
            self._masks = None
        else:
            self._masks = torch.as_tensor(
                masks, dtype=torch.bool, device=self._logits.device
            ).view(-1, self.n_cells, self.n_act)
        self._recompute()

    def _recompute(self):
        logits = self._logits
        if self._masks is not None:
            logits = torch.where(self._masks, logits, torch.full_like(logits, HUGE_NEG))
        self._logp = F.log_softmax(logits, dim=-1)

    def log_prob(self, actions):
        actions = actions.view(-1, self.n_cells, 1)
        return self._logp.gather(-1, actions).squeeze(-1).sum(dim=1)

    def entropy(self):
        plogp = self._logp.exp() * self._logp
        if self._masks is not None:
            plogp = torch.where(self._masks, plogp, torch.zeros_like(plogp))
        return -plogp.sum(-1).sum(dim=1)

    def sample(self):
        # Gumbel-max: argmax(logits + gumbel noise) draws from the categorical
        # per cell without a Python loop or a multinomial reshape.
        u = torch.empty_like(self._logp).uniform_(1e-10, 1.0)
        gumbel = -torch.log(-torch.log(u))
        return (self._logp + gumbel).argmax(dim=-1)

    def mode(self):
        return self._logp.argmax(dim=-1)


class CrawlMaskablePolicy(MaskableMultiInputActorCriticPolicy):
    """Fully-convolutional spatial policy for the grid action space.

    Replaces SB3's dense ``action_net``/``value_net`` with a 1x1 conv action head
    and a mean-pooled value head. Every base-class method routes through
    ``action_net(latent_pi)`` and ``value_net(latent_vf)``, so overriding
    ``_build`` is sufficient -- ``forward``, ``evaluate_actions``, masking, and
    sampling are unchanged. ``net_arch`` is forced empty so the ``mlp_extractor``
    is an identity and the (B, C, H, W) trunk map reaches the heads untouched.
    """

    def __init__(self, *args, n_residual_blocks: int = 3, **kwargs):
        # Stored before super().__init__ since that triggers make_features_extractor.
        self.n_residual_blocks = n_residual_blocks
        kwargs["net_arch"] = []
        super().__init__(*args, **kwargs)

    def make_features_extractor(self) -> CNNFeatureExtractor:
        # Own the trunk instead of receiving it via features_extractor_class, so
        # the policy is self-contained. This is the single seam SB3's inherited
        # methods route obs through, so building it here keeps every one of them
        # (forward, evaluate_actions, get_distribution, predict_values) working.
        return CNNFeatureExtractor(
            self.observation_space, n_residual_blocks=self.n_residual_blocks
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        # Swap the stock per-cell-loop distribution for the vectorized one. The
        # action logits still come from SpatialActionHead below, so only the
        # distribution math changes. Safe to replace here: this policy's heads
        # don't use action_dist.proba_distribution_net.
        self.action_dist = VecMaskableMultiCategoricalDistribution(
            list(self.action_space.nvec)
        )

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
