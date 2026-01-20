# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization


class ResBlock(nn.Module):
    """Simple residual MLP block used by NavBot PPO actor/critic."""

    def __init__(self, fin: int, fout: int, n_neurons: int = 512) -> None:
        super().__init__()
        self.fin = fin
        self.fout = fout

        self.fc1 = nn.Linear(fin, n_neurons)
        nn.init.uniform_(self.fc1.weight, -1.0 / math.sqrt(fin), 1.0 / math.sqrt(fin))

        self.fc2 = nn.Linear(n_neurons, fout)
        nn.init.uniform_(self.fc2.weight, -1.0 / math.sqrt(n_neurons), 1.0 / math.sqrt(n_neurons))

        self.fc_skip = nn.Linear(fin, fout) if fin != fout else None
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, apply_nl: bool = True) -> torch.Tensor:
        skip = x if self.fc_skip is None else self.act(self.fc_skip(x))

        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = skip + out

        return self.act(out) if apply_nl else out


class ResActor(nn.Module):
    """Actor network mirroring navbot_ppo: two residual blocks, dual heads."""

    def __init__(self, in_dim: int, out_dim: int, n_neurons: int = 512) -> None:
        super().__init__()
        if out_dim != 2:
            raise ValueError("ResActor expects exactly 2 actions (linear, angular).")

        self.rb1 = ResBlock(in_dim, in_dim, n_neurons)
        self.rb2 = ResBlock(in_dim + in_dim, in_dim + in_dim, n_neurons)

        self.out_linear = nn.Linear(in_dim + in_dim, 1)
        nn.init.uniform_(self.out_linear.weight, -1.0 / math.sqrt(in_dim), 1.0 / math.sqrt(in_dim))

        self.out_angular = nn.Linear(in_dim + in_dim, 1)
        nn.init.uniform_(self.out_angular.weight, -1.0 / math.sqrt(in_dim + in_dim), 1.0 / math.sqrt(in_dim + in_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x0 = obs
        x = self.rb1(x0, apply_nl=True)
        x = self.rb2(torch.cat([x0, x], dim=-1), apply_nl=True)

        linear = torch.sigmoid(self.out_linear(x))
        angular = torch.tanh(self.out_angular(x))
        return torch.cat((linear, angular), dim=-1)


class ResCritic(nn.Module):
    """Critic network mirroring navbot_ppo residual MLP."""

    def __init__(self, in_dim: int, out_dim: int = 1, n_neurons: int = 512) -> None:
        super().__init__()
        self.rb1 = ResBlock(in_dim, in_dim, n_neurons)
        self.rb2 = ResBlock(in_dim + in_dim, in_dim + in_dim, n_neurons)
        self.out = nn.Linear(in_dim + in_dim, out_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x0 = obs
        x = self.rb1(x0, apply_nl=True)
        x = self.rb2(torch.cat([x0, x], dim=-1), apply_nl=True)
        return self.out(x)


class ResActorCritic(nn.Module):
    """Actor-Critic with navbot_ppo-style residual heads, API-compatible with ActorCritic."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        n_neurons: int = 512,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ResActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "ResActorCritic only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "ResActorCritic only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std

        # Actor / Critic
        self.actor = ResActor(num_actor_obs, num_actions, n_neurons)
        self.critic = ResCritic(num_critic_obs, 1, n_neurons)

        # Observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            raise NotImplementedError("state_dependent_std not supported for ResActorCritic.")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError("noise_std_type must be 'scalar' or 'log'.")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:  # pragma: no cover - API parity
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'.")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
