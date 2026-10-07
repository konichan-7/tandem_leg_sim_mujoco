import copy

import torch
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, HiddenState

from tandem_mjlab.model import HEADING_ERROR_SCALE
from tensordict import TensorDict

STEERING_ACTION_LIMIT = 0.15


def steering_features(latent: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            latent[:, :25],
            latent[:, 25:26] / HEADING_ERROR_SCALE,
            (latent[:, 26:27] - 1.0) / HEADING_ERROR_SCALE,
            latent[:, 27:],
        ),
        dim=-1,
    )


def steering_actions(base: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    correction = STEERING_ACTION_LIMIT * correction.tanh()
    return torch.cat(
        (base[:, :4], base[:, 4:5] - correction, base[:, 5:6] + correction), dim=-1
    )


def stand_actions(
    actions: torch.Tensor, observation: torch.Tensor, leg_alpha: torch.Tensor
) -> torch.Tensor:
    standing = (observation[:, 6:8].abs() < 1e-6).all(dim=-1, keepdim=True)
    legs = leg_alpha * actions[:, :4] + (1 - leg_alpha) * observation[:, 19:23]
    return torch.cat(
        (torch.where(standing, legs, actions[:, :4]), actions[:, 4:]), dim=-1
    )


class WheelLegActor(MLPModel):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        self.current_dim = obs["actor"].shape[-1]
        self.history_dim = obs["history"].shape[-1]
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.encoder = MLP(self.history_dim, 3, (128, 64), activation)
        self.requires_grad_(False)
        self.steering = MLP(self.current_dim + 3, 1, (32, 16), activation)
        torch.nn.init.zeros_(self.steering[-1].weight)
        torch.nn.init.zeros_(self.steering[-1].bias)
        self.register_buffer("stand_leg_alpha", torch.tensor(0.25))

    def _get_latent_dim(self) -> int:
        return self.current_dim + 3

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        return torch.cat((obs["actor"], self.encoder(obs["history"]).detach()), dim=-1)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        actions = steering_actions(
            self.mlp(latent), self.steering(steering_features(latent))
        )
        actions = stand_actions(actions, obs["actor"], self.stand_leg_alpha)
        if self.distribution is None:
            return actions
        if stochastic_output:
            self.distribution.update(actions)
            return self.distribution.sample()
        return self.distribution.deterministic_output(actions)

    def as_onnx(self, verbose: bool = False) -> torch.nn.Module:
        del verbose
        return WheelLegOnnx(self)


class WheelLegOnnx(torch.nn.Module):
    input_names = ["obs", "obs_history"]
    output_names = ["actions"]

    def __init__(self, model: WheelLegActor) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(model.encoder)
        self.actor = copy.deepcopy(model.mlp)
        self.steering = copy.deepcopy(model.steering)
        self.register_buffer("stand_leg_alpha", model.stand_leg_alpha.clone())
        self.output = (
            torch.nn.Identity()
            if model.distribution is None
            else model.distribution.as_deterministic_output_module()
        )
        self.current_dim = model.current_dim
        self.history_dim = model.history_dim

    def forward(self, obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        latent = torch.cat((obs, self.encoder(obs_history)), dim=-1)
        actions = steering_actions(
            self.actor(latent), self.steering(steering_features(latent))
        )
        return self.output(stand_actions(actions, obs, self.stand_leg_alpha))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, self.current_dim), torch.zeros(1, self.history_dim)
