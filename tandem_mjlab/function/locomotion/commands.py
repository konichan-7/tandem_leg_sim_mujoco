from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

from tandem_mjlab.keyboard import LINEAR_ACCELERATION, YAW_ACCELERATION


@dataclass(kw_only=True)
class LocomotionCommandCfg(UniformVelocityCommandCfg):
    turn_velocity_limit: float = 1.0
    turn_yaw_limit: float = 2.0

    def build(self, env: ManagerBasedRlEnv) -> LocomotionCommand:
        return LocomotionCommand(self, env)


class LocomotionCommand(UniformVelocityCommand):
    def __init__(self, cfg: LocomotionCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.target = torch.zeros_like(self.vel_command_b)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reference_heading = torch.zeros(self.num_envs, device=self.device)
        self._dt = torch.zeros(self.num_envs, device=self.device)
        self._was_straight = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._capture_heading = torch.ones_like(self._was_straight)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        sample = torch.rand(len(env_ids), device=self.device)
        boundaries = sample.new_tensor((0.1, 0.5, 0.7, 0.85))
        self.mode[env_ids] = torch.bucketize(sample, boundaries)
        mode = self.mode[env_ids]
        self.target[env_ids] = 0.0
        drive = env_ids[mode == 1]
        spin = env_ids[mode == 2]
        self.target[drive, 0] = torch.empty(len(drive), device=self.device).uniform_(
            *self.cfg.ranges.lin_vel_x
        )
        self.target[spin, 2] = torch.empty(len(spin), device=self.device).uniform_(
            *self.cfg.ranges.ang_vel_z
        )
        for turn_mode, direction in ((3, 1), (4, -1)):
            turn = env_ids[mode == turn_mode]
            self.target[turn, 0] = (
                torch.empty(len(turn), device=self.device).uniform_(
                    0.3, self.cfg.turn_velocity_limit
                )
                * direction
            )
            magnitude = torch.empty(len(turn), device=self.device).uniform_(
                0.3, self.cfg.turn_yaw_limit
            )
            sign = torch.where(
                torch.rand(len(turn), device=self.device) < 0.5, -1.0, 1.0
            )
            self.target[turn, 2] = magnitude * sign
        self.is_standing_env[env_ids] = mode == 0
        self._capture_heading[env_ids] = True
        self.time_left[drive] = torch.empty(len(drive), device=self.device).uniform_(
            10.0, 20.0
        )

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        self._dt.zero_()
        if env_ids is None:
            self._dt[:] = dt
        else:
            self._dt[env_ids] = dt
        super().compute(dt, env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        limits = self._dt[:, None] * self.target.new_tensor(
            (LINEAR_ACCELERATION, 0.0, YAW_ACCELERATION)
        )
        self.vel_command_b += (self.target - self.vel_command_b).clamp(-limits, limits)
        self.vel_command_w[:] = self.vel_command_b
        straight = self.vel_command_b[:, 2].abs() < 1e-6
        capture = ~straight | ~self._was_straight | self._capture_heading
        self.reference_heading[capture] = self.robot.data.heading_w[capture]
        error = self.robot.data.heading_w - self.reference_heading
        self.heading_error[:] = torch.atan2(error.sin(), error.cos()) * straight
        self._was_straight[:] = straight
        self._capture_heading.zero_()

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_w[env_ids] = 0.0
        self._was_straight[env_ids] = True
        self.heading_error[env_ids] = 0.0
        return super().reset(env_ids)
