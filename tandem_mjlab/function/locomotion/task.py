from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import (
    action_rate_l2,
    bad_orientation,
    joint_acc_l2,
    joint_vel_l2,
    reset_root_state_uniform,
    reset_scene_to_default,
    root_height_below_minimum,
    time_out,
)
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.lab_api.math import quat_apply_inverse


from tandem_mjlab.model import (
    ACTUATED_JOINTS,
    DECIMATION,
    HEADING_ERROR_SCALE,
    HISTORY_LENGTH,
    LEG_ACTION_SCALE,
    LEG_DAMPING,
    LEG_JOINTS,
    LEG_LENGTH_BODIES,
    LEG_STIFFNESS,
    LINEAR_COMMAND_RANGE,
    POLICY_PERIOD,
    SIMULATION_OPTIONS,
    WHEEL_ACTION_SCALE,
    WHEEL_DAMPING,
    YAW_COMMAND_RANGE,
    get_spec,
    initial_joint_positions,
    motor_limits,
    nominal_leg_lengths,
    standing_pose,
    wheel_action_limit,
)
from tandem_mjlab.function.locomotion.commands import LocomotionCommandCfg

REFERENCE_HEIGHT = standing_pose()[1]


@dataclass(kw_only=True)
class WheelLegActionCfg(ActionTermCfg):
    randomize_dynamics: bool = False
    gain_range: tuple[float, float] = (0.985, 1.015)
    motor_strength_range: tuple[float, float] = (0.985, 1.015)
    position_bias_range: tuple[float, float] = (-0.009, 0.009)
    action_delay_frames: int = 0

    def build(self, env: ManagerBasedRlEnv) -> WheelLegAction:
        return WheelLegAction(self, env)


class WheelLegAction(ActionTerm):
    def __init__(self, cfg: WheelLegActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        joint_ids, _ = self._entity.find_joints(ACTUATED_JOINTS, preserve_order=True)
        self._joint_ids = torch.tensor(joint_ids, device=self.device)
        self._raw_action = torch.zeros(self.num_envs, 6, device=self.device)
        self._wheel_action_limit = wheel_action_limit()
        self._kp_scale = torch.ones(self.num_envs, 4, device=self.device)
        self._kd_scale = torch.ones(self.num_envs, 6, device=self.device)
        self._motor_scale = torch.ones(self.num_envs, 6, device=self.device)
        self.position_bias = torch.zeros(self.num_envs, 4, device=self.device)
        self._action_history = torch.zeros(
            self.num_envs, cfg.action_delay_frames + 1, 6, device=self.device
        )
        self._delay = torch.randint(
            cfg.action_delay_frames + 1, (self.num_envs,), device=self.device
        )
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        if cfg.randomize_dynamics:
            self._kp_scale.uniform_(*cfg.gain_range)
            self._kd_scale.uniform_(*cfg.gain_range)
            self._motor_scale.uniform_(*cfg.motor_strength_range)
            self.position_bias.uniform_(*cfg.position_bias_range)
        self._stand_effort = torch.tensor(
            [standing_pose()[2][name] for name in ACTUATED_JOINTS[:4]],
            device=self.device,
        )
        self._effort_limits = torch.tensor(
            [motor_limits()[name] for name in ACTUATED_JOINTS],
            device=self.device,
        )

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_action[:, :4] = actions[:, :4].clamp(-0.5, 0.5)
        self._raw_action[:, 4:] = actions[:, 4:].clamp(
            -self._wheel_action_limit, self._wheel_action_limit
        )
        self._action_history = self._action_history.roll(1, dims=1)
        self._action_history[:, 0] = self._raw_action

    def apply_actions(self) -> None:
        action = self._action_history[self._env_ids, self._delay]
        positions = self._entity.data.joint_pos[:, self._joint_ids]
        velocities = self._entity.data.joint_vel[:, self._joint_ids]
        defaults = self._entity.data.default_joint_pos[:, self._joint_ids]
        leg_effort = (
            self._stand_effort
            + LEG_STIFFNESS
            * self._kp_scale
            * (
                defaults[:, :4]
                + self.position_bias
                + LEG_ACTION_SCALE * action[:, :4]
                - positions[:, :4]
            )
            - LEG_DAMPING * self._kd_scale[:, :4] * velocities[:, :4]
        )
        wheel_effort = (
            WHEEL_DAMPING
            * self._kd_scale[:, 4:]
            * (WHEEL_ACTION_SCALE * action[:, 4:] - velocities[:, 4:])
        )
        effort = (
            torch.cat((leg_effort, wheel_effort), dim=-1) * self._motor_scale
        ).clamp(-self._effort_limits, self._effort_limits)
        self._entity.set_joint_effort_target(effort, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_action[env_ids] = 0.0
        self._action_history[env_ids] = 0.0


def proprioception(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(ACTUATED_JOINTS, preserve_order=True)
    joint_ids = torch.tensor(joint_ids, device=env.device)
    commands = env.command_manager.get_command("twist")
    command_obs = torch.cat(
        (
            commands[:, 0:1] * 2.0,
            commands[:, 2:3] * 0.25,
            torch.full_like(commands[:, 0:1], REFERENCE_HEIGHT * 5.0),
        ),
        dim=-1,
    )
    observation = torch.cat(
        (
            robot.data.root_link_ang_vel_b * 0.25,
            robot.data.projected_gravity_b,
            command_obs,
            (
                robot.data.joint_pos[:, joint_ids[:4]]
                - robot.data.default_joint_pos[:, joint_ids[:4]]
                - env.action_manager.get_term("pd").position_bias
            ),
            robot.data.joint_vel[:, joint_ids] * 0.05,
            env.action_manager.action,
        ),
        dim=-1,
    )
    command = env.command_manager.get_term("twist")
    return torch.cat(
        (
            observation,
            command.heading_error.sin()[:, None],
            command.heading_error.cos()[:, None],
        ),
        dim=-1,
    )


def critic_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.cat((proprioception(env), robot.data.root_link_lin_vel_b), dim=-1)


def track_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = env.command_manager.get_command("twist")
    velocity = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    return torch.exp(-torch.square(command[:, :2] - velocity).sum(dim=-1) / 0.25)


def track_yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = env.command_manager.get_command("twist")
    yaw_rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(command[:, 2] - yaw_rate) / 0.25)


def upright(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b
    return torch.exp(-torch.sum(torch.square(gravity[:, :2]), dim=-1) / std**2)


def base_ang_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(dim=-1)


def straight_motion_cost(
    env: ManagerBasedRlEnv, lateral_scale: float, yaw_scale: float
) -> torch.Tensor:
    command = env.command_manager.get_command("twist")
    robot = env.scene["robot"]
    lateral_error = robot.data.root_link_lin_vel_b[:, 1] / lateral_scale
    yaw_error = robot.data.root_link_ang_vel_b[:, 2] / yaw_scale
    cost = torch.sqrt(1 + lateral_error.square()) - 1
    cost += torch.sqrt(1 + yaw_error.square()) - 1
    return (command[:, 2].abs() < 1e-6) * cost


def heading_cost(env: ManagerBasedRlEnv, scale: float) -> torch.Tensor:
    command = env.command_manager.get_term("twist")
    error = env.scene["robot"].data.heading_w - command.reference_heading
    error = torch.atan2(error.sin(), error.cos())
    return (command.vel_command_b[:, 2].abs() < 1e-6) * (
        torch.sqrt(1 + (error / scale).square()) - 1
    )


def yaw_error_cost(env: ManagerBasedRlEnv, scale: float) -> torch.Tensor:
    command = env.command_manager.get_command("twist")
    error = command[:, 2] - env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return torch.sqrt(1 + (error / scale).square()) - 1


def leg_length(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    target: tuple[float, float],
    std: float,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    positions = robot.data.body_link_pos_w[:, asset_cfg.body_ids]
    hips = positions[:, :4].reshape(env.num_envs, 2, 2, 3).mean(dim=2)
    offsets = quat_apply_inverse(
        robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1),
        positions[:, 4:] - hips,
    )
    lengths = torch.linalg.vector_norm(offsets[:, :, [0, 2]], dim=-1)
    reference = lengths.new_tensor(target)
    return torch.exp(-((lengths - reference) / std).square().mean(dim=-1))


def body_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    return torch.exp(-torch.square(height - REFERENCE_HEIGHT) / 0.0025)


def leg_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(ACTUATED_JOINTS[:4], preserve_order=True)
    difference = (
        robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
    )
    return torch.exp(-torch.sum(torch.square(difference), dim=-1) / 0.16)


def nonwheel_ground_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
    data = env.scene["nonwheel_ground"].data
    return torch.any(data.found > 0, dim=-1) | torch.any(
        data.force_history != 0.0, dim=(1, 2, 3)
    )


def missing_wheel_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.any(
        env.scene["wheel_ground"].data.current_air_time > 5 * POLICY_PERIOD,
        dim=-1,
    )


def wheel_contact_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (env.scene["wheel_ground"].data.current_air_time > 0.0).float().mean(dim=-1)


def leg_action_smoothness(env: ManagerBasedRlEnv) -> torch.Tensor:
    actions = env.action_manager
    return torch.sum(
        torch.square(
            actions.action[:, :4]
            - 2 * actions.prev_action[:, :4]
            + actions.prev_prev_action[:, :4]
        ),
        dim=-1,
    )


def env_cfg(randomize: bool = False) -> ManagerBasedRlEnvCfg:
    robot = EntityCfg(
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=ACTUATED_JOINTS),)
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, REFERENCE_HEIGHT),
            joint_pos=initial_joint_positions(),
            joint_vel={".*": 0.0},
        ),
    )
    observation = ObservationTermCfg(func=proprioception)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": robot},
            sensors=(
                ContactSensorCfg(
                    name="wheel_ground",
                    primary=ContactMatch(
                        mode="body",
                        pattern=("left_wheel_link", "right_wheel_link"),
                        entity="robot",
                    ),
                    fields=("found",),
                    track_air_time=True,
                ),
                ContactSensorCfg(
                    name="nonwheel_ground",
                    primary=ContactMatch(
                        mode="body",
                        pattern=".*",
                        entity="robot",
                        exclude=("left_wheel_link", "right_wheel_link"),
                    ),
                    fields=("found", "force"),
                    history_length=DECIMATION,
                ),
            ),
            num_envs=256,
            env_spacing=2.0,
        ),
        observations={
            "actor": ObservationGroupCfg({"proprio": observation}),
            "history": ObservationGroupCfg(
                {"proprio": ObservationTermCfg(func=proprioception)},
                history_length=HISTORY_LENGTH,
            ),
            "critic": ObservationGroupCfg(
                {"state": ObservationTermCfg(func=critic_state)}
            ),
        },
        actions={
            "pd": WheelLegActionCfg(
                entity_name="robot",
                randomize_dynamics=randomize,
                action_delay_frames=1 if randomize else 0,
            )
        },
        commands={
            "twist": LocomotionCommandCfg(
                entity_name="robot",
                resampling_time_range=(3.0, 6.0),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=LINEAR_COMMAND_RANGE,
                    lin_vel_y=(0.0, 0.0),
                    ang_vel_z=YAW_COMMAND_RANGE,
                ),
            )
        },
        events={
            **(
                {
                    "wheel_friction": EventTermCfg(
                        func=dr.geom_friction,
                        mode="startup",
                        params={
                            "asset_cfg": SceneEntityCfg(
                                "robot", geom_names=(".*_wheel_link_collision",)
                            ),
                            "ranges": (0.88, 1.12),
                            "operation": "abs",
                            "shared_random": True,
                        },
                    ),
                    "base_density": EventTermCfg(
                        func=dr.pseudo_inertia,
                        mode="startup",
                        params={
                            "asset_cfg": SceneEntityCfg(
                                "robot", body_names=("base_link",)
                            ),
                            "alpha_range": (math.log(0.97) / 2, math.log(1.09) / 2),
                        },
                    ),
                    "limb_density": EventTermCfg(
                        func=dr.pseudo_inertia,
                        mode="startup",
                        params={
                            "asset_cfg": SceneEntityCfg(
                                "robot", body_names=("(left|right)_.*",)
                            ),
                            "alpha_range": (math.log(0.97) / 2, math.log(1.03) / 2),
                        },
                    ),
                    "base_com": EventTermCfg(
                        func=dr.body_com_offset,
                        mode="startup",
                        params={
                            "asset_cfg": SceneEntityCfg(
                                "robot", body_names=("base_link",)
                            ),
                            "ranges": {axis: (-0.006, 0.006) for axis in range(3)},
                            "operation": "add",
                        },
                    ),
                }
                if randomize
                else {}
            ),
            "reset_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "reset_root": EventTermCfg(
                func=reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.02, 0.02),
                        "y": (-0.02, 0.02),
                        "yaw": (-0.2, 0.2),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
        },
        rewards={
            "track_velocity": RewardTermCfg(func=track_velocity, weight=4.0),
            "track_yaw": RewardTermCfg(func=track_yaw, weight=2.0),
            "upright": RewardTermCfg(func=upright, weight=4.0, params={"std": 0.2}),
            "base_ang_vel_xy": RewardTermCfg(func=base_ang_vel_xy, weight=-0.2),
            "straight_motion": RewardTermCfg(
                func=straight_motion_cost,
                weight=-1.0,
                params={"lateral_scale": 0.1, "yaw_scale": 0.03},
            ),
            "yaw_error": RewardTermCfg(
                func=yaw_error_cost, weight=-0.2, params={"scale": 0.3}
            ),
            "heading": RewardTermCfg(
                func=heading_cost, weight=-4.0, params={"scale": HEADING_ERROR_SCALE}
            ),
            "wheel_contact": RewardTermCfg(func=wheel_contact_cost, weight=-2.0),
            "leg_length": RewardTermCfg(
                func=leg_length,
                weight=1.0,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=LEG_LENGTH_BODIES, preserve_order=True
                    ),
                    "target": nominal_leg_lengths(),
                    "std": 0.01,
                },
            ),
            "height": RewardTermCfg(func=body_height, weight=1.0),
            "leg_pose": RewardTermCfg(func=leg_pose, weight=1.0),
            "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
            "leg_action_smoothness": RewardTermCfg(
                func=leg_action_smoothness, weight=-0.05
            ),
            "leg_joint_vel": RewardTermCfg(
                func=joint_vel_l2,
                weight=-5e-5,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
            ),
            "joint_acc": RewardTermCfg(
                func=joint_acc_l2,
                weight=-2.5e-7,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS)
                },
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "tilt": TerminationTermCfg(
                func=bad_orientation,
                params={
                    "limit_angle": math.pi / 3,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "low": TerminationTermCfg(
                func=root_height_below_minimum,
                params={
                    "minimum_height": REFERENCE_HEIGHT - 0.015,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "nonwheel_ground": TerminationTermCfg(func=nonwheel_ground_contact),
            "wheel_airborne": TerminationTermCfg(func=missing_wheel_contact),
        },
        sim=SimulationCfg(
            njmax=256,
            mujoco=MujocoCfg(**SIMULATION_OPTIONS),
        ),
        decimation=DECIMATION,
        episode_length_s=20.0,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.5,
        ),
    )
    return cfg


def ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 64, 32),
            activation="elu",
            obs_normalization=False,
            class_name="tandem_mjlab.policy.WheelLegActor",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.003,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.005,
            max_grad_norm=1.0,
        ),
        obs_groups={"actor": ("actor", "history"), "critic": ("critic",)},
        seed=1,
        experiment_name="tandem_wheelleg",
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=48,
        max_iterations=5000,
    )
