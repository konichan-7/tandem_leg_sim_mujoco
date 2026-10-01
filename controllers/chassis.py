from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from controllers.gimbal_controller import GimbalController
from controllers.jump_controller import BaseJumpController
from modelling.base import RobotModelling
from modelling.lqr_design import LqrDesign
from utils.control import move_towards
from utils.paths import MODEL_PATH


class ChassisController(ABC):
    model: mujoco.MjModel
    data: mujoco.MjData
    params: dict
    command_config: dict
    modelling: RobotModelling
    design: LqrDesign
    jump: BaseJumpController
    gimbal: GimbalController
    leg_pose: CubicSpline
    leg_torque: CubicSpline
    wheel_joints: list
    wheel_bodies: list[int]
    radii: np.ndarray
    target_xy: np.ndarray
    target_yaw: float
    target_velocity: float
    target_yaw_rate: float
    desired_velocity: float
    desired_yaw_rate: float
    leg_length_direction: float
    target_l0: float
    paused: bool
    spinning: bool
    returning: bool
    return_yaw: float | None

    def __init__(
        self,
        yaml_path: Path,
        model: mujoco.MjModel | Path = MODEL_PATH,
        modelling_type: type[RobotModelling] = RobotModelling,
    ) -> None:
        self.params = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        self.model = (
            model
            if isinstance(model, mujoco.MjModel)
            else mujoco.MjModel.from_xml_path(str(model))
        )
        self.data = mujoco.MjData(self.model)
        self.command_config = self.params["command"]
        self.modelling = modelling_type(self.model, **self.params.get("modelling", {}))
        self.design = self.build_design()
        heights = np.linspace(
            min(
                self.command_config["leg_length_min"],
                self.params["jump"]["leg_length_min"],
            ),
            max(
                self.command_config["leg_length_max"],
                self.params["jump"]["leg_length_max"],
            ),
            18,
        )
        references = [self.modelling.operating_point(height) for height in heights]
        self.leg_pose = CubicSpline(
            heights, np.stack([reference.qpos for reference in references])
        )
        self.leg_torque = CubicSpline(
            heights, np.stack([reference.torque for reference in references])
        )
        self.wheel_joints = [
            self.model.joint(f"{side}_wheel_joint") for side in ("left", "right")
        ]
        self.wheel_bodies = [
            self.model.body(f"{side}_wheel_link").id for side in ("left", "right")
        ]
        self.radii = np.array(
            [
                self.model.geom(f"{side}_wheel_link_collision").size[0]
                for side in ("left", "right")
            ]
        )
        self.target_xy = np.zeros(2)
        self.gimbal = GimbalController(
            self.model, self.data, self.design.qpos, self.design.limits
        )

    @abstractmethod
    def build_design(self) -> LqrDesign[Any]:
        raise NotImplementedError

    @abstractmethod
    def lqr_control(self, dt: float) -> None:
        raise NotImplementedError

    def reset(self, pitch: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.design.qpos
        displacement = np.zeros(self.model.nv)
        displacement[4] = pitch
        mujoco.mj_integratePos(self.model, self.data.qpos, displacement, 1)
        self.data.ctrl[:] = self.design.torque
        mujoco.mj_forward(self.model, self.data)
        self.target_xy[:] = self.design.qpos[:2]
        self.target_yaw = 0.0
        self.target_velocity = 0.0
        self.target_yaw_rate = 0.0
        self.desired_velocity = 0.0
        self.desired_yaw_rate = 0.0
        self.leg_length_direction = 0.0
        self.target_l0 = self.params["control"]["leg_height"]
        self.paused = False
        self.spinning = False
        self.returning = False
        self.return_yaw: float | None = None
        self.gimbal.reset()
        self.jump.reset()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def command(
        self,
        linear_direction: float,
        yaw_direction: float,
        leg_length_direction: float,
        spin: bool = False,
        jump: bool = False,
    ) -> None:
        self.jump.command(jump, self.target_l0)
        if self.jump.phase != "idle":
            yaw_direction = leg_length_direction = 0.0
            spin = False
            if self.spinning:
                self.spinning = False
                self.returning = True
                self.return_yaw = None
        desired_velocity = (
            0.0
            if spin
            else linear_direction
            * (
                self.jump.config["linear_velocity"]
                if self.jump.phase != "idle"
                else self.command_config["linear_velocity"]
            )
        )
        desired_yaw_rate = (
            self.command_config["spin_yaw_rate"]
            if spin
            else yaw_direction * self.command_config["yaw_rate"]
        )
        spin_changed = spin != self.spinning
        motion_changed = (
            desired_velocity != self.desired_velocity
            or desired_yaw_rate != self.desired_yaw_rate
        )
        if (
            not motion_changed
            and not spin_changed
            and leg_length_direction == self.leg_length_direction
        ):
            return
        if spin_changed:
            self.spinning = spin
            self.returning = not spin
            self.return_yaw = None
            if spin:
                self.gimbal.hold_world(True)
        if motion_changed and not self.returning:
            self.reset_position_reference()
            self.target_yaw = Rotation.from_quat(
                self.data.qpos[3:7], scalar_first=True
            ).as_euler("xyz")[2]
        self.desired_velocity = desired_velocity
        self.desired_yaw_rate = desired_yaw_rate
        self.leg_length_direction = leg_length_direction
        print(
            f"desired_velocity={self.desired_velocity:.2f} "
            f"desired_yaw_rate={self.desired_yaw_rate:.2f} "
            f"target_leg_length={self.target_l0:.3f} spin={spin}"
        )

    def update_command(self, dt: float) -> None:
        desired_yaw_rate = self.desired_yaw_rate
        acceleration = self.command_config["yaw_acceleration"]
        if self.returning:
            desired_yaw_rate = 0.0
            if self.return_yaw is None and self.target_yaw_rate == 0.0:
                direction = self.gimbal.direction
                error = np.arctan2(direction[1], direction[0]) - self.target_yaw
                self.return_yaw = self.target_yaw + np.arctan2(
                    np.sin(error), np.cos(error)
                )
            if self.return_yaw is not None:
                error = self.return_yaw - self.target_yaw
                rate = (
                    np.sqrt(
                        2 * acceleration * abs(error) + (acceleration * dt / 2) ** 2
                    )
                    - acceleration * dt / 2
                )
                desired_yaw_rate = np.sign(error) * min(
                    self.command_config["yaw_rate"], rate
                )
                if (
                    abs(error) <= acceleration * dt**2
                    and abs(self.target_yaw_rate) <= acceleration * dt
                ):
                    self.target_yaw = self.return_yaw
                    self.target_yaw_rate = 0.0
                    desired_yaw_rate = 0.0
                    self.returning = False
        elif not self.spinning and (
            self.desired_velocity != 0.0 or self.desired_yaw_rate != 0.0
        ):
            self.gimbal.hold_world(False)
        self.target_velocity = move_towards(
            self.target_velocity,
            0.0 if self.returning else self.desired_velocity,
            self.command_config["linear_acceleration"] * dt,
        )
        self.target_yaw_rate = move_towards(
            self.target_yaw_rate,
            desired_yaw_rate,
            acceleration * dt,
        )
        if self.jump.phase != "idle":
            return
        self.target_l0 = np.clip(
            self.target_l0
            + self.leg_length_direction
            * self.command_config["leg_length_velocity"]
            * dt,
            self.command_config["leg_length_min"],
            self.command_config["leg_length_max"],
        )

    def control(self) -> None:
        self.lqr_control(self.model.opt.timestep)

    def step(self) -> None:
        if not self.paused:
            self.control()
            mujoco.mj_step(self.model, self.data)

    def reset_position_reference(self) -> None:
        self.target_xy[:] = self.data.qpos[:2]
