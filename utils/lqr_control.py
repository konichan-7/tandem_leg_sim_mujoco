from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from demo import PATHS
from utils.gimbal_controller import GimbalController
from utils.math_tools import design_lqr, move_towards
from utils.mujoco_io import equilibrium


class DemoLqrController:
    def __init__(self, yaml_path: Path = PATHS.lqr_yaml) -> None:
        self.params = yaml.safe_load(yaml_path.read_text())
        self.model = mujoco.MjModel.from_xml_path(str(PATHS.xml))
        self.data = mujoco.MjData(self.model)
        self.design = design_lqr(self.model, self.params)
        self.command_config = self.params["command"]
        heights = np.linspace(
            self.command_config["leg_length_min"],
            self.command_config["leg_length_max"],
            5,
        )
        references = [equilibrium(self.model, height) for height in heights]
        self.leg_pose = CubicSpline(
            heights, np.stack([reference[0] for reference in references])
        )
        self.leg_torque = CubicSpline(
            heights, np.stack([reference[1] for reference in references])
        )
        self.active_dofs = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]
        self.wheel_joints = [
            self.model.joint(f"{side}_wheel_joint") for side in ("left", "right")
        ]
        wheel_bodies = np.array(
            [self.model.body(f"{side}_wheel_link").id for side in ("left", "right")]
        )
        self.radii = np.array(
            [
                self.model.geom(f"{side}_wheel_link_collision").size[0]
                for side in ("left", "right")
            ]
        )
        self.position_error = np.zeros(self.model.nv)
        self.reference_qpos = self.design.qpos.copy()
        self.reference_qvel = np.zeros(self.model.nv)
        self.target_xy = np.zeros(2)
        self.gimbal = GimbalController(self.model, self.data, self.design)
        self.chassis_actuators = np.setdiff1d(
            np.arange(self.model.nu), self.gimbal.actuators
        )
        self.reset()
        self.wheel_offsets = self.data.xpos[wheel_bodies, 1] - self.data.qpos[1]

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

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def command(
        self,
        linear_direction: float,
        yaw_direction: float,
        leg_length_direction: float,
        spin: bool = False,
    ) -> None:
        desired_velocity = (
            0.0 if spin else linear_direction * self.command_config["linear_velocity"]
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
            self.target_xy[:] = self.data.qpos[:2]
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
        self.target_l0 = np.clip(
            self.target_l0
            + self.leg_length_direction
            * self.command_config["leg_length_velocity"]
            * dt,
            self.command_config["leg_length_min"],
            self.command_config["leg_length_max"],
        )

    def lqr_control(self, dt: float) -> None:
        self.update_command(dt)
        self.target_yaw += self.target_yaw_rate * dt
        c, s = np.cos(self.target_yaw), np.sin(self.target_yaw)
        heading = np.array([c, s])
        self.target_xy += self.target_velocity * heading * dt
        self.reference_qpos[:] = self.leg_pose(self.target_l0)
        self.reference_qpos[:2] = self.target_xy
        yaw_quaternion = np.array(
            [np.cos(self.target_yaw / 2), 0, 0, np.sin(self.target_yaw / 2)]
        )
        mujoco.mju_mulQuat(
            self.reference_qpos[3:7], yaw_quaternion, self.design.qpos[3:7]
        )
        self.reference_qvel[:] = 0
        self.reference_qvel[:2] = self.target_velocity * heading
        self.reference_qvel[5] = self.target_yaw_rate
        for index, joint in enumerate(self.wheel_joints):
            self.reference_qvel[joint.dofadr[0]] = (
                self.target_velocity - self.target_yaw_rate * self.wheel_offsets[index]
            ) / (self.radii[index] * joint.axis[1])
        gimbal_position, gimbal_velocity = self.gimbal.update_reference(
            self.reference_qpos[3:7], dt
        )
        self.reference_qpos[self.gimbal.qpos] = gimbal_position
        self.reference_qvel[self.gimbal.dofs] = gimbal_velocity
        mujoco.mj_differentiatePos(
            self.model, self.position_error, 1, self.reference_qpos, self.data.qpos
        )
        self.position_error[0] = heading @ self.position_error[:2]
        velocity_error = self.data.qvel - self.reference_qvel
        velocity_error[0] = heading @ velocity_error[:2]
        state_error = np.concatenate(
            (
                self.position_error[self.design.position_dofs],
                velocity_error[self.design.velocity_dofs],
            )
        )
        damping = (
            self.model.dof_damping[self.active_dofs]
            * self.reference_qvel[self.active_dofs]
        )
        feedforward = self.leg_torque(self.target_l0) + damping
        chassis = self.chassis_actuators
        self.data.ctrl[chassis] = np.clip(
            feedforward[chassis] - self.design.gain[chassis] @ state_error,
            -self.design.limits[chassis],
            self.design.limits[chassis],
        )
        self.gimbal.control(state_error, feedforward)

    def control(self) -> None:
        self.lqr_control(self.model.opt.timestep)

    def step(self) -> None:
        if not self.paused:
            self.control()
            mujoco.mj_step(self.model, self.data)
