from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.interpolate import CubicSpline

from controllers.gimbal_controller import GimbalController
from controllers.lqr_controller import ChassisController
from controllers.jump_controller import WholeBodyJumpController
from controllers.lqr_design import design_whole_body_lqr
from modelling.whole_body_modelling import WholeBodyModelling
from utils.paths import WHOLE_BODY_LQR_CONFIG_PATH, MODEL_PATH


class WholeBodyLqrController(ChassisController):
    def __init__(
        self,
        yaml_path: Path = WHOLE_BODY_LQR_CONFIG_PATH,
        model: mujoco.MjModel | Path = MODEL_PATH,
        modelling_type: type[WholeBodyModelling] = WholeBodyModelling,
    ) -> None:
        self.params = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        self.model = (
            mujoco.MjModel.from_xml_path(str(model))
            if isinstance(model, Path)
            else model
        )
        self.data = mujoco.MjData(self.model)
        self.modelling = modelling_type(
            self.model, self.params["modelling"]["finite_difference_step"]
        )
        self.design = design_whole_body_lqr(self.modelling, self.params)
        self.command_config = self.params["command"]
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
        self.gimbal = GimbalController(
            self.model, self.data, self.design.qpos, self.design.limits
        )
        self.jump = WholeBodyJumpController(
            self.model, self.data, self.design, self.leg_pose, self.params
        )
        jump_heights = np.unique(
            np.append(
                np.linspace(self.jump.minimum, self.jump.maximum, 9), self.jump.normal
            )
        )
        self.jump_gain = CubicSpline(
            jump_heights,
            np.stack(
                [
                    design_whole_body_lqr(
                        self.modelling,
                        self.params
                        | {"control": self.params["control"] | {"leg_height": height}},
                    ).gain
                    for height in jump_heights
                ]
            ),
        )
        self.chassis_actuators = self.design.plant.chassis_actuators
        self.hip_gain_rows = np.array(
            [
                np.flatnonzero(self.chassis_actuators == actuator)[0]
                for actuator in self.jump.hips
            ]
        )
        self.reset()
        self.wheel_offsets = self.data.xpos[wheel_bodies, 1] - self.data.qpos[1]

    def lqr_control(self, dt: float) -> None:
        previous_phase = self.jump.phase
        self.jump.update(dt, not self.returning, self.target_velocity)
        if self.jump.phase == "landing" and previous_phase == "flight":
            self.target_xy[:] = self.data.qpos[:2]
        self.update_command(dt)
        if self.jump.phase != "idle":
            self.target_l0 = self.jump.height
        self.target_yaw += self.target_yaw_rate * dt
        c, s = np.cos(self.target_yaw), np.sin(self.target_yaw)
        heading = np.array([c, s])
        self.target_xy += self.target_velocity * heading * dt
        reference_height = (
            np.clip(self.jump.actual_height, self.jump.minimum, self.jump.maximum)
            if self.jump.phase in ("thrust", "landing")
            else self.target_l0
        )
        self.reference_qpos[:] = self.leg_pose(reference_height)
        self.reference_qpos[2] += self.jump.ground_height
        self.reference_qpos[:2] = self.target_xy
        yaw_quaternion = np.array(
            [np.cos(self.target_yaw / 2), 0, 0, np.sin(self.target_yaw / 2)]
        )
        mujoco.mju_mulQuat(
            self.reference_qpos[3:7], yaw_quaternion, self.design.qpos[3:7]
        )
        self.reference_qvel[:] = 0
        if self.jump.phase in ("thrust", "landing"):
            tangent = np.empty(self.model.nv)
            mujoco.mj_differentiatePos(
                self.model,
                tangent,
                2e-5,
                self.leg_pose(reference_height - 1e-5),
                self.leg_pose(reference_height + 1e-5),
            )
            self.reference_qvel[:] = tangent * np.mean(
                self.jump.leg_jacobian @ self.data.qvel
            )
        self.reference_qvel[:2] = self.target_velocity * heading
        self.reference_qvel[5] = self.target_yaw_rate
        if self.jump.phase == "flight":
            self.reference_qpos[:3] = self.data.qpos[:3]
            self.reference_qvel[:3] = self.data.qvel[:3]
        for index, joint in enumerate(self.wheel_joints):
            self.reference_qvel[joint.dofadr[0]] = (
                self.target_velocity - self.target_yaw_rate * self.wheel_offsets[index]
            ) / (self.radii[index] * joint.axis[1])
        gimbal_position, gimbal_velocity = self.gimbal.update_reference(
            self.data.qpos[3:7], dt
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
                self.position_error[self.design.plant.position_dofs],
                velocity_error[self.design.plant.velocity_dofs],
            )
        )
        damping = (
            self.model.dof_damping[self.active_dofs]
            * self.reference_qvel[self.active_dofs]
        )
        feedforward = self.leg_torque(self.target_l0) + damping
        gain = (
            self.jump_gain(reference_height)
            if self.jump.phase != "idle"
            else self.design.gain
        )
        chassis = self.chassis_actuators
        self.data.ctrl[chassis] = np.clip(
            feedforward[chassis] - gain @ state_error,
            -self.design.limits[chassis],
            self.design.limits[chassis],
        )
        if self.jump.phase == "flight" or (
            self.jump.phase != "idle" and not self.jump.contacting
        ):
            self.gimbal.control(
                np.zeros(self.model.nu),
                self.jump.config["air_gimbal_kp"],
                self.jump.config["air_gimbal_kd"],
            )
        else:
            self.gimbal.control(
                self.leg_torque(self.target_l0),
                self.params["gimbal"]["kp"],
                self.params["gimbal"]["kd"],
            )
        self.jump.control(
            self.position_error[3:6], -gain[self.hip_gain_rows] @ state_error
        )
