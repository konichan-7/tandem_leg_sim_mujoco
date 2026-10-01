from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

from controllers.chassis import ChassisController
from controllers.jump_controller import WholeBodyJumpController
from modelling.lqr_design import LqrDesign, design_whole_body_lqr
from modelling.whole_body_modelling import WholeBodyModel, WholeBodyModelling
from utils.paths import WHOLE_BODY_LQR_CONFIG_PATH, MODEL_PATH


class WholeBodyLqrController(ChassisController):
    def build_design(self) -> LqrDesign[WholeBodyModel]:
        return design_whole_body_lqr(self.modelling, self.params)

    def __init__(
        self,
        yaml_path: Path = WHOLE_BODY_LQR_CONFIG_PATH,
        model: mujoco.MjModel | Path = MODEL_PATH,
        modelling_type: type[WholeBodyModelling] = WholeBodyModelling,
    ) -> None:
        super().__init__(yaml_path, model, modelling_type)
        self.active_dofs = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]
        self.position_error = np.zeros(self.model.nv)
        self.reference_qpos = self.design.qpos.copy()
        self.reference_qvel = np.zeros(self.model.nv)
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
        self.wheel_offsets = self.data.xpos[self.wheel_bodies, 1] - self.data.qpos[1]

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
