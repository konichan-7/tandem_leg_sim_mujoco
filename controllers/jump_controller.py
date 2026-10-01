from abc import ABC, abstractmethod

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

from modelling.base import closed_chain_tangent
from modelling.kinematics import Leg
from modelling.lqr_design import LqrDesign
from utils.control import move_towards


class BaseJumpController(ABC):
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        design: LqrDesign,
        leg_pose: CubicSpline,
        params: dict,
    ) -> None:
        self.model = model
        self.data = data
        self.design = design
        self.leg_pose = leg_pose
        self.config = params["jump"]
        self.minimum = self.config["leg_length_min"]
        self.maximum = self.config["leg_length_max"]
        self.normal = params["control"]["leg_height"]
        self.wheels = np.array(
            [model.actuator(f"{side}_wheel_motor").id for side in ("left", "right")]
        )
        self.hips = np.array(
            [
                model.actuator(f"{side}_{part}_motor").id
                for side in ("left", "right")
                for part in ("front", "rear")
            ]
        )
        joints = model.actuator_trnid[self.hips, 0]
        self.hip_qpos = model.jnt_qposadr[joints]
        self.hip_dofs = model.jnt_dofadr[joints]
        self.base = model.body("base_link").id
        self.mass = np.empty((model.nv, model.nv))
        self.active_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.passive_dofs = np.setdiff1d(
            np.arange(model.nv), np.concatenate((np.arange(6), self.active_dofs))
        )
        self.weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
        self.support_force = np.zeros(len(self.wheels))
        self.hip_bias = np.zeros(len(self.hips))
        self.response = np.zeros((model.nv, model.nv))
        self.acceleration_bias = np.zeros(model.nv)
        self.leg_jacobian = np.zeros((len(self.wheels), model.nv))
        self.hip_bodies = np.array(
            [model.body(f"{side}_front_link").id for side in ("left", "right")]
        )
        self.wheel_bodies = np.array(
            [model.body(f"{side}_wheel_link").id for side in ("left", "right")]
        )

    def reset(self) -> None:
        self.phase = "idle"
        self.held = False
        self.released = False
        self.height = self.normal
        self.height_velocity = 0.0
        self.leg_jacobian[:] = 0.0
        self.trajectory_target = self.normal
        self.trajectory_start = self.normal
        self.trajectory_time = 0.0
        self.trajectory_duration = 0.0
        self.height_acceleration = 0.0
        self.actual_height = self.normal
        self.phase_time = 0.0
        self.contacting = True
        self.air_time = 0.0
        self.extended = False
        self.motion = "crouch"
        self.hold_time = 0.0
        self.support_force[:] = self.weight / 2

    def command(self, held: bool, height: float) -> None:
        if held and not self.held and self.phase == "idle":
            self.phase = "crouch"
            self.height = height
            self.phase_time = 0.0
            self.extended = False
            self.motion = "crouch"
            self.hold_time = 0.0
        if self.phase == "crouch":
            self.released = not held
        self.held = held

    def height_jacobian(self, data: mujoco.MjData) -> np.ndarray:
        jacobian = np.empty_like(self.leg_jacobian)
        hip_jacobian = np.empty((3, self.model.nv))
        wheel_jacobian = np.empty((3, self.model.nv))
        rotation_jacobian = np.empty((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data, None, rotation_jacobian, self.base)
        vertical = data.xmat[self.base].reshape(3, 3)[:, 2]
        for side, (hip, wheel) in enumerate(zip(self.hip_bodies, self.wheel_bodies)):
            mujoco.mj_jacBody(self.model, data, hip_jacobian, None, hip)
            mujoco.mj_jacBody(self.model, data, wheel_jacobian, None, wheel)
            relative = data.xpos[hip] - data.xpos[wheel]
            jacobian[side] = (
                vertical @ (hip_jacobian - wheel_jacobian)
                + np.cross(vertical, relative) @ rotation_jacobian
            )
        return jacobian

    def update_trajectory(self, dt: float, target: float, speed: float) -> None:
        if self.phase in ("flight", "retract", "landing") and self.extended:
            if target != self.trajectory_target:
                self.trajectory_start = self.height
                self.trajectory_target = target
                self.trajectory_time = 0.0
                self.trajectory_duration = max(
                    1.875 * abs(target - self.height) / speed, dt
                )
            self.trajectory_time = min(
                self.trajectory_time + dt, self.trajectory_duration
            )
            t = self.trajectory_time / self.trajectory_duration
            distance = target - self.trajectory_start
            self.height = self.trajectory_start + distance * (
                10 * t**3 - 15 * t**4 + 6 * t**5
            )
            self.height_velocity = (
                distance
                * (30 * t**2 - 60 * t**3 + 30 * t**4)
                / self.trajectory_duration
            )
            self.height_acceleration = (
                distance
                * (60 * t - 180 * t**2 + 120 * t**3)
                / self.trajectory_duration**2
            )
            return
        self.height_acceleration = 0.0
        previous_height = self.height
        self.height = move_towards(self.height, target, speed * dt)
        self.height_velocity = (self.height - previous_height) / dt

    def update(self, dt: float, ready_to_launch: bool, forward_speed: float) -> None:
        if self.phase == "idle":
            return
        mujoco.mj_forward(self.model, self.data)
        self.update_support()
        self.air_time = 0.0 if self.contacting else self.air_time + dt
        self.phase_time += dt
        leg_heights = (
            self.data.xpos[self.hip_bodies] - self.data.xpos[self.wheel_bodies]
        ) @ self.data.xmat[self.base].reshape(3, 3)[:, 2]
        leg_height = leg_heights.mean()
        self.actual_height = leg_height
        self.leg_jacobian[:] = self.height_jacobian(self.data)
        forward = self.data.xmat[self.base].reshape(3, 3)[:, 0]
        velocity_error = self.data.qvel[:3] - forward_speed * np.array(
            [forward[0], forward[1], 0.0]
        )
        self.extended |= (
            np.min(leg_heights) >= self.maximum - self.config["ready_height_error"]
        )
        self.advance_phase(dt, ready_to_launch, leg_heights, leg_height, velocity_error)
        if self.phase == "crouch":
            target, speed = self.minimum, self.config["crouch_velocity"]
        elif self.phase == "thrust":
            target, speed = self.maximum, self.config["extension_velocity"]
        elif self.phase in ("flight", "retract"):
            self.update_motion(dt, leg_heights)
            target = {
                "extend": self.maximum,
                "retract": self.minimum,
                "hold": self.minimum,
                "deploy": self.normal,
            }[self.motion]
            speed = self.config["retraction_velocity"]
        else:
            target, speed = self.normal, self.config["retraction_velocity"]
        self.update_trajectory(dt, target, speed)

    def advance_phase(
        self,
        dt: float,
        ready_to_launch: bool,
        leg_heights: np.ndarray,
        leg_height: float,
        velocity_error: np.ndarray,
    ) -> None:
        self.before_transition(dt, ready_to_launch, leg_heights, velocity_error)
        if self.launch_ready(ready_to_launch, leg_heights, velocity_error):
            self.phase = "thrust"
            self.motion = "extend"
            self.phase_time = 0.0
        elif (
            self.phase in ("thrust", "retract")
            and self.air_time >= self.config["liftoff_time"]
        ):
            self.phase = "flight"
            self.phase_time = 0.0
        elif self.phase == "thrust" and leg_height >= self.maximum:
            self.phase = "retract"
            self.phase_time = 0.0
        elif self.phase == "flight" and self.landing_trigger():
            self.phase = "landing"
            self.phase_time = 0.0
            self.on_landing()
        elif self.phase == "landing":
            self.leave_landing()

    def update_motion(self, dt: float, leg_heights: np.ndarray) -> None:
        if self.motion == "extend" and self.extended:
            self.motion = "retract"
        elif (
            self.motion == "retract"
            and self.trajectory_time >= self.trajectory_duration
            and np.max(np.abs(leg_heights - self.minimum))
            <= self.config["tuck_height_error"]
            and self.tuck_settled()
        ):
            self.motion = "hold"
            self.hold_time = 0.0
        elif self.motion == "hold":
            self.hold_time = (
                self.hold_time + dt
                if not self.contacting
                and np.max(np.abs(leg_heights - self.minimum))
                <= self.config["tuck_height_error"]
                else 0.0
            )
            if self.hold_time >= self.config["air_hold_time"]:
                self.motion = "deploy"

    def before_transition(
        self,
        dt: float,
        ready_to_launch: bool,
        leg_heights: np.ndarray,
        velocity_error: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def launch_ready(
        self,
        ready_to_launch: bool,
        leg_heights: np.ndarray,
        velocity_error: np.ndarray,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def landing_trigger(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def leave_landing(self) -> None:
        raise NotImplementedError

    def on_landing(self) -> None:
        pass

    def tuck_settled(self) -> bool:
        return True

    @abstractmethod
    def update_support(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def control_ground(self, balance_torque: np.ndarray) -> None:
        raise NotImplementedError

    def control(self, attitude_error: np.ndarray, balance_torque: np.ndarray) -> None:
        self.control_ground(balance_torque)
        if not self.contacting:
            self.data.ctrl[self.wheels] = 0.0
        if self.phase == "idle" or (self.contacting and self.phase != "flight"):
            return
        target = self.leg_pose(self.height)[self.hip_qpos]
        torque = (
            self.config["air_joint_kp"] * (target - self.data.qpos[self.hip_qpos])
            - self.config["air_joint_kd"] * self.data.qvel[self.hip_dofs]
        )
        other_torque = self.data.ctrl.copy()
        other_torque[self.hips] = other_torque[self.wheels] = 0.0
        full_bias = self.acceleration_bias + self.response[:, self.active_dofs] @ (
            other_torque * self.model.actuator_gear[:, 0]
        )
        desired = (
            -self.config["air_attitude_kp"] * attitude_error
            - self.config["air_attitude_kd"] * self.data.qvel[3:6]
        )
        limits = self.design.limits[self.hips]
        regularizer = np.eye(len(self.hips)) * self.config["air_posture_weight"]
        lengths = (
            self.data.xpos[self.hip_bodies] - self.data.xpos[self.wheel_bodies]
        ) @ self.data.xmat[self.base].reshape(3, 3)[:, 2]
        leg_acceleration = (
            self.height_acceleration
            + self.config["air_leg_kp"] * (self.height - lengths)
            + self.config["air_leg_kd"]
            * (self.height_velocity - self.leg_jacobian @ self.data.qvel)
        )
        leg_weight = self.config["air_leg_weight"]
        solution = lsq_linear(
            np.vstack(
                (
                    self.response[self.attitude_dofs, self.hip_dofs],
                    leg_weight * self.leg_jacobian @ self.response[:, self.hip_dofs],
                    regularizer,
                )
            ),
            np.concatenate(
                (
                    desired[self.attitude_axes] - full_bias[self.attitude_dofs],
                    leg_weight * (leg_acceleration - self.leg_jacobian @ full_bias),
                    regularizer @ torque,
                )
            ),
            bounds=(-limits, limits),
        )
        if not solution.success:
            raise RuntimeError("Airborne attitude torque allocation failed")
        self.data.ctrl[self.hips] = solution.x
        self.data.ctrl[self.wheels] = 0.0


class WholeBodyJumpController(BaseJumpController):
    attitude_axes = slice(0, 2)
    attitude_dofs = slice(3, 5)

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        design: LqrDesign,
        leg_pose: CubicSpline,
        params: dict,
    ) -> None:
        super().__init__(model, data, design, leg_pose, params)
        self.force_map = np.zeros((len(self.hips), len(self.wheels)))
        self.terrain_group = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.ground_height = 0.0

    def update_support(self) -> None:
        mujoco.mj_fullM(self.model, self.data, self.mass)
        jacobian = self.data.efc_J.reshape(self.data.nefc, self.model.nv)
        jacobian = jacobian[self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        tangent = np.zeros((self.model.nv, len(self.hips)))
        tangent[self.hip_dofs] = np.eye(len(self.hips))
        tangent[self.passive_dofs] = -np.linalg.lstsq(
            jacobian[:, self.passive_dofs], jacobian[:, self.hip_dofs], rcond=None
        )[0]
        residual = tangent.T @ (
            self.mass @ self.data.qacc
            + self.data.qfrc_bias
            - self.data.qfrc_passive
            - self.data.qfrc_actuator
            - self.data.qfrc_applied
        )
        forward = self.data.xmat[self.base].reshape(3, 3)[:, 0]
        forward = np.array([forward[0], forward[1], 0.0])
        forward /= np.linalg.norm(forward)
        force_map = np.empty((len(self.hips), 2 * len(self.wheels)))
        foot_jacobian = np.empty((3, self.model.nv))
        for side, body in enumerate(self.wheel_bodies):
            mujoco.mj_jacBody(self.model, self.data, foot_jacobian, None, body)
            virtual_jacobian = (
                np.vstack((forward @ foot_jacobian, foot_jacobian[2])) @ tangent
            )
            force_map[:, 2 * side : 2 * side + 2] = virtual_jacobian.T
        self.support_force[:] = np.linalg.solve(force_map, residual)[1::2]
        self.force_map[:] = force_map[:, 1::2]
        self.hip_bias[:] = tangent.T @ self.data.qfrc_bias
        threshold = (
            self.weight
            * self.config[
                "liftoff_force_ratio" if self.contacting else "landing_force_ratio"
            ]
        )
        self.contacting = np.maximum(self.support_force, 0.0).sum() > threshold
        if self.phase != "idle":
            inverse_mass = np.linalg.inv(self.mass)
            response = (
                inverse_mass
                - inverse_mass
                @ jacobian.T
                @ np.linalg.pinv(jacobian @ inverse_mass @ jacobian.T)
                @ jacobian
                @ inverse_mass
            )
            self.response[:] = response
            self.acceleration_bias[:] = (
                self.data.qacc - response @ self.data.qfrc_actuator
            )

    def terrain_height(self) -> float:
        heights = []
        for body in self.wheel_bodies:
            origin = self.data.xpos[body].copy()
            origin[2] += self.model.stat.extent
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                np.array([0.0, 0.0, -1.0]),
                self.terrain_group,
                True,
                -1,
                None,
            )
            if distance < 0:
                raise ValueError(
                    "Jump requires terrain in geom group 1 below the wheels"
                )
            heights.append(origin[2] - distance)
        return max(heights)

    def launch_ready(
        self,
        ready_to_launch: bool,
        leg_heights: np.ndarray,
        velocity_error: np.ndarray,
    ) -> bool:
        return (
            self.phase == "crouch"
            and self.released
            and ready_to_launch
            and self.contacting
            and np.max(np.abs(leg_heights - self.minimum))
            < self.config["ready_height_error"]
            and np.linalg.norm(velocity_error) < self.config["ready_speed"]
            and np.linalg.norm(self.data.qvel[3:6]) < self.config["ready_angular_speed"]
        )

    def landing_trigger(self) -> bool:
        return self.contacting and self.data.qvel[2] < 0

    def on_landing(self) -> None:
        self.ground_height = self.terrain_height()

    def leave_landing(self) -> None:
        if not self.contacting:
            self.phase = "flight"
            self.phase_time = 0.0
        elif self.phase_time >= self.config["landing_time"]:
            self.phase = "idle"
            self.phase_time = 0.0

    def control_ground(self, balance_torque: np.ndarray) -> None:
        if self.phase in ("thrust", "landing") and self.contacting:
            scale = self.config["thrust_gravity_scale"]
            if self.phase == "landing":
                acceleration = self.config["landing_leg_kp"] * (
                    self.normal - self.actual_height
                ) - self.config["landing_leg_kd"] * np.mean(
                    self.leg_jacobian @ self.data.qvel
                )
                scale = max(
                    0.0, 1 + acceleration / np.linalg.norm(self.model.opt.gravity)
                )
            force_torque = -self.force_map @ np.full(2, scale * self.weight / 2)
            limits = self.design.limits[self.hips]
            balance = np.clip(self.hip_bias + balance_torque, -limits, limits)
            fractions = np.divide(
                limits - np.sign(force_torque) * balance,
                np.abs(force_torque),
                out=np.ones(len(self.hips)),
                where=force_torque != 0,
            )
            self.data.ctrl[self.hips] = (
                balance + np.clip(np.min(fractions), 0.0, 1.0) * force_torque
            )


class JumpController(BaseJumpController):
    attitude_axes = slice(1, 2)
    attitude_dofs = slice(4, 5)

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        design: LqrDesign,
        leg_pose: CubicSpline,
        params: dict,
    ) -> None:
        super().__init__(model, data, design, leg_pose, params)
        self.leg_config = params["leg"]
        self.legs = [Leg(model, side) for side in ("left", "right")]
        self.inertia_force = np.empty(model.nv)
        self.momentum_matrix = np.empty((3, model.nv))
        self.wheel_radii = np.array(
            [
                model.geom(f"{side}_wheel_link_collision").size[0]
                for side in ("left", "right")
            ]
        )
        self.wheel_signs = model.jnt_axis[model.actuator_trnid[self.wheels, 0], 1]
        self.probe = mujoco.MjData(model)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.ready_time = 0.0

    def update_support(self) -> None:
        jacobian = self.data.efc_J.reshape(self.data.nefc, self.model.nv)
        jacobian = jacobian[self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        tangent = np.zeros((self.model.nv, len(self.hips)))
        tangent[self.hip_dofs] = np.eye(len(self.hips))
        tangent[self.passive_dofs] = -np.linalg.lstsq(
            jacobian[:, self.passive_dofs], jacobian[:, self.hip_dofs], rcond=None
        )[0]
        mujoco.mj_mulM(self.model, self.data, self.inertia_force, self.data.qacc)
        compensation = tangent.T @ (
            self.inertia_force
            + self.data.qfrc_bias
            - self.data.qfrc_passive
            - self.data.qfrc_applied
        )
        self.hip_bias[:] = tangent.T @ self.data.qfrc_bias
        motor_feedback = tangent.T @ self.data.qfrc_actuator
        pitch = np.arctan2(
            -self.data.xmat[self.base].reshape(3, 3)[2, 0],
            np.hypot(*self.data.xmat[self.base].reshape(3, 3)[:2, 0]),
        )
        for i, leg in enumerate(self.legs):
            leg.update(self.data)
            force, moment = leg.vmc.joint_torque_to_virtual_force(
                leg.j_t, (motor_feedback - compensation)[2 * i : 2 * i + 2]
            )
            theta = leg.angle - np.pi / 2 + pitch
            self.support_force[i] = (
                force * np.cos(theta) - moment * np.sin(theta) / leg.length
            )
        threshold = (
            self.weight
            * self.config[
                "liftoff_force_ratio" if self.contacting else "landing_force_ratio"
            ]
        )
        self.contacting = np.maximum(self.support_force, 0.0).sum() > threshold

        if not self.contacting:
            mujoco.mj_fullM(self.model, self.data, self.mass)
            tangent, retained = closed_chain_tangent(self.model, self.data)
            velocity = tangent @ self.data.qvel[retained]
            tangents = []
            for step in (-1e-5, 1e-5):
                self.probe.qpos[:] = self.data.qpos
                mujoco.mj_integratePos(self.model, self.probe.qpos, velocity, step)
                mujoco.mj_forward(self.model, self.probe)
                tangents.append(closed_chain_tangent(self.model, self.probe)[0])
            curvature = (tangents[1] - tangents[0]) @ self.data.qvel[retained] / 2e-5
            self.response[:] = tangent @ np.linalg.solve(
                tangent.T @ self.mass @ tangent, tangent.T
            )
            self.acceleration_bias[:] = curvature + self.response @ (
                self.data.qfrc_passive
                + self.data.qfrc_applied
                - self.data.qfrc_bias
                - self.mass @ curvature
            )

    def before_transition(
        self,
        dt: float,
        ready_to_launch: bool,
        leg_heights: np.ndarray,
        velocity_error: np.ndarray,
    ) -> None:
        ready = (
            self.phase == "crouch"
            and ready_to_launch
            and self.contacting
            and np.max(np.abs(leg_heights - self.minimum))
            < self.config["ready_height_error"]
            and np.linalg.norm(velocity_error) < self.config["ready_speed"]
            and np.linalg.norm(self.data.qvel[3:6]) < self.config["ready_angular_speed"]
        )
        self.ready_time = self.ready_time + dt if ready else 0.0

    def launch_ready(
        self,
        ready_to_launch: bool,
        leg_heights: np.ndarray,
        velocity_error: np.ndarray,
    ) -> bool:
        return (
            self.phase == "crouch"
            and self.released
            and self.ready_time >= self.config["ready_time"]
        )

    def landing_trigger(self) -> bool:
        return self.contacting and (self.data.qvel[2] < 0 or self.motion == "deploy")

    def leave_landing(self) -> None:
        if self.air_time >= self.config["liftoff_time"]:
            self.phase = "flight"
            self.phase_time = 0.0
        elif self.contacting and self.phase_time >= self.config["landing_time"]:
            self.phase = "idle"

    def tuck_settled(self) -> bool:
        return (
            np.max(np.abs(self.leg_jacobian @ self.data.qvel))
            < self.config["ready_speed"]
        )

    def control_ground(self, balance_torque: np.ndarray) -> None:
        if self.phase in ("thrust", "landing") and self.contacting:
            lengths = (
                self.data.xpos[self.hip_bodies] - self.data.xpos[self.wheel_bodies]
            ) @ self.data.xmat[self.base].reshape(3, 3)[:, 2]
            velocities = self.leg_jacobian @ self.data.qvel
            for i, leg in enumerate(self.legs):
                if self.phase == "thrust":
                    force = self.config["thrust_gravity_scale"] * self.weight / 2
                    force += self.leg_config["kp"] * (
                        lengths.mean() - lengths[i]
                    ) + self.leg_config["kd"] * (velocities.mean() - velocities[i])
                else:
                    acceleration = (
                        self.config["landing_leg_kp"] * (self.normal - lengths[i])
                        - self.config["landing_leg_kd"] * velocities[i]
                    )
                    force = (
                        self.weight
                        / 2
                        * max(
                            0.0,
                            1 + acceleration / np.linalg.norm(self.model.opt.gravity),
                        )
                    )
                limits = self.design.limits[leg.actuators]
                balance = np.clip(
                    (self.hip_bias + balance_torque)[2 * i : 2 * i + 2], -limits, limits
                )
                force_torque = leg.torque(force, 0.0)
                active = force_torque != 0.0
                scale = min(
                    1.0,
                    np.min(
                        (
                            limits[active]
                            - np.sign(force_torque[active]) * balance[active]
                        )
                        / np.abs(force_torque[active]),
                        initial=1.0,
                    ),
                )
                self.data.ctrl[leg.actuators] = balance + scale * force_torque
        if self.phase == "thrust" and self.contacting:
            mujoco.mj_angmomMat(self.model, self.data, self.momentum_matrix, self.base)
            forward = self.data.xmat[self.base].reshape(3, 3)[:, 0].copy()
            forward[2] = 0.0
            forward /= np.linalg.norm(forward)
            lateral = np.cross([0.0, 0.0, 1.0], forward)
            momentum = lateral @ self.momentum_matrix @ self.data.qvel
            offset = self.data.subtree_com[self.base] - self.data.xpos[
                self.wheel_bodies
            ].mean(axis=0)
            height = offset[2] + self.wheel_radii.mean()
            force = (
                self.config["momentum_gain"] * momentum
                + (offset @ forward) * np.maximum(self.support_force, 0.0).sum()
            ) / height
            self.data.ctrl[self.wheels] = np.clip(
                self.wheel_signs * self.wheel_radii * force / 2,
                -self.design.limits[self.wheels],
                self.design.limits[self.wheels],
            )
