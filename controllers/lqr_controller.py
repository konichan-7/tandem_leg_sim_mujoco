from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from controllers.chassis import ChassisController
from controllers.jump_controller import JumpController
from modelling.kinematics import Leg
from modelling.lqr_design import LqrDesign, design_lqr
from modelling.vmc_modelling import VmcModel, VmcModelling
from utils.control import PID
from utils.paths import LQR_CONFIG_PATH, MODEL_PATH


class LqrController(ChassisController):
    required_sections = ("leg",)

    def build_design(self) -> LqrDesign[VmcModel]:
        return design_lqr(self.modelling, self.params)

    def reset(self, pitch: float = 0.0) -> None:
        super().reset(pitch)
        self.distance = self.reference_distance = 0.0
        for pid in self.leg_pid:
            pid.clear()

    def reset_position_reference(self) -> None:
        super().reset_position_reference()
        self.reference_distance = self.distance

    def __init__(
        self,
        yaml_path: Path = LQR_CONFIG_PATH,
        model: mujoco.MjModel | Path = MODEL_PATH,
        modelling_type: type[VmcModelling] = VmcModelling,
    ) -> None:
        super().__init__(yaml_path, model, modelling_type)
        self.wheel_rotation_jacobian = np.empty((3, self.model.nv))
        self.jump = JumpController(
            self.model, self.data, self.design, self.leg_pose, self.params
        )
        jump_heights = np.unique(
            np.append(
                np.linspace(self.jump.minimum, self.jump.maximum, 9), self.jump.normal
            )
        )
        self.jump_gains = {}
        for phase, overrides in (
            ("default", {}),
            ("landing", self.params["lqr"]["landing_Q_weights"]),
        ):
            self.jump_gains[phase] = CubicSpline(
                jump_heights,
                np.stack(
                    [
                        design_lqr(
                            self.modelling,
                            self.params
                            | {
                                "control": self.params["control"]
                                | {"leg_height": height},
                                "lqr": self.params["lqr"]
                                | {
                                    "Q_weights": self.params["lqr"]["Q_weights"]
                                    | overrides
                                },
                            },
                        ).gain
                        for height in jump_heights
                    ]
                ),
            )
        self.legs = [Leg(self.model, side) for side in ("left", "right")]
        self.reference_legs = [Leg(self.model, side) for side in ("left", "right")]
        self.reference_data = mujoco.MjData(self.model)
        config = self.params["leg"]
        self.leg_pid = [
            PID(
                config["kp"],
                config["ki"],
                config["kd"],
                0.0,
                config["integral_limit"],
                config["force_limit"],
            )
            for _ in self.legs
        ]
        self.state = np.zeros(10)
        self.reference_state = np.zeros(10)
        self.reset()

    def lqr_control(self, dt: float) -> None:
        previous_phase = self.jump.phase
        self.jump.update(dt, not self.returning, self.target_velocity)
        if self.jump.phase == "landing" and previous_phase == "flight":
            self.target_xy[:] = self.data.qpos[:2]
            self.reference_distance = self.distance
        self.update_command(dt)
        if self.jump.phase != "idle":
            self.target_l0 = self.jump.height
        self.target_yaw += self.target_yaw_rate * dt
        heading = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw)])
        self.target_xy += self.target_velocity * heading * dt
        self.reference_distance += self.target_velocity * dt
        reference_height = (
            np.clip(self.jump.actual_height, self.jump.minimum, self.jump.maximum)
            if self.jump.phase in ("thrust", "landing")
            else self.target_l0
        )
        self.reference_data.qpos[:] = self.leg_pose(reference_height)
        self.reference_data.ctrl[:] = self.leg_torque(reference_height)
        mujoco.mj_forward(self.model, self.reference_data)
        for leg, reference in zip(self.legs, self.reference_legs):
            leg.update(self.data)
            reference.update(self.reference_data)
        roll, pitch, yaw = Rotation.from_quat(
            self.data.qpos[3:7], scalar_first=True
        ).as_euler("xyz")
        wheel_speed = np.empty(2)
        lateral = self.data.xmat[self.model.body("base_link").id].reshape(3, 3)[:, 1]
        for i, body in enumerate(self.wheel_bodies):
            mujoco.mj_jacBody(
                self.model, self.data, None, self.wheel_rotation_jacobian, body
            )
            wheel_speed[i] = lateral @ self.wheel_rotation_jacobian @ self.data.qvel
        speed = np.mean(self.radii * wheel_speed)
        self.distance += speed * dt
        yaw_error = np.arctan2(
            np.sin(yaw - self.target_yaw), np.cos(yaw - self.target_yaw)
        )
        self.state[:] = [
            self.distance,
            speed,
            yaw_error,
            self.data.qvel[5],
            self.legs[0].angle - np.pi / 2 + pitch,
            self.legs[0].velocity[1] + self.data.qvel[4],
            self.legs[1].angle - np.pi / 2 + pitch,
            self.legs[1].velocity[1] + self.data.qvel[4],
            pitch,
            self.data.qvel[4],
        ]
        self.reference_state[:] = [
            self.reference_distance,
            self.target_velocity,
            0.0,
            self.target_yaw_rate,
            self.reference_legs[0].angle - np.pi / 2,
            0.0,
            self.reference_legs[1].angle - np.pi / 2,
            0.0,
            0.0,
            0.0,
        ]
        gain = (
            self.jump_gains["landing" if self.jump.phase == "landing" else "default"](
                reference_height
            )
            if self.jump.phase != "idle"
            else self.design.gain
        )
        virtual = gain @ (self.reference_state - self.state)
        config = self.params["leg"]
        balance_torque = np.empty(4)
        for i, (leg, reference) in enumerate(zip(self.legs, self.reference_legs)):
            self.leg_pid[i].target = reference.length
            if previous_phase == "landing" and self.jump.phase == "idle":
                self.leg_pid[i].clear(reference.length - leg.length)
            force = reference.force[0] + self.leg_pid[i].calc(leg.length, dt)
            force = np.clip(force, -config["force_limit"], config["force_limit"])
            torque = leg.torque(force, reference.force[1] + virtual[2 + i])
            self.data.ctrl[leg.actuators] = np.clip(
                torque,
                -self.design.limits[leg.actuators],
                self.design.limits[leg.actuators],
            )
            balance_torque[2 * i : 2 * i + 2] = leg.torque(0.0, virtual[2 + i])
            wheel = self.wheel_joints[i]
            actuator = self.jump.wheels[i]
            self.data.ctrl[actuator] = np.clip(
                wheel.axis[1] * virtual[i],
                -self.design.limits[actuator],
                self.design.limits[actuator],
            )
        self.gimbal.update_reference(self.data.qpos[3:7], dt)
        self.gimbal.control(
            self.reference_data.ctrl,
            self.params["gimbal"]["kp"],
            self.params["gimbal"]["kd"],
        )
        self.jump.control(np.array([roll, pitch, yaw_error]), balance_torque)
