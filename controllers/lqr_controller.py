from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from controllers.gimbal_controller import GimbalController
from controllers.jump_controller import JumpController
from controllers.lqr_design import design_lqr
from modelling.vmc_modelling import Leg, VmcModelling
from utils.control import PID, move_towards
from utils.paths import LQR_CONFIG_PATH, MODEL_PATH


class ChassisController:
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


class LqrController(ChassisController):
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
        self.params = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        self.model = (
            mujoco.MjModel.from_xml_path(str(model))
            if isinstance(model, Path)
            else model
        )
        self.data = mujoco.MjData(self.model)
        self.modelling = modelling_type(self.model)
        self.design = design_lqr(self.modelling, self.params)
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
        self.wheel_joints = [
            self.model.joint(f"{side}_wheel_joint") for side in ("left", "right")
        ]
        self.radii = np.array(
            [
                self.model.geom(f"{side}_wheel_link_collision").size[0]
                for side in ("left", "right")
            ]
        )
        self.wheel_bodies = [
            self.model.body(f"{side}_wheel_link").id for side in ("left", "right")
        ]
        self.wheel_rotation_jacobian = np.empty((3, self.model.nv))
        self.target_xy = np.zeros(2)
        self.gimbal = GimbalController(
            self.model, self.data, self.design.qpos, self.design.limits
        )
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
