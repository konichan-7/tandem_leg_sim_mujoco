import math
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
import yaml

from demo import (
    ACTUATORS,
    CONTROL,
    JOINTS,
    OFFSETS,
    PATHS,
    SENSORS,
    VMC_GEOMETRY,
)
from utils.math_tools import angle_diff, move_towards, quat_to_euler, wrap
from utils.mujoco_io import (
    ImuData,
    MujocoActuatorWriter,
    MujocoSensorReader,
    place_free_body_on_floor,
)
from utils.pid import PID
from utils.vmc import VMC

sys.path.append(os.path.join(str(Path(__file__).resolve().parents[1]), "sp_lqr"))
from sjtu.lqr import compute_lqr_controller


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class DemoLqrController:
    def __init__(
        self,
        yaml_path: Path = PATHS.lqr_yaml,
    ) -> None:
        self.params = load_yaml(yaml_path)
        self.model = mujoco.MjModel.from_xml_path(str(PATHS.xml))
        self.data = mujoco.MjData(self.model)
        self.sensor = MujocoSensorReader(self.model, self.data)
        self.actuator = MujocoActuatorWriter(
            self.model,
            self.data,
            {
                JOINTS.left_front: ACTUATORS.left_front,
                JOINTS.left_rear: ACTUATORS.left_rear,
                JOINTS.right_front: ACTUATORS.right_front,
                JOINTS.right_rear: ACTUATORS.right_rear,
                JOINTS.left_wheel: ACTUATORS.left_wheel,
                JOINTS.right_wheel: ACTUATORS.right_wheel,
            },
        )
        self.vmc = VMC(
            VMC_GEOMETRY.l1,
            VMC_GEOMETRY.l2,
            VMC_GEOMETRY.l3,
            VMC_GEOMETRY.l4,
            VMC_GEOMETRY.l5,
        )
        self.k = np.asarray(
            compute_lqr_controller(str(yaml_path), verbose=False)[0],
            dtype=float,
        )
        self.control_limits = self.params["lqr"]["control_limits"]
        self.target_leg_angle = self.params["control"]["target_leg_angle"]
        self.command_config = self.params["command"]
        self.l0_pid_l = PID(
            CONTROL.leg_force_kp,
            CONTROL.leg_force_ki,
            CONTROL.leg_force_kd,
            CONTROL.target_l0,
            output_limit=CONTROL.leg_force_limit,
            integral_limit=CONTROL.leg_force_integral_limit,
        )
        self.l0_pid_r = PID(
            CONTROL.leg_force_kp,
            CONTROL.leg_force_ki,
            CONTROL.leg_force_kd,
            CONTROL.target_l0,
            output_limit=CONTROL.leg_force_limit,
            integral_limit=CONTROL.leg_force_integral_limit,
        )
        self.half_weight = (
            0.5 * float(np.sum(self.model.body_mass[1:])) * self.params["g"]
        )
        self.stand_targets = self.leg_targets(CONTROL.target_l0, CONTROL.target_phi0)
        self.last_phi0_l = 0.0
        self.last_phi0_r = 0.0
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.yaw_unwrapped = 0.0
        self.phi0_ready = False
        self.pitch_ready = False
        self.yaw_ready = False
        self.s = 0.0
        self.paused = False
        self.target_l0 = CONTROL.target_l0
        self.target_s = CONTROL.target_s
        self.target_velocity = CONTROL.target_velocity
        self.target_yaw = CONTROL.target_yaw
        self.target_yaw_rate = CONTROL.target_yaw_rate
        self.desired_velocity = CONTROL.target_velocity
        self.desired_yaw_rate = CONTROL.target_yaw_rate
        self.leg_length_direction = 0.0
        self.linear_error_integral = 0.0
        self.leg_length = np.zeros(2, dtype=float)
        self.leg_force = np.zeros(2, dtype=float)
        self.wheel_torque = np.zeros(2, dtype=float)
        place_free_body_on_floor(self.model, self.data, "floor")
        phi = self.leg_phi()
        l0_l = self.vmc.forward_kinematics(
            phi["left_phi1"],
            phi["left_phi4"],
        )[2]
        l0_r = self.vmc.forward_kinematics(
            phi["right_phi1"],
            phi["right_phi4"],
        )[2]
        self.l0_pid_l.clear(CONTROL.target_l0 - l0_l)
        self.l0_pid_r.clear(CONTROL.target_l0 - l0_r)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def command(
        self,
        linear_direction: float,
        yaw_direction: float,
        leg_length_direction: float,
    ) -> None:
        desired_velocity = linear_direction * self.command_config["linear_velocity"]
        desired_yaw_rate = yaw_direction * self.command_config["yaw_rate"]
        motion_changed = (
            desired_velocity != self.desired_velocity
            or desired_yaw_rate != self.desired_yaw_rate
        )
        if not motion_changed and leg_length_direction == self.leg_length_direction:
            return

        if motion_changed:
            self.target_s = self.s
            self.target_yaw = self.yaw_unwrapped if self.yaw_ready else 0.0
            self.linear_error_integral = 0.0
            self.desired_velocity = desired_velocity
            self.desired_yaw_rate = desired_yaw_rate
        self.leg_length_direction = leg_length_direction
        print(
            f"desired_velocity={self.desired_velocity:.2f} "
            f"desired_yaw_rate={self.desired_yaw_rate:.2f} "
            f"target_leg_length={self.target_l0:.3f}"
        )

    def update_command(self, dt: float) -> None:
        self.target_velocity = move_towards(
            self.target_velocity,
            self.desired_velocity,
            self.command_config["linear_acceleration"] * dt,
        )
        self.target_yaw_rate = move_towards(
            self.target_yaw_rate,
            self.desired_yaw_rate,
            self.command_config["yaw_acceleration"] * dt,
        )
        self.target_l0 = float(
            np.clip(
                self.target_l0
                + self.leg_length_direction
                * self.command_config["leg_length_velocity"]
                * dt,
                self.command_config["leg_length_min"],
                self.command_config["leg_length_max"],
            )
        )

    def leg_targets(self, l0: float, phi0: float) -> dict[str, float]:
        phi1, phi4 = self.vmc.inverse_kinematics(l0, phi0)
        return {
            JOINTS.left_front: wrap(phi1 + OFFSETS.left_phi1),
            JOINTS.left_rear: wrap(phi4 + OFFSETS.left_phi4),
            JOINTS.right_front: wrap(phi1 + OFFSETS.right_phi1),
            JOINTS.right_rear: wrap(phi4 + OFFSETS.right_phi4),
        }

    def leg_phi(self) -> dict[str, float]:
        positions = self.sensor.joint_positions(
            {
                JOINTS.left_front: SENSORS.left_front_pos,
                JOINTS.left_rear: SENSORS.left_rear_pos,
                JOINTS.right_front: SENSORS.right_front_pos,
                JOINTS.right_rear: SENSORS.right_rear_pos,
            }
        )
        return {
            "left_phi1": positions[JOINTS.left_front] - OFFSETS.left_phi1,
            "left_phi4": positions[JOINTS.left_rear] - OFFSETS.left_phi4,
            "right_phi1": positions[JOINTS.right_front] - OFFSETS.right_phi1,
            "right_phi4": positions[JOINTS.right_rear] - OFFSETS.right_phi4,
        }

    def imu(self) -> ImuData:
        return self.sensor.imu(SENSORS.quat, SENSORS.acc, SENSORS.gyro)

    def apply_stand_pd(self) -> None:
        pos_sensors = {
            JOINTS.left_front: SENSORS.left_front_pos,
            JOINTS.left_rear: SENSORS.left_rear_pos,
            JOINTS.right_front: SENSORS.right_front_pos,
            JOINTS.right_rear: SENSORS.right_rear_pos,
        }
        vel_sensors = {
            JOINTS.left_front: SENSORS.left_front_vel,
            JOINTS.left_rear: SENSORS.left_rear_vel,
            JOINTS.right_front: SENSORS.right_front_vel,
            JOINTS.right_rear: SENSORS.right_rear_vel,
        }
        values = {}
        for joint, target in self.stand_targets.items():
            q = self.sensor.scalar(pos_sensors[joint])
            dq = self.sensor.joint_velocity(vel_sensors[joint])
            values[joint] = CONTROL.stand_kp * wrap(target - q) - CONTROL.stand_kd * dq
        self.actuator.set_many(values)

    def base_state(self, dt: float) -> tuple[float, float, float, float]:
        pitch, yaw = quat_to_euler(self.imu().quat)

        if self.yaw_ready:
            yaw_delta = angle_diff(yaw, self.last_yaw)
            self.yaw_unwrapped += yaw_delta
            dot_yaw = yaw_delta / dt
        else:
            dot_yaw = 0.0
            self.yaw_unwrapped = yaw
            self.yaw_ready = True
        self.last_yaw = yaw

        if self.pitch_ready:
            dot_pitch = angle_diff(pitch, self.last_pitch) / dt
        else:
            dot_pitch = 0.0
            self.pitch_ready = True
        self.last_pitch = pitch

        return pitch, self.yaw_unwrapped, dot_pitch, dot_yaw

    def leg_angle_rates(
        self,
        phi0_l: float,
        phi0_r: float,
        dt: float,
    ) -> tuple[float, float]:
        if self.phi0_ready:
            dot_phi0_l = angle_diff(phi0_l, self.last_phi0_l) / dt
            dot_phi0_r = angle_diff(phi0_r, self.last_phi0_r) / dt
        else:
            dot_phi0_l = 0.0
            dot_phi0_r = 0.0
            self.phi0_ready = True

        self.last_phi0_l = phi0_l
        self.last_phi0_r = phi0_r
        return dot_phi0_l, dot_phi0_r

    def lqr_control(self, dt: float) -> None:
        self.update_command(dt)
        dtheta_l = self.sensor.joint_velocity(SENSORS.left_wheel_vel)
        dtheta_r = self.sensor.joint_velocity(SENSORS.right_wheel_vel)
        ds = 0.5 * self.params["R_w"] * (dtheta_l + dtheta_r)
        self.s += ds * dt

        pitch, yaw, dot_pitch, dot_yaw = self.base_state(dt)
        yaw_rate_reference = dot_yaw + float(
            np.clip(
                self.target_yaw_rate - dot_yaw,
                -self.command_config["yaw_tracking_error_limit"],
                self.command_config["yaw_tracking_error_limit"],
            )
        )
        self.target_yaw = yaw
        if self.target_velocity:
            if (
                self.linear_error_integral
                or self.target_velocity * (ds - self.target_velocity) > 0.0
            ):
                self.linear_error_integral = float(
                    np.clip(
                        self.linear_error_integral
                        + self.command_config["linear_integral_gain"]
                        * (self.target_velocity - ds)
                        * dt,
                        -self.command_config["linear_error_limit"],
                        self.command_config["linear_error_limit"],
                    )
                )
            self.target_s = self.s + self.linear_error_integral
        phi = self.leg_phi()
        p2l, p3l, l0_l, phi0_l = self.vmc.forward_kinematics(
            phi["left_phi1"],
            phi["left_phi4"],
        )
        p2r, p3r, l0_r, phi0_r = self.vmc.forward_kinematics(
            phi["right_phi1"],
            phi["right_phi4"],
        )
        theta_ll = wrap(-math.pi / 2 + phi0_l + pitch)
        theta_lr = wrap(-math.pi / 2 + phi0_r + pitch)
        dot_phi0_l, dot_phi0_r = self.leg_angle_rates(phi0_l, phi0_r, dt)
        real_state = np.array(
            [
                self.s,
                ds,
                -yaw,
                -dot_yaw,
                theta_ll,
                dot_phi0_l + dot_pitch,
                theta_lr,
                dot_phi0_r + dot_pitch,
                pitch,
                dot_pitch,
            ],
            dtype=float,
        )
        expect_state = np.array(
            [
                self.target_s,
                self.target_velocity,
                -self.target_yaw,
                -yaw_rate_reference,
                self.target_leg_angle,
                0.0,
                self.target_leg_angle,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        u = self.k @ (expect_state - real_state)
        self.l0_pid_l.target = self.target_l0
        self.l0_pid_r.target = self.target_l0
        f_bl = self.half_weight * math.cos(theta_ll) + self.l0_pid_l.calc(l0_l, dt)
        f_br = self.half_weight * math.cos(theta_lr) + self.l0_pid_r.calc(l0_r, dt)
        self.leg_length = np.array([l0_l, l0_r])
        self.leg_force = np.array([f_bl, f_br])
        j_l = self.vmc.mat_jrm(
            phi0_l,
            phi["left_phi1"],
            p2l,
            p3l,
            phi["left_phi4"],
            l0_l,
        )
        j_r = self.vmc.mat_jrm(
            phi0_r,
            phi["right_phi1"],
            p2r,
            p3r,
            phi["right_phi4"],
            l0_r,
        )
        tau_l = self.vmc.virtual_force_to_joint_torque(j_l, f_bl, float(u[2]))
        tau_r = self.vmc.virtual_force_to_joint_torque(j_r, f_br, float(u[3]))
        left_wheel_torque = float(
            np.clip(
                u[0],
                -self.control_limits["T_wl_max"],
                self.control_limits["T_wl_max"],
            )
        )
        right_wheel_torque = float(
            np.clip(
                u[1],
                -self.control_limits["T_wr_max"],
                self.control_limits["T_wr_max"],
            )
        )
        self.wheel_torque = np.array([left_wheel_torque, right_wheel_torque])
        self.actuator.set_many(
            {
                JOINTS.left_front: float(
                    np.clip(
                        tau_l[0],
                        -self.control_limits["T_bl_max"],
                        self.control_limits["T_bl_max"],
                    )
                ),
                JOINTS.left_rear: float(
                    np.clip(
                        tau_l[1],
                        -self.control_limits["T_bl_max"],
                        self.control_limits["T_bl_max"],
                    )
                ),
                JOINTS.right_front: float(
                    np.clip(
                        tau_r[0],
                        -self.control_limits["T_br_max"],
                        self.control_limits["T_br_max"],
                    )
                ),
                JOINTS.right_rear: float(
                    np.clip(
                        tau_r[1],
                        -self.control_limits["T_br_max"],
                        self.control_limits["T_br_max"],
                    )
                ),
                JOINTS.left_wheel: left_wheel_torque,
                JOINTS.right_wheel: right_wheel_torque,
            }
        )

    def control(self) -> None:
        if self.data.time < CONTROL.lqr_start:
            self.apply_stand_pd()
        else:
            self.lqr_control(self.model.opt.timestep)

    def step(self) -> None:
        if not self.paused:
            self.control()
            mujoco.mj_step(self.model, self.data)
