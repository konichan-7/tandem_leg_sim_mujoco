import argparse
from dataclasses import dataclass
import math
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
    KEY_COMMANDS,
    OFFSETS,
    PATHS,
    SENSORS,
    VMC_GEOMETRY,
)
from utils.math_tools import angle_diff, move_towards, quat_to_euler, wrap
from utils.mujoco_io import (
    MujocoActuatorWriter,
    MujocoSensorReader,
    place_free_body_on_floor,
)
from utils.pid import PID
from utils.vmc import VMC
from utils.viewer import run_interactive

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "sp_lqr"))

from mujoco_mpc import MujocoMpcController, load_or_export_mujoco_mpc
from sjtu.lqr import compute_lqr_controller

MPC_EXPORT_PATH = ROOT / "sp_lqr" / "generated" / "demo_mpc.npz"


@dataclass(frozen=True)
class PlantState:
    vector: np.ndarray
    left_leg_length: float
    right_leg_length: float
    left_jacobian: np.ndarray
    right_jacobian: np.ndarray
    left_leg_angle: float
    right_leg_angle: float


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class DemoMpcController:
    def __init__(self, yaml_path: Path = PATHS.mpc_yaml) -> None:
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
        mpc_config = self.params["mpc"]
        self.target_leg_angle = self.params["control"]["target_leg_angle"]
        self.command_config = self.params["command"]
        self.mpc_model = load_or_export_mujoco_mpc(
            yaml_path=yaml_path,
            export_path=MPC_EXPORT_PATH,
            sample_time=float(mpc_config["sample_time"]),
            horizon=int(mpc_config["horizon"]),
        )
        self.mpc = MujocoMpcController(self.mpc_model)
        self.mpc_interval = int(
            round(self.mpc_model.sample_time / self.model.opt.timestep)
        )
        self.lqr_gain = np.asarray(
            compute_lqr_controller(yaml_path, verbose=False)[0],
            dtype=float,
        )
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
        self.expected_state = np.array(
            [
                CONTROL.target_s,
                CONTROL.target_velocity,
                -CONTROL.target_yaw,
                0.0,
                self.target_leg_angle,
                0.0,
                self.target_leg_angle,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        self.desired_velocity = CONTROL.target_velocity
        self.desired_yaw_rate = CONTROL.target_yaw_rate
        self.cached_control = np.zeros(self.mpc_model.nu, dtype=float)
        self.last_phi0_l = 0.0
        self.last_phi0_r = 0.0
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.yaw_unwrapped = 0.0
        self.phi0_ready = False
        self.pitch_ready = False
        self.yaw_ready = False
        self.s = 0.0
        self.mpc_step = 0
        self.paused = False
        self.last_mpc_status: str | None = None
        self.linear_error_integral = 0.0

        place_free_body_on_floor(self.model, self.data, "floor")
        phi = self.leg_phi()
        left_leg_length = self.vmc.forward_kinematics(
            phi["left_phi1"],
            phi["left_phi4"],
        )[2]
        right_leg_length = self.vmc.forward_kinematics(
            phi["right_phi1"],
            phi["right_phi4"],
        )[2]
        self.l0_pid_l.clear(CONTROL.target_l0 - left_leg_length)
        self.l0_pid_r.clear(CONTROL.target_l0 - right_leg_length)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def command(self, linear_direction: float, yaw_direction: float) -> None:
        desired_velocity = linear_direction * self.command_config["linear_velocity"]
        desired_yaw_rate = -yaw_direction * self.command_config["yaw_rate"]
        if (
            desired_velocity == self.desired_velocity
            and desired_yaw_rate == self.desired_yaw_rate
        ):
            return

        self.expected_state[0] = self.s
        self.expected_state[2] = -self.yaw_unwrapped if self.yaw_ready else 0.0
        self.linear_error_integral = 0.0
        self.desired_velocity = desired_velocity
        self.desired_yaw_rate = desired_yaw_rate
        print(
            f"desired_velocity={self.desired_velocity:.2f} "
            f"desired_yaw_rate={-self.desired_yaw_rate:.2f}"
        )

    def update_command(self, dt: float) -> None:
        self.expected_state[1] = move_towards(
            self.expected_state[1],
            self.desired_velocity,
            self.command_config["linear_acceleration"] * dt,
        )
        self.expected_state[3] = move_towards(
            self.expected_state[3],
            self.desired_yaw_rate,
            self.command_config["yaw_acceleration"] * dt,
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

    def base_state(self, dt: float) -> tuple[float, float, float, float]:
        pitch, yaw = quat_to_euler(self.sensor.sensor(SENSORS.quat))

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

    def plant_state(self, dt: float) -> PlantState:
        dtheta_l = self.sensor.joint_velocity(SENSORS.left_wheel_vel)
        dtheta_r = self.sensor.joint_velocity(SENSORS.right_wheel_vel)
        ds = 0.5 * self.params["R_w"] * (dtheta_l + dtheta_r)
        self.s += ds * dt

        pitch, yaw, dot_pitch, dot_yaw = self.base_state(dt)
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
        vector = np.array(
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
        left_jacobian = self.vmc.mat_jrm(
            phi0_l,
            phi["left_phi1"],
            p2l,
            p3l,
            phi["left_phi4"],
            l0_l,
        )
        right_jacobian = self.vmc.mat_jrm(
            phi0_r,
            phi["right_phi1"],
            p2r,
            p3r,
            phi["right_phi4"],
            l0_r,
        )
        return PlantState(
            vector,
            l0_l,
            l0_r,
            left_jacobian,
            right_jacobian,
            theta_ll,
            theta_lr,
        )

    def apply_stand_pd(self) -> None:
        positions = {
            JOINTS.left_front: SENSORS.left_front_pos,
            JOINTS.left_rear: SENSORS.left_rear_pos,
            JOINTS.right_front: SENSORS.right_front_pos,
            JOINTS.right_rear: SENSORS.right_rear_pos,
        }
        velocities = {
            JOINTS.left_front: SENSORS.left_front_vel,
            JOINTS.left_rear: SENSORS.left_rear_vel,
            JOINTS.right_front: SENSORS.right_front_vel,
            JOINTS.right_rear: SENSORS.right_rear_vel,
        }
        values = {
            joint: CONTROL.stand_kp
            * wrap(target - self.sensor.scalar(positions[joint]))
            - CONTROL.stand_kd * self.sensor.joint_velocity(velocities[joint])
            for joint, target in self.stand_targets.items()
        }
        self.actuator.set_many(values)

    def reference_trajectory(self, state: np.ndarray) -> np.ndarray:
        reference = np.repeat(
            self.expected_state[None, :],
            self.mpc_model.horizon + 1,
            axis=0,
        )
        prediction_time = np.arange(self.mpc_model.horizon + 1) * (
            self.mpc_model.sample_time
        )
        linear_velocity = self.expected_state[1] + np.clip(
            self.desired_velocity - self.expected_state[1],
            -self.command_config["linear_acceleration"] * prediction_time,
            self.command_config["linear_acceleration"] * prediction_time,
        )
        planned_yaw_rate = self.expected_state[3] + np.clip(
            self.desired_yaw_rate - self.expected_state[3],
            -self.command_config["yaw_acceleration"] * prediction_time,
            self.command_config["yaw_acceleration"] * prediction_time,
        )
        yaw_rate = state[3] + np.clip(
            planned_yaw_rate - state[3],
            -self.command_config["yaw_tracking_error_limit"],
            self.command_config["yaw_tracking_error_limit"],
        )
        reference[:, 1] = linear_velocity
        reference[:, 3] = yaw_rate
        reference[:, 2] = state[2]
        reference[1:, 0] += np.cumsum(
            0.5
            * (linear_velocity[:-1] + linear_velocity[1:])
            * self.mpc_model.sample_time
        )
        reference[1:, 2] += np.cumsum(
            0.5 * (yaw_rate[:-1] + yaw_rate[1:]) * self.mpc_model.sample_time
        )
        return reference

    def mpc_control(self, dt: float) -> None:
        self.update_command(dt)
        state = self.plant_state(dt)
        self.expected_state[2] = state.vector[2]
        if self.expected_state[1]:
            if (
                self.linear_error_integral
                or self.expected_state[1] * (state.vector[1] - self.expected_state[1])
                > 0.0
            ):
                self.linear_error_integral = float(
                    np.clip(
                        self.linear_error_integral
                        + self.command_config["linear_integral_gain"]
                        * (self.expected_state[1] - state.vector[1])
                        * dt,
                        -self.command_config["linear_error_limit"],
                        self.command_config["linear_error_limit"],
                    )
                )
            self.expected_state[0] = state.vector[0] + self.linear_error_integral
        self.l0_pid_l.target = CONTROL.target_l0
        self.l0_pid_r.target = CONTROL.target_l0
        left_leg_force = self.half_weight * math.cos(
            state.left_leg_angle
        ) + self.l0_pid_l.calc(state.left_leg_length, dt)
        right_leg_force = self.half_weight * math.cos(
            state.right_leg_angle
        ) + self.l0_pid_r.calc(state.right_leg_length, dt)

        left_support_torque = np.clip(
            self.vmc.virtual_force_to_joint_torque(
                state.left_jacobian,
                left_leg_force,
                0.0,
            ),
            -self.mpc_model.hip_torque_limit,
            self.mpc_model.hip_torque_limit,
        )
        right_support_torque = np.clip(
            self.vmc.virtual_force_to_joint_torque(
                state.right_jacobian,
                right_leg_force,
                0.0,
            ),
            -self.mpc_model.hip_torque_limit,
            self.mpc_model.hip_torque_limit,
        )
        left_leg_force = float(
            self.vmc.joint_torque_to_virtual_force(
                state.left_jacobian,
                left_support_torque,
            )[0]
        )
        right_leg_force = float(
            self.vmc.joint_torque_to_virtual_force(
                state.right_jacobian,
                right_support_torque,
            )[0]
        )

        if self.mpc_step == 0:
            reference = self.reference_trajectory(state.vector)
            output = self.mpc.solve(
                x0=state.vector,
                x_ref=reference,
                left_leg_length=state.left_leg_length,
                right_leg_length=state.right_leg_length,
                left_j_t=state.left_jacobian,
                right_j_t=state.right_jacobian,
                left_leg_force=left_leg_force,
                right_leg_force=right_leg_force,
            )
            if output.solved:
                self.cached_control = output.control
                self.last_mpc_status = None
            else:
                lqr_control = self.lqr_gain @ (reference[0] - state.vector)
                self.cached_control = self.mpc.project_control(
                    lqr_control,
                    state.left_jacobian,
                    state.right_jacobian,
                    left_leg_force,
                    right_leg_force,
                )
            if not output.solved and output.status != self.last_mpc_status:
                print(f"MPC status={output.status} iterations={output.iterations}")
                self.last_mpc_status = output.status
        self.mpc_step = (self.mpc_step + 1) % self.mpc_interval

        wheel_l, wheel_r, hip_l, hip_r = self.cached_control
        left_torque = left_support_torque + self.vmc.virtual_force_to_joint_torque(
            state.left_jacobian,
            0.0,
            float(hip_l),
        )
        right_torque = right_support_torque + self.vmc.virtual_force_to_joint_torque(
            state.right_jacobian,
            0.0,
            float(hip_r),
        )
        torque_limit = self.mpc_model.hip_torque_limit
        self.actuator.set_many(
            {
                JOINTS.left_front: float(
                    np.clip(left_torque[0], -torque_limit, torque_limit)
                ),
                JOINTS.left_rear: float(
                    np.clip(left_torque[1], -torque_limit, torque_limit)
                ),
                JOINTS.right_front: float(
                    np.clip(right_torque[0], -torque_limit, torque_limit)
                ),
                JOINTS.right_rear: float(
                    np.clip(right_torque[1], -torque_limit, torque_limit)
                ),
                JOINTS.left_wheel: float(
                    np.clip(
                        wheel_l,
                        -self.mpc_model.wheel_torque_limit,
                        self.mpc_model.wheel_torque_limit,
                    )
                ),
                JOINTS.right_wheel: float(
                    np.clip(
                        wheel_r,
                        -self.mpc_model.wheel_torque_limit,
                        self.mpc_model.wheel_torque_limit,
                    )
                ),
            }
        )

    def control(self) -> None:
        if self.data.time < CONTROL.lqr_start:
            self.apply_stand_pd()
        else:
            self.mpc_control(self.model.opt.timestep)

    def step(self) -> None:
        if not self.paused:
            self.control()
            mujoco.mj_step(self.model, self.data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="?", type=Path, default=PATHS.mpc_yaml)
    return parser.parse_args()


def main(yaml_path: Path) -> None:
    controller = DemoMpcController(yaml_path)
    run_interactive(controller, KEY_COMMANDS, CONTROL.viewer_fps, "MPC Controller")


if __name__ == "__main__":
    args = parse_args()
    main(args.yaml)
